"""Query-independent ReKV observation with explicit Qwen2.5 three-axis RoPE.

This is an adaptation, not an upstream Qwen backend. Up to ``n_local`` tokens
the original HF attention is used, including its post-RoPE DynamicCache. Raw
archive keys are captured at k_proj, never obtained by inverting that cache.
Beyond that boundary, untouched upstream ReKV TorchMultiStage attention mixes
locally rebased mRoPE keys with unrotated initial sink keys in one softmax.

The upstream chunk-END sink exposure rule is intentional: at the first window
crossing an initial token can occur in both branches for an early query in the
chunk. Consequently chunk boundaries are part of the observation protocol.
The caller must use its fixed frame chunking, absolute mRoPE positions, eval(),
and inference_mode(). This module neither retrieves nor answers questions.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import torch
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb,
)


_POLICY = {
    "name": "rekv_qwen25_mrope_observation_v1",
    "raw_key_boundary": "k_proj_before_mrope",
    "short_prefix": "original_hf_attention_with_dynamic_cache",
    "short_mask": "native_transformers_create_causal_mask",
    "local_rope": "absolute_three_axis_positions_minus_first_retained_position",
    "global_query_rope": "all_three_axes_at_n_local_minus_one",
    "global_key_rope": "none_initial_keys_are_raw",
    "sink_exposure": "min(n_init,max(chunk_end_total_tokens-n_local,0))",
    "sink_mask": "all_exposed_init_tokens_for_every_query_in_chunk",
    "softmax": "official_torch_joint_local_and_sink_native_dtype",
}


def _official_attention():
    """Import only the unmodified CPU-capable upstream implementation."""
    root = Path(__file__).resolve().parents[2] / "streambench/vendor/rekv"
    provenance = json.loads((root / "provenance.json").read_text())
    selected = (
        "model/attention/dot_production_attention/base.py",
        "model/attention/dot_production_attention/torch_impl.py",
    )
    records = {item["path"]: item for item in provenance["files"]}
    verified = {}
    for relative in selected:
        record = records[relative]
        raw = (root / relative).read_bytes()
        if record.get("trailing_newline_added"):
            raw = raw[:-1]
        digest = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if digest != record["git_blob"]:
            raise RuntimeError(f"Modified official ReKV source: {relative}")
        verified[relative] = digest
    name = "_rekv_qwen25_official_dot_attention"
    directory = str(root / "model/attention/dot_production_attention")
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [directory]
        sys.modules[name] = package
    elif list(sys.modules[name].__path__) != [directory]:
        raise RuntimeError("Official ReKV import namespace collision")
    module = importlib.import_module(f"{name}.torch_impl")
    return module.TorchMultiStageDotProductionAttention, {
        "repository": provenance["repository"],
        "revision": provenance["revision"],
        "verified_git_blobs": verified,
        "implementation": "TorchMultiStageDotProductionAttention",
    }


def _cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", copy=True).contiguous()


def _rotate(attention, tensor: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    cos, sin = attention.rotary_emb(tensor, positions)
    rotated, _ = apply_multimodal_rotary_pos_emb(
        tensor, tensor, cos, sin, attention.rope_scaling["mrope_section"]
    )
    return rotated


def _language_model(model):
    if hasattr(model, "layers") and hasattr(model, "rotary_emb"):
        return model
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model
    if hasattr(model, "language_model"):
        return model.language_model
    raise TypeError("Expected Qwen2.5-VL model or its language_model")


class ReKVSession:
    """Install bounded, batch-one streaming observation on a Qwen text model.

    ``set_positions([3, 1, chunk_tokens])`` starts each call. Then invoke the
    language model with those positions, ``inputs_embeds``, ``use_cache=False``
    and ``attention_mask={'full_attention': None}``. ``latest_kv`` contains the
    current call's CPU raw tensors, indexed as ``layer_00.key``/``.value`` with
    shape [1, kv_heads, chunk_tokens, head_dim]. ``export_state`` is a nested
    dict containing CPU tensors plus metadata, suitable for torch.save. It is
    valid only between completed calls. ``import_state`` restores it exactly,
    including native post-RoPE keys while still in the short-prefix phase.
    """

    def __init__(self, model, n_init: int, n_local: int = 15000):
        self.model = _language_model(model)
        if self.model.training:
            raise ValueError("ReKV observation requires model.eval()")
        if n_local <= 0 or not 0 <= n_init <= n_local:
            raise ValueError("Require 0 <= n_init <= n_local and n_local > 0")
        self.n_init, self.n_local = int(n_init), int(n_local)
        self.attentions = [layer.self_attn for layer in self.model.layers]
        if not self.attentions:
            raise ValueError("No attention layers")
        for i, attention in enumerate(self.attentions):
            if attention.layer_idx != i or attention.sliding_window is not None:
                raise ValueError("Require ordered full-attention Qwen2.5 layers")
            rope_type = attention.rope_scaling.get(
                "rope_type", attention.rope_scaling.get("type", "default")
            )
            if rope_type not in ("default", "mrope"):
                raise ValueError("Rebasing requires fixed-frequency default mRoPE")
            if getattr(attention, "_rekv_session_owner", None) is not None:
                raise RuntimeError("Attention already has a ReKVSession")
        self._attention_class, self.upstream = _official_attention()
        self.policy = dict(_POLICY)
        self.total_seen = 0
        self.latest_kv: dict[str, torch.Tensor] = {}
        self._states: list[dict[str, torch.Tensor] | None] = [None] * len(self.attentions)
        self._native_cache: DynamicCache | None = DynamicCache()
        self._positions: torch.Tensor | None = None
        self._short_mask: torch.Tensor | None = None
        self._next_layer = 0
        self._failed = False
        self._installed = False
        self._originals = [attention.forward for attention in self.attentions]
        self.install()

    @property
    def seen_tokens(self):
        return self.total_seen

    @property
    def layer_states(self):
        """Bounded live state; callers must not mutate the returned tensors."""
        return tuple(self._states)

    def install(self):
        if self._installed:
            return self
        for index, attention in enumerate(self.attentions):
            if getattr(attention, "_rekv_session_owner", None) is not None:
                raise RuntimeError("Attention already has a ReKVSession")

            def forward(module, hidden_states, attention_mask=None, position_ids=None,
                        past_key_values=None, output_attentions=False, use_cache=False,
                        cache_position=None, position_embeddings=None, _index=index, **kwargs):
                try:
                    return self._forward(
                        _index, hidden_states, attention_mask, position_ids,
                        past_key_values, output_attentions, position_embeddings, kwargs,
                    )
                except Exception:
                    self._failed = True
                    raise

            attention.forward = types.MethodType(forward, attention)
            attention._rekv_session_owner = self
        self._installed = True
        return self

    def restore(self):
        if self._installed:
            for attention, original in zip(self.attentions, self._originals):
                attention.forward = original
                del attention._rekv_session_owner
            self._installed = False

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.restore()

    def set_positions(self, position_ids: torch.Tensor):
        if not self._installed or self._failed:
            raise RuntimeError("Session is uninstalled or failed; restore/import before reuse")
        if self._positions is not None:
            raise RuntimeError("Previous model call has not completed all layers")
        if position_ids.ndim != 3 or tuple(position_ids.shape[:2]) != (3, 1):
            raise ValueError("position_ids must have shape [3, 1, chunk_tokens]")
        if position_ids.dtype != torch.int64:
            raise ValueError("position_ids must use int64")
        if not 0 < position_ids.shape[-1] <= self.n_local:
            raise ValueError("A chunk must contain 1..n_local tokens")
        self._positions = position_ids.detach().clone()
        self._short_mask = None
        self._next_layer = 0
        self.latest_kv = {}
        if self.total_seen + position_ids.shape[-1] > self.n_local:
            # The native cache is post-RoPE and is never used by the long path.
            self._native_cache = None

    def _forward(self, index, hidden, attention_mask, position_ids, external_cache,
                 output_attentions, position_embeddings, kwargs):
        if self._failed or self._positions is None or index != self._next_layer:
            raise RuntimeError("Call set_positions once per ordered model forward")
        attention = self.attentions[index]
        if attention.training or torch.is_grad_enabled():
            raise RuntimeError("Use eval() and torch.no_grad()/inference_mode()")
        if hidden.shape[:2] != (1, self._positions.shape[-1]):
            raise ValueError("Only batch one with the declared chunk length is supported")
        if external_cache is not None:
            raise ValueError("Pass use_cache=False; ReKVSession owns its cache")
        if attention_mask is not None:
            raise ValueError("Pass attention_mask={'full_attention': None}; session builds its mask")
        if output_attentions:
            raise ValueError("Observation does not return attention probability matrices")
        positions = self._positions.to(hidden.device)
        if position_ids is not None and not torch.equal(position_ids, positions):
            raise ValueError("Model position_ids differ from set_positions")
        length = hidden.shape[1]
        end = self.total_seen + length
        state = self._states[index]

        if end <= self.n_local:
            # Capture actual native projections, before any rotary operation.
            projections = {}

            def capture(name):
                def hook(module, inputs, output):
                    projections[name] = output.detach().view(
                        1, length, attention.num_key_value_heads, attention.head_dim
                    ).transpose(1, 2)
                return hook

            handles = [attention.k_proj.register_forward_hook(capture("key")),
                       attention.v_proj.register_forward_hook(capture("value"))]
            query_indices = torch.arange(self.total_seen, end, device=hidden.device)
            if index == 0:
                # Compute once BEFORE layer 0 updates the shared cache. Match
                # native dispatch too: SDPA can use None/is_causal for a first
                # prefill or one-token decode, and a boolean mask otherwise.
                # A mathematically equal additive mask can choose a different
                # CUDA kernel and change BF16 values from layer 1 onward.
                self._short_mask = create_causal_mask(
                    config=attention.config, input_embeds=hidden,
                    attention_mask=torch.ones((1, end), dtype=torch.int64, device=hidden.device),
                    cache_position=query_indices, past_key_values=self._native_cache,
                    position_ids=None,
                )
            if position_embeddings is None:
                position_embeddings = attention.rotary_emb(hidden, positions)
            try:
                output, weights = self._originals[index](
                    hidden, attention_mask=self._short_mask, position_ids=positions,
                    past_key_values=self._native_cache, output_attentions=False,
                    use_cache=True, cache_position=query_indices,
                    position_embeddings=position_embeddings, **kwargs,
                )
            finally:
                for handle in handles:
                    handle.remove()
            key, value = projections["key"], projections["value"]
        else:
            query = attention.q_proj(hidden).view(
                1, length, attention.num_heads, attention.head_dim
            ).transpose(1, 2)
            key = attention.k_proj(hidden).view(
                1, length, attention.num_key_value_heads, attention.head_dim
            ).transpose(1, 2)
            value = attention.v_proj(hidden).view(
                1, length, attention.num_key_value_heads, attention.head_dim
            ).transpose(1, 2)
            local_key = torch.cat((state["key"], key), dim=-2)
            local_value = torch.cat((state["value"], value), dim=-2)
            local_positions = torch.cat((state["positions"], positions), dim=-1)
            offset = local_positions[:, :, :1]
            local_query = _rotate(attention, query, positions - offset)
            local_key_rotated = _rotate(attention, local_key, local_positions - offset)
            sink_count = min(self.n_init, max(end - self.n_local, 0))
            sink_key, sink_value = self._initial(state, key, value)
            global_query = _rotate(
                attention, query, torch.full_like(positions, self.n_local - 1)
            )
            multi = self._attention_class(query.shape, query.dtype, query.device)
            multi.append(local_query, local_key_rotated, local_value,
                         sliding_window=self.n_local)
            # Match upstream: None makes this an unrestricted sink branch,
            # even though complement_sliding_window is True.
            multi.append(global_query, sink_key[:, :, :sink_count], sink_value[:, :, :sink_count],
                         sliding_window=None, complement_sliding_window=True, end=True)
            attended, _ = multi.get_result()
            output = attention.o_proj(attended.transpose(1, 2).reshape(1, length, -1).contiguous())
            weights = None

        self.latest_kv[f"layer_{index:02d}.key"] = _cpu(key)
        self.latest_kv[f"layer_{index:02d}.value"] = _cpu(value)
        sink_key, sink_value = self._initial(state, key, value)
        if state is None:
            recent_key, recent_value, recent_positions = key, value, positions
        else:
            recent_key = torch.cat((state["key"], key), dim=-2)
            recent_value = torch.cat((state["value"], value), dim=-2)
            recent_positions = torch.cat((state["positions"], positions), dim=-1)
        # Clone slices so no retained view keeps an unbounded backing allocation.
        self._states[index] = {
            "key": recent_key[:, :, -self.n_local:].detach().clone(),
            "value": recent_value[:, :, -self.n_local:].detach().clone(),
            "positions": recent_positions[:, :, -self.n_local:].detach().clone(),
            "init_key": sink_key.detach().clone(),
            "init_value": sink_value.detach().clone(),
        }
        self._next_layer += 1
        if self._next_layer == len(self.attentions):
            self.total_seen = end
            self._positions = None
            self._short_mask = None
            self._next_layer = 0
        return output, weights

    def _initial(self, state, key, value):
        if state is None:
            return key[:, :, :self.n_init], value[:, :, :self.n_init]
        missing = self.n_init - state["init_key"].shape[-2]
        if missing:
            return (torch.cat((state["init_key"], key[:, :, :missing]), dim=-2),
                    torch.cat((state["init_value"], value[:, :, :missing]), dim=-2))
        return state["init_key"], state["init_value"]

    def _identity(self):
        return {
            "schema": 1, "n_init": self.n_init, "n_local": self.n_local,
            "layers": len(self.attentions), "policy": self.policy, "upstream": self.upstream,
            "geometry": [
                {"query_heads": a.num_heads, "kv_heads": a.num_key_value_heads,
                 "head_dim": a.head_dim, "mrope_section": a.rope_scaling["mrope_section"],
                 "rope_theta": a.config.rope_theta, "dtype": str(a.k_proj.weight.dtype)}
                for a in self.attentions
            ],
        }

    def export_state(self) -> dict[str, Any]:
        if self._failed or self._positions is not None:
            raise RuntimeError("Cannot export an incomplete or failed model call")
        native = None
        if self._native_cache is not None and self.total_seen:
            native = [tuple(_cpu(t) for t in pair) for pair in self._native_cache.to_legacy_cache()]
        return {
            **self._identity(), "total_seen": self.total_seen,
            "states": [None if state is None else {name: _cpu(t) for name, t in state.items()}
                       for state in self._states],
            "native_post_rope_cache": native,
        }

    def import_state(self, saved: dict[str, Any]):
        """Restore at a completed call boundary; model weights must also match."""
        for name, expected in self._identity().items():
            if saved.get(name) != expected:
                raise ValueError(f"Resume identity mismatch: {name}")
        total = saved["total_seen"]
        if not isinstance(total, int) or total < 0:
            raise ValueError("Invalid total_seen")
        if len(saved["states"]) != len(self.attentions):
            raise ValueError("Invalid resume layer count")
        restored = []
        for attention, state in zip(self.attentions, saved["states"]):
            if total == 0:
                if state is not None:
                    raise ValueError("Empty resume has nonempty state")
                restored.append(None)
                continue
            recent = min(total, self.n_local)
            initial = min(total, self.n_init)
            expected_shapes = {
                "key": (1, attention.num_key_value_heads, recent, attention.head_dim),
                "value": (1, attention.num_key_value_heads, recent, attention.head_dim),
                "positions": (3, 1, recent),
                "init_key": (1, attention.num_key_value_heads, initial, attention.head_dim),
                "init_value": (1, attention.num_key_value_heads, initial, attention.head_dim),
            }
            if not isinstance(state, dict) or set(state) != set(expected_shapes):
                raise ValueError("Invalid resume layer fields")
            current = {}
            for name, shape in expected_shapes.items():
                tensor = state[name]
                dtype = torch.int64 if name == "positions" else attention.k_proj.weight.dtype
                if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape or tensor.dtype != dtype:
                    raise ValueError(f"Invalid resume tensor {name}")
                current[name] = tensor.to(attention.k_proj.weight.device, copy=True)
            restored.append(current)
        native = saved.get("native_post_rope_cache")
        if 0 < total <= self.n_local:
            if native is None or len(native) != len(self.attentions):
                raise ValueError("Short-prefix resume requires native post-RoPE cache")
            pairs = []
            for attention, pair in zip(self.attentions, native):
                shape = (1, attention.num_key_value_heads, total, attention.head_dim)
                if len(pair) != 2 or any(tuple(t.shape) != shape or t.dtype != attention.k_proj.weight.dtype for t in pair):
                    raise ValueError("Invalid native post-RoPE resume cache")
                pairs.append(tuple(t.to(attention.k_proj.weight.device, copy=True) for t in pair))
            cache = DynamicCache.from_legacy_cache(tuple(pairs))
        elif total == 0:
            cache = DynamicCache()
        else:
            if native is not None:
                raise ValueError("Long-prefix resume must not retain a native cache")
            cache = None
        self._states, self._native_cache = restored, cache
        self.total_seen, self._positions, self._next_layer = total, None, 0
        self._short_mask = None
        self._failed, self.latest_kv = False, {}
        return self

    export = export_state
    load_state = import_state
