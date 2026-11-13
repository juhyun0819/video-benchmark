"""A persistent session around the unchanged official StreamForest memory.

Each arrival executes the original update and cascade. Reading a snapshot never
updates the manager. Compressed events are not assigned fabricated source IDs.
"""
from __future__ import annotations

from copy import deepcopy
import math
from numbers import Integral, Real
from typing import Any

import torch

from .upstream import load_memory_manager


_CONFIG_KEYS = {
    "st_memory_windows", "st_memory_tokens", "event_split_window",
    "long_memory_tokens_per_frame", "long_memory_tokens_quota",
    "sim_weight_g", "time_weight_a", "merge_weight_b",
}
_MANAGER_FIELDS = (
    "memory_now", "memory_short", "memory_long", "last_frame_flat",
    "frame_sim_buffer", "time_buffer", "mergecnt_buffer", "similarity_buffer",
    "long_memory_current_tokens", "long_memory_last_time_pos",
)


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def validate_memory_config(config: dict) -> dict:
    """Validate explicit official knobs and return an independent normalized copy."""
    if not isinstance(config, dict) or set(config) != _CONFIG_KEYS:
        raise ValueError("memory config must supply exactly: " + ", ".join(sorted(_CONFIG_KEYS)))
    result = deepcopy(config)
    for name in ("st_memory_windows", "st_memory_tokens"):
        if not isinstance(result[name], (list, tuple)) or len(result[name]) != 2:
            raise ValueError(f"{name} must contain two integers")
        result[name] = [_integer(value, name, 0 if name == "st_memory_windows" and i == 1 else 1)
                        for i, value in enumerate(result[name])]
    for name in ("event_split_window", "long_memory_tokens_per_frame", "long_memory_tokens_quota"):
        result[name] = _integer(result[name], name, 1)
    if not result["st_memory_tokens"][0] >= result["st_memory_tokens"][1] >= result["long_memory_tokens_per_frame"]:
        raise ValueError("token counts must satisfy now >= short >= long per frame")
    if result["long_memory_tokens_quota"] < result["event_split_window"] * result["long_memory_tokens_per_frame"]:
        raise ValueError("long quota must fit one maximum-size event (split window * long tokens)")
    for name in ("sim_weight_g", "time_weight_a", "merge_weight_b"):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
        result[name] = float(value)
    if sum(result[name] for name in ("sim_weight_g", "time_weight_a", "merge_weight_b")) <= 0:
        raise ValueError("at least one merge penalty weight must be positive")
    return result


def _copy_state(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device).clone()
    if isinstance(value, list):
        return [_copy_state(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_state(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _copy_state(item, device) for key, item in value.items()}
    return deepcopy(value)


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)


class StreamForestMemory:
    """Persist the official manager across single-frame arrivals and questions."""

    _STATE_VERSION = 1

    def __init__(self, hidden_size: int, num_attention_heads: int, *, config: dict):
        self.hidden_size = _integer(hidden_size, "hidden_size", 1)
        self.num_attention_heads = _integer(num_attention_heads, "num_attention_heads", 1)
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.config = validate_memory_config(config)
        self._manager = load_memory_manager()(self.hidden_size, self.num_attention_heads,
                                              **deepcopy(self.config))
        self._feature_shape = None
        self._dtype = self._device = None
        self._last_time_id = self._last_timestamp = None
        self._last_metadata = None
        self._total_ingested = 0
        self._failed = False

    def _validate_input(self, features, time_id, timestamp_seconds, metadata):
        if self._failed:
            raise RuntimeError("memory update previously failed; start or restore a session")
        if (not isinstance(features, torch.Tensor) or features.ndim != 2
                or features.shape[1] != self.hidden_size
                or features.shape[0] < self.config["st_memory_tokens"][0]):
            raise ValueError("features must have shape [tokens >= now tokens, hidden_size]")
        if not features.is_floating_point() or not bool(torch.isfinite(features).all()):
            raise ValueError("features must contain finite floating point values")
        if self._feature_shape is not None:
            if tuple(features.shape) != self._feature_shape:
                raise ValueError("all frames must share their feature shape")
            if features.dtype != self._dtype or features.device != self._device:
                raise ValueError("all frames must share dtype and device")
        time_id = _integer(time_id, "time_id")
        if self._last_time_id is not None and time_id <= self._last_time_id:
            raise ValueError("time_id must strictly increase")
        if (isinstance(timestamp_seconds, bool) or not isinstance(timestamp_seconds, Real)
                or not math.isfinite(timestamp_seconds) or timestamp_seconds < 0):
            raise ValueError("timestamp_seconds must be finite and nonnegative")
        if self._last_timestamp is not None and timestamp_seconds <= self._last_timestamp:
            raise ValueError("timestamp_seconds must strictly increase")
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a dictionary")

    @torch.no_grad()
    def ingest(self, features: torch.Tensor, *, time_id: int,
               timestamp_seconds: float, metadata: dict) -> dict:
        self._validate_input(features, time_id, timestamp_seconds, metadata)
        owned = features.detach().clone()
        owned_metadata = deepcopy(metadata)
        try:
            self._manager.update(owned)
            # Official get_memory_tokens cascades short/event/long memory. It
            # must run on every arrival, not only when a question happens.
            current = self._manager.get_memory_tokens()
            if not bool(torch.isfinite(current).all()):
                raise RuntimeError("official StreamForest memory produced nonfinite features")
        except Exception:
            self._failed = True
            raise
        self._feature_shape = tuple(features.shape)
        self._dtype, self._device = features.dtype, features.device
        self._last_time_id, self._last_timestamp = int(time_id), float(timestamp_seconds)
        self._last_metadata = owned_metadata
        self._total_ingested += 1
        return {"time_id": self._last_time_id, "timestamp_seconds": self._last_timestamp,
                "total_ingested": self._total_ingested}

    def _manager_state(self):
        return {name: getattr(self._manager, name) for name in _MANAGER_FIELDS}

    def stats(self) -> dict:
        tier_tokens = {name: sum(t.numel() // self.hidden_size for t in getattr(self._manager, "memory_" + name))
                       for name in ("long", "short", "now")}
        logical_bytes = storage_bytes = cpu_bytes = 0
        seen_storages = set()
        for tensor in _tensors(self._manager_state()):
            logical_bytes += tensor.numel() * tensor.element_size()
            storage = tensor.untyped_storage()
            identity = (str(tensor.device), storage.data_ptr())
            if identity not in seen_storages:
                seen_storages.add(identity)
                storage_bytes += storage.nbytes()
                if tensor.device.type == "cpu":
                    cpu_bytes += storage.nbytes()
        now, short = self.config["st_memory_windows"]
        now_tokens, short_tokens = self.config["st_memory_tokens"]
        bound = (now * now_tokens + (short + self.config["event_split_window"] - 1) * short_tokens
                 + self.config["long_memory_tokens_quota"])
        return {"total_tokens": sum(tier_tokens.values()), "tier_tokens": tier_tokens,
                "tier_records": {name: len(getattr(self._manager, "memory_" + name))
                                 for name in ("long", "short", "now")},
                "total_ingested": self._total_ingested,
                "retained_tensor_bytes": storage_bytes, "retained_cpu_bytes": cpu_bytes,
                "logical_tensor_bytes": logical_bytes,
                "frame_similarity_entries": len(self._manager.frame_sim_buffer),
                "visual_token_upper_bound": bound, "memory_config": deepcopy(self.config),
                "failed": self._failed}

    def snapshot(self) -> dict:
        if self._failed:
            raise RuntimeError("cannot answer from memory after a failed update")
        # Concatenation matches the original readout but deliberately does not
        # call its mutating get_memory_tokens() from the question path.
        parts = [tensor.reshape(1, -1, self.hidden_size)
                 for name in ("long", "short", "now")
                 for tensor in getattr(self._manager, "memory_" + name)]
        features = (torch.cat(parts, dim=1).detach().clone() if parts
                    else torch.empty((1, 0, self.hidden_size)))
        observed = None if self._last_time_id is None else {
            "time_id": self._last_time_id, "timestamp_seconds": self._last_timestamp,
            "metadata": deepcopy(self._last_metadata)}
        return {"features": features, "stats": self.stats(), "observed_time": observed,
                "time_id": self._last_time_id, "timestamp_seconds": self._last_timestamp,
                "metadata": deepcopy(self._last_metadata)}

    def state_dict(self) -> dict:
        """Save the bounded bank, not a complete video/session replay checkpoint."""
        if self._failed:
            raise RuntimeError("cannot checkpoint memory after a failed update")
        return {"kind": "streamforest_official_memory", "version": self._STATE_VERSION,
                "hidden_size": self.hidden_size, "num_attention_heads": self.num_attention_heads,
                "config": deepcopy(self.config), "manager": _copy_state(self._manager_state(), "cpu"),
                "feature_shape": self._feature_shape, "total_ingested": self._total_ingested,
                "last_time_id": self._last_time_id, "last_timestamp": self._last_timestamp,
                "last_metadata": deepcopy(self._last_metadata)}

    @classmethod
    def from_state_dict(cls, state: dict, *, device="cpu") -> "StreamForestMemory":
        """Restore trusted bank state directly; the caller restores stream position."""
        if (not isinstance(state, dict) or state.get("kind") != "streamforest_official_memory"
                or type(state.get("version")) is not int or state["version"] != cls._STATE_VERSION):
            raise ValueError("unsupported StreamForest memory checkpoint")
        bank = cls(state.get("hidden_size"), state.get("num_attention_heads"), config=state.get("config"))
        manager = state.get("manager")
        if not isinstance(manager, dict) or set(manager) != set(_MANAGER_FIELDS):
            raise ValueError("checkpoint must contain the complete official manager state")
        total = _integer(state.get("total_ingested"), "total_ingested")
        copied = _copy_state(manager, device)
        for name in _MANAGER_FIELDS:
            setattr(bank._manager, name, copied[name])
        bank._total_ingested = total
        if total:
            shape = state.get("feature_shape")
            if (not isinstance(shape, (tuple, list)) or len(shape) != 2
                    or shape[1] != bank.hidden_size):
                raise ValueError("checkpoint feature_shape is invalid")
            tokens = _integer(shape[0], "feature tokens", bank.config["st_memory_tokens"][0])
            raw = bank._manager.last_frame_flat
            if not isinstance(raw, torch.Tensor) or tuple(raw.shape) != (tokens * bank.hidden_size,):
                raise ValueError("checkpoint last_frame_flat has invalid shape")
            bank._feature_shape = (tokens, bank.hidden_size)
            bank._dtype, bank._device = raw.dtype, raw.device
            # Validate last observed metadata through the normal input checks.
            bank._validate_input(raw.reshape(bank._feature_shape), state.get("last_time_id"),
                                 state.get("last_timestamp"), state.get("last_metadata"))
            bank._last_time_id = int(state["last_time_id"])
            bank._last_timestamp = float(state["last_timestamp"])
            bank._last_metadata = deepcopy(state["last_metadata"])
        elif (state.get("feature_shape") is not None or state.get("last_time_id") is not None
              or state.get("last_timestamp") is not None or state.get("last_metadata") is not None
              or bank._manager.last_frame_flat is not None):
            raise ValueError("empty checkpoint contains nonempty stream state")
        bank._validate_restored_manager()
        return bank

    def _validate_restored_manager(self):
        manager = self._manager
        for name in ("memory_now", "memory_short", "memory_long", "frame_sim_buffer",
                     "time_buffer", "mergecnt_buffer", "similarity_buffer"):
            if not isinstance(getattr(manager, name), list):
                raise ValueError(f"checkpoint {name} must be a list")
        for name, tokens in (("now", self.config["st_memory_tokens"][0]),
                             ("short", self.config["st_memory_tokens"][1]),
                             ("long", self.config["long_memory_tokens_per_frame"])):
            for tensor in getattr(manager, "memory_" + name):
                if (not isinstance(tensor, torch.Tensor) or tensor.ndim != 3
                        or tensor.shape[0] < 1 or (name != "long" and tensor.shape[0] != 1)
                        or tuple(tensor.shape[1:]) != (tokens, self.hidden_size)
                        or tensor.dtype != self._dtype or tensor.device != self._device):
                    raise ValueError("checkpoint memory tier has invalid shape/dtype/device")
        for tensor in _tensors(self._manager_state()):
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("checkpoint contains nonfinite tensors")
        events = len(manager.memory_long)
        if (len(manager.time_buffer) != events or len(manager.mergecnt_buffer) != events
                or len(manager.similarity_buffer) != max(0, events - 1)
                or len(manager.frame_sim_buffer) != max(0, len(manager.memory_now) + len(manager.memory_short) - 1)):
            raise ValueError("checkpoint official memory buffers are inconsistent")
        stats = self.stats()
        if (len(manager.memory_now) > self.config["st_memory_windows"][0]
                or len(manager.memory_short) >= sum((self.config["st_memory_windows"][1], self.config["event_split_window"]))
                or stats["tier_tokens"]["long"] > self.config["long_memory_tokens_quota"]
                or manager.long_memory_current_tokens != stats["tier_tokens"]["long"]
                or bool(stats["total_tokens"]) != bool(self._total_ingested)):
            raise ValueError("checkpoint violates the completed-arrival memory bounds")
