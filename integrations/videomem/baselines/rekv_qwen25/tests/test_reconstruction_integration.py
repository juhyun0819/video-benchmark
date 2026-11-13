"""Real tiny Qwen CPU continuation through the lossless archive/state stack."""

import copy

import pytest
import torch
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel

from rekv_qwen25.archive import KVArchiveWriter, load_block
from rekv_qwen25.attention import ReKVSession
from rekv_qwen25.state_io import flatten_state, reconstruct_long_state, unflatten_state


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model(dtype=torch.float32):
    torch.manual_seed(21056)
    config = Qwen2_5_VLTextConfig(
        vocab_size=64, hidden_size=24, intermediate_size=40,
        num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1,
        rope_scaling={"type": "mrope", "mrope_section": [1, 2, 3]},
        rope_theta=10000.0, max_position_embeddings=1024,
    )
    config._attn_implementation = "sdpa"
    return Qwen2_5_VLTextModel(config).to(dtype=dtype).eval()


def positions(length):
    indices = torch.arange(length)
    base = (indices // 3) * 5
    return torch.stack((base, base + indices % 3, base + (indices % 3)//2)).unsqueeze(1)


def step(network, session, embeddings, pos):
    session.set_positions(pos)
    return network(inputs_embeds=embeddings, position_ids=pos,
                   attention_mask={"full_attention": None}, use_cache=False).last_hidden_state


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_archived_suffix_and_original_sinks_reconstruct_exact_continuation(tmp_path, dtype):
    original = model(dtype)
    resumed_model = copy.deepcopy(original)
    continuous = ReKVSession(original, n_init=2, n_local=7)
    resumed = ReKVSession(resumed_model, n_init=2, n_local=7)
    embeddings = torch.randn(1, 23, 24, dtype=dtype)
    pos = positions(23)
    config = dict(checkpoint="tiny-deterministic-Qwen-fixture", policy=continuous.policy,
                  upstream=continuous.upstream, n_init=2, n_local=7)
    prefix_writer = KVArchiveWriter(tmp_path/"prefix", config)
    writer = KVArchiveWriter(tmp_path/"frames", config)
    try:
        step(original, continuous, embeddings[:, :2], pos[:, :, :2])
        prefix_flat, descriptor = flatten_state(continuous.export_state())
        prefix_writer.commit(0, prefix_flat, dict(frame_ids=[0], timestamps=[0.0],
                             resume_state=descriptor, scope="text prefix only"))
        # Five causal three-token frames, with atomic raw-KV blocks only. The
        # complete resume state is deliberately NOT included in these blocks.
        for frame in range(5):
            start, end = 2+frame*3, 2+(frame+1)*3
            step(original, continuous, embeddings[:, start:end], pos[:, :, start:end])
            payload = dict(continuous.latest_kv, position_ids=pos[:, :, start:end])
            writer.commit(frame, payload, dict(frame_ids=[frame], timestamps=[frame*1.25]))
        expected = continuous.export_state()
        prefix_tensors, prefix_meta = load_block(prefix_writer.root, 0, prefix_writer.config_id)
        prefix_state = unflatten_state(prefix_tensors, prefix_meta["resume_state"])
        sinks = {f"layer_{i:02d}.{kind}": state[f"init_{kind}"]
                 for i, state in enumerate(prefix_state["states"]) for kind in ("key", "value")}
        blocks = [load_block(writer.root, frame, writer.config_id, cutoff_timestamp=5.0)[0]
                  for frame in (2, 3, 4)]
        joined = {name: torch.cat([block[name] for block in blocks],
                                  dim=-1 if name == "position_ids" else -2)
                  for name in blocks[0]}
        reconstructed = reconstruct_long_state(expected, joined, sinks, total_seen=17)
        assert reconstructed["native_post_rope_cache"] is None
        for left, right in zip(expected["states"], reconstructed["states"]):
            assert set(left) == set(right)
            for name in left:
                torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)
        resumed.import_state(reconstructed)
        for frame in (5, 6):
            start, end = 2+frame*3, 2+(frame+1)*3
            reference = step(original, continuous, embeddings[:, start:end], pos[:, :, start:end])
            actual = step(resumed_model, resumed, embeddings[:, start:end], pos[:, :, start:end])
            torch.testing.assert_close(reference, actual, rtol=0, atol=0)
            for name in continuous.latest_kv:
                torch.testing.assert_close(continuous.latest_kv[name], resumed.latest_kv[name], rtol=0, atol=0)
        with pytest.raises(ValueError, match="future"):
            load_block(writer.root, 4, writer.config_id, cutoff_timestamp=3.75)
        too_short = {name: value[..., -6:] if name == "position_ids" else value[:, :, -6:]
                     for name, value in joined.items()}
        with pytest.raises(ValueError, match="at least"):
            reconstruct_long_state(expected, too_short, sinks, total_seen=17)
    finally:
        continuous.restore()
        resumed.restore()


@torch.inference_mode()
def test_native_mask_dispatch_matches_prefix_cached_chunk_and_single_token():
    """Guard the CUDA-discovered bug: equivalent float/bool masks are not enough.

    The wrapper must preserve native None/is_causal dispatch as well as tensor
    dtype and values; otherwise CUDA can choose a different BF16 SDPA kernel.
    """
    native = model()
    adapted = copy.deepcopy(native)
    session = ReKVSession(adapted, n_init=2, n_local=16)
    cache = DynamicCache()
    embeddings, pos = torch.randn(1, 6, 24), positions(6)
    native_masks, adapted_masks = [], []

    def record_native(module, args, kwargs):
        value = kwargs["attention_mask"]
        native_masks.append(None if value is None else value.clone())

    handle = native.layers[0].self_attn.register_forward_pre_hook(record_native, with_kwargs=True)
    original = session._originals[0]

    def record_adapted(*args, **kwargs):
        value = kwargs["attention_mask"]
        adapted_masks.append(None if value is None else value.clone())
        return original(*args, **kwargs)

    session._originals[0] = record_adapted
    try:
        start = 0
        for count in (2, 3, 1):
            end = start+count
            expected = native(inputs_embeds=embeddings[:, start:end], position_ids=pos[:, :, start:end],
                              attention_mask=torch.ones(1, end, dtype=torch.int64),
                              past_key_values=cache, use_cache=True).last_hidden_state
            actual = step(adapted, session, embeddings[:, start:end], pos[:, :, start:end])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            start = end
        assert native_masks[0] is adapted_masks[0] is None
        assert native_masks[1].dtype == adapted_masks[1].dtype == torch.bool
        torch.testing.assert_close(native_masks[1], adapted_masks[1], rtol=0, atol=0)
        assert native_masks[2] is adapted_masks[2] is None
    finally:
        handle.remove()
        session._originals[0] = original
        session.restore()


@torch.inference_mode()
def test_short_chunk_shapes_preserve_causal_semantics_in_float64():
    """Separate semantic alignment from BF16 CUDA chunk-size sensitivity."""
    frozen = model(torch.float64)
    embeddings, pos = torch.randn(1, 50, 24, dtype=torch.float64), positions(50)
    reference_hidden, reference_kv = None, None
    for frames_per_call in (1, 4, 8, 16):
        network = copy.deepcopy(frozen)
        session = ReKVSession(network, n_init=2, n_local=64)
        hidden_parts, raw_parts = [], {}
        try:
            step(network, session, embeddings[:, :2], pos[:, :, :2])
            for frame in range(0, 16, frames_per_call):
                start, end = 2+frame*3, 2+(frame+frames_per_call)*3
                hidden_parts.append(step(network, session, embeddings[:, start:end], pos[:, :, start:end]))
                for name, value in session.latest_kv.items():
                    raw_parts.setdefault(name, []).append(value)
            hidden = torch.cat(hidden_parts, dim=1)
            raw = {name: torch.cat(parts, dim=-2) for name, parts in raw_parts.items()}
            if reference_hidden is None:
                reference_hidden, reference_kv = hidden, raw
            else:
                torch.testing.assert_close(hidden, reference_hidden, rtol=1e-12, atol=1e-12)
                for name in raw:
                    torch.testing.assert_close(raw[name], reference_kv[name], rtol=1e-12, atol=1e-12)
        finally:
            session.restore()
