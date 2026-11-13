"""CPU scientific fidelity checks; no model downloads or CUDA execution."""

import copy
import io
import math

import pytest
import torch
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel

from rekv_qwen25.attention import ReKVSession


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_model(layers=2, backend="eager"):
    torch.manual_seed(9347)
    config = Qwen2_5_VLTextConfig(
        vocab_size=64, hidden_size=24, intermediate_size=40,
        num_hidden_layers=layers, num_attention_heads=2,
        num_key_value_heads=1, max_position_embeddings=2048,
        rope_theta=10000.0,
        rope_scaling={"type": "mrope", "mrope_section": [2, 2, 2]},
    )
    config._attn_implementation = backend
    return Qwen2_5_VLTextModel(config).eval()


def positions(length):
    """Nontrivial independent spatial axes, including within-frame resets."""
    i = torch.arange(length, dtype=torch.int64)
    base = (i // 4) * 7
    return torch.stack((base, base + (i % 4) // 2, base + i % 2)).unsqueeze(1)


def run_chunk(model, session, embeddings, pos):
    session.set_positions(pos)
    return model(inputs_embeds=embeddings, position_ids=pos,
                 attention_mask={"full_attention": None}, use_cache=False).last_hidden_state


def independent_rotate(attention, x, pos):
    """Dense reference derives mRoPE channels without production rotate helper."""
    phase = pos.float().unsqueeze(-1) * attention.rotary_emb.inv_freq.float()[None, None, None, :]
    phase = torch.cat((phase, phase), dim=-1)
    routed, start = [], 0
    for index, width in enumerate(attention.rope_scaling["mrope_section"] * 2):
        routed.append(phase[index % 3, :, :, start:start + width])
        start += width
    angle = torch.cat(routed, dim=-1).unsqueeze(1)
    cosine = angle.cos().to(x.dtype)
    sine = angle.sin().to(x.dtype)
    half = x.shape[-1] // 2
    rotated_half = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cosine + rotated_half * sine


def dense_reference(attention, hidden, pos, history, n_init, window, *, drop_overlap=False):
    """Full-history oracle, independent masks/concatenated joint softmax."""
    length = hidden.shape[1]
    q = attention.q_proj(hidden).view(1, length, attention.num_heads, attention.head_dim).transpose(1, 2)
    k = attention.k_proj(hidden).view(1, length, attention.num_key_value_heads, attention.head_dim).transpose(1, 2)
    v = attention.v_proj(hidden).view(1, length, attention.num_key_value_heads, attention.head_dim).transpose(1, 2)
    if history is None:
        all_k, all_v, all_pos = k, v, pos
    else:
        all_k = torch.cat((history[0], k), dim=-2)
        all_v = torch.cat((history[1], v), dim=-2)
        all_pos = torch.cat((history[2], pos), dim=-1)
    total = all_k.shape[-2]
    start = max(0, total - length - window)
    local_k, local_v, local_pos = all_k[:, :, start:], all_v[:, :, start:], all_pos[:, :, start:]
    offset = local_pos[:, :, :1] if total > window else torch.zeros_like(pos[:, :, :1])
    rq = independent_rotate(attention, q, pos - offset)
    rk = independent_rotate(attention, local_k, local_pos - offset)
    repeats = attention.num_heads // attention.num_key_value_heads
    rk, local_v = rk.repeat_interleave(repeats, dim=1), local_v.repeat_interleave(repeats, dim=1)
    query_index = torch.arange(total - length, total)
    key_index = torch.arange(start, total)
    distance = query_index[:, None] - key_index[None, :]
    allowed = (distance >= 0) & (distance < window)
    scores = torch.einsum("bhqd,bhkd->bhqk", rq, rk) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~allowed, -torch.inf)
    values = local_v
    count = min(n_init, max(total - window, 0))
    if count:
        sq = independent_rotate(attention, q, torch.full_like(pos, window - 1))
        sk = all_k[:, :, :count].repeat_interleave(repeats, dim=1)
        sv = all_v[:, :, :count].repeat_interleave(repeats, dim=1)
        sink_scores = torch.einsum("bhqd,bhkd->bhqk", sq, sk) / math.sqrt(q.shape[-1])
        if drop_overlap:
            sink_scores = sink_scores.masked_fill(query_index[:, None] - torch.arange(count)[None, :] < window, -torch.inf)
        scores = torch.cat((scores, sink_scores), dim=-1)
        values = torch.cat((values, sv), dim=-2)
    probabilities = torch.softmax(scores, dim=-1)
    result = torch.einsum("bhqk,bhkd->bhqd", probabilities, values)
    output = attention.o_proj(result.transpose(1, 2).reshape(1, length, -1))
    return output, (all_k, all_v, all_pos), (k, v)


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@torch.inference_mode()
def test_short_prefix_is_original_hf_and_captures_actual_raw_projections(backend):
    native = make_model(backend=backend)
    adapted = copy.deepcopy(native)
    session = ReKVSession(adapted, n_init=2, n_local=32)
    embeddings = torch.randn(1, 14, 24)
    pos = positions(14)
    cache = DynamicCache()
    start = 0
    captured = {}
    handles = []
    for layer_index, layer in enumerate(native.layers):
        for name, module in (("key", layer.self_attn.k_proj), ("value", layer.self_attn.v_proj)):
            def hook(module, inputs, output, i=layer_index, n=name):
                captured[f"layer_{i:02d}.{n}"] = output.view(1, -1, 1, 12).transpose(1, 2).clone()
            handles.append(module.register_forward_hook(hook))
    try:
        for length in (2, 4, 4, 4):
            end = start + length
            query = torch.arange(start, end)
            expected = native(
                inputs_embeds=embeddings[:, start:end], position_ids=pos[:, :, start:end],
                attention_mask=torch.ones((1, end), dtype=torch.int64), past_key_values=cache,
                cache_position=query, use_cache=True,
            ).last_hidden_state
            actual = run_chunk(adapted, session, embeddings[:, start:end], pos[:, :, start:end])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for name in captured:
                torch.testing.assert_close(session.latest_kv[name], captured[name], rtol=0, atol=0)
            start = end
        raw = session.layer_states[0]["key"]
        native_rotated = session._native_cache.to_legacy_cache()[0][0]
        assert not torch.equal(raw, native_rotated)
        assert session.seen_tokens == 14
    finally:
        for handle in handles:
            handle.remove()
        session.restore()


@pytest.mark.parametrize("n_init", [0, 2, 5])
@torch.inference_mode()
def test_window_crossing_matches_independent_dense_reference_and_is_bounded(n_init):
    model = make_model(layers=1)
    attention = model.layers[0].self_attn
    session = ReKVSession(model, n_init=n_init, n_local=8)
    hidden = torch.randn(1, 24, 24)
    pos = positions(24)
    history, start = None, 0
    for length in (2, 4, 4, 3, 4, 4, 3):
        end = start + length
        current_h, current_p = hidden[:, start:end], pos[:, :, start:end]
        expected, history, raw = dense_reference(attention, current_h, current_p, history, n_init, 8)
        session.set_positions(current_p)
        actual, _ = attention(current_h, position_ids=current_p)
        torch.testing.assert_close(actual, expected, rtol=3e-6, atol=1e-7)
        torch.testing.assert_close(session.latest_kv["layer_00.key"], raw[0], rtol=0, atol=0)
        torch.testing.assert_close(session.latest_kv["layer_00.value"], raw[1], rtol=0, atol=0)
        state = session.layer_states[0]
        assert state["key"].shape[-2] == min(end, 8)
        assert state["init_key"].shape[-2] == min(end, n_init)
        assert state["positions"].shape[-1] == min(end, 8)
        assert state["key"].untyped_storage().nbytes() == state["key"].numel() * state["key"].element_size()
        torch.testing.assert_close(state["positions"], pos[:, :, max(0, end-8):end], rtol=0, atol=0)
        start = end
    assert session._native_cache is None
    session.restore()


@torch.inference_mode()
def test_first_cross_preserves_upstream_chunk_end_sink_overlap():
    model = make_model(layers=1)
    attention = model.layers[0].self_attn
    session = ReKVSession(model, n_init=2, n_local=8)
    hidden, pos = torch.randn(1, 10, 24), positions(10)
    session.set_positions(pos[:, :, :6])
    attention(hidden[:, :6], position_ids=pos[:, :, :6])
    _, history, _ = dense_reference(attention, hidden[:, :6], pos[:, :, :6], None, 2, 8)
    expected, _, _ = dense_reference(attention, hidden[:, 6:], pos[:, :, 6:], history, 2, 8)
    deduplicated, _, _ = dense_reference(attention, hidden[:, 6:], pos[:, :, 6:], history, 2, 8, drop_overlap=True)
    session.set_positions(pos[:, :, 6:])
    actual, _ = attention(hidden[:, 6:], position_ids=pos[:, :, 6:])
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=1e-7)
    assert (actual[:, :2] - deduplicated[:, :2]).abs().max().item() > 1e-5
    session.restore()


@pytest.mark.parametrize("resume_at", [0, 6, 10, 14])
@torch.inference_mode()
def test_exact_export_reload_continuation_short_and_long(resume_at):
    continuous = make_model()
    restored_model = copy.deepcopy(continuous)
    first = ReKVSession(continuous, n_init=2, n_local=8)
    second = ReKVSession(restored_model, n_init=2, n_local=8)
    hidden, pos = torch.randn(1, 22, 24), positions(22)
    bounds = (0, 2, 6, 10, 14, 18, 22)
    for start, end in zip(bounds, bounds[1:]):
        if start == resume_at:
            # Exercise the actual tensor serialization, not shared live views.
            buffer = io.BytesIO()
            torch.save(first.export_state(), buffer)
            buffer.seek(0)
            saved = torch.load(buffer, weights_only=True)
            second.import_state(saved)
        expected = run_chunk(continuous, first, hidden[:, start:end], pos[:, :, start:end])
        if start >= resume_at:
            actual = run_chunk(restored_model, second, hidden[:, start:end], pos[:, :, start:end])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for name in first.latest_kv:
                torch.testing.assert_close(second.latest_kv[name], first.latest_kv[name], rtol=0, atol=0)
            assert first.total_seen == second.total_seen
    first.restore()
    second.restore()


@torch.inference_mode()
def test_restore_and_resume_validation():
    model = make_model()
    originals = [layer.self_attn.forward for layer in model.layers]
    session = ReKVSession(model, n_init=2, n_local=8)
    run_chunk(model, session, torch.randn(1, 2, 24), positions(2))
    saved = session.export_state()
    invalid = dict(saved, n_local=9)
    with pytest.raises(ValueError, match="n_local"):
        session.import_state(invalid)
    assert session.total_seen == 2
    invalid = dict(saved, native_post_rope_cache=None)
    with pytest.raises(ValueError, match="post-RoPE"):
        session.import_state(invalid)
    session.restore()
    session.restore()
    assert [layer.self_attn.forward for layer in model.layers] == originals
    assert "revision" in session.upstream
    assert len(session.upstream["verified_git_blobs"]) == 2


@torch.inference_mode()
def test_incomplete_call_cannot_be_exported_or_silently_restarted():
    model = make_model()
    session = ReKVSession(model, n_init=2, n_local=8)
    session.set_positions(positions(2))
    model.layers[0].self_attn(torch.randn(1, 2, 24), position_ids=positions(2))
    with pytest.raises(RuntimeError, match="incomplete"):
        session.export_state()
    with pytest.raises(RuntimeError, match="not completed"):
        session.set_positions(positions(2))
    session.restore()


@torch.inference_mode()
def test_finite_warmup_forgets_arbitrary_boundary_state_with_identical_fixed_sinks():
    """Fixture proof only: production temporal shards still require real-data parity.

    The two streams have identical frozen per-layer initial KV, absolute mRoPE
    positions and chunk boundaries. All recent boundary KV is deliberately
    wrong in the second stream. L*W subsequent tokens suffice to forget it for
    this causal local-plus-fixed-sink operator. This does not license resetting
    the sinks, changing chunking, or treating full-attention HF as finite-range.
    """
    layers, window, boundary = 3, 5, 12
    continuous = make_model(layers=layers)
    second_model = copy.deepcopy(continuous)
    first = ReKVSession(continuous, n_init=2, n_local=window)
    second = ReKVSession(second_model, n_init=2, n_local=window)
    warmup = layers * window
    length = boundary + warmup + 3
    hidden, pos = torch.randn(1, length, 24), positions(length)
    for index in range(boundary):
        run_chunk(continuous, first, hidden[:, index:index+1], pos[:, :, index:index+1])
    perturbed = first.export_state()
    for layer in perturbed["states"]:
        layer["key"] = torch.randn_like(layer["key"])
        layer["value"] = torch.randn_like(layer["value"])
    second.import_state(perturbed)
    early_difference = None
    for index in range(boundary, length):
        expected = run_chunk(continuous, first, hidden[:, index:index+1], pos[:, :, index:index+1])
        actual = run_chunk(second_model, second, hidden[:, index:index+1], pos[:, :, index:index+1])
        if index == boundary:
            early_difference = (expected - actual).abs().max().item()
        if index >= boundary + warmup - 1:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for name in first.latest_kv:
                torch.testing.assert_close(second.latest_kv[name], first.latest_kv[name], rtol=0, atol=0)
    assert early_difference > 1e-4
    first.restore()
    second.restore()
