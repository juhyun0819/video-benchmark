"""Structural tests: real memory kernels with explicitly synthetic features."""
from copy import deepcopy
import importlib
import io
import random
import subprocess
import sys
from pathlib import Path

import pytest
import torch

torch.set_num_threads(2)


def adapter(method, budget=1_000_000, **options):
    native = importlib.import_module("baselines.streambench.adapters.native")
    config = {"method": method, "memory_budget_bytes": budget,
              "synthetic_test_mode": True, "encoder_id": "fixture-v1", **options}
    def encode(frames):
        if method == "streamforest":
            return frames
        value = (frames[0] + frames[1]) / 2
        return {"embeddings": value, "height": 2, "width": 2,
                "deepstack": [value + i for i in (1, 2, 3)]}
    result = native.NativeVisualMemory(config, encoder=encode,
        answerer=lambda snapshot, prompt: {"prediction": str(float(
            snapshot.get("embeddings", snapshot.get("features")).sum())),
            "stopped_on_eos": True})
    result.initialize()
    return result


def timestamps(start, count, day=1):
    return [{"stream_id": "subject", "day": day, "global_frame_id": i,
             "source_timestamp": float(i), "global_timestamp": float(i),
             "source_id": f"source-{day}", "source_frame_index": i}
            for i in range(start, start + count)]


def frames(start, count, tokens=4):
    generator = torch.Generator().manual_seed(90)
    base = torch.randn(tokens, 8, generator=generator)
    return [base + i / 10 for i in range(start, start + count)]


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for left, right in zip(a, b):
            assert_tree_equal(left, right)
    else:
        assert a == b


@pytest.mark.parametrize("method", ["fifo", "reservoir", "fluxmem_qwen3vl", "fluxmem_streaming"])
def test_pending_frame_day_checkpoint_continues_without_reencoding(method):
    options = {"capacity_grids": 2, "seed": 7} if method in {"fifo", "reservoir"} else {
        "short_frames": 2, "medium_frames": 2, "long_frames": None}
    continuous = adapter(method, **options)
    continuous.observe(frames(0, 3), timestamps(0, 3))
    continuous.finalize_chunk()
    state = deepcopy(continuous.state_dict())
    assert state["pending_frame"] is not None
    resumed = adapter(method, **options)
    resumed.load_state_dict(state)
    for item in (continuous, resumed):
        item.observe(frames(3, 13), timestamps(3, 13, day=2))
        item.finalize_chunk()
    assert_tree_equal(continuous.state_dict(), resumed.state_dict())
    assert continuous.answer("what?") == resumed.answer("what?")
    assert continuous.bank.stats()["total_ingested"] == 8


def test_fifo_keeps_newest_whole_groups_and_answers_do_not_mark_responses():
    method = adapter("fifo", capacity_grids=2)
    method.observe(frames(0, 8), timestamps(0, 8))
    assert [r["time_id"] for r in method.bank.snapshot()["records"]] == [2, 3]
    before = deepcopy(method.state_dict())
    first = method.answer("first?")
    assert first["answer"] == first["prediction"]
    method.answer("second?", ["A", "B"])
    assert_tree_equal(before, method.state_dict())
    assert first["query_read_bytes"] > 0
    assert first["num_visual_tokens_used"] == 8
    assert method.bank.stats()["response_count"] == 0


def test_reservoir_preserves_algorithm_r_slot_order_and_rng_after_save():
    method = adapter("reservoir", capacity_grids=3, seed=419)
    method.observe(frames(0, 50), timestamps(0, 50))
    expected = list(range(3))
    rng = random.Random(419)
    for index in range(3, 25):
        draw = rng.randrange(index + 1)
        if draw < 3:
            expected[draw] = index
    assert [grid.time_id for grid in method.bank._slots] == expected
    payload = io.BytesIO()
    torch.save(method.state_dict(), payload)
    payload.seek(0)
    resumed = adapter("reservoir", capacity_grids=3, seed=419)
    resumed.load_state_dict(torch.load(payload, weights_only=False))
    assert_tree_equal(method.state_dict(), resumed.state_dict())


def test_fifo_and_reservoir_byte_budget_derives_fixed_native_capacity():
    for name in ("fifo", "reservoir"):
        small = adapter(name, budget=30_000, seed=7)
        large = adapter(name, budget=70_000, seed=7)
        for method in (small, large):
            method.observe(frames(0, 4), timestamps(0, 4))
            capacity = method.bank._capacity()
            method.observe(frames(4, 200), timestamps(4, 200, day=2))
            assert method.bank._capacity() == capacity
            assert method.memory_bytes()["persistent_total_bytes"] <= method.memory_budget_bytes
        assert small.bank._capacity() < large.bank._capacity()


SF_CONFIG = {"st_memory_windows": [1, 2], "st_memory_tokens": [16, 4],
             "event_split_window": 2, "long_memory_tokens_per_frame": 2,
             "long_memory_tokens_quota": 8,
             "sim_weight_g": 0.4, "time_weight_a": 0.2, "merge_weight_b": 0.4}


def test_streamforest_checkpoint_preserves_raw_frame_alias_and_future_merges():
    options = {"memory_config": SF_CONFIG, "hidden_size": 8, "num_attention_heads": 2}
    method = adapter("streamforest", **options)
    method.observe(frames(0, 1, tokens=16), timestamps(0, 1))
    before_bytes = method.memory_bytes()
    saved = deepcopy(method.state_dict())
    other = adapter("streamforest", **options)
    other.load_state_dict(saved)
    manager = other.bank._manager
    assert manager.last_frame_flat.untyped_storage().data_ptr() == manager.memory_now[-1].untyped_storage().data_ptr()
    assert other.memory_bytes() == before_bytes
    for item in (method, other):
        item.observe(frames(1, 25, tokens=16), timestamps(1, 25, day=2))
    assert_tree_equal(method.state_dict(), other.state_dict())
    assert method.bank.stats()["tier_tokens"]["long"] <= 8


def test_tiny_budget_and_stateless_reference_are_explicitly_unsupported():
    from baselines.streambench.method import UnsupportedBudget
    with pytest.raises(UnsupportedBudget):
        adapter("fifo", budget=1)
    native = importlib.import_module("baselines.streambench.adapters.native")
    with pytest.raises(NotImplementedError, match="prefill"):
        native.create_method({"method": "fluxmem_reference", "memory_budget_bytes": 1_000_000})


def test_budget_overflow_rolls_back_and_never_replaces_fluxmem_with_fifo():
    from baselines.streambench.method import UnsupportedBudget
    method = adapter("fluxmem_qwen3vl", budget=9_000, short_frames=2, medium_frames=0)
    raised = False
    for index in range(0, 100, 2):
        before = deepcopy(method.state_dict())
        try:
            method.observe(frames(index, 2), timestamps(index, 2))
        except UnsupportedBudget:
            assert_tree_equal(before, method.state_dict())
            raised = True
            break
    assert raised
    assert method.bank.long_capacity is None


def test_restore_rejects_different_budget_or_encoder_identity_before_mutating():
    method = adapter("fifo", capacity_grids=2)
    method.observe(frames(0, 2), timestamps(0, 2))
    for key, value in (("memory_budget_bytes", 99), ("encoder_id", "different")):
        saved = deepcopy(method.state_dict())
        saved["identity"][key] = value
        before = deepcopy(method.state_dict())
        with pytest.raises(ValueError, match="identity"):
            method.load_state_dict(saved)
        assert_tree_equal(before, method.state_dict())


def test_checkpoint_resumes_reservoir_in_a_new_process(tmp_path):
    method = adapter("reservoir", capacity_grids=3, seed=419)
    method.observe(frames(0, 13), timestamps(0, 13))
    torch.save(method.state_dict(), tmp_path / "input.pt")
    script = """
import runpy, sys, torch
t = runpy.run_path('baselines/streambench/tests/test_streambench_native.py')
m = t['adapter']('reservoir', capacity_grids=3, seed=419)
m.load_state_dict(torch.load(sys.argv[1], weights_only=False))
m.observe(t['frames'](13, 19), t['timestamps'](13, 19, day=2))
torch.save(m.state_dict(), sys.argv[2])
"""
    subprocess.run([sys.executable, "-c", script, str(tmp_path / "input.pt"), str(tmp_path / "output.pt")],
                   check=True, cwd=Path(__file__).resolve().parents[3])
    method.observe(frames(13, 19), timestamps(13, 19, day=2))
    assert_tree_equal(method.state_dict(), torch.load(tmp_path / "output.pt", weights_only=False))


def test_native_compression_and_eviction_counts_survive_checkpoints():
    method = adapter("fluxmem_qwen3vl", short_frames=2, medium_frames=0, long_frames=2)
    method.observe(frames(0, 16), timestamps(0, 16))
    assert method.compression_count > 0
    assert method.eviction_count == 6
    other = adapter("fluxmem_qwen3vl", short_frames=2, medium_frames=0, long_frames=2)
    other.load_state_dict(method.state_dict())
    assert (other.compression_count, other.eviction_count) == (method.compression_count, method.eviction_count)


def test_tensor_input_is_rejected_outside_explicit_structural_mode_before_loading(tmp_path):
    native = importlib.import_module("baselines.streambench.adapters.native")
    method = native.create_method({"method": "fifo", "memory_budget_bytes": 1_000_000,
                                   "model_path": str(tmp_path)})
    method.initialize()
    with pytest.raises(TypeError, match="synthetic"):
        method.observe(frames(0, 1), timestamps(0, 1))


def test_automatic_streamforest_quota_is_restored_into_native_manager():
    config = {"hidden_size": 8, "num_attention_heads": 2}
    method = adapter("streamforest", budget=230_000, **config)
    method.observe(frames(0, 1, tokens=729), timestamps(0, 1))
    other = adapter("streamforest", budget=230_000, **config)
    other.load_state_dict(method.state_dict())
    for item in (method, other):
        item.observe(frames(1, 89, tokens=729), timestamps(1, 89, day=2))
    assert_tree_equal(method.state_dict(), other.state_dict())


def test_pending_rgb_frame_is_owned_and_survives_original_image_close():
    import numpy as np
    from PIL import Image
    native = importlib.import_module("baselines.streambench.adapters.native")
    seen = []
    def encode(pair):
        seen.append([int(np.array(image)[0, 0, 0]) for image in pair])
        value = torch.ones(4, 8)
        return {"embeddings": value, "height": 2, "width": 2,
                "deepstack": [value.clone() for _ in range(3)]}
    config = {"method": "fifo", "memory_budget_bytes": 1_000_000,
              "capacity_grids": 1, "synthetic_test_mode": True, "encoder_id": "rgb-fixture"}
    method = native.NativeVisualMemory(config, encoder=encode).initialize()
    image = Image.new("RGB", (8, 8), (13, 0, 0))
    method.observe([image], timestamps(0, 1))
    image.close()
    restored = native.NativeVisualMemory(config, encoder=encode).initialize()
    restored.load_state_dict(method.state_dict())
    next_image = Image.new("RGB", (8, 8), (27, 0, 0))
    try:
        restored.observe([next_image], timestamps(1, 1, day=2))
    finally:
        next_image.close()
    assert seen == [[13, 27]]
    assert restored.bank.stats()["total_ingested"] == 1


def test_native_local_model_identity_detects_replaced_weights_at_same_path(tmp_path):
    native = importlib.import_module("baselines.streambench.adapters.native")
    (tmp_path / "config.json").write_text('{"architecture":"fixture"}')
    (tmp_path / "model.safetensors").write_bytes(b"original")
    before = native.local_model_fingerprint(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"modified")
    assert native.local_model_fingerprint(tmp_path) != before


def test_restore_rejects_changed_local_model_before_installing_cached_features(tmp_path):
    native = importlib.import_module("baselines.streambench.adapters.native")
    config = {"method": "fifo", "memory_budget_bytes": 1_000_000,
              "model_path": str(tmp_path)}
    (tmp_path / "config.json").write_text('{}')
    (tmp_path / "model.safetensors").write_bytes(b"old")
    method = native.create_method(config).initialize()
    state = method.state_dict()
    state["model_fingerprint"] = native.local_model_fingerprint(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"new")
    with pytest.raises(ValueError, match="model.*identity"):
        method.load_state_dict(state)


def test_qwen25_temporal_collision_rejects_sampling_before_encoding():
    method = adapter("fluxmem_streaming", short_frames=2, medium_frames=0)
    records = timestamps(0, 6)
    for row in records:
        row["source_timestamp"] = row["global_timestamp"] = row["global_frame_id"] / 5
    calls = []
    encoder = method._encoder
    def counted(pair):
        calls.append(1)
        return encoder(pair)
    method._encoder = counted
    before = deepcopy(method.state_dict())
    with pytest.raises(ValueError, match="temporal_tokens_per_second=2"):
        method.observe(frames(0, 6), records)
    assert calls == []
    assert_tree_equal(method.state_dict(), before)
