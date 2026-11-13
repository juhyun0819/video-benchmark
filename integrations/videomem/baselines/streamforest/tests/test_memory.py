"""Prefix parity with the real official manager and persistent-bank contracts."""
from copy import deepcopy

import pytest
import torch

from streamforest_videomem.memory import StreamForestMemory, validate_memory_config
from streamforest_videomem.upstream import load_memory_manager


CONFIG = {"st_memory_windows": [1, 2], "st_memory_tokens": [8, 4],
          "event_split_window": 2, "long_memory_tokens_per_frame": 2,
          "long_memory_tokens_quota": 4, "sim_weight_g": 0.4,
          "time_weight_a": 0.2, "merge_weight_b": 0.4}


def frame(index, *, tokens=16, dtype=torch.float32):
    # Independent, positive nondegenerate features avoid zero-norm ToMe metrics.
    return torch.rand(tokens, 8, generator=torch.Generator().manual_seed(913 + index), dtype=dtype) + 0.1


def ingest(bank, index, features=None):
    return bank.ingest(frame(index) if features is None else features, time_id=index,
                       timestamp_seconds=index * 0.5,
                       metadata={"original_time": {"seconds": index * 0.5}, "memory_id": f"f{index}"})


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for one, two in zip(left, right):
            assert_equal(one, two)
    else:
        assert left == right


@pytest.mark.parametrize("changes", [
    {"unknown": 1}, {"st_memory_windows": [0, 2]}, {"st_memory_windows": [1, -1]},
    {"st_memory_windows": [1]}, {"st_memory_tokens": [True, 4]},
    {"st_memory_tokens": [4, 8]}, {"st_memory_tokens": [8, 1]},
    {"event_split_window": 0}, {"event_split_window": 2.0},
    {"long_memory_tokens_quota": 3}, {"long_memory_tokens_per_frame": 0},
    {"sim_weight_g": float("nan")}, {"time_weight_a": -1}, {"merge_weight_b": True},
    {"sim_weight_g": 0, "time_weight_a": 0, "merge_weight_b": 0},
])
def test_memory_config_rejects_unsupported_or_unbounded_settings(changes):
    with pytest.raises(ValueError):
        validate_memory_config({**deepcopy(CONFIG), **changes})


def test_config_is_explicit_and_owned():
    with pytest.raises(ValueError):
        validate_memory_config({})
    config = deepcopy(CONFIG)
    bank = StreamForestMemory(8, 2, config=config)
    config["st_memory_windows"][0] = 100
    assert bank.config["st_memory_windows"] == [1, 2]
    assert validate_memory_config(CONFIG) == CONFIG
    with pytest.raises(ValueError, match="divisible"):
        StreamForestMemory(8, 3, config=CONFIG)


def test_every_prefix_matches_official_updates_including_event_and_long_merges():
    bank = StreamForestMemory(8, 2, config=CONFIG)
    official = load_memory_manager()(8, 2, **deepcopy(CONFIG))
    saw_long = saw_merged_event = False
    for index in range(60):
        incoming = frame(index)
        ingest(bank, index, incoming)
        with torch.no_grad():
            official.update(incoming.clone())
            expected = official.get_memory_tokens()
        snapshot = bank.snapshot()
        torch.testing.assert_close(snapshot["features"], expected, rtol=0, atol=0)
        for name in ("memory_now", "memory_short", "memory_long", "last_frame_flat",
                     "frame_sim_buffer", "time_buffer", "mergecnt_buffer", "similarity_buffer",
                     "long_memory_current_tokens", "long_memory_last_time_pos"):
            assert_equal(bank.state_dict()["manager"][name], getattr(official, name))
        stats = snapshot["stats"]
        assert stats["total_ingested"] == index + 1
        assert stats["total_tokens"] == expected.shape[1]
        assert stats["total_tokens"] <= stats["visual_token_upper_bound"] == 24
        assert stats["tier_tokens"]["long"] <= CONFIG["long_memory_tokens_quota"]
        assert stats["frame_similarity_entries"] <= 3
        assert stats["retained_tensor_bytes"] < 4096
        assert snapshot["time_id"] == index
        assert snapshot["observed_time"]["timestamp_seconds"] == index * 0.5
        assert "records" not in snapshot  # No invented attribution of compressed events.
        saw_long |= bool(official.memory_long)
        saw_merged_event |= any(value > 1 for value in official.mergecnt_buffer)
    assert saw_long and saw_merged_event


def test_snapshot_never_calls_mutating_official_readout_or_changes_state(monkeypatch):
    bank = StreamForestMemory(8, 2, config=CONFIG)
    for index in range(12):
        ingest(bank, index)
    before = bank.state_dict()

    def forbidden():
        raise AssertionError("question path called mutating get_memory_tokens")

    monkeypatch.setattr(bank._manager, "get_memory_tokens", forbidden)
    first = bank.snapshot()
    for _ in range(5):
        assert_equal(bank.snapshot(), first)
    assert_equal(bank.state_dict(), before)


def test_inputs_outputs_metadata_and_checkpoints_do_not_alias():
    bank = StreamForestMemory(8, 2, config=CONFIG)
    features = frame(0).requires_grad_(True)
    metadata = {"nested": {"seconds": 0}}
    bank.ingest(features, time_id=0, timestamp_seconds=0, metadata=metadata)
    expected = bank.snapshot()
    with torch.no_grad():
        features.fill_(-100)
    metadata["nested"]["seconds"] = 999
    assert_equal(bank.snapshot(), expected)
    snapshot = bank.snapshot()
    snapshot["features"].fill_(-200)
    snapshot["metadata"]["nested"]["seconds"] = 1000
    snapshot["observed_time"]["metadata"]["nested"]["seconds"] = 1001
    snapshot["stats"]["memory_config"]["st_memory_tokens"][0] = 1000
    assert_equal(bank.snapshot(), expected)
    checkpoint = bank.state_dict()
    checkpoint["manager"]["memory_now"][0].fill_(-300)
    checkpoint["manager"]["last_frame_flat"].fill_(-400)
    checkpoint["last_metadata"]["nested"]["seconds"] = 1002
    assert_equal(bank.snapshot(), expected)
    assert not expected["features"].requires_grad


def test_tensor_accounting_includes_last_raw_frame_and_deduplicates_aliases():
    config = {**deepcopy(CONFIG), "st_memory_tokens": [16, 4]}
    bank = StreamForestMemory(8, 2, config=config)
    ingest(bank, 0)
    # No first-tier merging: memory_now and last_frame_flat share the owned
    # 16x8 tensor's storage. Count its physical allocation once.
    stats = bank.stats()
    assert stats["retained_tensor_bytes"] == 16 * 8 * 4
    assert stats["logical_tensor_bytes"] == 2 * 16 * 8 * 4
    compressed = StreamForestMemory(8, 2, config=CONFIG)
    ingest(compressed, 0)
    assert compressed.stats()["retained_tensor_bytes"] == (16 + 8) * 8 * 4


@pytest.mark.parametrize("invalid", ["shape", "dtype", "hidden", "few_tokens", "nan",
                                      "id", "timestamp", "metadata"])
def test_invalid_arrival_leaves_state_unchanged(invalid):
    bank = StreamForestMemory(8, 2, config=CONFIG)
    ingest(bank, 0)
    before = bank.state_dict()
    args = {"features": frame(1), "time_id": 1, "timestamp_seconds": 0.5, "metadata": {}}
    if invalid == "shape":
        args["features"] = frame(1, tokens=20)
    elif invalid == "dtype":
        args["features"] = frame(1, dtype=torch.float64)
    elif invalid == "hidden":
        args["features"] = frame(1)[:, :4]
    elif invalid == "few_tokens":
        args["features"] = frame(1, tokens=4)
    elif invalid == "nan":
        args["features"][0, 0] = float("nan")
    elif invalid == "id":
        args["time_id"] = 0
    elif invalid == "timestamp":
        args["timestamp_seconds"] = 0
    elif invalid == "metadata":
        args["metadata"] = []
    with pytest.raises(ValueError):
        bank.ingest(**args)
    assert_equal(bank.state_dict(), before)
    ingest(bank, 1)
    assert bank.stats()["total_ingested"] == 2


@pytest.mark.parametrize("prefix", [0, 1, 5, 30])
def test_trusted_bank_checkpoint_continues_without_replaying_frames(prefix):
    bank = StreamForestMemory(8, 2, config=CONFIG)
    for index in range(prefix):
        ingest(bank, index)
    state = bank.state_dict()
    restored = StreamForestMemory.from_state_dict(state)
    assert_equal(restored.state_dict(), bank.state_dict())
    torch.testing.assert_close(restored.snapshot()["features"], bank.snapshot()["features"], rtol=0, atol=0)
    if prefix:
        state["manager"]["memory_now"][0].fill_(-1)
        state["last_metadata"]["original_time"]["seconds"] = 999
        assert_equal(restored.state_dict(), bank.state_dict())
        with pytest.raises(ValueError, match="strictly increase"):
            ingest(restored, prefix - 1)
    for index in range(prefix, prefix + 12):
        ingest(bank, index)
        ingest(restored, index)
        assert_equal(restored.state_dict(), bank.state_dict())
        torch.testing.assert_close(restored.snapshot()["features"], bank.snapshot()["features"], rtol=0, atol=0)


@pytest.mark.parametrize("invalid", ["version", "fields", "shape", "nonfinite", "counter", "buffer"])
def test_checkpoint_rejects_inconsistent_manager_state(invalid):
    bank = StreamForestMemory(8, 2, config=CONFIG)
    for index in range(12):
        ingest(bank, index)
    state = bank.state_dict()
    if invalid == "version":
        state["version"] = 999
    elif invalid == "fields":
        state["manager"].pop("time_buffer")
    elif invalid == "shape":
        state["manager"]["memory_now"][0] = torch.ones(1, 2, 8)
    elif invalid == "nonfinite":
        state["manager"]["memory_now"][0][0, 0, 0] = float("nan")
    elif invalid == "counter":
        state["manager"]["long_memory_current_tokens"] += 1
    elif invalid == "buffer":
        state["manager"]["frame_sim_buffer"].append(torch.tensor(1.0))
    with pytest.raises(ValueError):
        StreamForestMemory.from_state_dict(state)
