"""FIFO policy checks with known temporal grids; no model or GPU is needed."""
from copy import deepcopy

import pytest
import torch

from fifo_videomem.memory import Qwen3FIFOMemory


def payload(index, *, height=2, width=2, dtype=torch.float32):
    features = torch.arange(height * width * 3, dtype=dtype).reshape(-1, 3) + index * 100
    return {"embeddings": features, "height": height, "width": width,
            "time_id": index, "timestamp_seconds": index * 2.0,
            "metadata": {"memory_id": f"pair_{index}", "nested": {"value": index}},
            "deepstack": [features + (level + 1) * 1000 for level in range(3)]}


def assert_snapshot_equal(left, right):
    assert left["records"] == right["records"]
    assert left["stats"] == right["stats"]
    assert torch.equal(left["embeddings"], right["embeddings"])
    assert torch.equal(left["positions"], right["positions"])
    for one, two in zip(left["deepstack"], right["deepstack"]):
        assert torch.equal(one, two)


@pytest.mark.parametrize("config", [
    {}, {"memory_frames": 2, "memory_tokens": 4}, {"memory_frames": 0},
    {"memory_frames": 1}, {"memory_frames": 3}, {"memory_frames": True},
    {"memory_frames": 4.0}, {"memory_tokens": 0}, {"memory_tokens": -1},
    {"memory_tokens": True}, {"memory_tokens": 8.0},
])
def test_constructor_requires_one_valid_budget(config):
    with pytest.raises(ValueError):
        Qwen3FIFOMemory(**config)


def test_latest_complete_grids_survive_before_advance():
    bank = Qwen3FIFOMemory(memory_frames=6)
    for index in range(7):
        bank.ingest(**payload(index))
        snapshot = bank.snapshot()
        expected_ids = list(range(max(0, index - 2), index + 1))
        assert [record["time_id"] for record in snapshot["records"]] == expected_ids
        assert snapshot["stats"]["retained_frames"] == 2 * len(expected_ids)
        assert snapshot["stats"]["retained_frames"] <= 6
        assert snapshot["stats"]["evicted_grids"] == max(0, index - 2)
        assert snapshot["stats"]["pending_advance"] is True
        assert torch.equal(snapshot["embeddings"],
                           torch.cat([payload(i)["embeddings"] for i in expected_ids]))
        bank.advance()
        assert bank.stats()["pending_advance"] is False


def test_token_budget_keeps_whole_grids_and_counts_every_visual_stream():
    bank = Qwen3FIFOMemory(memory_tokens=10)
    for index in range(5):
        bank.ingest(**payload(index))
        assert bank.stats()["total_tokens"] <= 10
        bank.advance()
    snapshot = bank.snapshot()
    assert [r["time_id"] for r in snapshot["records"]] == [3, 4]
    assert snapshot["stats"]["total_tokens"] == 8  # Two unused slots cannot split a grid.
    assert snapshot["stats"]["evicted_tokens"] == 12
    assert snapshot["stats"]["budget"]["capacity_grids"] == 2
    for level in range(3):
        expected = torch.cat([payload(i)["deepstack"][level] for i in (3, 4)])
        assert torch.equal(snapshot["deepstack"][level], expected)
    expected_positions = torch.tensor([
        [3, 3, 3, 3, 4, 4, 4, 4],
        [0, 0, 1, 1, 0, 0, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 1],
    ])
    assert torch.equal(snapshot["positions"], expected_positions)
    # Four float32 streams of [8, 3], plus int64 positions of [3, 8].
    assert snapshot["stats"]["retained_tensor_bytes"] == 4 * 8 * 3 * 4 + 3 * 8 * 8
    assert snapshot["stats"]["retained_cpu_bytes"] == snapshot["stats"]["retained_tensor_bytes"]


def test_too_small_token_budget_rejects_before_initializing_stream():
    bank = Qwen3FIFOMemory(memory_tokens=3)
    before = bank.snapshot()
    with pytest.raises(ValueError, match="at least one complete"):
        bank.ingest(**payload(0))
    assert_snapshot_equal(bank.snapshot(), before)
    # Failed input did not pin a shape or advance time.
    bank.ingest(**payload(0, height=1, width=2))
    assert bank.stats()["total_tokens"] == 2


def test_ingest_snapshot_and_checkpoint_never_alias_owned_memory():
    bank = Qwen3FIFOMemory(memory_frames=2)
    incoming = payload(0)
    incoming["embeddings"].requires_grad_(True)
    for tensor in incoming["deepstack"]:
        tensor.requires_grad_(True)
    bank.ingest(**incoming)
    original = bank.snapshot()
    with torch.no_grad():
        incoming["embeddings"].fill_(-1)
        for tensor in incoming["deepstack"]:
            tensor.fill_(-2)
    incoming["metadata"]["nested"]["value"] = 999
    assert_snapshot_equal(bank.snapshot(), original)
    snapshot = bank.snapshot()
    snapshot["embeddings"].fill_(-3)
    snapshot["positions"].fill_(-4)
    for tensor in snapshot["deepstack"]:
        tensor.fill_(-5)
    snapshot["records"][0]["metadata"]["nested"]["value"] = 1000
    assert_snapshot_equal(bank.snapshot(), original)
    state = bank.state_dict()
    state["grids"][0]["embeddings"].fill_(-6)
    state["grids"][0]["deepstack"][0].fill_(-7)
    state["grids"][0]["metadata"]["nested"]["value"] = 1001
    assert_snapshot_equal(bank.snapshot(), original)
    assert not original["embeddings"].requires_grad
    assert all(not tensor.requires_grad for tensor in original["deepstack"])
    bank.advance()
    bank.ingest(**payload(1))
    assert [r["time_id"] for r in original["records"]] == [0]
    assert torch.equal(original["embeddings"], payload(0)["embeddings"])


@pytest.mark.parametrize("invalid", [
    "shape", "dtype", "id_duplicate", "time_duplicate", "time_nan", "time_inf",
    "time_bool", "id_bool", "height_bool", "features_nan", "features_integer",
    "metadata", "levels_missing", "level_shape", "level_dtype", "level_nan",
])
def test_invalid_input_leaves_existing_state_unchanged(invalid):
    bank = Qwen3FIFOMemory(memory_frames=4)
    bank.ingest(**payload(0))
    bank.advance()
    before = bank.snapshot()
    incoming = payload(1)
    if invalid == "shape":
        incoming = payload(1, height=1, width=4)
    elif invalid == "dtype":
        incoming = payload(1, dtype=torch.float64)
    elif invalid == "id_duplicate":
        incoming["time_id"] = 0
    elif invalid == "time_duplicate":
        incoming["timestamp_seconds"] = 0
    elif invalid == "time_nan":
        incoming["timestamp_seconds"] = float("nan")
    elif invalid == "time_inf":
        incoming["timestamp_seconds"] = float("inf")
    elif invalid == "time_bool":
        incoming["timestamp_seconds"] = True
    elif invalid == "id_bool":
        incoming["time_id"] = True
    elif invalid == "height_bool":
        incoming["height"] = True
    elif invalid == "features_nan":
        incoming["embeddings"][0, 0] = float("nan")
    elif invalid == "features_integer":
        incoming["embeddings"] = incoming["embeddings"].long()
    elif invalid == "metadata":
        incoming["metadata"] = []
    elif invalid == "levels_missing":
        incoming["deepstack"].pop()
    elif invalid == "level_shape":
        incoming["deepstack"][1] = incoming["deepstack"][1][:-1]
    elif invalid == "level_dtype":
        incoming["deepstack"][1] = incoming["deepstack"][1].double()
    elif invalid == "level_nan":
        incoming["deepstack"][1][0, 0] = float("nan")
    with pytest.raises(ValueError):
        bank.ingest(**incoming)
    assert_snapshot_equal(bank.snapshot(), before)
    bank.ingest(**payload(1))
    assert [r["time_id"] for r in bank.snapshot()["records"]] == [0, 1]


def test_phase_and_response_contract_has_no_proactive_trigger():
    bank = Qwen3FIFOMemory(memory_frames=2)
    with pytest.raises(RuntimeError):
        bank.advance()
    with pytest.raises(RuntimeError):
        bank.mark_response()
    assert not bank.should_respond()
    observation = bank.ingest(**payload(0))
    observation["triggered"] = True
    assert bank.last_observation["triggered"] is False
    before = bank.snapshot()
    with pytest.raises(RuntimeError, match="advance"):
        bank.ingest(**payload(1))
    assert_snapshot_equal(bank.snapshot(), before)
    bank.mark_response()
    bank.mark_response()
    assert bank.stats()["response_count"] == 2
    assert not bank.should_respond()
    bank.advance()
    with pytest.raises(RuntimeError):
        bank.advance()


@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("budget", [{"memory_frames": 4}, {"memory_tokens": 9}])
def test_checkpoint_restores_evictions_phase_and_future_behavior(pending, budget):
    bank = Qwen3FIFOMemory(**budget)
    for index in range(5):
        bank.ingest(**payload(index))
        if index in (0, 2):
            bank.mark_response()
        if index < 4 or not pending:
            bank.advance()
    checkpoint = bank.state_dict()
    restored = Qwen3FIFOMemory.from_state_dict(checkpoint)
    assert_snapshot_equal(restored.snapshot(), bank.snapshot())
    assert restored.last_observation == bank.last_observation
    checkpoint["grids"][0]["embeddings"].fill_(-1)
    checkpoint["grids"][0]["deepstack"][2].fill_(-2)
    checkpoint["grids"][0]["metadata"]["nested"]["value"] = 999
    assert_snapshot_equal(restored.snapshot(), bank.snapshot())
    if pending:
        bank.advance()
        restored.advance()
    bank.ingest(**payload(5))
    restored.ingest(**payload(5))
    assert_snapshot_equal(restored.snapshot(), bank.snapshot())
    assert [r["time_id"] for r in restored.snapshot()["records"]] == [4, 5]


def test_empty_checkpoint_round_trip():
    bank = Qwen3FIFOMemory(memory_tokens=8)
    restored = Qwen3FIFOMemory.from_state_dict(bank.state_dict())
    assert_snapshot_equal(bank.snapshot(), restored.snapshot())
    restored.ingest(**payload(0))
    assert bank.stats()["total_tokens"] == 0


@pytest.mark.parametrize("invalid", ["budget", "count", "tokens", "pending", "response", "version"])
def test_checkpoint_rejects_inconsistent_state(invalid):
    bank = Qwen3FIFOMemory(memory_frames=4)
    for index in range(2):
        bank.ingest(**payload(index))
        bank.advance()
    checkpoint = deepcopy(bank.state_dict())
    if invalid == "budget":
        checkpoint["config"]["memory_frames"] = 2
    elif invalid == "count":
        checkpoint["evicted_grids"] = 1
    elif invalid == "tokens":
        checkpoint["total_encoded_tokens"] = 7
    elif invalid == "pending":
        checkpoint["pending_advance"] = 1
    elif invalid == "response":
        checkpoint["last_response_time_id"] = 0
    elif invalid == "version":
        checkpoint["version"] = 999
    with pytest.raises(ValueError):
        Qwen3FIFOMemory.from_state_dict(checkpoint)
