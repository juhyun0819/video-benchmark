"""Exact Algorithm R and reservoir ownership/continuation contract checks."""
from collections import Counter
from copy import deepcopy
from itertools import combinations, product
import random

import pytest
import torch

from reservoir_videomem.memory import Qwen3ReservoirMemory


def payload(index, *, height=2, width=2, dtype=torch.float32):
    features = torch.arange(height * width * 3, dtype=dtype).reshape(-1, 3) + index * 100
    return {"embeddings": features, "height": height, "width": width,
            "time_id": index, "timestamp_seconds": index * 2.0,
            "metadata": {"memory_id": f"pair_{index}", "nested": {"value": index}},
            "deepstack": [features + (level + 1) * 1000 for level in range(3)]}


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


def ingest_all(bank, count):
    for index in range(count):
        bank.ingest(**payload(index))
        bank.advance()


def retained_ids(bank):
    return tuple(record["time_id"] for record in bank.snapshot()["records"])


@pytest.mark.parametrize("config", [
    {}, {"memory_frames": 2, "memory_tokens": 4}, {"memory_frames": 0},
    {"memory_frames": 1}, {"memory_frames": 3}, {"memory_frames": True},
    {"memory_frames": 4.0}, {"memory_tokens": 0}, {"memory_tokens": -1},
    {"memory_tokens": True}, {"memory_tokens": 8.0},
])
def test_requires_one_valid_budget(config):
    with pytest.raises(ValueError):
        Qwen3ReservoirMemory(seed=0, **config)


@pytest.mark.parametrize("seed", [-1, True, False, 1.0, None, "1"])
def test_requires_explicit_nonnegative_integer_seed(seed):
    with pytest.raises(ValueError):
        Qwen3ReservoirMemory(memory_frames=4, seed=seed)


def test_seed_has_no_default():
    with pytest.raises(TypeError):
        Qwen3ReservoirMemory(memory_frames=4)


@pytest.mark.parametrize("capacity", [1, 2, 3])
def test_exact_equiprobable_draw_paths_give_uniform_subsets(monkeypatch, capacity):
    # At each overflow, Algorithm R draws uniformly from [0, n). All products
    # of those choices are equiprobable; enumerate them instead of sampling.
    total = 4
    counts = Counter()
    for choices in product(*(range(n) for n in range(capacity + 1, total + 1))):
        bank = Qwen3ReservoirMemory(memory_frames=capacity * 2, seed=123)
        draws = iter(choices)
        expected_n = iter(range(capacity + 1, total + 1))

        def forced_draw(n):
            assert n == next(expected_n)
            return next(draws)

        monkeypatch.setattr(bank._rng, "randrange", forced_draw)
        ingest_all(bank, total)
        counts[retained_ids(bank)] += 1
        assert next(draws, None) is None
    expected_subsets = set(combinations(range(total), capacity))
    assert set(counts) == expected_subsets
    assert len(set(counts.values())) == 1


def test_snapshot_sorting_never_reorders_sampling_slots(monkeypatch):
    bank = Qwen3ReservoirMemory(memory_frames=4, seed=0)
    choices = iter((0, 1))
    monkeypatch.setattr(bank._rng, "randrange", lambda n: next(choices))
    ingest_all(bank, 3)
    assert [record["time_id"] for record in bank.state_dict()["slots"]] == [2, 1]
    snapshot = bank.snapshot()
    assert retained_ids(bank) == (1, 2)
    assert [record["time_id"] for record in bank.state_dict()["slots"]] == [2, 1]
    for level in range(3):
        assert torch.equal(snapshot["deepstack"][level],
                           torch.cat([payload(i)["deepstack"][level] for i in (1, 2)]))
    assert torch.equal(snapshot["embeddings"],
                       torch.cat([payload(i)["embeddings"] for i in (1, 2)]))
    assert torch.equal(snapshot["positions"], torch.tensor([
        [1, 1, 1, 1, 2, 2, 2, 2], [0, 0, 1, 1, 0, 0, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 1],
    ]))
    assert [r["metadata"]["nested"]["value"] for r in snapshot["records"]] == [1, 2]
    bank.ingest(**payload(3))
    assert retained_ids(bank) == (2, 3)  # Replace slot 1 containing grid 1, not sorted grid 2.


def test_rejection_and_replacement_have_distinct_counts_and_truncation():
    rejected = Qwen3ReservoirMemory(memory_frames=4, seed=5)
    ingest_all(rejected, 2)
    assert rejected.snapshot()["memory_truncated"] is False
    observation = rejected.ingest(**payload(2))
    assert observation["accepted"] is False
    assert retained_ids(rejected) == (0, 1)
    assert rejected.snapshot()["memory_truncated"] is True
    stats = rejected.stats()
    assert (stats["evicted_grids"], stats["rejected_grids"], stats["discarded_grids"]) == (0, 1, 1)
    assert (stats["evicted_tokens"], stats["rejected_tokens"], stats["discarded_tokens"]) == (0, 4, 4)
    assert stats["total_ingested"] == 3

    replaced = Qwen3ReservoirMemory(memory_frames=4, seed=0)
    ingest_all(replaced, 2)
    observation = replaced.ingest(**payload(2))
    assert observation["accepted"] is True and observation["replaced_time_id"] == 1
    assert retained_ids(replaced) == (0, 2)
    assert replaced.snapshot()["memory_truncated"] is True
    stats = replaced.stats()
    assert (stats["evicted_grids"], stats["rejected_grids"], stats["discarded_grids"]) == (1, 0, 1)


@pytest.mark.parametrize("budget", [{"memory_frames": 6}, {"memory_tokens": 14}])
def test_strict_capacity_before_answer_and_all_tensor_bytes(budget):
    bank = Qwen3ReservoirMemory(**budget, seed=19)
    for index in range(25):
        bank.ingest(**payload(index))
        snapshot = bank.snapshot()
        grids = min(index + 1, 3)
        assert len(snapshot["records"]) == grids
        assert snapshot["stats"]["retained_frames"] == grids * 2
        assert snapshot["stats"]["total_tokens"] == grids * 4
        assert snapshot["stats"]["discarded_grids"] == index + 1 - grids
        assert snapshot["stats"]["retained_tensor_bytes"] == grids * (4 * 4 * 3 * 4 + 3 * 4 * 8)
        assert snapshot["stats"]["retained_cpu_bytes"] == snapshot["stats"]["retained_tensor_bytes"]
        assert snapshot["stats"]["pending_advance"] is True
        bank.advance()


def test_too_small_budget_does_not_pin_shape_or_consume_rng():
    bank = Qwen3ReservoirMemory(memory_tokens=3, seed=8)
    before = bank.state_dict()
    with pytest.raises(ValueError, match="at least one complete"):
        bank.ingest(**payload(0))
    assert_equal(bank.state_dict(), before)
    bank.ingest(**payload(0, height=1, width=2))
    assert bank.stats()["total_tokens"] == 2


def test_deterministic_local_rng_and_global_rng_isolation():
    original_global_state = random.getstate()
    try:
        random.seed(9823)
        expected_global_state = random.getstate()
        first = Qwen3ReservoirMemory(memory_frames=6, seed=41)
        second = Qwen3ReservoirMemory(memory_frames=6, seed=41)
        reference_rng = random.Random(41)
        expected_slots = []
        for index in range(30):
            first.ingest(**payload(index))
            assert random.getstate() == expected_global_state
            random.random()  # Unrelated randomness cannot influence the bank.
            expected_global_state = random.getstate()
            second.ingest(**payload(index))
            assert random.getstate() == expected_global_state
            if index < 3:
                expected_slots.append(index)
            else:
                slot = reference_rng.randrange(index + 1)
                if slot < 3:
                    expected_slots[slot] = index
            assert retained_ids(first) == tuple(sorted(expected_slots))
            assert_equal(first.state_dict(), second.state_dict())
            first.advance()
            second.advance()
    finally:
        random.setstate(original_global_state)


@pytest.mark.parametrize("invalid", [
    "shape", "dtype", "id_duplicate", "time_duplicate", "time_nan", "time_inf",
    "time_bool", "id_bool", "height_bool", "features_nan", "features_integer",
    "metadata", "levels_missing", "level_shape", "level_dtype", "level_nan",
])
def test_invalid_input_keeps_rng_last_seen_and_memory_unchanged(invalid):
    bank = Qwen3ReservoirMemory(memory_frames=4, seed=5)
    ingest_all(bank, 3)  # Latest grid 2 is rejected, so retained max is only 1.
    before = bank.state_dict()
    incoming = payload(3)
    if invalid == "shape":
        incoming = payload(3, height=1, width=4)
    elif invalid == "dtype":
        incoming = payload(3, dtype=torch.float64)
    elif invalid == "id_duplicate":
        incoming["time_id"] = 2
    elif invalid == "time_duplicate":
        incoming["timestamp_seconds"] = 4
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
    assert_equal(bank.state_dict(), before)
    expected = Qwen3ReservoirMemory.from_state_dict(before)
    bank.ingest(**payload(3))
    expected.ingest(**payload(3))
    assert_equal(bank.state_dict(), expected.state_dict())


def test_owned_inputs_snapshots_and_checkpoints_are_isolated():
    bank = Qwen3ReservoirMemory(memory_frames=2, seed=0)
    incoming = payload(0)
    for tensor in (incoming["embeddings"], *incoming["deepstack"]):
        tensor.requires_grad_(True)
    bank.ingest(**incoming)
    expected = bank.snapshot()
    with torch.no_grad():
        for tensor in (incoming["embeddings"], *incoming["deepstack"]):
            tensor.fill_(-1)
    incoming["metadata"]["nested"]["value"] = 999
    assert_equal(bank.snapshot(), expected)
    snapshot = bank.snapshot()
    for tensor in (snapshot["embeddings"], snapshot["positions"], *snapshot["deepstack"]):
        tensor.fill_(-2)
    snapshot["records"][0]["metadata"]["nested"]["value"] = 1000
    assert_equal(bank.snapshot(), expected)
    checkpoint = bank.state_dict()
    checkpoint["slots"][0]["embeddings"].fill_(-3)
    checkpoint["slots"][0]["deepstack"][1].fill_(-4)
    checkpoint["slots"][0]["metadata"]["nested"]["value"] = 1001
    checkpoint["last_observation"]["time_id"] = 555
    assert_equal(bank.snapshot(), expected)
    assert bank.last_observation["time_id"] == 0
    assert not expected["embeddings"].requires_grad
    assert all(not tensor.requires_grad for tensor in expected["deepstack"])


@pytest.mark.parametrize("pending", [True, False])
@pytest.mark.parametrize("seed", [0, 5])
@pytest.mark.parametrize("budget", [{"memory_frames": 4}, {"memory_tokens": 9}])
def test_checkpoint_preserves_slots_rng_and_rejected_latest_stream_state(pending, seed, budget):
    bank = Qwen3ReservoirMemory(**budget, seed=seed)
    ingest_all(bank, 3)
    bank.ingest(**payload(3))
    assert bank.last_observation["accepted"] is False
    bank.mark_response()
    if not pending:
        bank.advance()
    saved = bank.state_dict()
    assert saved["last_time_id"] == 3 and max(retained_ids(bank)) < 3
    restored = Qwen3ReservoirMemory.from_state_dict(saved)
    assert_equal(restored.state_dict(), saved)
    assert_equal(restored.snapshot(), bank.snapshot())
    saved["slots"][0]["embeddings"].fill_(-1)
    saved["slots"][0]["deepstack"][0].fill_(-2)
    saved["slots"][0]["metadata"]["nested"]["value"] = 900
    saved["last_observation"]["time_id"] = 999
    assert_equal(restored.state_dict(), bank.state_dict())
    if pending:
        restored.advance()
        bank.advance()
    with pytest.raises(ValueError, match="including rejected"):
        restored.ingest(**payload(3))
    for index in range(4, 20):
        bank.ingest(**payload(index))
        restored.ingest(**payload(index))
        assert_equal(restored.state_dict(), bank.state_dict())
        bank.advance()
        restored.advance()


def test_checkpoint_keeps_unsorted_slots_and_acceptance_phase():
    bank = Qwen3ReservoirMemory(memory_frames=4, seed=1)
    ingest_all(bank, 2)
    bank.ingest(**payload(2))
    assert [r["time_id"] for r in bank.state_dict()["slots"]] == [2, 1]
    restored = Qwen3ReservoirMemory.from_state_dict(bank.state_dict())
    assert_equal(restored.state_dict(), bank.state_dict())
    bank.advance()
    restored.advance()
    bank.ingest(**payload(3))
    restored.ingest(**payload(3))
    assert_equal(restored.state_dict(), bank.state_dict())


def test_empty_checkpoint_round_trip():
    bank = Qwen3ReservoirMemory(memory_tokens=9, seed=7)
    restored = Qwen3ReservoirMemory.from_state_dict(bank.state_dict())
    assert_equal(bank.state_dict(), restored.state_dict())
    restored.ingest(**payload(0))
    assert bank.stats()["total_ingested"] == 0


def test_phase_and_response_operations_never_consume_rng():
    bank = Qwen3ReservoirMemory(memory_frames=2, seed=0)
    with pytest.raises(RuntimeError):
        bank.advance()
    with pytest.raises(RuntimeError):
        bank.mark_response()
    bank.ingest(**payload(0))
    before = bank.state_dict()
    with pytest.raises(RuntimeError, match="advance"):
        bank.ingest(**payload(1))
    assert_equal(bank.state_dict(), before)
    observation = bank.last_observation
    observation["accepted"] = False
    assert bank.last_observation["accepted"] is True
    for _ in range(3):
        bank.snapshot()
        bank.mark_response()
        assert bank.should_respond() is False
    bank.advance()
    assert bank.stats()["response_count"] == 3
    assert bank.state_dict()["rng_state"] == before["rng_state"]


@pytest.mark.parametrize("invalid", [
    "budget", "count", "tokens", "pending", "response", "version", "rng",
    "last_seen", "last_time", "observation", "duplicate_slot",
])
def test_checkpoint_rejects_inconsistent_stream_or_sampling_state(invalid):
    bank = Qwen3ReservoirMemory(memory_frames=4, seed=5)
    ingest_all(bank, 3)
    checkpoint = deepcopy(bank.state_dict())
    if invalid == "budget":
        checkpoint["config"]["memory_frames"] = 2
    elif invalid == "count":
        checkpoint["rejected_grids"] = 2
    elif invalid == "tokens":
        checkpoint["total_encoded_tokens"] = 7
    elif invalid == "pending":
        checkpoint["pending_advance"] = 1
    elif invalid == "response":
        checkpoint["last_response_time_id"] = 0
    elif invalid == "version":
        checkpoint["version"] = 999
    elif invalid == "rng":
        checkpoint["rng_state"] = (0, (), None)
    elif invalid == "last_seen":
        checkpoint["last_time_id"] = 1
    elif invalid == "last_time":
        checkpoint["last_timestamp"] = 2
    elif invalid == "observation":
        checkpoint["last_observation"]["accepted"] = True
    elif invalid == "duplicate_slot":
        checkpoint["slots"][1] = deepcopy(checkpoint["slots"][0])
    with pytest.raises(ValueError):
        Qwen3ReservoirMemory.from_state_dict(checkpoint)
