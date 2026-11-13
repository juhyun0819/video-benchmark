"""Boundedness, algorithm semantics, ownership and serialization checks."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from baselines.feature_memory.memory import TokenMemory
from baselines.streambench.accounting import measure_state


def feed(memory, lengths=(3, 8, 2, 9, 1), width=3):
    cursor = 0
    for frame, length in enumerate(lengths):
        values = torch.arange(cursor, cursor + length, dtype=torch.float32).view(-1, 1).repeat(1, width)
        memory.update(values, frame, float(frame))
        cursor += length
        assert memory.stats()['retained_visual_tokens'] <= memory.budget
        yield cursor, memory.snapshot()


@pytest.mark.parametrize('budget', [1, 4, 11])
def test_fifo_matches_exact_tail_after_every_frame(budget):
    memory = TokenMemory('fifo', budget)
    for seen, snapshot in feed(memory):
        expected = list(range(max(0, seen - budget), seen))
        assert snapshot['tensors']['global_token_ids'].tolist() == expected
        assert snapshot['tensors']['features'][:, 0].tolist() == expected


@pytest.mark.parametrize('budget', [1, 4, 11])
def test_uniform_is_causal_global_dyadic_spacing(budget):
    memory = TokenMemory('uniform', budget)
    for seen, snapshot in feed(memory):
        stride = memory.stats()['uniform_stride']
        expected = list(range(0, seen, stride))
        assert snapshot['tensors']['global_token_ids'].tolist() == expected
        assert snapshot['tensors']['features'][:, 0].tolist() == expected
        assert stride & (stride - 1) == 0
        assert len(expected) <= budget
        assert seen <= budget or len(expected) > budget // 2


def test_reservoir_vectorized_matches_sequential_algorithm_r():
    seed, budget = 891, 7
    memory = TokenMemory('reservoir', budget, seed=seed)
    rng = np.random.Generator(np.random.PCG64(seed))
    reference = []
    previous = 0
    for seen, snapshot in feed(memory, lengths=(3, 23, 8, 19)):
        for index in range(previous, seen):
            if len(reference) < budget:
                reference.append(index)
            else:
                slot = rng.integers(0, index + 1)
                if slot < budget:
                    reference[slot] = index
        assert snapshot['tensors']['global_token_ids'].tolist() == sorted(reference)
        previous = seen


def test_reservoir_inclusion_is_uniform_across_history():
    # A regression to recent-window or prefix retention fails by a large margin.
    runs, count, budget = 500, 20, 4
    frequencies = np.zeros(count, dtype=np.int64)
    values = torch.arange(count, dtype=torch.float32).view(-1, 1)
    for seed in range(runs):
        memory = TokenMemory('reservoir', budget, seed)
        memory.update(values, 0, 0.0)
        ids = memory.snapshot()['tensors']['global_token_ids'].numpy()
        frequencies[ids] += 1
    expected = runs * budget / count
    assert np.max(np.abs(frequencies - expected)) < 42


@pytest.mark.parametrize('method', ['fifo', 'uniform', 'reservoir'])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_owned_storage_and_complete_allocation_accounting(method, dtype):
    base = torch.ones(100, 12, dtype=dtype)
    features = base[:5, :3]  # Input view owns a much larger allocation.
    memory = TokenMemory(method, 8, 123)
    memory.update(features, 4, 2.5)
    base.zero_()
    snapshot = memory.snapshot()
    assert torch.equal(snapshot['tensors']['features'], torch.ones(5, 3, dtype=dtype))
    stats = memory.stats()
    assert stats['feature_bytes'] == 5 * 3 * features.element_size()
    assert stats['allocated_feature_bytes'] == 8 * 3 * features.element_size()
    assert stats['metadata_bytes'] >= 8 * 4 * 8
    assert stats['actual_memory_bytes'] == measure_state(memory.persistent_state())['persistent_total_bytes']
    assert stats['compression_ratio'] == 1.0
    snapshot['tensors']['features'].zero_()
    assert memory.snapshot()['tensors']['features'].sum().item() == 15
    assert snapshot['tensors']['features'].untyped_storage().nbytes() == stats['feature_bytes']


@pytest.mark.parametrize('method', ['fifo', 'uniform', 'reservoir'])
def test_causality_and_stable_schema_validation_are_nonmutating(method):
    memory = TokenMemory(method, 4)
    features = torch.ones(3, 2)
    memory.update(features, 7, 2.0)
    before = deepcopy(memory.snapshot()['state'])
    for fid, stamp in [(8, 1.0), (7, 3.0), (8, float('nan')), (True, 3.0)]:
        with pytest.raises(ValueError):
            memory.update(features, fid, stamp)
        assert memory.snapshot()['state'] == before
    for bad in [torch.ones(2, 3), torch.ones(2, 2, dtype=torch.float16), torch.ones(0, 2)]:
        with pytest.raises(ValueError):
            memory.update(bad, 8, 3.0)
        assert memory.snapshot()['state'] == before


@pytest.mark.parametrize('method', ['fifo', 'uniform', 'reservoir'])
def test_snapshot_roundtrip_preserves_token_identity_and_payload(tmp_path, method):
    memory = TokenMemory(method, 11, 42)
    list(feed(memory))
    destination = tmp_path / 'snapshot.pt'
    metadata = memory.save_snapshot(destination, timestamp=5.0)
    recovered = torch.load(destination, weights_only=True)
    expected = memory.snapshot()
    assert metadata['serialized_file_bytes'] == destination.stat().st_size
    assert metadata['serialized_file_bytes'] > metadata['feature_bytes']
    for key, tensor in expected['tensors'].items():
        assert torch.equal(recovered['tensors'][key], tensor)
    assert recovered['state'] == expected['state']
    assert recovered['metadata']['timestamp'] == 5.0
    assert recovered['tensors']['frame_timestamps'].max().item() < 5.0
    with pytest.raises(FileExistsError):
        memory.save_snapshot(destination, timestamp=5.0)


def test_empty_and_invalid_budgets():
    for budget in [0, -1, True, 1.5]:
        with pytest.raises(ValueError):
            TokenMemory('fifo', budget)
    empty = TokenMemory('fifo', 4)
    assert empty.snapshot()['tensors'] == {}
    assert empty.stats()['retained_visual_tokens'] == 0
    assert empty.stats()['feature_bytes'] == 0
