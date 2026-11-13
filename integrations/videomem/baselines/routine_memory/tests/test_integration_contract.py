"""Independent checks of ownership, total accounting, causality and persisted queries."""
import hashlib
import json

import pytest
import torch

from baselines.routine_memory.budget import measure_state_bytes
from baselines.routine_memory.checkpoint import (
    CheckpointError, RoutineSnapshotReader, load_checkpoint, save_checkpoint,
)
from baselines.routine_memory.config import RoutineConfig
from baselines.routine_memory.memory import RoutineAwareMemory


def fingerprint(value):
    digest = hashlib.sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            digest.update(str((str(x.dtype), list(x.shape))).encode())
            digest.update(x.contiguous().view(torch.uint8).numpy().tobytes())
        elif isinstance(x, dict):
            for key in sorted(x):
                digest.update(key.encode())
                visit(x[key])
        elif isinstance(x, (list, tuple)):
            digest.update(str((type(x).__name__, len(x))).encode())
            for item in x:
                visit(item)
        else:
            digest.update(json.dumps(x, sort_keys=True, allow_nan=False).encode())
    visit(value)
    return digest.hexdigest()


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensors(item)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
def test_caller_feature_shard_and_metadata_never_alias_retained_state(dtype):
    memory = RoutineAwareMemory()
    backing = torch.arange(8 * 16 * 12, dtype=torch.float32).reshape(8, 16, 12).to(dtype)
    metadata = {'source_id': 'video1', 'day_id': 'Day1'}
    memory.update(backing[2], frame_id=0, timestamp=0.0, metadata=metadata)
    state = memory.state_dict()
    before = fingerprint(state)
    input_address = backing.untyped_storage().data_ptr()
    assert all(x.untyped_storage().data_ptr() != input_address for x in tensors(state))
    backing.fill_(-100)
    metadata['day_id'] = 'a future day'
    assert fingerprint(memory.state_dict()) == before


def test_invalid_or_future_order_does_not_partially_mutate_memory():
    memory = RoutineAwareMemory()
    features = torch.ones(16, 12, dtype=torch.bfloat16)
    memory.update(features, frame_id=0, timestamp=10.0)
    before = fingerprint(memory.state_dict())
    invalid = [(features, 1, 9.0), (features, 0, 11.0),
               (features, 1, float('nan')), (features[:, :8], 1, 11.0),
               (torch.full_like(features, float('nan')), 1, 11.0)]
    for value, frame, stamp in invalid:
        with pytest.raises((ValueError, TypeError)):
            memory.update(value, frame_id=frame, timestamp=stamp)
        assert fingerprint(memory.state_dict()) == before


def test_16gb_is_one_total_and_checkpoint_query_preserves_state(tmp_path):
    config = RoutineConfig()
    assert config.budget_bytes == 16_000_000_000
    memory = RoutineAwareMemory(config=config)
    for i in range(12):
        memory.update(torch.ones(16, 12, dtype=torch.bfloat16), frame_id=i, timestamp=float(i))
    state = memory.state_dict()
    account = measure_state_bytes(state)
    assert sum(account[key] for key in ('short_term_bytes', 'routine_memory_bytes',
                                      'event_memory_bytes', 'metadata_bytes')) == account['actual_memory_bytes']
    assert account['actual_memory_bytes'] <= config.budget_bytes
    target = tmp_path / 'checkpoint'
    save_checkpoint(memory, target)
    assert sum(f.stat().st_size for f in target.iterdir()) == account['serialized_bytes']
    restored = load_checkpoint(target)
    assert fingerprint(restored.state_dict()) == fingerprint(state)
    reader = RoutineSnapshotReader(target)
    before = fingerprint(reader.memory.state_dict())
    evidence = reader.retrieve(torch.ones(12), k=2, as_of=11.0)
    assert evidence
    for row in evidence:
        assert row['timestamp'] <= 11.0
        row['features'].zero_()
    assert fingerprint(reader.memory.state_dict()) == before
    with pytest.raises(ValueError, match='Historical'):
        reader.retrieve(torch.ones(12), as_of=10.0)
    assert fingerprint(reader.memory.state_dict()) == before


def test_corrupted_retained_features_fail_before_resume(tmp_path):
    memory = RoutineAwareMemory()
    memory.update(torch.ones(16, 12, dtype=torch.bfloat16), frame_id=0, timestamp=0.0)
    target = tmp_path / 'checkpoint'
    save_checkpoint(memory, target)
    path = target / 'features.bin'
    with path.open('r+b') as f:
        old = f.read(1)
        assert old
        f.seek(0)
        f.write(bytes([old[0] ^ 1]))
    with pytest.raises(CheckpointError, match='SHA256'):
        load_checkpoint(target)
