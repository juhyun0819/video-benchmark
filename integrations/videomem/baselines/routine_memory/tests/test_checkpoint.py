"""Safe atomic checkpoints and retrieval without rebuilding a video stream."""
import hashlib
import json

import pytest
import torch

from baselines.routine_memory.budget import canonical_json, manifest_for, measure_state_bytes
from baselines.routine_memory.checkpoint import CheckpointError, load_state, save_checkpoint
from baselines.routine_memory.retrieval import retrieve, iter_evidence_batches
from baselines.routine_memory.tests.test_budget import tiny_state


class FixtureMemory:
    def __init__(self, state):
        self.state = state

    def state_dict(self):
        return self.state

    def retained_evidence(self):
        return self.state['short_term']['frames']


def fixture_memory():
    state = tiny_state()
    owner = torch.arange(96, dtype=torch.float32).reshape(12, 8)
    evidence = {'features': owner[::2, 1:7], 'token_indices': torch.arange(6),
                'kind': 'short_term', 'frame_id': 10, 'timestamp': 10.0}
    state['short_term']['frames'].append(evidence)
    state['routine_memory']['routines'] = [{'spatial': owner.T, 'temporal': (owner[:2],), 'same': evidence['features']}]
    state['control'].update(last_timestamp=10.0, total_frames_seen=11)
    return FixtureMemory(state)


def test_roundtrip_exact_state_storage_aliases_and_actual_file_bytes(tmp_path):
    memory = fixture_memory()
    expected = measure_state_bytes(memory.state_dict())
    destination = tmp_path / 'snapshot'
    manifest = save_checkpoint(memory, destination)
    assert sum(p.stat().st_size for p in destination.iterdir()) == expected['serialized_bytes'] == manifest['total_file_bytes']
    state, actual_manifest = load_state(destination)
    assert manifest == actual_manifest
    a = state['short_term']['frames'][0]['features']
    b = state['routine_memory']['routines'][0]['spatial']
    assert torch.equal(a, memory.state['short_term']['frames'][0]['features'])
    assert torch.equal(b, memory.state['routine_memory']['routines'][0]['spatial'])
    assert a.stride() == memory.state['short_term']['frames'][0]['features'].stride()
    assert a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
    assert state['routine_memory']['routines'][0]['same'] is a
    assert isinstance(state['routine_memory']['routines'][0]['temporal'], tuple)
    accounting = measure_state_bytes(state)
    for key in ('actual_memory_bytes', 'serialized_bytes', 'tensor_storage_bytes', 'python_object_reserve_bytes'):
        assert accounting[key] == expected[key]
    with pytest.raises(FileExistsError):
        save_checkpoint(memory, destination)


@pytest.mark.parametrize('filename', ['features.bin', 'state.json', 'manifest.json'])
def test_corruption_refused(tmp_path, filename):
    directory = tmp_path / filename
    save_checkpoint(fixture_memory(), directory)
    path = directory / filename
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 1
    path.write_bytes(data)
    with pytest.raises(CheckpointError):
        load_state(directory)


def test_tensor_layout_bounds_verified_before_materialization(tmp_path):
    directory = tmp_path / 'snapshot'
    save_checkpoint(fixture_memory(), directory)
    index = json.loads((directory / 'state.json').read_text())
    index['tensors'][0]['shape'] = [2 ** 50, 6]
    raw = canonical_json(index)
    (directory / 'state.json').write_bytes(raw)
    old = json.loads((directory / 'manifest.json').read_text())
    new = manifest_for(len(raw), old['features_bytes'], old['features_sha256'], hashlib.sha256(raw).hexdigest())
    (directory / 'manifest.json').write_bytes(canonical_json(new))
    with pytest.raises(CheckpointError, match='storage'):
        load_state(directory)


def test_atomic_failure_and_over_budget_state_leave_no_checkpoint(tmp_path, monkeypatch):
    from baselines.routine_memory import checkpoint
    memory = fixture_memory()
    original = checkpoint._write_file

    def broken(path, data):
        if path.name == 'state.json':
            raise OSError('injected failure')
        original(path, data)

    monkeypatch.setattr(checkpoint, '_write_file', broken)
    with pytest.raises(OSError, match='injected'):
        save_checkpoint(memory, tmp_path / 'failed')
    assert not list(tmp_path.iterdir())
    memory.state['control']['config']['budget_bytes'] = 1
    with pytest.raises(CheckpointError, match='hard memory budget'):
        save_checkpoint(memory, tmp_path / 'too_large')
    assert not list(tmp_path.iterdir())


def test_retrieval_causal_cutoff_no_hidden_history_and_owned_results():
    memory = fixture_memory()
    record = memory.state['short_term']['frames'][0]
    query = record['features'].mean(dim=0)
    with pytest.raises(ValueError, match='earlier snapshot'):
        retrieve(memory, query, as_of=9)
    found = retrieve(memory, query, k=1, as_of=10)
    assert found[0]['frame_id'] == 10 and found[0]['score'] == pytest.approx(1.0)
    found[0]['features'].zero_()
    assert torch.count_nonzero(record['features']) > 0
    batches = list(iter_evidence_batches(memory, max_tokens=2))
    assert len(batches) == 3 and all(len(x['features']) <= 2 for x in batches)
    assert torch.equal(torch.cat([x['features'] for x in batches]), record['features'])
    assert torch.equal(torch.cat([x['token_indices'] for x in batches]), record['token_indices'])


def test_resume_actual_routine_memory_matches_uninterrupted_updates(tmp_path):
    from baselines.routine_memory.config import RoutineConfig
    from baselines.routine_memory.memory import RoutineAwareMemory
    from baselines.routine_memory.checkpoint import load_checkpoint, RoutineSnapshotReader
    memory = RoutineAwareMemory(RoutineConfig(budget_bytes=2_000_000))
    generator = torch.Generator().manual_seed(924)
    features = [torch.randn(16, 12, generator=generator) for _ in range(10)]
    for i, frame in enumerate(features[:5]):
        memory.update(frame, i, float(i))
    directory = tmp_path / 'resume'
    save_checkpoint(memory, directory)
    resumed = load_checkpoint(directory)
    reader = RoutineSnapshotReader(directory)
    assert reader.stats()['total_frames_seen'] == 5
    assert reader.retrieve(features[4], k=1)
    for i in range(5, 10):
        first_decision = memory.update(features[i], i, float(i))
        second_decision = resumed.update(features[i], i, float(i))
        first_decision.pop('processing_seconds', None)
        second_decision.pop('processing_seconds', None)
        assert first_decision == second_decision
    first, second = tmp_path / 'first', tmp_path / 'second'
    a, b = save_checkpoint(memory, first), save_checkpoint(resumed, second)
    assert a == b
