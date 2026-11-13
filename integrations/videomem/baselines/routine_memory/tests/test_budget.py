"""Accounting covers tensor owners, candidates, metadata and checkpoint bytes."""
import copy
import json

import pytest
import torch

from baselines.routine_memory.budget import canonical_json, manifest_for, measure_state_bytes, pack_state


def tiny_state():
    return {'short_term': {'frames': []}, 'routine_memory': {'routines': []},
            'event_memory': {'events': [], 'candidates': []},
            'control': {'config': {'budget_bytes': 2_000_000}, 'total_frames_seen': 0,
                        'last_timestamp': None}}


def test_whole_tensor_storage_aliases_and_candidates_charged_once():
    state = tiny_state()
    owner = torch.arange(120, dtype=torch.float32).reshape(12, 10)
    state['short_term']['frames'] = [{'features': owner[2:3, :2]}]
    state['routine_memory']['routines'] = [{'prototype': owner.T}]
    a = measure_state_bytes(state)
    assert a['tensor_storage_bytes'] == owner.untyped_storage().nbytes()
    assert a['short_term_bytes'] + a['routine_memory_bytes'] + a['event_memory_bytes'] + a['metadata_bytes'] == a['actual_memory_bytes']
    assert a['actual_memory_bytes'] >= a['serialized_bytes']
    assert a['actual_memory_bytes'] >= a['resident_state_bytes_estimate']
    state['event_memory']['candidates'].append({'features': torch.ones(5, 10), 'occurrences': [1, 5, 9]})
    b = measure_state_bytes(state)
    assert b['tensor_storage_bytes'] == a['tensor_storage_bytes'] + 200
    assert b['event_memory_bytes'] > a['event_memory_bytes'] + 200


def test_serialized_measure_is_exact_without_materializing_feature_copy():
    state = tiny_state()
    state['short_term']['frames'].append({'features': torch.ones(7, 8, dtype=torch.bfloat16)})
    packed = pack_state(state)
    size = measure_state_bytes(state)['serialized_bytes']
    manifest = manifest_for(len(packed['index_raw']), packed['feature_bytes'])
    assert size == packed['feature_bytes'] + len(packed['index_raw']) + len(canonical_json(manifest))
    assert packed['storage_handles'][0].data_ptr() == state['short_term']['frames'][0]['features'].data_ptr()


def test_metadata_charge_deterministic_after_container_reconstruction():
    state = tiny_state()
    state['event_memory']['events'] = [{'description': '작업 변화', 'interval': [1.0, 20.0], 'flags': [True, None]}]
    rebuilt = json.loads(json.dumps(state))
    a, b = measure_state_bytes(state), measure_state_bytes(rebuilt)
    for name in ('actual_memory_bytes', 'metadata_bytes', 'serialized_bytes', 'python_object_reserve_bytes'):
        assert a[name] == b[name]
    assert a['python_object_reserve_bytes'] >= a['python_object_bytes_estimate']


@pytest.mark.parametrize('bad', [object(), float('nan'), {1: 'non-string key'}])
def test_unknown_or_nonfinite_state_fails_closed(bad):
    state = tiny_state()
    state['event_memory']['bad'] = bad
    with pytest.raises((TypeError, ValueError)):
        measure_state_bytes(state)


def test_no_hidden_fourth_bank_or_cyclic_state():
    state = tiny_state()
    state['candidate_bank'] = []
    with pytest.raises(ValueError, match='exactly'):
        measure_state_bytes(state)
    del state['candidate_bank']
    state['control']['cycle'] = state
    with pytest.raises(ValueError, match='Cyclic'):
        measure_state_bytes(state)


def test_independent_partial_frombuffer_owners_rejected():
    backing = bytearray(100)
    state = tiny_state()
    state['short_term']['frames'] = [torch.frombuffer(backing, dtype=torch.uint8, count=60),
                                    torch.frombuffer(backing, dtype=torch.uint8, offset=40, count=60)]
    with pytest.raises(ValueError, match='partially overlapping'):
        measure_state_bytes(state)
