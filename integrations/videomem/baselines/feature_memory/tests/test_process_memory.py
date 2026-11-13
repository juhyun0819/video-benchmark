"""Real spawned CPU consumers, shared pages, causal snapshots and cleanup."""
import json
from multiprocessing import shared_memory
import os
from pathlib import Path
import signal

import pytest
import torch

from baselines.feature_memory.byte_memory import ByteBudgetMemory, SnapshotReader
from baselines.feature_memory.fluxmem import FluxMemByteMemory, FluxMemSnapshotReader
from baselines.feature_memory.process_memory import MemoryFarm


def plan():
    return {'methods': ['fifo', 'uniform', 'reservoir', 'fluxmem'], 'dtype': 'bfloat16',
            'hidden_dim': 8, 'tokens_per_frame': 9, 'bytes_per_token': 16,
            'post_merger_grid_hw': [3, 3], 'full_feature_bytes': 2_000_000,
            'settings': [{'label': f'{int(r*100)}pct', 'retention_ratio': r,
                          'hard_budget_bytes': int(2_000_000*r)} for r in (.05, .1, .25, .5)]}


def values(count):
    generator = torch.Generator().manual_seed(419)
    result = torch.randn(count, 9, 8, generator=generator).to(torch.bfloat16)
    for i in range(count):
        result[i, 1] = result[i, 0] * .97 + result[i, 1] * .03
        if i % 3 == 1:
            result[i, :3] = result[i-1, :3]
    return result


def records(reader):
    batches = list(reader.iter_batches(max_tokens=7))
    features = torch.cat([x['features'] for x in batches])
    metadata = {key: torch.cat([x['metadata'][key] for x in batches]) for key in batches[0]['metadata']}
    order = torch.argsort(metadata['global_token_id'])
    return features[order], {key: value[order] for key, value in metadata.items()}


def serial_states(tmp_path, config, features):
    result = {}
    for method in config['methods']:
        for setting in config['settings']:
            common = dict(directory=tmp_path/method/setting['label'],
                          hard_budget_bytes=setting['hard_budget_bytes'],
                          retention_ratio=setting['retention_ratio'],
                          hidden_dim=8, tokens_per_frame=9, dtype='bfloat16', seed=17, shard_bytes=4096)
            manager = (FluxMemByteMemory(height=3, width=3, **common) if method == 'fluxmem'
                       else ByteBudgetMemory(method=method, **common))
            for index, feature in enumerate(features):
                manager.update(feature, index, float(index))
            result[(method, setting['label'])] = manager
    return result


def assert_gone(farm, name):
    assert not any(row['process'].is_alive() for row in farm._handles.values())
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name, create=False)


def test_process_farm_matches_serial_all_states_and_causal_subchunk_snapshots(tmp_path):
    torch.set_num_threads(1)
    config, features = plan(), values(12)
    original = features.clone()
    farm = MemoryFarm(config, tmp_path/'parallel', seed=17, shard_bytes=4096,
                      cpu_threads=1, max_batch_frames=5)
    name = farm.shared_memory_name
    assert len(farm.worker_pids) == 5 and len(set(farm.worker_pids.values())) == 5
    assert farm.shared_memory_bytes == 5 * 9 * 8 * 2
    serial = {}
    try:
        first = farm.update_batch(features[:3], [{'frame_id': i, 'timestamp': float(i)} for i in range(3)])
        assert len(first['rows']) == 16 and len(first['worker_timings']) == 5
        assert first['all_workers_acknowledged_before_buffer_reuse']
        prefix = farm.snapshot(3., '3s')
        assert len(prefix['states']) == 16
        for state in prefix['states']:
            reader = (FluxMemSnapshotReader(state['path']) if state['method'] == 'fluxmem'
                      else SnapshotReader(state['path']))
            try:
                for batch in reader.iter_batches(max_tokens=7):
                    assert float(batch['metadata']['timestamp'].max()) < 3
                    assert int(batch['metadata']['frame_id'].max()) < 3
            finally:
                reader.close()
        for low, high in [(3, 8), (8, 12)]:
            farm.update_batch(features[low:high], [{'frame_id': i, 'timestamp': float(i)}
                                                  for i in range(low, high)])
        torch.testing.assert_close(features, original, rtol=0, atol=0)
        with pytest.raises(ValueError, match='split batches'):
            farm.snapshot(11., 'invalid')
        final = farm.snapshot(12., '12s')
        serial = serial_states(tmp_path/'serial', config, features)
        for state in final['states']:
            assert state['validation']['passed']
            assert state['actual_memory_bytes'] <= state['hard_budget_bytes']
            reader = (FluxMemSnapshotReader(state['path']) if state['method'] == 'fluxmem'
                      else SnapshotReader(state['path']))
            try:
                actual, meta = records(reader)
                expected, expected_meta = records(serial[(state['method'], state['label'])])
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for key in meta:
                    torch.testing.assert_close(meta[key], expected_meta[key], rtol=0, atol=0)
            finally:
                reader.close()
        rows = farm.stats()
        assert len(rows) == 16 and all(row['total_frames_seen'] == 12 for row in rows)
        statuses = list((tmp_path/'parallel'/'worker_status').glob('*.json'))
        assert len(statuses) == 5
        assert all(json.loads(p.read_text())['phase'] == 'ready' for p in statuses)
    finally:
        for bank in serial.values():
            bank.close()
        receipt = farm.close()
    assert receipt['worker_processes_alive_after_close'] == 0
    assert receipt['shared_memory_unlinked']
    assert receipt == farm.close()
    assert_gone(farm, name)


def test_noncausal_input_rejected_before_any_worker_update(tmp_path):
    features = values(2)
    with MemoryFarm(plan(), tmp_path/'farm', shard_bytes=4096, max_batch_frames=2) as farm:
        with pytest.raises(ValueError, match='consecutive'):
            farm.update_batch(features, [{'frame_id': 0, 'timestamp': 0.}, {'frame_id': 1, 'timestamp': 0.}])
        assert all(row['total_frames_seen'] == 0 for row in farm.stats())
        with pytest.raises(ValueError, match='bounded'):
            farm.update_batch(values(3), [{'frame_id': i, 'timestamp': float(i)} for i in range(3)])
        assert all(row['total_frames_seen'] == 0 for row in farm.stats())
        farm.update_batch(features, [{'frame_id': 0, 'timestamp': 0.}, {'frame_id': 1, 'timestamp': 1.}])
        assert all(row['total_frames_seen'] == 2 for row in farm.stats())


def test_worker_death_fails_fast_and_cleans_all_processes_and_shared_memory(tmp_path):
    farm = MemoryFarm(plan(), tmp_path/'farm', shard_bytes=4096, max_batch_frames=2)
    name = farm.shared_memory_name
    os.kill(farm.worker_pids['fluxmem_5pct'], signal.SIGKILL)
    with pytest.raises((RuntimeError, BrokenPipeError, ConnectionResetError)):
        farm.stats()
    assert_gone(farm, name)
    receipt = farm.close()
    assert receipt['failed'] and receipt['shared_memory_unlinked']


def test_parent_exception_context_aborts_workers_and_buffer(tmp_path):
    farm = MemoryFarm(plan(), tmp_path/'farm', shard_bytes=4096, max_batch_frames=2)
    name = farm.shared_memory_name
    with pytest.raises(RuntimeError, match='parent failure'):
        with farm:
            farm.update_batch(values(1), [{'frame_id': 0, 'timestamp': 0.}])
            raise RuntimeError('parent failure')
    assert_gone(farm, name)
