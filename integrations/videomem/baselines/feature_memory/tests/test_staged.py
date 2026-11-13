"""A single archive reader feeds five real processes at causal snapshot barriers."""
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace

import torch

from baselines.feature_memory.byte_memory import SnapshotReader
from baselines.feature_memory.fluxmem import FluxMemSnapshotReader
from baselines.feature_memory.storage import write_json


class NoTelemetry:
    def start(self): pass
    def close(self): pass
    def report(self):
        return {'peak_cpu_ram': 0, 'gpus': {}}


def test_real_five_process_consumer_one_reader_fixed_caps_causal_snapshots(tmp_path, monkeypatch):
    from baselines.feature_memory import archive, staged

    settings = [{'retention_ratio': ratio, 'label': str(int(ratio * 100)) + 'pct',
                 'hard_budget_bytes': int(2_000_000 * ratio)} for ratio in (.05, .1, .25, .5)]
    plan = {'settings': settings, 'methods': ['fifo', 'uniform', 'reservoir', 'fluxmem'],
            'hidden_dim': 8, 'tokens_per_frame': 9, 'bytes_per_token': 16,
            'post_merger_grid_hw': [3, 3], 'full_feature_bytes': 2_000_000,
            'reference_frames': 10000, 'planned_frames': 12, 'dtype': 'bfloat16',
            'planned_run_feature_bytes': 12 * 9 * 16, 'duration_seconds': 12}
    archive_path = tmp_path / 'archive'
    write_json(archive_path / 'archive.json', {'plan': plan})
    features = torch.arange(8 * 9 * 8, dtype=torch.float32).reshape(8, 9, 8).to(torch.bfloat16)
    frame_rows = [{'frame_id': index, 'timestamp': float(index),
                   'shared_feature_sha256': hashlib.sha256(features[index].view(torch.uint8).numpy().tobytes()).hexdigest(),
                   'decode_seconds': .01, 'encoder': {'encode_seconds': .02}}
                  for index in range(8)]
    reads, closed = [], []
    def read_once(path, *, wait, max_frames):
        assert path == archive_path and wait and max_frames == 8
        try:
            for start in range(0, 8, 3):
                reads.append(start)
                yield {'features': features[start:start+3], 'frames': frame_rows[start:start+3],
                       'read_seconds': .001}
        finally:
            closed.append(True)

    monkeypatch.setattr(archive, 'iter_archive', read_once)
    monkeypatch.setattr(staged, 'Telemetry', NoTelemetry)
    args = SimpleNamespace(archive=archive_path, output=tmp_path / 'memory', max_frames=8,
                           snapshot_seconds=[2, 5, 8], stream_id='fixture', seed=17,
                           shard_bytes=4096, cpu_threads=1, batch_frames=3)
    expected_features = features.clone()
    summary = staged.consume_archive(args)
    assert reads == [0, 3, 6] and closed == [True]
    assert torch.equal(features, expected_features)
    assert summary['num_encoder_calls'] == 0
    assert summary['consumer_count'] == 16
    assert summary['total_frames'] == 8
    assert summary['payload_bytes'] == 8 * 9 * 16
    assert summary['archive_read_once_by_coordinator'] is True
    assert summary['workers_open_archive'] is False
    assert summary['plan']['full_feature_bytes'] == 2_000_000
    assert summary['plan']['planned_frames'] == 12
    assert len(set(summary['worker_pids'].values())) == 5
    assert not set(summary['worker_pids'].values()) & {child.pid for child in mp.active_children()}
    expected_digest = hashlib.sha256(b''.join(bytes.fromhex(row['shared_feature_sha256']) for row in frame_rows))
    assert summary['ordered_shared_features_sha256'] == expected_digest.hexdigest()
    logs = [json.loads(line) for line in (args.output / 'frames.jsonl').read_text().splitlines()]
    assert [row['shared_feature_sha256'] for row in logs] == [row['shared_feature_sha256'] for row in frame_rows]
    assert all(row['consumers'] == 16 and row['encoder_calls_in_cpu_stage'] == 0 for row in logs)
    assert [snapshot['timestamp'] for snapshot in summary['snapshots']] == [2, 5, 8]
    for snapshot in summary['snapshots']:
        cutoff = snapshot['timestamp']
        assert snapshot['frames_seen'] == cutoff
        assert len(snapshot['states']) == 16
        for state in snapshot['states']:
            metadata = json.loads((Path(state['path']) / 'benchmark_metadata.json').read_text())
            assert metadata['total_frames_seen'] == cutoff
            assert metadata['fixed_reference_feature_bytes'] == 2_000_000
            assert state['hard_budget_bytes'] == int(2_000_000 * state['ratio'])
            assert metadata['actual_memory_bytes'] <= state['hard_budget_bytes']
            assert metadata['effective_retention_ratio'] == metadata['actual_memory_bytes'] / (cutoff * 9 * 16)
            assert state['validation']['passed'] is True
            reader = FluxMemSnapshotReader(state['path']) if state['method'] == 'fluxmem' else SnapshotReader(state['path'])
            try:
                count = 0
                for batch in reader.iter_batches(max_tokens=7):
                    count += len(batch['features'])
                    for key in ('frame_id', 'timestamp', 'start_frame', 'end_frame', 'start_timestamp', 'end_timestamp'):
                        if key in batch['metadata']:
                            assert float(batch['metadata'][key].max()) < cutoff
                    if state['method'] != 'fluxmem':
                        for position, (frame_id, token_index) in enumerate(zip(batch['metadata']['frame_id'], batch['metadata']['token_index'])):
                            assert torch.equal(batch['features'][position], expected_features[int(frame_id), int(token_index)])
                assert count == metadata['retained_visual_tokens']
            finally:
                reader.close()
    assert json.loads((args.output / 'status.json').read_text())['status'] == 'completed'
