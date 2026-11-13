"""Exercise actual sixteen-bank builder, snapshot barriers and fixed caps."""
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import torch

from baselines.feature_memory import construction, video
from baselines.feature_memory.byte_memory import SnapshotReader
from baselines.feature_memory.fluxmem import FluxMemSnapshotReader


class TinyEncoder:
    def __init__(self, model_path, device='cpu', dtype='bfloat16'):
        self.metadata = {'fixture': True}

    def encode(self, image):
        value = image.getpixel((0, 0))[0]
        return torch.full((9, 8), float(value), dtype=torch.bfloat16), {'encode_seconds': 0.0}


class NoTelemetry:
    def start(self): pass
    def close(self): pass
    def report(self):
        return {'peak_cpu_ram': 0, 'gpus': {}}


def test_threaded_builder_fixed_caps_causal_snapshots_shared_once(tmp_path, monkeypatch):
    def frames(inventory, max_seconds=None, stats=None):
        for index in range(8):
            image = Image.new('RGB', (8, 8), (index, 0, 0))
            try:
                yield {'frame': image, 'frame_id': index, 'timestamp': float(index), 'decode_seconds': 0.01}
            finally:
                image.close()

    settings = [{'retention_ratio': ratio, 'label': str(int(ratio * 100)) + 'pct',
                 'hard_budget_bytes': int(2_000_000 * ratio)} for ratio in (.05, .1, .25, .5)]
    plan = {'settings': settings, 'methods': ['fifo', 'uniform', 'reservoir', 'fluxmem'],
            'hidden_dim': 8, 'tokens_per_frame': 9, 'bytes_per_token': 16,
            'post_merger_grid_hw': [3, 3], 'full_feature_bytes': 2_000_000,
            'reference_frames': 10000, 'planned_frames': 8, 'planned_run_feature_bytes': 8 * 9 * 16,
            'duration_seconds': 8}
    for setting in settings:
        setting['hard_budget_GB'] = setting['hard_budget_bytes'] / 1e9
    monkeypatch.setattr(video, 'inspect_sources', lambda *a, **k: {})
    monkeypatch.setattr(video, 'iter_stream', frames)
    monkeypatch.setattr(construction, 'make_plan', lambda *a, **k: plan)
    monkeypatch.setattr(construction, 'Telemetry', NoTelemetry)
    args = SimpleNamespace(cpu_threads=1, flux_workers=4, snapshot_seconds=[2, 5],
        videos=[], dataset_status=None, timeline=None, model='fixture', max_seconds=8,
        ratios=[.05, .1, .25, .5], output=tmp_path/'build', stream_id='fixture', mode='sequential',
        devices=['cpu'], chunk_frames=2, max_inflight_chunks=2, max_queue_bytes=100000,
        seed=17, shard_bytes=4096, progress_frames=2)
    summary = construction.build(args, encoder_factory=TinyEncoder)
    assert summary['num_encoder_calls'] == summary['total_frames'] == 8
    assert summary['consumer_count'] == 16
    assert summary['full_feature_archive_created'] is False
    assert summary['full_stream_completed'] is False
    assert summary['payload_bytes'] == 8 * 9 * 16
    assert len(summary['snapshots']) == 3
    assert summary['timing_seconds']['memory_fanout_wall_seconds'] > 0
    for record in summary['snapshots']:
        cutoff = record['timestamp']
        assert record['frames_seen'] == cutoff
        assert len(record['states']) == 16
        for state in record['states']:
            assert state['hard_budget_bytes'] == int(2_000_000 * state['ratio'])
            assert state['actual_memory_bytes'] <= state['hard_budget_bytes']
            assert state['effective_retention_ratio'] == state['actual_memory_bytes'] / (cutoff * 144)
            path = state['path']
            metadata = json.loads((Path(path)/'benchmark_metadata.json').read_text())
            assert metadata['fixed_reference_feature_bytes'] == 2_000_000
            reader = FluxMemSnapshotReader(path) if state['method'] == 'fluxmem' else SnapshotReader(path)
            try:
                for batch in reader.iter_batches():
                    for name, values in batch['metadata'].items():
                        if name in ('timestamp', 'end_timestamp', 'frame_id', 'end_frame'):
                            assert float(values.max()) < cutoff
            finally:
                reader.close()
