"""Atomic publication, integrity, prefix reuse and one-call feature generation."""
from argparse import Namespace
import hashlib
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from baselines.feature_memory.archive import ArchiveError, ArchiveWriter, build_archive, iter_archive
from baselines.feature_memory.byte_memory import ByteBudgetMemory
from baselines.feature_memory.tests.test_video import clips
from baselines.feature_memory.video import inspect_sources, iter_stream


def _values(frame):
    return (torch.arange(72).reshape(9, 8).float() / 10 + frame).bfloat16()


def _metadata(frame, features):
    return {'frame_id': frame, 'timestamp': float(frame), 'decode_seconds': .01,
            'encoder': {'encode_seconds': .02},
            'shared_feature_sha256': hashlib.sha256(features.view(torch.uint8).numpy().tobytes()).hexdigest()}


def _writer(path, shard_frames=2):
    return ArchiveWriter(path, {'frame_shape': [9, 8], 'plan': {'planned_frames': 5}}, shard_frames)


def test_partial_files_are_invisible_and_shard_roundtrip_is_exact(tmp_path):
    writer = _writer(tmp_path)
    writer.append(_values(0), _metadata(0, _values(0)))
    assert list(iter_archive(tmp_path, wait=False)) == []
    assert len(list(tmp_path.glob('*.partial'))) == 1
    for frame in range(1, 5):
        writer.append(_values(frame), _metadata(frame, _values(frame)))
    # The last uncommitted frame cannot be read until final publication.
    assert sum(len(row['frames']) for row in iter_archive(tmp_path, wait=False)) == 4
    receipt = writer.complete(5)
    assert receipt['payload_bytes'] == 5 * 9 * 8 * 2
    batches = list(iter_archive(tmp_path))
    assert [row['shard_id'] for row in batches] == [0, 1, 2]
    assert [frame['frame_id'] for row in batches for frame in row['frames']] == list(range(5))
    actual = torch.cat([row['features'] for row in batches])
    torch.testing.assert_close(actual, torch.stack([_values(i) for i in range(5)]), rtol=0, atol=0)
    assert actual.dtype == torch.bfloat16
    limited = list(iter_archive(tmp_path, max_frames=3))
    assert [len(row['frames']) for row in limited] == [2, 1]
    assert limited[-1]['features'].untyped_storage().nbytes() == 9 * 8 * 2
    assert limited[-1]['file_bytes_read'] == 2 * 9 * 8 * 2
    assert not list(tmp_path.glob('*.partial'))


@pytest.mark.parametrize('which', ['feature', 'index'])
def test_corrupted_committed_shard_fails_before_yield(tmp_path, which):
    writer = _writer(tmp_path)
    for frame in range(2):
        writer.append(_values(frame), _metadata(frame, _values(frame)))
    path = tmp_path / ('features_00000000.bin' if which == 'feature' else 'index_00000000.json')
    with path.open('r+b') as handle:
        old = handle.read(1)
        handle.seek(0)
        handle.write(bytes([old[0] ^ 1]))
    with pytest.raises(ArchiveError, match='SHA256'):
        next(iter_archive(tmp_path, wait=False))


def test_failed_producer_wakes_reader_without_complete_receipt(tmp_path):
    _writer(tmp_path)
    (tmp_path / 'status.json').write_text(json.dumps({'status': 'failed', 'error_type': 'FixtureError', 'error': 'failed intentionally'}))
    with pytest.raises(ArchiveError, match='producer failed'):
        next(iter_archive(tmp_path, wait=True))


class FakeEncoder:
    calls = []

    def __init__(self, *args, **kwargs):
        self.metadata = {'fixture': True}

    def encode(self, image):
        frame = image.getpixel((0, 0))[0]
        self.calls.append(frame)
        return _values(frame), {'encode_seconds': .02, 'grid_thw': [1, 6, 6]}


def _setup_build(tmp_path, monkeypatch, seed=False):
    import baselines.feature_memory.video as video
    import baselines.feature_memory.planning as planning
    count = 12
    inventory = {'sources': [{'source_fingerprint': '0' * 64}], 'total_1fps_frames': count}
    plan = {'dtype': 'bfloat16', 'hidden_dim': 8, 'tokens_per_frame': 9,
            'post_merger_grid_hw': [3, 3], 'planned_frames': count,
            'planned_run_feature_bytes': count * 9 * 8 * 2}
    monkeypatch.setattr(video, 'inspect_sources', lambda *a, **k: inventory)
    monkeypatch.setattr(planning, 'make_plan', lambda *a, **k: plan)

    def frames(inv, max_seconds=None, stats=None, start_frame=0):
        stats['start_frame'] = start_frame
        for frame in range(start_frame, count):
            with Image.new('RGB', (12, 12), (frame, 0, 0)) as image:
                yield {'frame_id': frame, 'timestamp': float(frame), 'frame': image, 'decode_seconds': .01}
    monkeypatch.setattr(video, 'iter_stream', frames)
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'config.json').write_text('{}')
    (model / 'preprocessor_config.json').write_text('{}')
    seed_path = None
    if seed:
        seed_path = tmp_path / 'seed'
        seed_path.mkdir()
        (seed_path / 'status.json').write_text(json.dumps({'status': 'failed', 'frames_processed': 5}))
        (seed_path / 'inventory.json').write_text(json.dumps(inventory))
        (seed_path / 'budget_plan.json').write_text(json.dumps(plan))
        encoder_file = Path(__file__).parents[1] / 'encoder.py'
        (seed_path / 'run_config.json').write_text(json.dumps({'source_sha256': {'encoder.py': hashlib.sha256(encoder_file.read_bytes()).hexdigest()}}))
        bank = ByteBudgetMemory('fifo', 100000, .05, hidden_dim=8, tokens_per_frame=9,
                               directory=seed_path / 'live' / 'fifo' / '5pct', shard_bytes=512)
        with (seed_path / 'frames.jsonl').open('w') as handle:
            for frame in range(5):
                features = _values(frame)
                bank.update(features, frame, float(frame))
                handle.write(json.dumps(_metadata(frame, features)) + '\n')
        # An interrupted next frame can have extra retained data, which must
        # not be reused unless its commit journal record exists.
        bank.update(_values(5), 5, 5.0)
        bank.close()
    return Namespace(videos=['unused'], model=model, output=tmp_path / 'archive',
                     mode='sequential', devices=['cpu'], archive_frames_per_shard=4,
                     seed_run=seed_path), plan


@pytest.mark.parametrize('seed', [False, True])
def test_build_counts_new_encoding_once_and_reuses_only_committed_seed(tmp_path, monkeypatch, seed):
    args, plan = _setup_build(tmp_path, monkeypatch, seed)
    FakeEncoder.calls = []
    receipt = build_archive(args, encoder_factory=FakeEncoder)
    first = 5 if seed else 0
    assert FakeEncoder.calls == list(range(first, 12))
    assert receipt['seed_frames_reused'] == first
    assert receipt['new_encoder_calls'] == 12 - first
    assert receipt['total_frames'] == 12
    rows = list(iter_archive(args.output))
    actual = torch.cat([row['features'] for row in rows])
    torch.testing.assert_close(actual, torch.stack([_values(i) for i in range(12)]), rtol=0, atol=0)
    assert json.loads((args.output / 'status.json').read_text())['status'] == 'completed'


def test_resumed_video_sampling_keeps_exact_global_ids_and_timestamps(clips):
    inventory = inspect_sources(clips)
    expected = [(row['frame_id'], row['timestamp']) for row in iter_stream(inventory)]
    stats = {}
    actual = [(row['frame_id'], row['timestamp']) for row in iter_stream(inventory, start_frame=4, stats=stats)]
    assert actual == expected[4:]
    assert stats['prefix_frames_decoded_skipped'] == 4
    assert stats['frames_yielded'] == 2
    assert stats['decode_prefix_skipped_seconds'] > 0
