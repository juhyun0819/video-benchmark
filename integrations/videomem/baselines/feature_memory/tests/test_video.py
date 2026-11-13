"""Real decoder checks for a single-pass, audio-free cumulative stream."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from baselines.feature_memory.video import inspect_sources, iter_stream


@pytest.fixture
def clips(tmp_path):
    pytest.importorskip('av')
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg/ffprobe required')
    paths = [tmp_path / 'day1.mp4', tmp_path / 'day2.mp4']
    for path, color in zip(paths, ['red', 'blue']):
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                        f'color=c={color}:s=64x64:r=4:d=2.25', '-f', 'lavfi', '-i',
                        'sine=frequency=400:sample_rate=8000:duration=2.25',
                        '-c:v', 'mpeg4', '-c:a', 'aac', '-shortest', str(path)],
                       check=True, capture_output=True)
    return paths


def test_single_pass_cross_source_clock_audio_never_decoded(clips, monkeypatch):
    import av
    inventory = inspect_sources(clips)
    assert inventory['total_duration_seconds'] == 4.5
    assert inventory['total_1fps_frames'] == 6
    assert inventory['all_verified']
    opened = []
    original = av.open

    class VideoOnly:
        def __init__(self, *args, **kwargs):
            opened.append(Path(args[0]).name)
            self.inner = original(*args, **kwargs)

        def __enter__(self):
            self.inner.__enter__()
            return self

        def __exit__(self, *args):
            return self.inner.__exit__(*args)

        @property
        def streams(self):
            return self.inner.streams

        def decode(self, stream):
            assert stream.type == 'video'
            yield from self.inner.decode(stream)

    monkeypatch.setattr(av, 'open', VideoOnly)
    stats, seen, images = {}, [], []
    for row in iter_stream(inventory, stats=stats):
        seen.append((row['frame_id'], row['timestamp'], row['source_timestamp']))
        images.append(row['frame'])
        assert row['decode_seconds'] >= 0
        assert row['frame'].size == (64, 64)
    assert seen == [(0, 0, 0), (1, 1, 1), (2, 2, 2),
                    (3, 2.25, 0), (4, 3.25, 1), (5, 4.25, 2)]
    assert opened == ['day1.mp4', 'day2.mp4']
    assert stats['sampled_frames'] == stats['expected_1fps_frames'] == 6
    assert stats['source_open_count'] == 2
    assert stats['max_live_sampled_frames'] == 1
    assert stats['decode_seconds'] >= stats['decode_tail_seconds'] >= 0
    for image in images:
        with pytest.raises(ValueError, match='closed'):
            image.getpixel((0, 0))


def test_partial_prefix_causal_end_exclusive_and_close(clips):
    inventory = inspect_sources(clips)
    times = [r['timestamp'] for r in iter_stream(inventory, max_seconds=3.25)]
    assert times == [0, 1, 2, 2.25]
    stream = iter_stream(inventory, max_seconds=1)
    first = next(stream)
    stream.close()
    with pytest.raises(ValueError, match='closed'):
        first['frame'].getpixel((0, 0))


def test_pending_future_source_allowed_for_smoke_but_not_full(clips, tmp_path):
    status = tmp_path / 'download_status.json'
    first, second = clips
    digest = hashlib.sha256(first.read_bytes()).hexdigest()
    status.write_text(json.dumps({'files': [
        {'path': first.name, 'status': 'verified', 'local_path': str(first),
         'size': first.stat().st_size, 'sha256': digest, 'lfs_sha256': digest},
        {'path': second.name, 'status': 'downloading'}]}))
    timeline = tmp_path / 'timeline.json'
    timeline.write_text(json.dumps({'days': [{'day': 2, 'duration_seconds': 2.25}]}))
    inventory = inspect_sources(clips, status_path=status, timeline_path=timeline)
    assert not inventory['all_verified']
    assert inventory['counts_provisional']
    assert inventory['total_1fps_frames'] == 6
    assert inventory['sources'][1]['duration_provisional']
    assert inventory['sources'][0]['provenance']['fingerprint_reused_from_verified_download']
    assert [r['timestamp'] for r in iter_stream(inventory, max_seconds=2)] == [0, 1]
    with pytest.raises(ValueError, match='every selected source'):
        list(iter_stream(inventory))
    with pytest.raises(ValueError, match='unverified source'):
        list(iter_stream(inventory, max_seconds=3))


def test_changed_source_and_duplicates_fail(clips):
    with pytest.raises(ValueError, match='Duplicate'):
        inspect_sources([clips[0], clips[0]])
    inventory = inspect_sources(clips)
    with clips[0].open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match='Source changed'):
        list(iter_stream(inventory, max_seconds=1))


def test_distinct_paths_with_duplicate_day_basename_fail_before_probe(tmp_path, monkeypatch):
    from baselines.feature_memory import video
    first = tmp_path / 'first' / 'day1.mp4'
    second = tmp_path / 'second' / 'day1.mp4'
    for path in (first, second):
        path.parent.mkdir()
        path.write_bytes(b'not decoded or probed')

    def unexpected(*args, **kwargs):
        pytest.fail('Duplicate Day IDs must fail before probing or hashing files')

    monkeypatch.setattr(video, '_probe', unexpected)
    monkeypatch.setattr(video, 'prepare_manifest', unexpected)
    with pytest.raises(ValueError, match='Source Day IDs must be distinct'):
        inspect_sources([first, second])
