"""Single-frame causal sampling from small real synthetic video files."""
from copy import deepcopy
from fractions import Fraction
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from streamforest_videomem.video import frame_key, iter_frames, time_key
from local_videomem.data import decode_prefix, prepare_manifest


ROOT = Path(__file__).resolve().parents[1]


def cutoff(seconds=8.5, day_id='Day2', day_order=2):
    return {'day_id': day_id, 'day_order': day_order, 'seconds': seconds}


@pytest.fixture(scope='session')
def source_media(tmp_path_factory):
    pytest.importorskip('av')
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('Real media tests require FFmpeg and FFprobe')
    directory = tmp_path_factory.mktemp('streamforest_test_only')
    green, red_blue = directory / 'green.mp4', directory / 'red_blue.mp4'
    common = ['ffmpeg', '-v', 'error']
    commands = [
        common + ['-f', 'lavfi', '-i', 'color=c=green:s=96x64:r=4:d=3',
                  '-an', '-c:v', 'mpeg4', str(green)],
        common + ['-f', 'lavfi', '-i', 'color=c=red:s=96x64:r=4:d=2',
                  '-f', 'lavfi', '-i', 'color=c=blue:s=96x64:r=4:d=2',
                  '-filter_complex', '[0:v][1:v]concat=n=2:v=1:a=0[out]', '-map', '[out]',
                  '-an', '-c:v', 'mpeg4', str(red_blue)],
    ]
    for command in commands:
        subprocess.run(command, check=True, capture_output=True, timeout=30)
    return green, red_blue


@pytest.fixture
def sources(tmp_path, source_media):
    green, red_blue = source_media
    rows = []
    for name, original, day_id, order, offset in (
        ('prior.mp4', green, 'Day1', 1, 0),
        ('current.mp4', red_blue, 'Day2', 2, 7),
        ('later.mp4', green, 'Day2', 2, 11),
        ('future.mp4', green, 'Day3', 3, 0),
    ):
        path = tmp_path / name
        shutil.copyfile(original, path)
        rows.append({'path': str(path), 'day_id': day_id, 'day_order': order,
                     'day_start_seconds': offset, 'timeline_status': 'direct_mp4'})
    return rows


def assert_closed(image):
    with pytest.raises(ValueError, match='closed'):
        image.getpixel((0, 0))


def test_import_needs_no_decoder_or_model_runtime():
    result = subprocess.run([sys.executable, '-c',
        "import sys; import streamforest_videomem.video; "
        "assert not {'av', 'torch', 'transformers'}.intersection(sys.modules)"],
        cwd=ROOT, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr


def test_native_resolution_prior_days_and_future_pixels_excluded(sources, monkeypatch):
    import av
    manifest = prepare_manifest(list(reversed(sources)))
    requested = {**cutoff(), 'ground_truth': {'answer': 'PRIVATE_LABEL'}}
    saved = deepcopy(requested)
    opened = []
    original_open = av.open

    def track(path, *args, **kwargs):
        opened.append(Path(path).name)
        return original_open(path, *args, **kwargs)

    original_read = Path.read_bytes

    def reject_whole_video(path):
        if path.suffix == '.mp4':
            pytest.fail('Decoder must stream video rather than read the entire file')
        return original_read(path)

    monkeypatch.setattr(av, 'open', track)
    monkeypatch.setattr(Path, 'read_bytes', reject_whole_video)
    stats, records, images = {}, [], []
    for item in iter_frames(manifest, requested, stats=stats):
        image, metadata = item['frame'], item['metadata']
        assert image.mode == 'RGB' and image.size == (96, 64)
        assert metadata['pixel_sha256'] == hashlib.sha256(image.tobytes()).hexdigest()
        assert metadata['decode_device'] == 'cpu'
        assert frame_key(item) < time_key(cutoff())
        red, green, blue = image.getpixel((0, 0))
        if metadata['day_id'] == 'Day1':
            assert green > max(red, blue) + 60
        else:
            assert red > max(green, blue) + 100
        records.append((item['frame_index'], metadata))
        images.append(image)
    assert requested == saved
    assert opened == ['prior.mp4', 'current.mp4']
    assert [index for index, _ in records] == list(range(5))
    assert [row['file_seconds'] for _, row in records] == [0, 1, 2, 0, 1]
    assert stats['sampled_frames'] == stats['yielded_frames'] == 5
    assert stats['max_live_sampled_frames'] == 1
    for image in images:
        assert_closed(image)


def test_optional_thumbnail_matches_existing_single_frame_sampler(sources):
    manifest = prepare_manifest([sources[1]])
    question = {'question_id': 'comparison', 'question': 'What is visible?',
                'choices': [], 'query_time': cutoff(11)}
    reference = decode_prefix(manifest, question, fps=1, max_frames=4, max_side=32)
    try:
        for index, item in enumerate(iter_frames(manifest, cutoff(11), max_side=32)):
            assert item['metadata'] == reference['frame_metadata'][index]
            assert item['frame'].tobytes() == reference['frames'][index].tobytes()
            assert max(item['frame'].size) <= 32
        assert index == 3
    finally:
        for image in reference['frames']:
            image.close()


@pytest.mark.parametrize('seconds,expected', [(7, []), (7.4, [0]), (7.5, [0]),
                                            (8, [0, 0.5, 0.75])])
def test_exact_pts_half_open_cutoff_includes_unpaired_single_frames(sources, seconds, expected):
    manifest = prepare_manifest([sources[1]])
    items = list(iter_frames(manifest, cutoff(seconds), fps=3))
    assert [item['metadata']['file_seconds'] for item in items] == expected
    assert all(frame_key(item) < time_key(cutoff(seconds)) for item in items)


def test_high_sampling_rate_never_duplicates_actual_pts(sources):
    manifest = prepare_manifest([sources[1]])
    items = list(iter_frames(manifest, cutoff(8), fps=20))
    assert [item['metadata']['file_seconds'] for item in items] == [0, 0.25, 0.5, 0.75]
    assert len({item['metadata']['pts'] for item in items}) == 4


def test_advance_and_early_close_release_each_image(sources):
    iterator = iter_frames(prepare_manifest([sources[1]]), cutoff(11), fps=4)
    previous = None
    try:
        for _ in range(7):
            item = next(iterator)
            if previous is not None:
                assert_closed(previous)
            previous = item['frame']
            previous.getpixel((0, 0))
    finally:
        iterator.close()
    assert_closed(previous)


def test_changed_source_rejected_before_open(sources, monkeypatch):
    import av
    manifest = prepare_manifest([sources[1]])
    with Path(sources[1]['path']).open('ab') as handle:
        handle.write(b'changed')
    monkeypatch.setattr(av, 'open', lambda *args, **kwargs: pytest.fail('Changed source must not open'))
    with pytest.raises(ValueError, match='Source changed'):
        next(iter_frames(manifest, cutoff(11)))


@pytest.mark.parametrize('finish', ['next', 'close'])
def test_changed_source_during_iteration_rejected_and_image_closed(sources, finish):
    iterator = iter_frames(prepare_manifest([sources[1]]), cutoff(11))
    item = next(iterator)
    with Path(sources[1]['path']).open('ab') as handle:
        handle.write(b'changed during iteration')
    with pytest.raises(ValueError, match='Source changed'):
        next(iterator) if finish == 'next' else iterator.close()
    assert_closed(item['frame'])


def test_time_helpers_preserve_exact_fraction_and_reject_mismatched_envelope():
    metadata = {'day_id': 'Day1', 'day_order': 1, 'day_seconds': 1 / 3,
                'day_seconds_fraction': '1/3'}
    assert frame_key(metadata) == (1, Fraction(1, 3))
    assert time_key({'query_time': cutoff()}) == (2, Fraction(17, 2))
    with pytest.raises(ValueError, match='disagree'):
        frame_key({'time': cutoff(), 'metadata': metadata})


@pytest.mark.parametrize('kwargs', [{'fps': 0}, {'fps': float('nan')}, {'fps': True},
                                    {'max_side': 0}, {'max_side': True}, {'stats': []}])
def test_invalid_sampling_options_rejected(kwargs):
    with pytest.raises(ValueError):
        next(iter_frames({'videos': []}, cutoff(), **kwargs))
