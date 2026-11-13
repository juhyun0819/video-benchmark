"""Real CFR fixtures check prefix boundaries and exact source pixel selection."""
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from baselines.qwen3_offline.video import (
    build_inventory, decode_indices, map_indices, prefix_frame_count,
)


@pytest.fixture
def videos(tmp_path):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg/ffprobe required')
    paths = []
    for day, color in enumerate(['red', 'blue'], 1):
        path = tmp_path / f'day{day}.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                        f'color=c={color}:s=64x64:r=4:d=1', '-c:v', 'libx264',
                        '-pix_fmt', 'yuv420p', str(path)], check=True)
        paths.append(str(path))
    return paths


def test_inventory_counts_and_day_offsets(videos):
    value = build_inventory(videos)
    assert value['fps'] == '4'
    assert value['total_frames'] == 8
    assert value['duration_fraction'] == '2'
    assert [x['global_start_fraction'] for x in value['sources']] == ['0', '1']
    assert [x['day_id'] for x in value['sources']] == ['Day1', 'Day2']
    assert [x['num_frames'] for x in value['sources']] == [4, 4]


@pytest.mark.parametrize('cutoff,count', [(0, 1), (.249, 1), (.25, 2), (.999, 4), (1, 5), (2, 8)])
def test_inclusive_cutoff_and_cross_day(videos, cutoff, count):
    assert prefix_frame_count(build_inventory(videos), cutoff) == count


@pytest.mark.parametrize('cutoff', [-1, 2.001, float('nan'), float('inf'), True, '1'])
def test_invalid_cutoff_rejected(videos, cutoff):
    with pytest.raises(ValueError):
        prefix_frame_count(build_inventory(videos), cutoff)


def test_index_mapping_keeps_order_duplicates_and_raw_timestamps(videos):
    rows = map_indices(build_inventory(videos), [0, 3, 4, 4, 5], 1.25)
    assert [r['frame_id'] for r in rows] == [0, 3, 4, 4, 5]
    assert [r['source_frame_index'] for r in rows] == [0, 3, 0, 0, 1]
    assert [r['global_seconds_fraction'] for r in rows] == ['0', '3/4', '1', '1', '5/4']
    assert [r['source_seconds_fraction'] for r in rows] == ['0', '3/4', '0', '0', '1/4']
    assert rows[2]['day_id'] == 'Day2'


@pytest.mark.parametrize('indices', [[-1], [4], [True], [1.2], [1, 0]])
def test_invalid_or_future_selection_rejected(videos, indices):
    with pytest.raises(ValueError):
        map_indices(build_inventory(videos), indices, .75)


def test_decode_matches_full_raw_decode_without_resizing(videos):
    import av
    reference = []
    for path in videos:
        with av.open(path) as container:
            reference.extend(f.to_ndarray(format='rgb24') for f in container.decode(video=0))
    selected = [0, 3, 4, 4, 5]
    frames, rows = decode_indices(build_inventory(videos), selected, 1.25)
    assert len(frames) == len(rows) == 5
    for frame, index in zip(frames, selected):
        assert frame.shape == (64, 64, 3)
        np.testing.assert_array_equal(frame, reference[index])
    assert all(Fraction(r['global_seconds_fraction']) <= Fraction(5, 4) for r in rows)


def test_source_change_rejected(videos):
    value = build_inventory(videos)
    with Path(videos[0]).open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='changed'):
        decode_indices(value, [0], 0)


@pytest.mark.parametrize('field,value', [('global_start_fraction', '1/4'), ('fps', '5'),
                                         ('num_frames', 5), ('width', 32), ('day_order', 3)])
def test_inventory_tampering_rejected(videos, field, value):
    inventory = build_inventory(videos)
    inventory['sources'][1][field] = value
    with pytest.raises(ValueError):
        prefix_frame_count(inventory, 1)


def test_different_rates_rejected(videos, tmp_path):
    other = tmp_path / 'other.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                    'color=c=green:s=64x64:r=5:d=1', '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p', str(other)], check=True)
    with pytest.raises(ValueError, match='rate|CFR'):
        build_inventory([videos[0], str(other)])


def test_non_cfr_metadata_rejected(videos, monkeypatch):
    import baselines.qwen3_offline.video as module
    original = module.subprocess.run
    def altered(*args, **kwargs):
        result = original(*args, **kwargs)
        data = json.loads(result.stdout)
        data['streams'][0]['avg_frame_rate'] = '5/1'
        result.stdout = json.dumps(data)
        return result
    monkeypatch.setattr(module.subprocess, 'run', altered)
    with pytest.raises(ValueError, match='CFR'):
        build_inventory(videos)


def test_exact_fraction_cutoff(videos):
    assert prefix_frame_count(build_inventory(videos), Fraction(5, 4)) == 6


def test_public_inventory_validation(videos):
    from baselines.qwen3_offline.video import validate_inventory
    inventory = build_inventory(videos)
    validate_inventory(inventory)
    inventory['total_frames'] = 9
    with pytest.raises(ValueError):
        validate_inventory(inventory)
