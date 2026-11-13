"""Real decoding plus cheap encoded-feature fixtures for runner integration."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import pytest
import torch
from safetensors.torch import load_file

from baselines.feature_memory.runner import execute
from baselines.feature_memory.storage import HistoryWriter


class FixtureEncoder:
    """Explicit test fixture; never represents a pretrained visual model."""

    calls = 0
    instances = 0

    def __init__(self, model, device, dtype):
        type(self).instances += 1
        self.dtype = getattr(torch, dtype)
        self.metadata = {'implementation': 'test_fixture_only', 'dtype': dtype}

    def encode(self, frame):
        assert frame.mode == 'RGB'
        offset = type(self).calls * 16
        type(self).calls += 1
        features = (torch.arange(15).reshape(3, 5) + offset).to(self.dtype)
        return features, {'grid_thw': [1, 2, 6], 'preprocess_seconds': 0.001,
                          'encode_seconds': 0.002, 'h2d_seconds': 0.0,
                          'd2h_seconds': 0.0}


@pytest.fixture(scope='module')
def tiny_video(tmp_path_factory):
    pytest.importorskip('av')
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('FFmpeg and FFprobe are required for real decoder integration')
    path = tmp_path_factory.mktemp('feature_runner_video') / 'day1.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                    'testsrc2=s=96x64:r=4:d=3', '-an', '-c:v', 'mpeg4', str(path)],
                   check=True, capture_output=True, timeout=30)
    return path


def arguments(video, output):
    return argparse.Namespace(videos=[Path(video)], dataset_status=None, timeline=None,
                              output=Path(output), model=Path('/not-loaded/test-fixture'),
                              device='cpu', dtype='bfloat16', budgets=[512, 1024, 2048, 4096],
                              snapshot_seconds=[1.0, 2.0, 3.0, 4.0], max_seconds=3.0,
                              seed=1729, no_history_archive=False)


def read_history(directory):
    index = [json.loads(line) for line in (directory / 'index.jsonl').read_text().splitlines()]
    raw = (directory / 'features.bin').read_bytes()
    rows = []
    for entry in index:
        part = bytearray(raw[entry['offset']:entry['offset'] + entry['nbytes']])
        tensor = torch.frombuffer(part, dtype=getattr(torch, entry['dtype'])).clone().reshape(entry['shape'])
        rows.append((entry, tensor))
    return raw, rows


@pytest.fixture
def completed_run(tmp_path, tiny_video):
    FixtureEncoder.calls = FixtureEncoder.instances = 0
    args = arguments(tiny_video, tmp_path / 'run')
    result = execute(args, encoder_factory=FixtureEncoder)
    return args, result


def test_one_traversal_causal_snapshots_all_budgets_and_no_raw_frames(completed_run):
    args, result = completed_run
    assert FixtureEncoder.calls == 3
    assert FixtureEncoder.instances == 1
    assert result['total_frames_seen'] == 3
    assert result['decoder_stats']['source_open_count'] == 1
    assert result['decoder_stats']['sampled_frames'] == 3
    assert result['decoder_stats']['max_live_sampled_frames'] == 1
    assert result['requested_snapshots_not_reached'] == [4.0]
    assert len(result['final_summary']) == 12
    for row in result['final_summary']:
        assert row['total_frames_seen'] == 3
        assert row['retained_visual_tokens'] == 9
        assert row['retained_items'] == 3
        assert row['actual_memory_bytes'] >= row['allocated_feature_bytes']
    for summary_path in sorted((args.output / 'snapshots').glob('*/summary.json')):
        summary = json.loads(summary_path.read_text())
        cutoff = summary['timestamp']
        assert summary['total_frames_seen'] == int(cutoff)
        assert summary['last_retained_input_timestamp'] < cutoff
        assert summary['history_serialization']['frames'] == int(cutoff)
        for row in summary['rows']:
            assert row['total_frames_seen'] == int(cutoff)
            assert all(t < cutoff for t in row['retained_frame_timestamps'])
            policy_dir = summary_path.parent / f"{row['method']}_B{row['budget']}"
            tensors = load_file(str(policy_dir / 'features.safetensors'))
            assert tensors['features'].dtype == torch.bfloat16
            assert tensors['frame_timestamps'].max().item() < cutoff
            actual_disk = sum(p.stat().st_size for p in policy_dir.iterdir() if p.is_file())
            assert row['snapshot_serialized_bytes'] == actual_disk
    prohibited = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.mp4', '.avi'}
    assert not any(p.suffix.lower() in prohibited for p in args.output.rglob('*') if p.is_file())
    assert result['timing_seconds']['preprocess_seconds'] == pytest.approx(0.003)
    assert result['timing_seconds']['encode_seconds'] == pytest.approx(0.006)
    assert result['timing_seconds']['decode_seconds'] > 0


def test_bfloat16_history_offsets_payload_sizes_and_dtype_probes(completed_run):
    args, result = completed_run
    directory = args.output / 'history'
    raw, recovered = read_history(directory)
    assert len(raw) == result['full_history_observed_tensor_bytes'] == 3 * 3 * 5 * 2
    assert len(recovered) == 3
    for i, (entry, tensor) in enumerate(recovered):
        assert entry['offset'] == i * 30
        assert entry['nbytes'] == 30
        assert entry['frame_id'] == i
        assert entry['timestamp'] == i
        assert torch.equal(tensor, (torch.arange(15).reshape(3, 5) + i * 16).to(torch.bfloat16))
    measured = result['full_history_observed_serialization']
    assert measured['serialized_file_bytes'] == sum(p.stat().st_size for p in directory.iterdir())
    assert measured['serialized_file_bytes'] > measured['tensor_payload_bytes']
    for dtype, element in [('float16', 2), ('bfloat16', 2), ('float32', 4)]:
        probe = result['dtype_measurements'][dtype]
        assert probe['tensor_payload_bytes'] == 15 * element
        file = args.output / 'dtype_probes' / (dtype + '.safetensors')
        assert probe['serialized_file_bytes'] == file.stat().st_size
        assert load_file(str(file))['features'].dtype == getattr(torch, dtype)


def test_existing_output_is_not_reused_or_overwritten(completed_run):
    args, _ = completed_run
    before = (args.output / 'summary.json').read_bytes()
    with pytest.raises(FileExistsError):
        execute(args, encoder_factory=FixtureEncoder)
    assert FixtureEncoder.calls == 3
    assert FixtureEncoder.instances == 1
    assert (args.output / 'summary.json').read_bytes() == before


def test_invalid_source_fails_before_encoder_or_output_creation(tmp_path):
    FixtureEncoder.calls = FixtureEncoder.instances = 0
    args = arguments(tmp_path / 'missing.mp4', tmp_path / 'invalid_run')
    args.max_seconds = None
    with pytest.raises((ValueError, FileNotFoundError)):
        execute(args, encoder_factory=FixtureEncoder)
    assert FixtureEncoder.instances == 0
    assert not args.output.exists()


def test_encoder_initialization_failure_is_recorded(tmp_path, tiny_video):
    args = arguments(tiny_video, tmp_path / 'encoder_failed')

    def fail_factory(*a, **kw):
        raise RuntimeError('fixture initialization failure')

    with pytest.raises(RuntimeError, match='fixture initialization failure'):
        execute(args, encoder_factory=fail_factory)
    status = json.loads((args.output / 'status.json').read_text())
    assert status['status'] == 'failed'
    assert status['frames_seen'] == 0
    assert status['error_type'] == 'RuntimeError'
    assert not (args.output / 'history').exists()


def test_archive_roundtrip_variable_shapes_and_noncontiguous_rejection(tmp_path):
    directory = tmp_path / 'history'
    writer = HistoryWriter(directory)
    frames = [torch.arange(6, dtype=torch.float32).reshape(2, 3).to(torch.bfloat16),
              torch.arange(9, dtype=torch.float32).reshape(3, 3).to(torch.bfloat16)]
    try:
        writer.append(frames[0], 0, 0.0, {})
        before = writer.sizes()
        with pytest.raises(ValueError):
            writer.append(frames[0].T, 1, 1.0, {})
        assert writer.sizes() == before
        writer.append(frames[1], 1, 1.0, {})
    finally:
        writer.close()
    _, recovered = read_history(directory)
    assert len(recovered) == 2
    for expected, (_, actual) in zip(frames, recovered):
        assert torch.equal(expected, actual)
    assert recovered[1][0]['offset'] == 12
