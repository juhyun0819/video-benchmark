import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / 'extract_visual_tokens.py'
spec = importlib.util.spec_from_file_location('extract_visual_tokens_tested', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_cli_and_repository_imports_work_outside_checkout(tmp_path):
    result = subprocess.run([sys.executable, '-I', str(SCRIPT), '--help'], cwd=tmp_path,
                            text=True, capture_output=True, check=True)
    assert '--dataset-root' in result.stdout and '--max-frames-per-source' in result.stdout
    # -I removes inherited PYTHONPATH and the current directory from imports.
    # The entrypoint must locate its own checkout before importing the decoder.
    program = ('import runpy; '
               f'runpy.run_path({str(SCRIPT)!r}, run_name="portability_check"); '
               'from baselines.feature_memory import video; print(video.__file__)')
    imported = subprocess.run([sys.executable, '-I', '-c', program], cwd=tmp_path,
                              text=True, capture_output=True, check=True)
    assert str(module.REPO / 'baselines/feature_memory/video.py') in imported.stdout
    rejected = subprocess.run([sys.executable, '-I', str(SCRIPT), '--model', '/model',
                               '--output', '/output'], cwd=tmp_path, text=True, capture_output=True)
    assert rejected.returncode == 2 and '--dataset-root' in rejected.stderr


def test_cli_uses_only_gpu_zero_unless_explicitly_selected(monkeypatch):
    captured = []
    monkeypatch.setattr(module, 'run', lambda args: captured.append(args))
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), '--dataset-root', '/dataset',
                                    '--model', '/model', '--output', '/output'])
    module.main()
    assert captured[0].devices == ['0']
    assert captured[0].dataset_root == '/dataset'
    assert captured[0].source_map is None


@pytest.mark.parametrize('mask,devices', [(None, ['0', 'cuda:0']),
                                         ('2,3', ['0', 'cuda:0']),
                                         ('2,2', ['0', '1'])])
def test_device_aliases_rejected_before_output_creation(tmp_path, monkeypatch, mask, devices):
    if mask is None:
        monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    else:
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', mask)
    output = tmp_path / 'must-not-exist'
    # The duplicate must fail before even looking up the nonexistent inputs.
    args = SimpleNamespace(devices=devices, dataset_root=tmp_path / 'missing', output=output)
    with pytest.raises(ValueError, match='same CUDA GPU'):
        module.run(args)
    assert not output.exists()


def test_distinct_logical_gpu_indices_resolve_within_visible_mask(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2,3')
    assert module.validate_device_selection(['0', 'cuda:1']) == ['2', '3']


class FakeEncoder:
    metadata = {'llm_loaded': False, 'qa_inference': False, 'test_only': True}

    def __init__(self):
        self.calls = 0

    def encode(self, image):
        self.calls += 1
        return torch.full((image.width // 16, 8), self.calls, dtype=torch.bfloat16), {'encode_seconds': 0.01}


@pytest.fixture
def videos(tmp_path):
    pytest.importorskip('av')
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg/ffprobe required')
    root = tmp_path / 'cook'
    for group, name, width, duration in [('work','alpha',64,'2.4'),('home','beta',80,'1.6')]:
        path = root / 'videos' / group / (name + '.mp4')
        path.parent.mkdir(parents=True)
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i',f'color=c=red:size={width}x48:rate=10',
                        '-t',duration,'-an','-c:v','libx264','-pix_fmt','yuv420p','-threads','1',str(path)], check=True)
    return root


def test_per_source_native_shapes_local_clocks_and_full_tail(videos, tmp_path):
    from baselines.feature_memory.archive import iter_archive
    sources = module.discover_sources(videos)
    encoder = FakeEncoder()
    output = tmp_path / 'output'
    receipts = []
    for source in sources:
        receipt = module.extract_source(source, encoder, output, {'test_model': True}, frames_per_shard=2)
        receipts.append(receipt)
        assert receipt['full_source_encoded'] is True
        assert receipt['decoder_stats']['sampled_frames'] == receipt['frames']
        assert 'decode_tail_seconds' in receipt['decoder_stats']
        assert receipt['timings']['archive_write_seconds'] > 0
        directory = output / source['archive_relative_path']
        loaded = list(iter_archive(directory, wait=False))
        frames = [row for batch in loaded for row in batch['frames']]
        assert [row['frame_id'] for row in frames] == list(range(receipt['frames']))
        assert [row['timestamp'] for row in frames] == [float(i) for i in range(receipt['frames'])]
        assert all(row['source_relative_path'] == source['relative_path'] for row in frames)
        assert module.validate_completed(directory, source, {'test_model': True}, None, deep=True)['frames'] == receipt['frames']
    assert sorted(r['frames'] for r in receipts) == [2,3]
    assert encoder.calls == 5
    shapes = [json.loads((output / s['archive_relative_path'] / 'archive.json').read_text())['frame_shape'] for s in sources]
    assert sorted(shapes) == [[4,8],[5,8]]


def test_smoke_prefix_and_changed_archive_not_reused(videos, tmp_path):
    source = module.discover_sources(videos, ['videos/work/alpha.mp4'])[0]
    output = tmp_path / 'smoke'
    encoder = FakeEncoder()
    receipt = module.extract_source(source, encoder, output, {'test_model': True}, max_frames=2)
    assert receipt['frames'] == encoder.calls == 2
    assert receipt['expected_full_source_frames'] == 3 and not receipt['full_source_encoded']
    directory = output / source['archive_relative_path']
    module.validate_completed(directory, source, {'test_model': True}, 2)
    with pytest.raises(ValueError, match='scope'):
        module.validate_completed(directory, source, {'test_model': True}, None)
    path = directory / 'features_00000000.bin'
    data = bytearray(path.read_bytes()); data[0] ^= 1; path.write_bytes(data)
    with pytest.raises(ValueError, match='changed'):
        module.validate_completed(directory, source, {'test_model': True}, 2)


def test_incomplete_source_and_duplicate_physical_files(videos, tmp_path):
    source = module.discover_sources(videos)[0]
    incomplete = tmp_path / 'incomplete'; incomplete.mkdir()
    with pytest.raises(ValueError, match='incomplete'):
        module.validate_completed(incomplete, source, {'test_model': True}, None)
    with pytest.raises(ValueError, match='Duplicate'):
        module.discover_sources(videos, [source['relative_path'], source['relative_path']])


def test_source_change_during_encoding_rejected(videos, tmp_path):
    source = module.discover_sources(videos, ['videos/home/beta.mp4'])[0]
    class ChangingEncoder(FakeEncoder):
        def encode(self, image):
            value = super().encode(image)
            if self.calls == 1:
                path = Path(source['path'])
                path.touch()
            return value
    with pytest.raises(ValueError, match='changed'):
        module.extract_source(source, ChangingEncoder(), tmp_path / 'changed', {'test_model': True}, max_frames=1)


def test_missing_terminal_sample_preserves_partial_without_false_completion(videos, tmp_path, monkeypatch):
    from baselines.feature_memory import video
    from baselines.feature_memory.archive import iter_archive
    source = module.discover_sources(videos, ['videos/home/beta.mp4'])[0]
    original_iterator = video.iter_stream
    def unavailable_terminal_sample(inventory, **kwargs):
        # A decoder can reach EOF with no eligible PTS for its final 1Hz target.
        # Preserve its strict failure after an earlier real frame was committed.
        stream = original_iterator(inventory, **kwargs)
        try:
            yield next(stream)
            raise ValueError('Actual 1fps samples differ from inventory: 1/2; terminal sample unavailable')
        finally:
            stream.close()
    monkeypatch.setattr(video, 'iter_stream', unavailable_terminal_sample)
    output = tmp_path / 'missing-tail'
    with pytest.raises(ValueError, match='samples differ'):
        module.extract_source(source, FakeEncoder(), output, {'test_model': True}, frames_per_shard=1)
    directory = output / source['archive_relative_path']
    assert (directory / 'ready_00000000.json').is_file()
    assert not (directory / 'source_receipt.json').exists()
    assert not (directory / 'complete.json').exists()
    assert not (output / 'complete.json').exists()
    # The completed early shard remains valid evidence, but resume must not
    # treat this interrupted source as a complete video or silently skip it.
    shard = next(iter_archive(directory, wait=False, max_frames=1))
    assert shard['frames'][0]['frame_id'] == 0
    with pytest.raises(ValueError, match='incomplete'):
        module.validate_completed(directory, source, {'test_model': True}, None)


def make_source_map(videos, tmp_path):
    from baselines.feature_memory.video import _probe
    path = videos / 'videos/work/alpha.mp4'
    probe = _probe(path)
    value = {'schema': 'cook_source_map_v1', 'original_source_count': 2,
             'selected_source_count': 1,
             'sources': [{'relative_path': 'videos/work/logical-original.mp4',
                          'path': str(path), 'original_path': '/missing/cook/videos/work/logical-original.mp4',
                          'original_source_file_state': module.file_state(path),
                          'expected_probe': {'width': probe['width'], 'height': probe['height'],
                                             'duration_fraction': probe['duration_fraction'],
                                             'expected_1fps_frames': 3}}],
             'omitted_sources': [{'relative_path': 'videos/home/unavailable.mp4',
                                  'reason': 'Original source unavailable; no substitute selected'}]}
    output = tmp_path / 'source_map.json'
    output.write_text(json.dumps(value))
    return output, value


def test_source_map_external_file_logical_identity_and_exact_probe(videos, tmp_path):
    from baselines.feature_memory.archive import iter_archive
    path, value = make_source_map(videos, tmp_path)
    sources, scope = module.load_source_map(path)
    source = sources[0]
    assert source['path'] == value['sources'][0]['path']
    assert source['relative_path'] == 'videos/work/logical-original.mp4'
    assert not source['recovery_provenance']['original_sha256_verified']
    assert scope['original_source_count'] == 2 and scope['omitted_source_count'] == 1
    receipt = module.extract_source(source, FakeEncoder(), tmp_path / 'mapped', {'test_model': True})
    directory = tmp_path / 'mapped' / source['archive_relative_path']
    assert receipt['frames'] == 3
    header = json.loads((directory / 'archive.json').read_text())
    assert header['source_sha256'] == module.file_sha256(source['path'])
    rows = [row for batch in iter_archive(directory, wait=False) for row in batch['frames']]
    assert all(row['source_relative_path'] == source['relative_path'] for row in rows)
    source['recovery_provenance']['expected_probe']['width'] += 1
    with pytest.raises(ValueError, match='probe changed'):
        module.extract_source(source, FakeEncoder(), tmp_path / 'bad-probe', {'test_model': True})


def test_source_map_rejects_duplicates_traversal_identity_and_hidden_omission(videos, tmp_path):
    path, original = make_source_map(videos, tmp_path)
    def check(value, message):
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError, match=message):
            module.load_source_map(path)
    value = json.loads(json.dumps(original)); value['omitted_sources'] = []
    check(value, 'account')
    value = json.loads(json.dumps(original)); value['sources'][0]['relative_path'] = 'videos/../escaped.mp4'
    check(value, 'logical')
    value = json.loads(json.dumps(original)); value['sources'][0]['original_source_file_state']['size'] += 1
    check(value, 'size/mtime')
    value = json.loads(json.dumps(original)); value['sources'].append(dict(value['sources'][0], relative_path='videos/work/another.mp4'))
    value.update(original_source_count=3, selected_source_count=2)
    check(value, 'Duplicate path/hardlink')
    value = json.loads(json.dumps(original)); value['omitted_sources'][0]['relative_path'] = value['sources'][0]['relative_path']
    check(value, 'Duplicate logical')
