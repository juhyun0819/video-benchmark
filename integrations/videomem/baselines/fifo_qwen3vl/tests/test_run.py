"""CLI and orchestration contracts without a GPU, checkpoint, or network."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType

import pytest
import torch

from fifo_videomem import run, shared


def required_arguments(tmp_path):
    return [argument for name in ('manifest', 'questions', 'output', 'model', 'verification')
            for argument in ('--' + name, str(tmp_path / name))]


def config_file(tmp_path, **changes):
    values = {name: str(tmp_path / name)
              for name in ('manifest', 'questions', 'output', 'model', 'verification')}
    values.update(gpu=0, memory_frames=None, memory_tokens=None)
    values.update(changes)
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(values))
    return path


@pytest.mark.parametrize('missing', ['budget', 'gpu'])
def test_runner_never_chooses_a_memory_budget_or_gpu(tmp_path, capsys, missing):
    arguments = required_arguments(tmp_path)
    arguments += ['--gpu', '0'] if missing == 'budget' else ['--memory-frames', '4']
    with pytest.raises(SystemExit) as error:
        run.parse_args(arguments)
    assert error.value.code == 2
    assert ('no default budget' if missing == 'budget' else 'explicit') in capsys.readouterr().err
    assert not (tmp_path / 'output').exists()


def test_deferred_config_can_be_completed_and_overridden_on_cli(tmp_path):
    config = config_file(tmp_path, gpu=None, fps=0.5, max_new_tokens=12)
    args = run.parse_args(['--config', str(config), '--gpu', '3', '--memory-tokens', '3072',
                           '--fps', '2.5'])
    assert (args.gpu, args.memory_frames, args.memory_tokens) == (3, None, 3072)
    assert args.fps == 2.5
    assert args.max_new_tokens == 12


def test_cli_override_of_same_budget_field_and_config_conflicts(tmp_path):
    config = config_file(tmp_path, memory_frames=4)
    assert run.parse_args(['--config', str(config), '--memory-frames', '8']).memory_frames == 8
    with pytest.raises(SystemExit):
        run.parse_args(['--config', str(config), '--memory-tokens', '1024'])


@pytest.mark.parametrize('changes', [
    {'memory_frames': 3}, {'memory_frames': 0}, {'memory_frames': True},
    {'memory_frames': 4.5}, {'memory_frames': 4, 'memory_tokens': 100},
    {'memory_tokens': 0}, {'memory_tokens': False}, {'memory_tokens': 2.5},
    {'memory_frames': 4, 'gpu': True}, {'memory_frames': 4, 'gpu': -1},
    {'memory_frames': 4, 'fps': float('nan')}, {'memory_frames': 4, 'fps': 0},
    {'memory_frames': 4, 'max_side': True}, {'memory_frames': 4, 'max_pixels': 1},
    {'memory_frames': 4, 'max_new_tokens': -1}, {'memory_frames': 4, 'model': []},
    {'memory_frames': 4, 'unknown_setting': 10},
])
def test_invalid_config_is_rejected_before_output_creation(tmp_path, changes):
    config = config_file(tmp_path, **changes)
    with pytest.raises(SystemExit) as error:
        run.parse_args(['--config', str(config)])
    assert error.value.code == 2
    assert not (tmp_path / 'output').exists()


def test_gpu_isolation_uses_verified_uuid_and_offline_mode(monkeypatch):
    for key in ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'TOKENIZERS_PARALLELISM',
                'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        monkeypatch.setenv(key, 'original')
    commands = []

    def query(command, **kwargs):
        commands.append(command)
        return '2, GPU-test-uuid, Test GPU, 0\n'

    monkeypatch.setattr(run.subprocess, 'check_output', query)
    assert run.bind_gpu(2) == {'physical_index': 2, 'uuid': 'GPU-test-uuid', 'name': 'Test GPU'}
    assert len(commands) == 1 and '--id=2' in commands[0]
    assert os.environ['CUDA_VISIBLE_DEVICES'] == 'GPU-test-uuid'
    assert os.environ['HF_HUB_OFFLINE'] == os.environ['TRANSFORMERS_OFFLINE'] == '1'


@pytest.mark.parametrize('response', ['2, GPU-test, Test GPU, 2000\n',
                                     '1, GPU-test, Test GPU, 0\n'])
def test_busy_or_wrong_gpu_does_not_modify_environment(monkeypatch, response):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'original')
    monkeypatch.setattr(run.subprocess, 'check_output', lambda *args, **kwargs: response)
    with pytest.raises(RuntimeError):
        run.bind_gpu(2)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == 'original'


def test_shared_import_rejects_another_variant_already_in_process(monkeypatch):
    foreign = ModuleType('streaming.accidentally_mixed')
    foreign.__file__ = '/unrelated/baseline/streaming/accidentally_mixed.py'
    monkeypatch.setitem(sys.modules, foreign.__name__, foreign)
    with pytest.raises(RuntimeError, match='fresh Python process'):
        shared.load_shared()


class FakeNativeModel:
    """Stand-in only for loading; the real FIFO and scheduler still execute."""
    calls = []

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.calls.append((path, kwargs))
        assert kwargs['local_files_only'] is True
        return cls()

    def eval(self):
        return self

    def requires_grad_(self, value):
        assert value is False


class FakeProcessor:
    @classmethod
    def from_pretrained(cls, path, **kwargs):
        assert kwargs == {'local_files_only': True}
        return cls()


class Pairs:
    def __init__(self):
        self.index = 0
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= 3:
            raise StopIteration
        index = self.index
        self.index += 1
        cutoff = lambda seconds: {'day_id': 'Day1', 'day_order': 1, 'seconds': seconds}
        return {'pair_index': index, 'frames': [index, index],
                'available_time': cutoff(index * 2 + 1), 'frontier_time': cutoff(index * 2 + 2),
                'metadata': {'source_id': 'synthetic'},
                'frame_metadata': [{'day_id': 'Day1', 'day_order': 1, 'day_seconds': index * 2 + offset}
                                   for offset in (0, 1)]}

    def close(self):
        self.closed = True


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    shared.load_shared()
    from local_videomem import data
    from streaming import checkpoint, qwen3_bridge, video

    media = tmp_path / 'synthetic.media'
    media.write_bytes(b'test-only; decoder is replaced')
    manifest = {'videos': [{'day_id': 'Day1', 'day_order': 1, 'path': str(media),
                           'day_start_seconds': 0, 'day_end_seconds': 6, 'duration_seconds': 6,
                           'provenance': {'source_state': data._state(media)}}]}
    questions = [{'question_id': f'q{second}', 'question': 'What is visible?', 'choices': [],
                  'query_time': {'day_id': 'Day1', 'day_order': 1, 'seconds': second},
                  'answer': 'PRIVATE_ANSWER', 'evidence': 'PRIVATE_EVIDENCE'} for second in (2, 6)]
    (tmp_path / 'manifest').write_text(json.dumps(manifest))
    (tmp_path / 'questions').write_text(json.dumps(questions))
    transformer_stub = ModuleType('transformers')
    transformer_stub.__version__ = 'test-only'
    transformer_stub.Qwen3VLForConditionalGeneration = FakeNativeModel
    transformer_stub.AutoProcessor = FakeProcessor
    monkeypatch.setitem(sys.modules, 'transformers', transformer_stub)
    monkeypatch.setattr(FakeNativeModel, 'calls', [])
    monkeypatch.setattr(checkpoint, 'verify_weights', lambda *args: {'revision': 'test-revision'})
    monkeypatch.setattr(run, 'bind_gpu', lambda index: {'physical_index': index, 'uuid': 'GPU-test'})
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda: 0)
    pairs, snapshots = Pairs(), []
    monkeypatch.setattr(video, 'iter_pairs', lambda *args, **kwargs: pairs)

    def encode(model, processor, frames, max_pixels):
        features = torch.full((4, 8), float(frames[0]))
        return {'embeddings': features, 'deepstack': [features + i for i in (1, 2, 3)],
                'height': 2, 'width': 2}

    def answer(model, processor, snapshot, prompt, max_new_tokens):
        assert 'PRIVATE_' not in prompt
        snapshots.append(snapshot)
        return {'prediction': 'A visible scene.', 'stopped_on_eos': True}

    monkeypatch.setattr(qwen3_bridge, 'encode_pair', encode)
    monkeypatch.setattr(qwen3_bridge, 'generate_from_memory', answer)
    arguments = required_arguments(tmp_path) + ['--gpu', '0', '--memory-frames', '2']
    return arguments, tmp_path / 'output', pairs, snapshots, qwen3_bridge


def test_runner_connects_fifo_and_records_an_explicit_only_experiment(runtime):
    arguments, output, pairs, snapshots, _ = runtime
    assert run.main(arguments) == 0
    assert pairs.closed
    assert [[row['time_id'] for row in snapshot['records']] for snapshot in snapshots] == [[0], [2]]
    assert all(snapshot['stats']['total_tokens'] == 4 for snapshot in snapshots)
    rows = json.loads((output / 'predictions.json').read_text())
    assert all(row['kind'] == 'explicit' for row in rows)
    assert rows[-1]['memory_stats']['evicted_grids'] == 2
    assert 'PRIVATE_' not in (output / 'public_questions.json').read_text()
    provenance = json.loads((output / 'provenance.json').read_text())
    assert provenance['implementation'] == 'fifo_qwen3vl_complete_grid_v1'
    assert provenance['proactive'] is False and provenance['training'] is False
    assert provenance['settings']['memory_frames'] == 2
    assert any(path.endswith('fifo_videomem/memory.py') for path in provenance['source_hashes'])
    assert not any('upstream_kernel' in path for path in provenance['source_hashes'])
    status = json.loads((output / 'status.json').read_text())
    assert status['phase'] == 'finished' and status['encoder_calls'] == 3
    assert status['complete_answers'] == 2 and status['incomplete_answers'] == 0
    assert (output / 'memory.pt').is_file()
    assert len(FakeNativeModel.calls) == 1


def test_incomplete_generation_returns_nonzero_with_outputs(runtime, monkeypatch):
    arguments, output, pairs, _, bridge = runtime
    monkeypatch.setattr(bridge, 'generate_from_memory', lambda *args: {
        'prediction': 'truncated answer', 'stopped_on_eos': False})
    assert run.main(arguments) == 2
    assert pairs.closed
    status = json.loads((output / 'status.json').read_text())
    assert status['phase'] == 'finished' and status['incomplete_answers'] == 2
    assert len(json.loads((output / 'predictions.json').read_text())) == 2


def test_execution_error_closes_stream_and_preserves_diagnostic_status(runtime, monkeypatch):
    arguments, output, pairs, _, bridge = runtime

    def fail(*args):
        raise RuntimeError('injected encoder failure')

    monkeypatch.setattr(bridge, 'encode_pair', fail)
    with pytest.raises(RuntimeError, match='injected encoder failure'):
        run.main(arguments)
    assert pairs.closed
    status = json.loads((output / 'status.json').read_text())
    assert status['phase'] == 'execution_error'
    assert status['error_type'] == 'RuntimeError'
    assert not (output / 'predictions.json').exists()


def test_existing_output_is_preserved_and_no_model_is_loaded(runtime):
    arguments, output, _, _, _ = runtime
    output.mkdir()
    sentinel = output / 'prior-result.txt'
    sentinel.write_text('keep this result')
    with pytest.raises(FileExistsError):
        run.main(arguments)
    assert sentinel.read_text() == 'keep this result'
    assert FakeNativeModel.calls == []


def test_invalid_observation_cutoff_fails_before_model_or_output(runtime):
    arguments, output, _, _, _ = runtime
    question_path = Path(arguments[arguments.index('--questions') + 1])
    questions = json.loads(question_path.read_text())
    questions[0]['query_time']['seconds'] = 7
    question_path.write_text(json.dumps(questions))
    with pytest.raises(ValueError, match='playback range'):
        run.main(arguments)
    assert not output.exists()
    assert FakeNativeModel.calls == []


def test_prepare_and_decode_real_synthetic_video_keep_half_open_cutoff(tmp_path):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('Synthetic-media smoke requires FFmpeg and FFprobe')
    pytest.importorskip('av')
    from fifo_videomem import prepare
    shared.load_shared()
    from streaming.video import iter_pairs

    media = tmp_path / 'synthetic-test-only.mp4'
    subprocess.run([
        'ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
        'testsrc2=size=64x64:rate=4:duration=4', '-c:v', 'mpeg4', str(media),
    ], check=True, capture_output=True)
    sources = tmp_path / 'sources.json'
    sources.write_text(json.dumps({'videos': [{
        'path': str(media), 'day_id': 'Day1', 'day_order': 1, 'timeline_status': 'direct_mp4',
    }]}))
    output = tmp_path / 'manifest.json'
    prepare.main(['--sources', str(sources), '--output', str(output)])
    manifest = json.loads(output.read_text())
    assert manifest['source_video_reencoded'] is False
    assert len(manifest['videos'][0]['source_fingerprint']) == 64
    stats = {}
    pairs = iter_pairs(manifest, {'day_id': 'Day1', 'day_order': 1, 'seconds': 2},
                       fps=1, max_side=64, stats=stats)
    observed = []
    try:
        for pair in pairs:
            assert len(pair['frames']) == 2
            assert all(image.mode == 'RGB' for image in pair['frames'])
            observed.extend(frame['day_seconds'] for frame in pair['frame_metadata'])
    finally:
        pairs.close()
    assert observed == [0.0, 1.0]
    saved = output.read_bytes()
    with pytest.raises(SystemExit):
        prepare.main(['--sources', str(sources), '--output', str(output)])
    assert output.read_bytes() == saved
