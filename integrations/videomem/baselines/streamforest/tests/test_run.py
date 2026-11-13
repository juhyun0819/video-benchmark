"""CLI integration with the real memory and scheduler; no pretrained model/GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from streamforest_videomem import run, shared


def required_arguments(tmp_path):
    return [argument for name in ('manifest', 'questions', 'output-dir', 'model-path',
                                 'verification-file', 'memory-config')
            for argument in ('--' + name, str(tmp_path / name))]


def config_file(tmp_path, **changes):
    values = {name.replace('-', '_'): str(tmp_path / name) for name in (
        'manifest', 'questions', 'output-dir', 'model-path', 'verification-file', 'memory-config')}
    values.update(gpu=0)
    values.update(changes)
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(values))
    return path


@pytest.mark.parametrize('changes', [
    {'memory_config': None}, {'gpu': None}, {'gpu': True}, {'gpu': -1},
    {'fps': 0}, {'fps': float('nan')}, {'fps': True}, {'max_side': 0},
    {'max_side': True}, {'max_new_tokens': -1}, {'max_new_tokens': True},
    {'model_path': []}, {'unknown_setting': 10},
])
def test_invalid_or_unselected_settings_fail_before_outputs(tmp_path, changes):
    config = config_file(tmp_path, **changes)
    with pytest.raises(SystemExit) as error:
        run.parse_args(['--config', str(config)])
    assert error.value.code == 2
    assert not (tmp_path / 'output-dir').exists()


def test_later_choices_override_deferred_config(tmp_path):
    config = config_file(tmp_path, gpu=None, memory_config=None, max_new_tokens=12)
    args = run.parse_args(['--config', str(config), '--gpu', '3', '--memory-config', 'my-memory.json'])
    assert args.gpu == 3 and args.memory_config == 'my-memory.json'
    assert args.max_new_tokens == 12 and args.fps == 1 and args.max_side is None


def test_cli_import_does_not_initialize_gpu_or_models():
    result = subprocess.run([sys.executable, '-c',
        "import sys; from streamforest_videomem import run; "
        "assert not {'torch', 'transformers', 'av'}.intersection(sys.modules)"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


def test_gpu_isolation_uses_uuid_and_offline_mode(monkeypatch):
    for key in ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'TOKENIZERS_PARALLELISM',
                'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'USE_TF', 'USE_FLAX'):
        monkeypatch.setenv(key, 'original')
    commands = []
    def query(command, **kwargs):
        commands.append(command)
        return '2, GPU-test-uuid, Test GPU, 0\n'
    monkeypatch.setattr(run.subprocess, 'check_output', query)
    assert run.bind_gpu(2)['uuid'] == 'GPU-test-uuid'
    assert len(commands) == 1 and '--id=2' in commands[0]
    assert os.environ['CUDA_VISIBLE_DEVICES'] == 'GPU-test-uuid'
    assert os.environ['HF_HUB_OFFLINE'] == os.environ['TRANSFORMERS_OFFLINE'] == '1'


@pytest.mark.parametrize('response', ['2, GPU-test, Test GPU, 2000\n', '1, GPU-test, Test GPU, 0\n'])
def test_gpu_selection_rejects_occupied_or_wrong_device(monkeypatch, response):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'original')
    monkeypatch.setattr(run.subprocess, 'check_output', lambda *args, **kwargs: response)
    with pytest.raises(RuntimeError):
        run.bind_gpu(2)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == 'original'


class Frames:
    def __init__(self):
        self.index = 0
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= 12:
            raise StopIteration
        index = self.index
        self.index += 1
        time = {'day_id': 'Day1', 'day_order': 1, 'seconds': index}
        return {'frame_index': index, 'frame': index, 'time': time,
                'metadata': {'source_id': 'synthetic', 'day_id': 'Day1', 'day_order': 1,
                             'day_seconds': index}}

    def close(self):
        self.closed = True


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    shared.load_shared()
    from local_videomem import data
    from streamforest_videomem import bridge, checkpoint, video
    media = tmp_path / 'synthetic.media'
    media.write_bytes(b'test-only; decoder is replaced')
    manifest = {'videos': [{'day_id': 'Day1', 'day_order': 1, 'path': str(media),
                           'day_start_seconds': 0, 'day_end_seconds': 12, 'duration_seconds': 12,
                           'provenance': {'source_state': data._state(media)}}]}
    questions = [{'question_id': f'q{second}', 'question': 'What is visible?', 'choices': [],
                  'query_time': {'day_id': 'Day1', 'day_order': 1, 'seconds': second},
                  'answer': 'PRIVATE_ANSWER', 'evidence': 'PRIVATE_EVIDENCE'} for second in (2, 12)]
    memory = {'st_memory_windows': [1, 2], 'st_memory_tokens': [4, 2], 'event_split_window': 2,
              'long_memory_tokens_per_frame': 1, 'long_memory_tokens_quota': 4,
              'sim_weight_g': .4, 'time_weight_a': .2, 'merge_weight_b': .4}
    for name, value in [('manifest', manifest), ('questions', questions), ('memory-config', memory)]:
        (tmp_path / name).write_text(json.dumps(value))
    monkeypatch.setattr(checkpoint, 'verify_checkpoint', lambda *args: {'revision': 'test-revision'})
    monkeypatch.setattr(run, 'bind_gpu', lambda index: {'physical_index': index, 'uuid': 'GPU-test'})
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda: 0)
    calls, snapshots, frames = [], [], Frames()
    def model(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(get_vision_tower=lambda: SimpleNamespace(config=SimpleNamespace(
            hidden_size=8, num_attention_heads=2))), None, None
    monkeypatch.setattr(bridge, 'load_model', model)
    monkeypatch.setattr(video, 'iter_frames', lambda *args, **kwargs: frames)
    def encode(model, processor, frame):
        return torch.randn((4, 8), generator=torch.Generator().manual_seed(frame))
    def answer(model, tokenizer, snapshot, prompt, max_new_tokens):
        assert 'PRIVATE_' not in prompt
        snapshots.append(snapshot)
        return {'prediction': 'A visible scene.', 'stopped_on_eos': True}
    monkeypatch.setattr(bridge, 'encode_frame', encode)
    monkeypatch.setattr(bridge, 'generate_from_memory', answer)
    arguments = required_arguments(tmp_path) + ['--gpu', '0']
    return arguments, tmp_path / 'output-dir', frames, snapshots, bridge, calls


def test_runner_connects_real_memory_and_persists_an_explicit_session(runtime):
    arguments, output, frames, snapshots, _, calls = runtime
    assert run.main(arguments) == 0
    assert frames.closed and len(calls) == 1
    assert [s['stats']['total_ingested'] for s in snapshots] == [2, 12]
    assert snapshots[-1]['stats']['tier_tokens']['long'] > 0
    assert 'PRIVATE_' not in (output / 'public_questions.json').read_text()
    provenance = json.loads((output / 'provenance.json').read_text())
    assert provenance['implementation'] == 'streamforest_native_online_persistent_v1'
    assert provenance['proactive'] is False and provenance['training'] is False
    assert provenance['paper_evaluation_reproduction'] is False
    assert any(path.endswith('streamforest_videomem/memory.py') for path in provenance['source_hashes'])
    status = json.loads((output / 'status.json').read_text())
    assert status['phase'] == 'finished' and status['encoder_calls'] == 12
    assert status['complete_answers'] == 2 and status['incomplete_answers'] == 0
    assert (output / 'memory.pt').is_file()
    assert len(json.loads((output / 'predictions.json').read_text())) == 2


def test_incomplete_generation_is_nonzero_and_preserves_outputs(runtime, monkeypatch):
    arguments, output, frames, _, bridge, _ = runtime
    monkeypatch.setattr(bridge, 'generate_from_memory', lambda *args: {
        'prediction': 'truncated answer', 'stopped_on_eos': False})
    assert run.main(arguments) == 2 and frames.closed
    assert json.loads((output / 'status.json').read_text())['incomplete_answers'] == 2


def test_failure_closes_stream_and_records_error(runtime, monkeypatch):
    arguments, output, frames, _, bridge, _ = runtime
    def fail(*args):
        raise RuntimeError('injected encoder failure')
    monkeypatch.setattr(bridge, 'encode_frame', fail)
    with pytest.raises(RuntimeError, match='injected encoder failure'):
        run.main(arguments)
    assert frames.closed
    status = json.loads((output / 'status.json').read_text())
    assert status['phase'] == 'execution_error' and status['error_type'] == 'RuntimeError'


def test_existing_output_never_overwritten(runtime):
    arguments, output, _, _, _, calls = runtime
    output.mkdir()
    marker = output / 'existing.txt'
    marker.write_text('keep')
    with pytest.raises(FileExistsError):
        run.main(arguments)
    assert marker.read_text() == 'keep' and not calls


def test_invalid_cutoff_fails_before_model_or_output(runtime):
    arguments, output, _, _, _, calls = runtime
    question_path = Path(arguments[arguments.index('--questions') + 1])
    questions = json.loads(question_path.read_text())
    questions[0]['query_time']['seconds'] = 13
    question_path.write_text(json.dumps(questions))
    with pytest.raises(ValueError, match='playback range'):
        run.main(arguments)
    assert not output.exists() and not calls
