"""Published operational tools work from a checkout, without the old workspace."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[3]
TOOLS = REPO / 'tools' / 'feature_memory'
NAMES = ('measure_qwen25_kv', 'watch_feature_memory', 'watch_visual_archive_job', 'start_feature_memory_staged')


def load_tool(name):
    spec = importlib.util.spec_from_file_location('published_' + name, TOOLS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('name', NAMES)
def test_tool_help_from_unrelated_directory_without_workspace_imports(tmp_path, name):
    result = subprocess.run([sys.executable, str(TOOLS / (name + '.py')), '--help'],
                            cwd=tmp_path, text=True, capture_output=True, timeout=15, check=True)
    assert 'usage:' in result.stdout and '/home/zzuhyun' not in result.stdout
    if name == 'measure_qwen25_kv':
        assert '--archive ARCHIVE' in result.stdout and '--model MODEL' in result.stdout


def test_tool_roots_are_derived_from_the_repository():
    assert load_tool('measure_qwen25_kv').REPO == REPO
    assert load_tool('start_feature_memory_staged').REPO == REPO
    watch = load_tool('watch_feature_memory')
    assert watch.DEFAULT_JOB.parent == REPO.parent / 'evaluation'


def test_published_kv_probe_native_cpu_equivalence_and_cache_roundtrip():
    import torch
    probe = load_tool('measure_qwen25_kv')
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            result = probe.cpu_native_equivalence_check()
    finally:
        torch.set_num_threads(previous_threads)
    assert result['passed'] and result['native_kv_exact_equal']
    assert result['native_logits_exact_equal'] and result['serialization_reload_exact_equal']
    assert result['layers'] == 2 and result['sequence_length'] == 14


def test_staged_launcher_normalizes_paths_and_masks_selected_gpus(tmp_path, monkeypatch):
    launcher = load_tool('start_feature_memory_staged')
    golden = tmp_path / 'golden'
    golden.mkdir()
    interpreter = tmp_path / 'env' / 'bin' / 'python'
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    summary = {'plan': {'sum_all_state_hard_budgets': 16, 'full_feature_bytes': 100,
                        'settings': [{'hard_budget_bytes': 10}]}}
    summary_raw = json.dumps(summary).encode()
    (golden / 'summary.json').write_bytes(summary_raw)
    (golden / 'validation.json').write_text(json.dumps({'passed': True,
        'summary_sha256': hashlib.sha256(summary_raw).hexdigest()}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(launcher.shutil, 'disk_usage', lambda path: SimpleNamespace(free=10**12))
    monkeypatch.setattr(launcher, 'validate_smoke', lambda *args: {'passed': True})
    calls = []
    class FinishedProcess:
        returncode = 0
        def __init__(self, command, **kwargs):
            self.pid = 100 + len(calls)
            calls.append((command, kwargs))
            if 'archive' in command:
                output = Path(command[command.index('--output') + 1])
                output.mkdir()
                (output / 'ready_00000002.json').touch()
        def poll(self):
            return 0
    monkeypatch.setattr(launcher.subprocess, 'Popen', FinishedProcess)
    monkeypatch.setattr(sys, 'argv', [str(TOOLS/'start_feature_memory_staged.py'),
        '--archive', 'archive', '--output', 'memory', '--smoke-output', 'smoke',
        '--job-status', 'job.json', '--seed-run', 'seed', '--golden-smoke', 'golden',
        '--project-root', 'workspace', '--python', 'env/bin/python', '--videos', 'day1.mp4', 'day2.mp4',
        '--model', 'checkpoint', '--dataset-status', 'inventory.json', '--timeline', 'timeline.json',
        '--devices', '2', '5'])
    assert launcher.main() == 0
    assert len(calls) == 3
    command, options = calls[0]
    assert command[0] == str(interpreter)
    assert command[command.index('--model') + 1] == str(tmp_path / 'checkpoint')
    assert command[command.index('--videos') + 1:command.index('--dataset-status')] == [str(tmp_path/'day1.mp4'), str(tmp_path/'day2.mp4')]
    assert command[command.index('--devices') + 1:command.index('--archive-frames-per-shard')] == ['0', '1']
    assert command[command.index('--timeline') + 1] == str(tmp_path / 'timeline.json')
    assert options['cwd'] == REPO and options['env']['CUDA_VISIBLE_DEVICES'] == '2,5'
    assert all(options['env']['CUDA_VISIBLE_DEVICES'] == '' for _, options in calls[1:])
    assert json.loads((tmp_path / 'job.json').read_text())['state'] == 'completed'
