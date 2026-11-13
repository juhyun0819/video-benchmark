"""The terminal monitor reads bounded progress and survives staged startup."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[3] / 'tools' / 'feature_memory' / 'watch_feature_memory.py'
spec = importlib.util.spec_from_file_location('watch_feature_memory', SCRIPT)
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_staged_once_shows_independent_progress_and_snapshot_phase(tmp_path):
    archive, output = tmp_path / 'archive', tmp_path / 'memory'
    plan = {'planned_frames': 1000, 'reference_frames': 1000,
            'full_feature_bytes': 1_074_398_101_504}
    save(archive / 'archive.json', {'plan': plan})
    save(archive / 'status.json', {'status': 'running', 'phase': 'encoding',
        'frames_processed': 400, 'total_frames': 1000, 'total_throughput': 8,
        'elapsed_seconds': 50, 'payload_bytes': 1_938_227_200, 'published_shards': 4,
        'seed_frames_reused': 300, 'num_encoder_calls': 100})
    save(output / 'budget_plan.json', plan)
    save(output / 'status.json', {'status': 'running', 'phase': 'snapshot',
         'frames_processed': 300, 'total_frames': 300, 'snapshot_timestamp': 300})
    rows = [{'method': 'fifo', 'label': '5pct', 'hard_budget_bytes': 53_719_905_075,
             'actual_memory_bytes': 1_460_000_000, 'effective_retention_ratio': 1.004,
             'memory_update_time': 3.1}]
    with (output / 'progress.jsonl').open('w') as handle:
        for frames, elapsed in ((200, 40), (300, 50)):
            handle.write(json.dumps({'frames_processed': frames, 'elapsed_seconds': elapsed,
                'total_throughput': frames / elapsed, 'states': rows,
                'worker_timings': {'baseline': {'pid': 123, 'seconds': .2}}}) + '\n')
        handle.write('{"partial":')
    job = tmp_path / 'job.json'
    save(job, {'archive': str(archive), 'output': str(output), 'producer_pid': 10,
               'consumer_pid': 11, 'state': 'running'})
    result = subprocess.run([sys.executable, str(SCRIPT), '--job-status', str(job), '--once'],
                            text=True, capture_output=True, check=True, timeout=10)
    assert 'Stage 1 archive: running' in result.stdout
    assert '400 / 1,000' in result.stdout
    assert 'Payload: 1.938 GB' in result.stdout
    assert 'Stage 2 memory: running  phase: snapshot' in result.stdout
    assert '300 / 300' in result.stdout
    assert 'Recent: 10.000 fps' in result.stdout
    assert 'snapshot_timestamp: 300' in result.stdout
    assert '1074.398102 GB (full video inventory)' in result.stdout
    assert '100.40%' in result.stdout
    assert 'Producer PID: 10  Consumer PID: 11' in result.stdout


def test_staged_archive_visible_before_consumer_output_exists(tmp_path):
    archive = tmp_path / 'archive'
    save(archive / 'status.json', {'status': 'running', 'phase': 'seed_reuse',
                                'frames_processed': 100, 'total_frames': 221728})
    job = tmp_path / 'job.json'
    save(job, {'archive': str(archive), 'output': str(tmp_path / 'future')})
    rendered = watch.display(job)
    assert '100 / 221,728' in rendered
    assert 'Stage 2 memory: pending' in rendered
    assert '0 / pending' in rendered


@pytest.mark.parametrize('phase', ['cpu_smoke', 'validating_cpu_smoke'])
def test_staged_smoke_phase_follows_smoke_output_and_its_frame_target(tmp_path, phase):
    archive, smoke, full = tmp_path / 'archive', tmp_path / 'smoke', tmp_path / 'full'
    save(archive / 'archive.json', {'plan': {'planned_frames': 221728}})
    save(smoke / 'status.json', {'status': 'running', 'phase': 'snapshot',
                               'frames_processed': 300, 'total_frames': 300})
    save(full / 'status.json', {'status': 'running', 'phase': 'stream',
                              'frames_processed': 100, 'total_frames': 221728})
    job = tmp_path / 'job.json'
    save(job, {'archive': str(archive), 'output': str(full), 'smoke_output': str(smoke),
               'state': 'running', 'phase': phase})
    rendered = watch.display(job)
    assert 'Stage 2 smoke: running  phase: snapshot' in rendered
    assert 'Frames: 300 / 300 (100.000%)' in rendered
    assert 'Job: running  phase: ' + phase in rendered
    assert f'Memory: {smoke}' in rendered
    assert f'Full run after smoke validation: {full}' in rendered
    assert '100 / 221,728' not in rendered


@pytest.mark.parametrize('stop_fields', [
    {'memory_construction_state': 'stopped_by_user'},
    {'phase': 'visual_archive_only'},
])
def test_stopped_memory_hides_stale_workers_eta_and_displays_archive_and_kv(tmp_path, stop_fields):
    archive, memory = tmp_path / 'archive', tmp_path / 'memory'
    save(archive / 'archive.json', {'plan': {'planned_frames': 221728}})
    save(archive / 'status.json', {'status': 'running', 'phase': 'encoding',
        'frames_processed': 12300, 'total_frames': 221728, 'total_throughput': 7,
        'elapsed_seconds': 1757, 'payload_bytes': 59_600_486_400})
    # Even unreadable old state must not hide the producer's active progress.
    memory.mkdir()
    (memory / 'status.json').write_text('{stale unfinished JSON')
    (memory / 'budget_plan.json').write_text('{stale unfinished JSON')
    (memory / 'progress.jsonl').write_text('{stale unfinished JSON')
    save(memory / 'worker_status' / 'lightweight.json', {'pid': 123, 'phase': 'updating', 'frames_seen': 800})
    probe = tmp_path / 'kv' / 'status.json'
    save(probe, {'status': 'planned', 'phase': 'waiting_for_gpu', 'model': 'Qwen2.5-VL-7B',
                 'frames_processed': 0, 'total_frames': 10})
    job = tmp_path / 'job.json'
    save(job, {'archive': str(archive), 'output': str(memory), 'state': 'running',
               'consumer_pid': 123, 'kv_probe_status_path': str(probe), **stop_fields})
    rendered = watch.display(job)
    assert 'Stage 1 archive: running  phase: encoding' in rendered
    assert '12,300 / 221,728' in rendered and 'Archive ETA:' in rendered
    assert 'Stage 2 memory: stopped_by_user' in rendered
    assert 'Consumer PID: stopped by user' in rendered
    assert 'Rate-only ETA:' not in rendered
    assert 'CPU workers' not in rendered and 'Last batch worker seconds' not in rendered
    assert 'Method / budget' not in rendered and 'Fixed budget reference' not in rendered
    assert 'KV probe: planned  phase: waiting_for_gpu' in rendered
    assert 'Probe frames: 0 / 10' in rendered
    assert 'model: Qwen2.5-VL-7B' in rendered


def test_old_job_format_still_works_and_gpu_failure_is_nonfatal(tmp_path, monkeypatch):
    run = tmp_path / 'run'
    save(run / 'status.json', {'status': 'running', 'frames_processed': 30})
    save(run / 'budget_plan.json', {'planned_frames': 300})
    (run / 'progress.jsonl').write_text(json.dumps({'frames_processed': 30,
        'elapsed_seconds': 3, 'total_throughput': 10}) + '\n')
    job = tmp_path / 'job.json'
    save(job, {'output': str(run), 'state': 'running'})
    def unavailable(*args, **kwargs):
        raise FileNotFoundError('nvidia-smi')
    monkeypatch.setattr(watch.subprocess, 'run', unavailable)
    rendered = watch.display(job, gpu=True)
    assert 'Frames: 30 / 300 (10.000%)' in rendered
    assert 'GPU status unavailable:' in rendered


def test_status_and_progress_reads_are_size_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(watch, 'MAX_JSON_BYTES', 16)
    path = tmp_path / 'oversized.json'
    path.write_text(json.dumps({'padding': 'x' * 32}))
    with pytest.raises(ValueError, match='exceeds 16 bytes'):
        watch.read_json(path)
    monkeypatch.setattr(watch, 'MAX_TAIL_BYTES', 256)
    path = tmp_path / 'progress.jsonl'
    with path.open('w') as handle:
        handle.write('x' * 1000 + '\n')
        for frame in range(3):
            handle.write(json.dumps({'frames_processed': frame, 'elapsed_seconds': frame}) + '\n')
        handle.write('{"incomplete":')
    assert [row['frames_processed'] for row in watch.recent_progress(path)] == [0, 1, 2]
