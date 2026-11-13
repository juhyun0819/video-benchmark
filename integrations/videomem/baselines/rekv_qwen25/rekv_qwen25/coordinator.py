"""Sequential 1/2/4-GPU scaling orchestration; never launches the full archive.

Workers replay the exact original prefix and frames before their assigned
100-frame output range, then wait at a shared barrier. Only post-barrier work
is timed. This module's planning and validation functions do not use GPUs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from .archive import committed_blocks, _json, _normalize_config

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ATTENTION_PATH = Path(__file__).with_name('attention.py')
PREFIX_PROTOCOL = 'separate_fixed_prefix_then_one_frame'
CHECKPOINT = 'Qwen/Qwen2.5-VL-7B-Instruct'
REVISION = 'cc594898137f460bfe9f0759e9844b3ce807cfb5'


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    """Atomic status/report update, with no shared temporary filename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
    try:
        with temp.open('x') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def sha256_file(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            hasher.update(part)
    return hasher.hexdigest()


def config_id(config):
    return hashlib.sha256(_json(_normalize_config(config))).hexdigest()


def balanced_partition(total_frames, workers, blocksize=100):
    """Return contiguous [start,end) ranges balanced by archive chunk count.

    Nonfinal boundaries are block aligned. If fewer chunks than workers exist,
    unused workers are omitted; no empty GPU assignment is created.
    """
    if type(total_frames) is not int or total_frames < 0:
        raise ValueError('total_frames must be a nonnegative integer')
    if type(workers) is not int or workers <= 0 or type(blocksize) is not int or blocksize <= 0:
        raise ValueError('workers and blocksize must be positive integers')
    chunks = (total_frames + blocksize - 1) // blocksize
    if not chunks:
        return []
    active = min(workers, chunks)
    quotient, remainder = divmod(chunks, active)
    result, first = [], 0
    for worker in range(active):
        end = min(total_frames, first + (quotient + (worker < remainder)) * blocksize)
        result.append((first, end))
        first = end
    return result


def scaling_plan(root):
    return [dict(gpu_count=count, archive_root=str(Path(root) / 'benchmarks' / f'scaling_n{count}'),
                 workers=[dict(worker_id=f'gpu{gpu}', gpu=gpu,
                               start_frame=((1-gpu if count == 2 else gpu)+1)*100,
                               end_frame=((1-gpu if count == 2 else gpu)+2)*100)
                          for gpu in range(count)]) for count in (1, 2, 4)]


def validate_config(config_path, initial_state_root):
    """Pin the current code, passed finite-halo proof, and initial prefix state."""
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    required = dict(n_init=17, n_local=15000, chunk_frames=1, frames_per_block=100,
                    prefix_protocol=PREFIX_PROTOCOL)
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError('Config does not use the validated prefix/frame/window protocol')
    if type(config.get('total_frames')) is not int or config['total_frames'] < 500:
        raise ValueError('Config has insufficient benchmark frame coverage')
    if not isinstance(config.get('feature_archive_id'), str) or not config['feature_archive_id']:
        raise ValueError('Missing pinned feature archive identity')
    backbone = config.get('backbone', {})
    if (backbone.get('repo_id'), backbone.get('revision')) != (CHECKPOINT, REVISION):
        raise ValueError('Unexpected checkpoint/revision in config')
    proof = config.get('halo_validation', {})
    if proof.get('status') != 'passed' or proof.get('halo_frames') != 620:
        raise ValueError('Config lacks a passed 620-frame halo validation')
    report_path = Path(proof.get('report_path', ''))
    if not report_path.is_absolute():
        report_path = config_path.parent / report_path
    if not report_path.is_file() or sha256_file(report_path) != proof.get('report_sha256'):
        raise ValueError('Halo report checksum mismatch or missing report')
    report = json.loads(report_path.read_text())
    attention_hash = sha256_file(ATTENTION_PATH)
    if proof.get('attention_sha256') != attention_hash or report.get('adaptation_sha256') != attention_hash:
        raise ValueError('Attention implementation changed since the halo validation')
    if (report.get('status') != 'passed' or report.get('halo_frames') != 620 or
            report.get('n_init') != config['n_init'] or report.get('n_local') != config['n_local'] or
            report.get('chunk_frames') != 1 or report.get('absolute_positions') is not True or
            report.get('fixed_original_per_layer_sinks') is not True or
            report.get('policy') != config.get('policy')):
        raise ValueError('Halo report/config protocol mismatch')
    checkpoint = report.get('checkpoint', {})
    if (checkpoint.get('repo_id'), checkpoint.get('revision')) != (CHECKPOINT, REVISION):
        raise ValueError('Halo proof uses a different checkpoint/revision')
    boundary = report.get('boundary_frame')
    if type(boundary) is not int or boundary < 0:
        raise ValueError('Invalid halo proof boundary')
    verified = [row for row in report.get('checks', []) if row.get('frame_id', -1) >= boundary + 620]
    if (not verified or len(verified) != report.get('verification_frames') or
            not any(row.get('bounded_state_exact') is True for row in verified) or
            any(row.get('all_kv_exact') is not True or row.get('logits', {}).get('exact') is not True
                for row in verified)):
        raise ValueError('Halo proof does not establish exact post-halo continuation')
    initial = Path(initial_state_root)
    initial_config = json.loads((initial / 'config.json').read_text())
    initial_id = config_id(initial_config)
    receipt = json.loads((initial / 'receipts' / 'chunk_00000000.ready.json').read_text())
    if initial_id != config.get('initial_state_config_id') or receipt.get('config_id') != initial_id:
        raise ValueError('Initial prefix state config identifier mismatch')
    if (initial_config.get('input_protocol') != PREFIX_PROTOCOL or
            initial_config.get('attention_sha256') != attention_hash or
            initial_config.get('n_init') != config['n_init'] or
            initial_config.get('n_local') != config['n_local'] or
            initial_config.get('policy') != config['policy']):
        raise ValueError('Initial prefix artifact uses a different attention protocol')
    return config, dict(config_id=config_id(config), config_path=str(config_path),
                        config_file_sha256=sha256_file(config_path),
                        halo_report_path=str(report_path.resolve()),
                        halo_report_sha256=proof['report_sha256'], attention_sha256=attention_hash,
                        initial_state_config_id=initial_id)


def worker_command(worker, archive_root, config_path, initial_state_root, *,
                   barrier_dir=None, benchmark=False, python_executable=None,
                   barrier_timeout_seconds=7200):
    command = [str(python_executable or sys.executable), '-m', 'rekv_qwen25.runner',
               '--archive-root', str(archive_root), '--config', str(config_path),
               '--initial-state-root', str(initial_state_root),
               '--start-frame', str(worker['start_frame']), '--end-frame', str(worker['end_frame']),
               '--gpu', str(worker['gpu']), '--worker-id', worker['worker_id']]
    if barrier_dir is not None:
        command += ['--barrier-dir', str(barrier_dir), '--barrier-timeout-seconds', str(barrier_timeout_seconds)]
    if benchmark:
        command += ['--benchmark', '--no-resume']
    return command


def stop_workers(records, grace_seconds=10):
    """Terminate only subprocess groups created by this coordinator."""
    for record in records:
        process = record['process']
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + grace_seconds
    while any(record['process'].poll() is None for record in records) and time.monotonic() < deadline:
        time.sleep(0.2)
    for record in records:
        process = record['process']
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=10)


def worker_snapshot(records, archive_root):
    snapshots = []
    for record in records:
        worker = record['worker']
        path = Path(archive_root) / 'logs' / f"worker_{worker['worker_id']}.status.json"
        progress = json.loads(path.read_text()) if path.exists() else None
        snapshots.append(dict(worker, pid=record['process'].pid,
                              returncode=record['process'].poll(), progress=progress))
    return snapshots


def validate_worker_report(report, worker, expected_config_id, *, benchmark=False):
    if (report.get('status') != 'complete' or report.get('config_id') != expected_config_id or
            any(report.get(key) != worker[key] for key in ('worker_id', 'start_frame', 'end_frame'))):
        raise ValueError('Worker completion report identity/status mismatch')
    count = worker['end_frame'] - worker['start_frame']
    committed, resumed = report.get('committed_frames'), report.get('resumed_frames')
    if (type(committed) is not int or type(resumed) is not int or min(committed, resumed) < 0 or
            committed + resumed != count):
        raise ValueError('Worker completion report frame coverage mismatch')
    if benchmark and (resumed != 0 or committed != count or report.get('warmup_frames') != worker['start_frame']):
        raise ValueError('Benchmark must use fresh outputs and replay the full earlier prefix')
    for key in ('measured_seconds', 'read_seconds', 'infer_seconds', 'atomic_nas_seconds'):
        value = report.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'Invalid worker timing: {key}')
    if committed and report['measured_seconds'] <= 0:
        raise ValueError('Worker measured time must be positive for generated frames')
    for key in ('serialized_bytes', 'peak_vram_bytes'):
        if type(report.get(key)) is not int or report[key] < 0:
            raise ValueError(f'Invalid worker byte count: {key}')
    return report


def interval_union_seconds(intervals):
    """Wall-clock union duration; overlapping writers count time only once."""
    spans = []
    for interval in intervals:
        start, end = interval['started_time'], interval['finished_time']
        if (isinstance(start, bool) or isinstance(end, bool) or
                not isinstance(start, (float, int)) or not isinstance(end, (float, int)) or
                not math.isfinite(start) or not math.isfinite(end) or end <= start):
            raise ValueError('Invalid atomic-commit wall-clock interval')
        spans.append((start, end))
    duration = 0.0
    previous_end = -math.inf
    for start, end in sorted(spans):
        duration += max(0.0, end-max(start, previous_end))
        previous_end = max(previous_end, end)
    return duration


def compare_shared_chunks(reference, candidate):
    """Verify the same global chunks are bit-identical across GPU assignments.

    Committed receipt/index hashes are checked before comparing each tensor's
    shape, dtype, byte count and SHA-256. Whole safetensors hashes are not used
    for equality because container header ordering is not tensor identity.
    """
    if reference['config_id'] != candidate['config_id']:
        raise ValueError('Scaling stages have different model/config identifiers')
    stages = []
    for stage in (reference, candidate):
        root = Path(stage['archive_root'])
        entries = committed_blocks(root, expected_config_id=stage['config_id'])
        expected = {frame//100 for worker in stage['workers']
                    for frame in range(worker['start_frame'], worker['end_frame'], 100)}
        if {entry['chunk_id'] for entry in entries} != expected:
            raise ValueError('Scaling archive does not match its assigned chunk coverage')
        indices = {entry['chunk_id']:json.loads((root/'indices'/f"chunk_{entry['chunk_id']:08d}.json").read_text())
                   for entry in entries}
        stages.append(indices)
    common = sorted(set(stages[0]) & set(stages[1]))
    if not common:
        raise ValueError('Scaling stages have no common chunks for fidelity comparison')
    comparisons = []
    for chunk in common:
        left, right = stages[0][chunk], stages[1][chunk]
        if left['tensors'] != right['tensors']:
            differing = sorted(name for name in set(left['tensors'])|set(right['tensors'])
                               if left['tensors'].get(name) != right['tensors'].get(name))
            raise ValueError(f'Cross-GPU tensor identity mismatch in chunk {chunk}: {differing}')
        if left['metadata'] != right['metadata']:
            raise ValueError(f'Cross-GPU frame/source/position metadata mismatch in chunk {chunk}')
        assigned = []
        for stage in (reference, candidate):
            worker = next(worker for worker in stage['workers']
                          if worker['start_frame'] <= chunk*100 < worker['end_frame'])
            assigned.append(worker['gpu'] if 'gpu' in worker else int(worker['worker_id'].removeprefix('gpu')))
        comparisons.append(dict(status='passed',chunk_id=chunk,
            reference_gpu_count=reference['gpu_count'],candidate_gpu_count=candidate['gpu_count'],
            reference_gpu=assigned[0],candidate_gpu=assigned[1],
            frame_count=left['metadata']['frame_count'],tensor_count=len(left['tensors']),
            all_tensor_sha256_exact=True,all_tensor_geometry_exact=True,metadata_exact=True))
    return comparisons


def run_stage(stage, config_path, expected_config_id, initial_state_root, *,
              python_executable=None, poll_seconds=1.0, heartbeat_seconds=10.0,
              barrier_timeout_seconds=7200, run_timeout_seconds=7200):
    """Run exactly this stage's assigned GPUs and synchronize steady timing."""
    root = Path(stage['archive_root'])
    if root.exists() and any(root.iterdir()):
        raise ValueError(f'Refusing to reuse a scaling stage directory: {root}')
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    barrier = root / 'barrier'
    barrier.mkdir()
    records, outputs = [], []
    began = time.monotonic()
    steady_start = None
    last_heartbeat = -math.inf
    try:
        for worker in stage['workers']:
            command = worker_command(worker, root, config_path, initial_state_root,
                barrier_dir=barrier, benchmark=True, python_executable=python_executable,
                barrier_timeout_seconds=barrier_timeout_seconds)
            output = (root / 'logs' / f"worker_{worker['worker_id']}.stdout.log").open('x')
            outputs.append(output)
            process = subprocess.Popen(command, cwd=PACKAGE_ROOT, stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            records.append(dict(worker=worker, command=command, process=process))
        while True:
            now = time.monotonic()
            for record in records:
                returncode = record['process'].poll()
                if returncode is not None and (returncode != 0 or steady_start is None):
                    raise RuntimeError(f"Worker {record['worker']['worker_id']} exited with {returncode}")
            if steady_start is None:
                ready = []
                for record in records:
                    worker = record['worker']
                    path = barrier / f"{worker['worker_id']}.ready.json"
                    if path.exists():
                        message = json.loads(path.read_text())
                        if (message.get('config_id') != expected_config_id or
                                any(message.get(key) != worker[key] for key in ('worker_id', 'start_frame', 'end_frame'))):
                            raise ValueError('Worker barrier readiness identity mismatch')
                        ready.append(worker['worker_id'])
                if len(ready) == len(records):
                    steady_start = time.monotonic()
                    write_json(barrier / 'start.json', dict(workers=ready, started_at=utc_now(),
                                                           config_id=expected_config_id))
                    last_heartbeat = -math.inf
                elif now - began > barrier_timeout_seconds:
                    raise TimeoutError('Timed out waiting for all workers to finish exact warmup')
            elif now - steady_start > run_timeout_seconds:
                raise TimeoutError('Scaling stage exceeded its post-barrier timeout')
            if now - last_heartbeat >= heartbeat_seconds:
                write_json(root / 'status.json', dict(status='running' if steady_start is not None else 'warming',
                    updated_at=utc_now(), elapsed_seconds=now-began, gpu_count=stage['gpu_count'],
                    assigned_gpus=[w['gpu'] for w in stage['workers']],
                    workers=worker_snapshot(records, root), full_run_started=False))
                last_heartbeat = now
            if steady_start is not None and all(record['process'].poll() is not None for record in records):
                break
            time.sleep(poll_seconds)
        reports = []
        for record in records:
            worker = record['worker']
            path = root / 'logs' / f"worker_{worker['worker_id']}.report.json"
            reports.append(validate_worker_report(json.loads(path.read_text()), worker,
                                                  expected_config_id, benchmark=True))
        elapsed = max(report['measured_seconds'] for report in reports)
        frames = sum(report['committed_frames'] for report in reports)
        size = sum(report['serialized_bytes'] for report in reports)
        intervals = [interval for report in reports for interval in report.get('commit_intervals', [])]
        if sum(interval['serialized_bytes'] for interval in intervals) != size:
            raise ValueError('Atomic-commit timing intervals do not cover all serialized bytes')
        commit_union = interval_union_seconds(intervals)
        result = dict(status='complete', gpu_count=stage['gpu_count'], config_id=expected_config_id,
            archive_root=str(root), assigned_gpus=[w['gpu'] for w in stage['workers']],
            committed_frames=frames, serialized_bytes=size, steady_seconds=elapsed,
            steady_fps=frames/elapsed, serialized_bytes_per_second=size/elapsed,
            coordinator_seconds_after_barrier=time.monotonic()-steady_start,
            total_stage_seconds=time.monotonic()-began,
            timing_basis='max_worker_measured_seconds_after_shared_barrier',
            atomic_nas_union_seconds=commit_union,
            atomic_nas_effective_bytes_per_second=size/commit_union,
            atomic_nas_scope='Atomic commit including CPU copies, checksums, serialization and fsync',
            peak_worker_vram_bytes=max(report['peak_vram_bytes'] for report in reports),
            workers=reports, completed_at=utc_now(), full_run_started=False)
        write_json(root / 'report.json', result)
        write_json(root / 'status.json', result)
        return result
    except BaseException as exc:
        stop_workers(records)
        write_json(root / 'status.json', dict(status='failed', error=f'{type(exc).__name__}: {exc}',
            updated_at=utc_now(), workers=worker_snapshot(records, root), full_run_started=False))
        raise
    finally:
        for output in outputs:
            output.close()


def run_scaling(root, config_path, initial_state_root=None, *, python_executable=None):
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    initial = Path(initial_state_root or root / 'initial_state').resolve()
    _, validation = validate_config(config_path, initial)
    stages = scaling_plan(root)
    for stage in stages:
        path = Path(stage['archive_root'])
        if path.exists() and any(path.iterdir()):
            raise ValueError(f'Refusing to overwrite prior scaling artifacts: {path}')
    summary = dict(status='running', validation=validation, stages=[], started_at=utc_now(),
                   cross_gpu_fidelity=dict(status='pending',comparisons=[]),
                   full_run_started=False)
    report_path = root / 'benchmarks' / 'scaling_report.json'
    try:
        for stage in stages:
            _, current = validate_config(config_path, initial)
            if current != validation:
                raise ValueError('Validated configuration changed between scaling stages')
            write_json(report_path, dict(summary, current_gpu_count=stage['gpu_count']))
            report = run_stage(stage, config_path, validation['config_id'], initial,
                               python_executable=python_executable)
            reference = summary['stages'][0]['steady_fps'] if summary['stages'] else report['steady_fps']
            report['speedup_over_one_gpu'] = report['steady_fps'] / reference
            report['parallel_efficiency'] = report['speedup_over_one_gpu'] / stage['gpu_count']
            for previous in summary['stages']:
                summary['cross_gpu_fidelity']['comparisons'].extend(compare_shared_chunks(previous, report))
            summary['stages'].append(report)
        summary['cross_gpu_fidelity']['status'] = 'passed'
        summary.update(status='complete', completed_at=utc_now(),
            fastest_stage_gpu_count=max(summary['stages'], key=lambda r:r['steady_fps'])['gpu_count'],
            total_serialized_bytes=sum(r['serialized_bytes'] for r in summary['stages']))
        write_json(report_path, summary)
        return summary
    except BaseException as exc:
        summary.update(status='failed', error=f'{type(exc).__name__}: {exc}', updated_at=utc_now())
        write_json(report_path, summary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--initial-state-root')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--plan-only', action='store_true', help='Validate and print assignments without starting workers')
    args = parser.parse_args()
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f'Coordinator received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    initial = args.initial_state_root or str(Path(args.archive_root) / 'initial_state')
    if args.plan_only:
        _, validation = validate_config(args.config, initial)
        print(json.dumps(dict(validation=validation, stages=scaling_plan(args.archive_root),
                              full_run_started=False), indent=2))
    else:
        print(json.dumps(run_scaling(args.archive_root, args.config, initial,
                                    python_executable=args.python), indent=2))


if __name__ == '__main__':
    main()
