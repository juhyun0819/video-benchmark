"""Explicit-GPU, resumable production supervisor; no automatic full-run launch.

Run only after scaling results have selected --gpus. Workers start independently
and may wait for their source feature receipts. Any failure stops this run's
own workers; a later explicit invocation can resume verified committed chunks.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from .archive import KVArchiveWriter, committed_blocks, _atomic_json, _lock, _read_header
from .coordinator import (PACKAGE_ROOT, balanced_partition, config_id, sha256_file,
    stop_workers, utc_now, validate_config, validate_worker_report, worker_command,
    worker_snapshot, write_json)


def production_plan(total_frames, gpus, blocksize=100):
    if (not isinstance(gpus, (tuple, list)) or not gpus or
            any(type(gpu) is not int or not 0 <= gpu <= 7 for gpu in gpus) or
            len(set(gpus)) != len(gpus)):
        raise ValueError('Explicit --gpus must be distinct GPU indices in 0..7')
    ranges = balanced_partition(total_frames, len(gpus), blocksize)
    return [dict(worker_id=f'full_gpu{gpu}', gpu=gpu, start_frame=start, end_frame=end)
            for gpu, (start, end) in zip(gpus, ranges)]


def final_audit(root, config):
    """Audit all ready metadata and payload sizes without rereading TB of KV.

    committed_blocks verifies config, receipt/index hashes and causal continuity.
    This pass checks exact full coverage, production tensor geometry, continuation
    metadata, and each immutable payload's serialized size. Payload SHA-256 was
    computed by commit; this audit does not claim to detect same-size later bitrot.
    """
    root, identifier = Path(root), config_id(config)
    total, blocksize = config['total_frames'], config['frames_per_block']
    entries = committed_blocks(root, expected_config_id=identifier, require_contiguous=True)
    expected_chunks = (total + blocksize - 1) // blocksize
    if len(entries) != expected_chunks:
        raise ValueError('Final archive is missing chunks or contains extra chunks')
    serialized_bytes = tensor_bytes = 0
    for chunk_id, entry in enumerate(entries):
        start, end = chunk_id*blocksize, min(total, (chunk_id+1)*blocksize)
        if (entry['chunk_id'], entry['first_frame_id'], entry['last_frame_id'], entry['frame_count']) != (
                chunk_id, start, end-1, end-start):
            raise ValueError('Final archive does not cover the exact intended frame ranges')
        payload = root/'shards'/f'chunk_{chunk_id:08d}.safetensors'
        if not payload.is_file() or payload.stat().st_size != entry['serialized_bytes']:
            raise ValueError(f'Committed payload missing or truncated: chunk {chunk_id}')
        index = json.loads((root/'indices'/f'chunk_{chunk_id:08d}.json').read_text())
        metadata, specs = index['metadata'], index['tensors']
        tokens = (end-start)*678
        expected_meta = dict(config_id=identifier, next_frame_id=end, total_seen=17+end*678,
            initial_state_config_id=config['initial_state_config_id'],
            feature_archive_id=config['feature_archive_id'], key_representation='pre_rope',
            prefix_protocol=config['prefix_protocol'], token_count=tokens, tokens_per_frame=678,
            query_independent=True, resume_strategy='last_raw_kv_window_plus_canonical_init')
        if any(metadata.get(key) != value for key, value in expected_meta.items()):
            raise ValueError(f'Final archive continuation metadata mismatch: chunk {chunk_id}')
        expected_specs = {'position_ids':([3,1,tokens],'torch.int64',3*tokens*8),
                          'input_ids':([1,tokens],'torch.int64',tokens*8)}
        expected_specs.update({f'layer_{layer:02d}.{kind}':([1,4,tokens,128],'torch.bfloat16',4*tokens*128*2)
                              for layer in range(28) for kind in ('key','value')})
        if set(specs) != set(expected_specs):
            raise ValueError(f'Final archive tensor coverage mismatch: chunk {chunk_id}')
        for name, expected in expected_specs.items():
            spec = specs[name]
            if (spec['shape'], spec['dtype'], spec['bytes']) != expected:
                raise ValueError(f'Final archive tensor geometry mismatch: chunk {chunk_id}, {name}')
        chunk_tensor_bytes = sum(spec['bytes'] for spec in specs.values())
        if entry['serialized_bytes'] < chunk_tensor_bytes:
            raise ValueError(f'Serialized payload is too small for its declared tensors: chunk {chunk_id}')
        rows = metadata.get('frames', [])
        if (len(rows) != end-start or [row.get('frame_id') for row in rows] != list(range(start,end)) or
                [row.get('timestamp_seconds') for row in rows] != metadata['timestamps'] or
                any((row.get('token_start'),row.get('token_end')) != (offset*678,(offset+1)*678)
                    for offset,row in enumerate(rows))):
            raise ValueError(f'Final archive per-frame/token index mismatch: chunk {chunk_id}')
        serialized_bytes += entry['serialized_bytes']
        tensor_bytes += chunk_tensor_bytes
    if sum(entry['frame_count'] for entry in entries) != total:
        raise ValueError('Final archive frame count is incomplete')
    return dict(status='passed', config_id=identifier, frame_count=total, chunk_count=len(entries),
        first_frame_id=0, last_frame_id=total-1, serialized_bytes=serialized_bytes,
        tensor_payload_bytes=tensor_bytes, config_receipt_index_hashes_verified=True,
        exact_frame_and_token_coverage_verified=True, tensor_geometry_verified=True,
        payload_serialized_sizes_verified=True, payload_hashes_recomputed=False,
        payload_hash_basis='Immutable ready receipts from atomic commit; no full TB reread',
        audited_at=utc_now())


def run_full(root, config_path, gpus, initial_state_root=None, *, python_executable=None,
             poll_seconds=1.0, heartbeat_seconds=10.0, feature_wait_timeout_seconds=86400):
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    initial = Path(initial_state_root or root/'initial_state').resolve()
    config, validation = validate_config(config_path, initial)
    workers = production_plan(config['total_frames'], gpus, config['frames_per_block'])
    logs = root/'logs'
    logs.mkdir(parents=True, exist_ok=True)
    supervisor_lock = (logs/'.full_run.lock').open('a')
    try:
        fcntl.flock(supervisor_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        supervisor_lock.close()
        raise RuntimeError('A full-run supervisor already owns this archive') from None
    records, outputs = [], []
    began, run_id = time.monotonic(), uuid.uuid4().hex
    report_path, status_path = logs/'full-report.json', logs/'full-status.json'
    last_heartbeat = -math.inf
    try:
        archive = KVArchiveWriter(root, config)
        # Existing committed coverage is validated before workers can resume it.
        committed_blocks(root, expected_config_id=archive.config_id)
        if sha256_file(config_path) != validation['config_file_sha256']:
            raise ValueError('Config changed during production initialization')
        for worker in workers:
            command = worker_command(worker, root, config_path, initial,
                                     python_executable=python_executable)
            command += ['--wait-for-features', '--feature-wait-timeout-seconds', str(feature_wait_timeout_seconds)]
            output = (logs/f"worker_{worker['worker_id']}.run_{run_id}.stdout.log").open('x')
            outputs.append(output)
            process = subprocess.Popen(command, cwd=PACKAGE_ROOT, stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            records.append(dict(worker=worker, command=command, process=process))
        write_json(logs/f'full-plan.{run_id}.json', dict(run_id=run_id, config_id=archive.config_id,
            started_at=utc_now(), workers=[dict(record['worker'], command=record['command'],
                                              pid=record['process'].pid) for record in records]))
        while True:
            now = time.monotonic()
            for record in records:
                returncode = record['process'].poll()
                if returncode is not None and returncode != 0:
                    raise RuntimeError(f"Worker {record['worker']['worker_id']} exited with {returncode}")
            if now-last_heartbeat >= heartbeat_seconds:
                write_json(status_path, dict(status='running', run_id=run_id, updated_at=utc_now(),
                    config_id=archive.config_id, elapsed_seconds=now-began,
                    assigned_gpus=[worker['gpu'] for worker in workers],
                    workers=worker_snapshot(records, root), resumable=True, full_run_started=True))
                last_heartbeat = now
            if all(record['process'].poll() is not None for record in records):
                break
            time.sleep(poll_seconds)
        reports = []
        for worker in workers:
            path = logs/f"worker_{worker['worker_id']}.report.json"
            reports.append(validate_worker_report(json.loads(path.read_text()), worker, archive.config_id))
        _, current_validation = validate_config(config_path, initial)
        if current_validation != validation:
            raise ValueError('Validated model/config/proof identity changed during production')
        audit = final_audit(root, config)
        result = dict(status='complete', run_id=run_id, validation=validation, audit=audit,
            assigned_gpus=[worker['gpu'] for worker in workers], workers=reports,
            elapsed_seconds=time.monotonic()-began, completed_at=utc_now(),
            all_workers_exited_successfully=True, full_run_started=True)
        # Only the successful supervisor marks whole-archive completion.
        with _lock(root):
            _, _, manifest = _read_header(root, archive.config_id)
            if (set(manifest['chunks']) != {str(i) for i in range(audit['chunk_count'])} or
                    any(entry['state'] != 'ready' for entry in manifest['chunks'].values()) or
                    sum(entry['frame_count'] for entry in manifest['chunks'].values()) != audit['frame_count']):
                raise ValueError('Archive coverage changed between final audit and completion transaction')
            manifest['status'] = 'complete'
            manifest['completion'] = dict(run_id=run_id, completed_at=result['completed_at'],
                                         frame_count=audit['frame_count'], chunk_count=audit['chunk_count'],
                                         audit_path=str(report_path))
            _atomic_json(root/'manifest.json', manifest)
        write_json(report_path, result)
        write_json(status_path, result)
        return result
    except BaseException as exc:
        stop_workers(records)
        result = dict(status='failed', run_id=run_id, error=f'{type(exc).__name__}: {exc}',
            updated_at=utc_now(), validation=validation, workers=worker_snapshot(records, root),
            resumable=True, retry_policy='Fail-stop; explicit restart resumes verified chunks',
            full_run_started=bool(records))
        write_json(report_path, result)
        write_json(status_path, result)
        raise
    finally:
        for output in outputs:
            output.close()
        supervisor_lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--gpus', required=True, type=int, nargs='+')
    parser.add_argument('--initial-state-root')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--feature-wait-timeout-seconds', type=float, default=86400)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.feature_wait_timeout_seconds) or args.feature_wait_timeout_seconds <= 0:
        parser.error('--feature-wait-timeout-seconds must be positive and finite')
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f'Full-run supervisor received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    initial = args.initial_state_root or str(Path(args.archive_root)/'initial_state')
    if args.plan_only:
        config, validation = validate_config(args.config, initial)
        print(json.dumps(dict(validation=validation,
            workers=production_plan(config['total_frames'], args.gpus, config['frames_per_block']),
            full_run_started=False), indent=2))
    else:
        print(json.dumps(run_full(args.archive_root, args.config, args.gpus, initial,
            python_executable=args.python, feature_wait_timeout_seconds=args.feature_wait_timeout_seconds), indent=2))


if __name__ == '__main__':
    main()
