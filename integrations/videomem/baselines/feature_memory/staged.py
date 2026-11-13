"""Consume published feature shards once and broadcast to causal CPU processes."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

from .construction import actual_table, emit
from .storage import write_json
from .telemetry import Telemetry


def consume_archive(args):
    from .archive import iter_archive
    from .process_memory import MemoryFarm

    archive = Path(args.archive).resolve()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError('max-frames must be positive')
    if any(not math.isfinite(t) or t <= 0 for t in args.snapshot_seconds):
        raise ValueError('Snapshot times must be finite and positive')
    started = time.perf_counter()
    while not (archive / 'archive.json').exists():
        if time.perf_counter() - started > 600:
            raise TimeoutError('Feature producer did not publish its header within 600 seconds')
        status_path = archive / 'status.json'
        if status_path.exists() and json.loads(status_path.read_text()).get('status') == 'failed':
            raise RuntimeError('Feature producer failed before publishing its header')
        time.sleep(0.5)
    header = json.loads((archive / 'archive.json').read_text())
    plan = dict(header['plan'])
    target = min(plan['planned_frames'], args.max_frames or plan['planned_frames'])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'budget_plan.json', plan)
    write_json(output / 'run_config.json', {
        'stream_id': args.stream_id, 'architecture': 'published archive -> one reader -> shared RAM -> five CPU processes',
        'archive': str(archive), 'max_frames': args.max_frames,
        'snapshot_seconds': args.snapshot_seconds, 'seed': args.seed,
        'cpu_threads_per_worker': args.cpu_threads,
        'archive_is_input_cache_outside_policy_budget': True,
        'workers_read_archive': False, 'raw_frame_files': False, 'qa_used': False,
        'memory_commit_barrier': 'all workers finish the current batch before buffer reuse or snapshots',
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(Path(__file__).parent.glob('*.py'))},
        'accounting_source_sha256': hashlib.sha256((Path(__file__).parent.parent/'streambench/accounting.py').read_bytes()).hexdigest()})
    write_json(output / 'status.json', {'status': 'initializing', 'phase': 'cpu_workers',
                                      'frames_processed': 0, 'total_frames': target})
    telemetry = Telemetry()
    telemetry.start()
    farm = iterator = None
    frames = tokens = payload = 0
    last_timestamp = -1.0
    rows, snapshots = [], []
    pending = sorted(set(args.snapshot_seconds))
    timing = Counter()
    shared_digest = hashlib.sha256()
    try:
        farm = MemoryFarm(plan, output, args.seed, args.shard_bytes,
                          cpu_threads=args.cpu_threads, max_batch_frames=args.batch_frames)
        iterator = iter_archive(archive, wait=True, max_frames=target)

        def enriched(source_rows):
            result = []
            for source in source_rows:
                row = dict(source)
                if row['total_frames_seen'] != frames or row['total_visual_tokens_seen'] != tokens:
                    raise RuntimeError('A memory worker did not commit the same causal prefix')
                row['fixed_reference_feature_bytes'] = plan['full_feature_bytes']
                row['effective_retention_ratio'] = row['actual_memory_bytes'] / payload if payload else 0
                row['effective_visual_retention_ratio'] = row['retained_visual_tokens'] * plan['bytes_per_token'] / payload if payload else 0
                row['budget_utilization'] = row['actual_memory_bytes'] / row['hard_budget_bytes']
                if row['actual_memory_bytes'] > row['hard_budget_bytes']:
                    raise RuntimeError('Worker reported a fixed byte-cap violation')
                result.append(row)
            return result

        def status(phase='stream', **extra):
            elapsed = time.perf_counter() - started
            record = {'status': 'running', 'phase': phase, 'stream_id': args.stream_id,
                'frames_processed': frames, 'total_frames': target, 'timestamp': last_timestamp,
                'elapsed_seconds': elapsed, 'total_throughput': frames / max(elapsed, 1e-9),
                'decode_fps': frames / max(timing['decode_seconds'], 1e-9),
                'encoder_compute_fps': frames / max(timing['encode_seconds'], 1e-9),
                'timing_seconds': dict(timing), 'queue_depth': 1 if phase == 'updating' else 0,
                'queue_reserved_bytes': farm.shared_memory_bytes,
                'worker_pids': farm.worker_pids, 'resources': telemetry.report(), 'states': rows, **extra}
            write_json(output / 'status.json', record)
            return record

        def snapshot(timestamp):
            nonlocal rows
            if not frames:
                return
            status('snapshot', snapshot_timestamp=timestamp)
            key = f'{timestamp/3600:g}h' if timestamp % 3600 == 0 else f'{timestamp:g}s'
            saved = farm.snapshot(timestamp, key)
            rows = enriched(saved['rows'])
            by_key = {(row['method'], row['label']): row for row in rows}
            report = telemetry.report()
            for state in saved['states']:
                label = state.get('label') or Path(state['path']).parent.name
                row = by_key[(state['method'], label)]
                write_json(Path(state['path']) / 'benchmark_metadata.json', {
                    **row, 'timestamp': timestamp, 'dtype': plan['dtype'],
                    'tensor_shapes': [row['retained_visual_tokens'], plan['hidden_dim']],
                    'total_visual_tokens_seen': tokens,
                    'retained_frame_indices': 'frame_id/start_frame/end_frame columns in record shards and native tiers',
                    'retained_timestamps': 'timestamp/start_timestamp/end_timestamp columns in records and tiers',
                    'peak_cpu_ram': report['peak_cpu_ram'],
                    'peak_gpu_vram': 'encoder resources recorded separately in archive/status.json and complete.json',
                    'decoding_time': timing['decode_seconds'], 'encoder_time': timing['encode_seconds'],
                    'total_processing_time': time.perf_counter() - started,
                    'snapshot_validation': state.get('validation'),
                    'snapshot_serialized_bytes': state.get('snapshot_serialized_bytes'),
                    'manager_update_uses_future_frames': False,
                    'worker_prefetch_is_bounded_to_current_batch': True,
                    'snapshot_contains_future_frames': False})
                state.update(hard_budget_bytes=row['hard_budget_bytes'],
                             effective_retention_ratio=row['effective_retention_ratio'],
                             effective_visual_retention_ratio=row['effective_visual_retention_ratio'],
                             budget_utilization=row['budget_utilization'])
            timing['snapshot_seconds'] += saved['seconds']
            record = {'timestamp': timestamp, 'frames_seen': frames, 'states': saved['states'],
                      'save_and_verify_seconds': saved['seconds']}
            snapshots.append(record)
            write_json(output / 'snapshots' / f'{key}_summary.json', record)
            (output / 'snapshots' / f'{key}_summary.txt').write_text(actual_table(rows))
            emit('snapshot', timestamp=timestamp, frames=frames, states=len(rows), verified=True)

        with (output / 'progress.jsonl').open('x') as progress_log, (output / 'frames.jsonl').open('x') as frame_log:
            for chunk in iterator:
                timing['archive_read_seconds'] += chunk['read_seconds']
                values, metadata = chunk['features'], chunk['frames']
                cursor = 0
                while cursor < len(metadata):
                    while pending and pending[0] <= metadata[cursor]['timestamp']:
                        snapshot(pending.pop(0))
                    end = min(len(metadata), cursor + args.batch_frames)
                    if pending:
                        end = next((i for i in range(cursor, end) if metadata[i]['timestamp'] >= pending[0]), end)
                    selected = metadata[cursor:end]
                    if not selected:
                        raise RuntimeError('Empty archive segment at snapshot boundary')
                    previous = last_timestamp
                    for index, item in enumerate(selected, frames):
                        if item['frame_id'] != index or item['timestamp'] <= previous:
                            raise RuntimeError('Noncausal or duplicated archive frame')
                        previous = item['timestamp']
                    status('updating')
                    begin = time.perf_counter()
                    updated = farm.update_batch(values[cursor:end], selected)
                    timing['memory_fanout_wall_seconds'] += time.perf_counter() - begin
                    for item in selected:
                        frames += 1
                        tokens += int(plan['tokens_per_frame'])
                        payload += int(plan['tokens_per_frame']) * plan['bytes_per_token']
                        last_timestamp = item['timestamp']
                        timing['decode_seconds'] += item.get('decode_seconds', 0)
                        for key, value in item.get('encoder', {}).items():
                            if key.endswith('_seconds') and isinstance(value, (int, float)):
                                timing[key] += value
                        digest = item['shared_feature_sha256']
                        shared_digest.update(bytes.fromhex(digest))
                        frame_log.write(json.dumps({'frame_id': item['frame_id'], 'timestamp': item['timestamp'],
                            'shared_feature_sha256': digest, 'consumers': 16, 'encoder_calls_in_cpu_stage': 0}) + '\n')
                    rows = enriched(updated['rows'])
                    cursor = end
                    record = status(worker_timings=updated['worker_timings'])
                    progress_log.write(json.dumps(record) + '\n'); progress_log.flush(); frame_log.flush()
                    emit('progress', stage='memory', frames=frames, total=target,
                         timestamp=last_timestamp, throughput=record['total_throughput'])
                del values, metadata, chunk
            if frames != target:
                raise RuntimeError(f'Archive ended early: {frames}/{target}')
            endpoint = plan['duration_seconds'] if target == plan['planned_frames'] else min(plan['duration_seconds'], last_timestamp + 1)
            while pending and pending[0] <= endpoint:
                snapshot(pending.pop(0))
            if not snapshots or snapshots[-1]['timestamp'] != endpoint:
                snapshot(endpoint)
        summary = {'status': 'completed', 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
            'stream_id': args.stream_id, 'architecture': 'archive_shared_memory_processes',
            'archive': str(archive), 'plan': plan, 'rows': rows, 'total_frames': frames,
            'total_visual_tokens': tokens, 'payload_bytes': payload,
            'actual_cumulative_full_feature_bytes': payload,
            'bytes_per_frame': payload / frames, 'tokens_per_frame': tokens / frames,
            'bytes_per_token': plan['bytes_per_token'], 'full_memory_GB': payload / 1e9,
            'full_memory_TB': payload / 1e12, 'full_stream_completed': frames == plan['reference_frames'],
            'consumer_count': 16, 'num_encoder_calls': 0, 'feature_frames_reused_from_archive': frames,
            'full_feature_archive_created': True, 'archive_payload_is_outside_policy_caps': True,
            'archive_read_once_by_coordinator': True, 'workers_open_archive': False,
            'ordered_shared_features_sha256': shared_digest.hexdigest(),
            'all_hard_caps_verified_each_update': True, 'snapshot_save_load_consistency_verified': True,
            'timing_seconds': {**dict(timing), 'memory_management_seconds': sum(r['memory_update_time'] for r in rows),
                               'total_wall_seconds': time.perf_counter() - started},
            'resources': telemetry.report(), 'snapshots': snapshots, 'unreached_snapshot_times': pending,
            'worker_pids': farm.worker_pids, 'shared_memory_bytes': farm.shared_memory_bytes}
        summary['cleanup'] = farm.close()
        write_json(output / 'summary.json', summary)
        (output / 'summary.txt').write_text(actual_table(rows))
        write_json(output / 'status.json', {'status': 'completed', 'phase': 'complete', 'frames_processed': frames,
            'total_frames': target, 'elapsed_seconds': time.perf_counter() - started,
            'summary': str(output / 'summary.json')})
        print(actual_table(rows), flush=True)
        return summary
    except BaseException as exc:
        write_json(output / 'status.json', {'status': 'failed', 'frames_processed': frames,
            'total_frames': target, 'error_type': type(exc).__name__, 'error': str(exc)[:2000]})
        raise
    finally:
        if iterator is not None:
            iterator.close()
        if farm is not None:
            farm.close()
        telemetry.close()
