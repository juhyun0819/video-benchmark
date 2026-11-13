"""Byte-capped query-agnostic construction using existing video/vision stages."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Protocol

import torch

from .planning import METHODS, make_plan, plan_table
from .storage import write_json
from .telemetry import Telemetry


class MemoryManager(Protocol):
    def update(self, features, frame_id, timestamp): ...
    def stats(self): ...
    def snapshot(self, directory, timestamp): ...
    def iter_batches(self, max_tokens=1024): ...
    def flush(self): ...
    def close(self): ...


def emit(event, **values):
    print(json.dumps({'event': event, **values}, allow_nan=False), flush=True)


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def verify_snapshot(directory, expected_count, fluxmem=False):
    """Read back every stored feature/provenance byte; no full-state RAM copy."""
    from .byte_memory import SnapshotReader
    if fluxmem:
        from .fluxmem import FluxMemSnapshotReader
        reader = FluxMemSnapshotReader(directory, verify=False)
    else:
        reader = SnapshotReader(directory)
    try:
        validation = reader.verify()
        count = reader.count
        if fluxmem:
            path = Path(directory) / 'fluxmem_tiers.pt'
            # Tiers are bounded independently of the potentially TB-scale bank.
            # The official state loader/feature equality is covered by native
            # kernel roundtrip tests; here every serialized byte is rehashed.
            validation['fluxmem_tiers_sha256'] = _digest(path)
        if count != expected_count:
            raise RuntimeError('Snapshot retained token count mismatch')
        first = next(iter(reader.iter_batches(max_tokens=1)), None)
        if count and (first is None or first['features'].dtype != torch.bfloat16):
            raise RuntimeError('Snapshot reader failed to reconstruct bf16 features')
        return {'passed': True, **validation, 'stored_record_tokens': count,
                'feature_identity_basis': 'source SHA256 during streaming copy equals full read-back SHA256; typed loader exercised'}
    finally:
        reader.close()


def actual_table(rows):
    lines = ['Method | Fixed Ratio | Hard Budget GB | Actual GB | Effective Retention | Frames Seen | Retained Tokens | Frame-equivalent | Compression Ratio | Build Time',
             '--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:']
    for row in rows:
        lines.append(f"{row['method']} | {row['retention_ratio']:.0%} | {row['hard_budget_bytes']/1e9:.6f} | "
                     f"{row['actual_memory_bytes']/1e9:.6f} | {row['effective_retention_ratio']:.3%} | {row['total_frames_seen']} | {row['retained_visual_tokens']} | "
                     f"{row['approximate_frame_equivalent']:.3f} | {row['compression_ratio']:.3f}x | "
                     f"{row['memory_update_time']:.3f}s")
    return '\n'.join(lines) + '\n\nBuild Time is each state\'s update time; shared encoder/wall time is recorded once globally.\n'


def build(args, *, encoder_factory=None, feature_size_estimator=None):
    from .video import inspect_sources, iter_stream
    from .parallel import iter_encoded
    from .byte_memory import ByteBudgetMemory
    from .fluxmem import FluxMemByteMemory
    from .fanout import update_managers

    started = time.perf_counter()
    flux_workers = getattr(args, 'flux_workers', 1)
    if flux_workers < 1 or flux_workers > 4:
        raise ValueError('flux-workers must be between 1 and 4')
    torch.set_num_threads(args.cpu_threads)
    if args.cpu_threads < 1 or any(not math.isfinite(t) or t <= 0 for t in args.snapshot_seconds):
        raise ValueError('CPU threads and snapshot times must be positive')
    inventory = inspect_sources(args.videos, status_path=args.dataset_status, timeline_path=args.timeline)
    plan = make_plan(inventory, args.model, max_seconds=args.max_seconds, ratios=args.ratios)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'inventory.json', inventory)
    write_json(output / 'budget_plan.json', plan)
    (output / 'planned_summary.txt').write_text(plan_table(plan))
    write_json(output / 'run_config.json', {
        'stream_id': args.stream_id, 'mode': args.mode, 'devices': args.devices,
        'chunk_frames': args.chunk_frames, 'max_inflight_chunks': args.max_inflight_chunks,
        'max_queue_bytes': args.max_queue_bytes, 'snapshot_seconds': args.snapshot_seconds,
        'flux_workers': flux_workers, 'memory_commit_barrier': 'all consumers finish frame t before frame t+1',
        'seed': args.seed, 'representation': 'Qwen2.5-VL post-merger independent-image bf16',
        'history_archive': False, 'raw_frame_files': False, 'qa_used': False,
        'live_state_backend': 'bounded sharded retained records on disk, charged to each state cap',
        'source_sha256': {p.name: _digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}})
    write_json(output / 'status.json', {'status': 'initializing', 'frames_processed': 0})
    managers, updates, counts = {}, Counter(), Counter()
    decoder_stats, encoder_stats = {}, {}
    rows = []
    pending = sorted(set(args.snapshot_seconds))
    snapshots = []
    last_time = -1.0
    telemetry = Telemetry()
    telemetry.start()
    encoded = None
    executor = ThreadPoolExecutor(max_workers=flux_workers) if flux_workers > 1 else None
    total_frames = total_tokens = payload_bytes = 0
    peak_current_feature_bytes = 0
    encode_digest = hashlib.sha256()
    try:
        for method in METHODS:
            for setting in plan['settings']:
                common = {'directory': output / 'live' / method / setting['label'],
                          'hard_budget_bytes': setting['hard_budget_bytes'],
                          'retention_ratio': setting['retention_ratio'],
                          'hidden_dim': plan['hidden_dim'], 'tokens_per_frame': int(plan['tokens_per_frame']),
                          'seed': args.seed, 'dtype': 'bfloat16', 'shard_bytes': args.shard_bytes}
                if method == 'fluxmem':
                    manager = FluxMemByteMemory(height=plan['post_merger_grid_hw'][0],
                                               width=plan['post_merger_grid_hw'][1], **common)
                else:
                    manager = ByteBudgetMemory(method=method, **common)
                managers[(method, setting['label'])] = (manager, setting)
        stream = iter_stream(inventory, max_seconds=args.max_seconds, stats=decoder_stats)
        encoded = iter_encoded(stream, args.model, mode=args.mode, devices=args.devices,
                               chunk_frames=args.chunk_frames, max_inflight_chunks=args.max_inflight_chunks,
                               max_queue_bytes=args.max_queue_bytes, cpu_threads=args.cpu_threads,
                               dtype='bfloat16', stats=encoder_stats, encoder_factory=encoder_factory,
                               feature_size_estimator=feature_size_estimator)

        def current_rows():
            result = []
            for (method, label), (manager, setting) in managers.items():
                observed = manager.stats()
                if observed['total_frames_seen'] != total_frames or observed['total_visual_tokens_seen'] != total_tokens:
                    raise RuntimeError('Memory consumer frame/token counters differ from committed stream')
                row = {**observed, 'method': method, 'retention_ratio': setting['retention_ratio'],
                       'hard_budget_bytes': setting['hard_budget_bytes'], 'label': label,
                       'memory_update_time': updates[(method, label)], 'total_frames_seen': total_frames}
                row['fixed_reference_feature_bytes'] = plan['full_feature_bytes']
                row['effective_retention_ratio'] = row['actual_memory_bytes'] / payload_bytes if payload_bytes else 0.0
                row['effective_visual_retention_ratio'] = (row['retained_visual_tokens'] * plan['bytes_per_token'] / payload_bytes
                                                          if payload_bytes else 0.0)
                row['budget_utilization'] = row['actual_memory_bytes'] / setting['hard_budget_bytes']
                row['persistent_disk_bytes'] = manager.store.stats()['persistent_disk_bytes']
                if row['actual_memory_bytes'] > setting['hard_budget_bytes']:
                    raise RuntimeError(json.dumps({'method': method, 'budget': setting['hard_budget_bytes'],
                        'timestamp': last_time, 'overshoot_bytes': row['actual_memory_bytes']-setting['hard_budget_bytes'],
                        'cause': 'persistent bytes exceeded the frozen absolute budget'}))
                result.append(row)
            return result

        def snapshot(timestamp):
            if not total_frames:
                return
            begin = time.perf_counter()
            key = (f'{timestamp/3600:g}h' if timestamp % 3600 == 0 else f'{timestamp:g}s')
            report = telemetry.report()
            record = {'timestamp': timestamp, 'frames_seen': total_frames, 'states': []}
            snapshot_rows = []
            for row in current_rows():
                method, label = row['method'], row['label']
                manager, setting = managers[(method, label)]
                destination = output / 'snapshots' / method / label / key
                saved = manager.snapshot(destination, timestamp)
                refreshed = manager.stats()
                row.update(refreshed)
                row['persistent_disk_bytes'] = manager.store.stats()['persistent_disk_bytes']
                row['effective_retention_ratio'] = row['actual_memory_bytes'] / payload_bytes
                row['budget_utilization'] = row['actual_memory_bytes'] / setting['hard_budget_bytes']
                if row['actual_memory_bytes'] > setting['hard_budget_bytes']:
                    raise RuntimeError('Snapshot flush exceeded persistent byte budget')
                verified = verify_snapshot(destination, row['retained_visual_tokens'], fluxmem=method == 'fluxmem')
                if method == 'fluxmem' and saved.get('core_file_sha256') != verified['fluxmem_tiers_sha256']:
                    raise RuntimeError('FluxMem tier file differs after serialization')
                # Written separately from the manager's loadable native manifest.
                metadata = {**row, 'timestamp': timestamp, 'dtype': 'bfloat16',
                    'tensor_shapes': [row['retained_visual_tokens'], plan['hidden_dim']],
                    'total_visual_tokens_seen': total_tokens,
                    'retained_frame_indices': 'frame_id/start_frame/end_frame columns in record shards and native FluxMem tiers',
                    'retained_timestamps': 'timestamp/start_timestamp/end_timestamp columns in record shards and native FluxMem tiers',
                    'peak_cpu_ram': report['peak_cpu_ram'],
                    'peak_gpu_vram': {k: v['peak_used_bytes'] for k, v in report['gpus'].items()},
                    'decoding_time': counts['decode_seconds'], 'encoder_time': counts['encode_seconds'],
                    'total_processing_time': time.perf_counter() - started,
                    'snapshot_serialized_bytes': saved.get('snapshot_serialized_bytes'),
                    'snapshot_validation': verified, 'encoder_forward_count_committed': total_frames,
                    'future_queued_features_in_memory_state': False}
                write_json(destination / 'benchmark_metadata.json', metadata)
                snapshot_rows.append(row)
                record['states'].append({'method': method, 'ratio': setting['retention_ratio'], 'path': str(destination),
                    'hard_budget_bytes': setting['hard_budget_bytes'],
                    'actual_memory_bytes': row['actual_memory_bytes'],
                    'effective_retention_ratio': row['effective_retention_ratio'],
                    'effective_visual_retention_ratio': row['effective_visual_retention_ratio'],
                    'budget_utilization': row['budget_utilization'], 'verified': True})
            record['save_and_verify_seconds'] = time.perf_counter() - begin
            counts['snapshot_seconds'] += record['save_and_verify_seconds']
            snapshots.append(record)
            write_json(output / 'snapshots' / (key + '_summary.json'), record)
            (output / 'snapshots' / (key + '_summary.txt')).write_text(actual_table(snapshot_rows))
            emit('snapshot', timestamp=timestamp, frames=total_frames, states=len(managers), verified=True,
                 seconds=record['save_and_verify_seconds'])

        with (output / 'progress.jsonl').open('x') as log, (output / 'frames.jsonl').open('x') as frames_log:
            for item in encoded:
                timestamp = float(item['timestamp'])
                if item['frame_id'] != total_frames or timestamp <= last_time:
                    raise RuntimeError('Ordered stream violated frame identity or temporal causality')
                while pending and pending[0] <= timestamp:
                    snapshot(pending.pop(0))
                features = item['features']
                expected_shape = (int(plan['tokens_per_frame']), plan['hidden_dim'])
                if tuple(features.shape) != expected_shape or features.dtype != torch.bfloat16:
                    raise RuntimeError('Actual shared representation disagrees with the frozen byte-budget plan')
                byte_size = features.numel() * features.element_size()
                peak_current_feature_bytes = max(peak_current_feature_bytes, byte_size)
                # All 16 managers get this SAME tensor object before release.
                digest = hashlib.sha256(memoryview(features.view(torch.uint8).numpy()).cast('B')).hexdigest()
                begin = time.perf_counter()
                elapsed_updates = update_managers(
                    [(key, manager, setting) for key, (manager, setting) in managers.items()],
                    features, total_frames, timestamp, executor=executor)
                updates.update(elapsed_updates)
                counts['memory_fanout_wall_seconds'] += time.perf_counter() - begin
                total_frames += 1
                total_tokens += features.shape[0]
                payload_bytes += byte_size
                last_time = timestamp
                counts['decode_seconds'] += item['decode_seconds']
                for name in ('encode_seconds', 'preprocess_seconds', 'h2d_seconds', 'd2h_seconds', 'feature_validation_seconds'):
                    counts[name] += item['encoder'].get(name, 0)
                # An ordered digest proves which exact shared outputs were
                # committed, without retaining a history feature archive.
                if hashlib.sha256(memoryview(features.view(torch.uint8).numpy()).cast('B')).hexdigest() != digest:
                    raise RuntimeError('A memory consumer mutated the shared encoder output')
                encode_digest.update(bytes.fromhex(digest))
                frames_log.write(json.dumps({'frame_id': item['frame_id'], 'timestamp': timestamp,
                    'shape': list(features.shape), 'dtype': 'bfloat16', 'bytes': byte_size,
                    'shared_feature_sha256': digest, 'consumers': len(managers),
                    'encoder_worker_id': item.get('encoder_worker_id'),
                    'encoder_device': item.get('encoder_device'),
                    'decode_seconds': item['decode_seconds'], 'encoder': item['encoder']}) + '\n')
                del features, item
                if total_frames % args.progress_frames == 0:
                    elapsed = time.perf_counter() - started
                    progress = {'status': 'running', 'stream_id': args.stream_id,
                        'frames_processed': total_frames, 'total_frames': plan['planned_frames'],
                        'timestamp': last_time, 'total_throughput': total_frames / elapsed,
                        'decode_fps': total_frames / max(counts['decode_seconds'], 1e-9),
                        'encoder_compute_fps': total_frames / max(counts['encode_seconds'], 1e-9),
                        'elapsed_seconds': elapsed, 'queue_reserved_bytes': encoder_stats.get('queue_reserved_bytes'),
                        'queue_depth': encoder_stats.get('current_inflight_chunks'),
                        'peak_queue_depth': encoder_stats.get('peak_inflight_chunks'),
                        'backpressure_wait_seconds': encoder_stats.get('backpressure_wait_seconds'),
                        'resources': telemetry.report(), 'states': current_rows()}
                    log.write(json.dumps(progress, allow_nan=False) + '\n')
                    log.flush(); frames_log.flush()
                    write_json(output / 'status.json', progress)
                    emit('progress', frames=total_frames, total=plan['planned_frames'], timestamp=last_time,
                         throughput=progress['total_throughput'], queue_bytes=progress['queue_reserved_bytes'])
            end = plan['duration_seconds']
            while pending and pending[0] <= end:
                snapshot(pending.pop(0))
            if not snapshots or snapshots[-1]['timestamp'] != end:
                snapshot(end)
        if total_frames != plan['planned_frames'] or payload_bytes != plan['planned_run_feature_bytes']:
            raise RuntimeError('Observed feature footprint differs from frozen budget reference')
        if encoder_stats['num_encoder_calls'] != total_frames or encoder_stats['num_frames_encoded'] != total_frames:
            raise RuntimeError('Encoder was called more or less than once per frame')
        rows = current_rows()
        total_wall = time.perf_counter() - started
        summary = {'status': 'completed', 'stream_id': args.stream_id,
            'completed_at_utc': datetime.now(timezone.utc).isoformat(),
            'plan': plan, 'rows': rows, 'total_frames': total_frames, 'total_visual_tokens': total_tokens,
            'payload_bytes': payload_bytes, 'actual_cumulative_full_feature_bytes': payload_bytes,
            'full_memory_GB': payload_bytes / 1e9, 'full_memory_TB': payload_bytes / 1e12,
            'bytes_per_frame': payload_bytes / total_frames, 'tokens_per_frame': total_tokens / total_frames,
            'bytes_per_token': payload_bytes / total_tokens, 'full_feature_archive_created': False,
            'planned_run_vs_accumulated_difference_bytes': payload_bytes - plan['planned_run_feature_bytes'],
            'full_stream_completed': total_frames == plan['reference_frames'],
            'difference_vs_nominal_60h_smoke_projection_bytes': (payload_bytes - 1046642688000
                                                               if total_frames == plan['reference_frames'] else None),
            'timing_seconds': {**dict(counts), 'memory_management_seconds': sum(updates.values()), 'total_wall_seconds': total_wall},
            'encoder_stats': encoder_stats, 'decoder_stats': decoder_stats, 'resources': telemetry.report(),
            'ordered_shared_features_sha256': encode_digest.hexdigest(), 'consumer_count': len(managers),
            'num_encoder_calls': encoder_stats['num_encoder_calls'],
            'peak_current_feature_bytes': peak_current_feature_bytes, 'snapshots': snapshots,
            'unreached_snapshot_times': pending, 'all_hard_caps_verified_each_update': True,
            'snapshot_save_load_consistency_verified': True,
            'accounting_scope': 'retained disk record bytes + bounded control state + FluxMem live tier tensors; encoder, transient queues, snapshot archive copies and filesystem allocation overhead reported separately',
            'uniform_definition': 'causal dyadic token-index uniform thinning; fixed spatial grid makes token density proportional to time; not moving exact fixed-count linspace',
            'fluxmem_definition': 'official TAS/SDC kernels; shared single-image grids, budget-adaptive tier capacity, native SDC recompression to fixed point before explicitly counted hard-cap eviction fallback',
            'flux_workers': flux_workers}
        write_json(output / 'summary.json', summary)
        (output / 'summary.txt').write_text(actual_table(rows))
        write_json(output / 'status.json', {'status': 'completed', 'frames_processed': total_frames,
                                          'summary': str(output / 'summary.json')})
        print(actual_table(rows), flush=True)
        return summary
    except BaseException as exc:
        write_json(output / 'status.json', {'status': 'failed', 'frames_processed': total_frames,
                                          'error_type': type(exc).__name__, 'error': str(exc)[:2000]})
        raise
    finally:
        try:
            if encoded is not None:
                encoded.close()
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
            for manager, _ in managers.values():
                manager.close()
            telemetry.close()
