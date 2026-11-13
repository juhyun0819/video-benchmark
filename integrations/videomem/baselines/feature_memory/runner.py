"""Measure encoder features and bounded policies in one causal traversal."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import time

import torch
from safetensors.torch import save_file

from .storage import HistoryWriter, save_memory_snapshot, write_json


METHODS = ('fifo', 'uniform', 'reservoir')
DTYPES = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}


def emit(event, **fields):
    print(json.dumps({'event': event, **fields}, ensure_ascii=False, allow_nan=False), flush=True)


def dtype_probes(features, directory):
    directory.mkdir(exist_ok=False)
    result = {}
    for name, dtype in DTYPES.items():
        tensor = features.to(dtype=dtype).contiguous()
        path = directory / (name + '.safetensors')
        save_file({'features': tensor}, str(path))
        result[name] = {'shape': list(tensor.shape), 'element_size': tensor.element_size(),
                        'bytes_per_token': tensor.shape[1] * tensor.element_size(),
                        'tensor_payload_bytes': tensor.numel() * tensor.element_size(),
                        'serialized_file_bytes': path.stat().st_size,
                        'origin': 'same encoded frame cast to storage dtype; not separate inference'}
    return result


def summary_table(rows):
    lines = ['Method | Budget(tokens) | Actual Memory(MB / GiB) | Frames Seen | Frames/Items Retained | Compression Ratio',
             '--- | ---: | ---: | ---: | ---: | ---:']
    for row in rows:
        size = row['actual_memory_bytes']
        lines.append(f"{row['method']} | {row['budget']} | {size / 1e6:.3f} / {size / 2**30:.6f} | "
                     f"{row['total_frames_seen']} | {row['retained_items']} | {row['actual_memory_compression_ratio']:.2f}x")
    return '\n'.join(lines) + '\n'


def execute(args, *, encoder_factory=None):
    from .encoder import Qwen25VisionEncoder
    from .memory import TokenMemory
    from .video import inspect_sources, iter_stream

    if args.max_seconds is not None and (not math.isfinite(args.max_seconds) or args.max_seconds <= 0):
        raise ValueError('max-seconds must be positive and finite')
    if any(b <= 0 for b in args.budgets) or len(set(args.budgets)) != len(args.budgets):
        raise ValueError('Budgets must be unique positive integer token counts')
    if any(not math.isfinite(s) or s <= 0 for s in args.snapshot_seconds):
        raise ValueError('Snapshot times must be positive and finite')
    cpu_threads = getattr(args, 'cpu_threads', 1)
    if cpu_threads < 1:
        raise ValueError('cpu-threads must be positive')
    torch.set_num_threads(cpu_threads)
    inventory = inspect_sources(args.videos, status_path=args.dataset_status, timeline_path=args.timeline)
    if inventory['total_duration_seconds'] is None:
        raise ValueError('All source durations must be known before profiling')
    requested_end = min(inventory['total_duration_seconds'], args.max_seconds or float('inf'))
    for source in inventory['sources']:
        if source['global_start_seconds'] < requested_end and not source['verified']:
            raise ValueError('Requested stream includes an unverified video: ' + source['path'])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'inventory.json', inventory)
    write_json(output / 'run_config.json', {
        'videos': [str(path) for path in args.videos], 'model': str(args.model),
        'device': args.device, 'dtype': args.dtype, 'budgets': args.budgets,
        'snapshot_seconds': args.snapshot_seconds, 'max_seconds': args.max_seconds,
        'seed': args.seed, 'cpu_threads': cpu_threads,
        'save_full_history': not args.no_history_archive,
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(Path(__file__).parent.glob('*.py'))}})
    run_started = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    status = {'status': 'initializing', 'started_at_utc': started, 'qa_inference': False}
    write_json(output / 'status.json', status)
    try:
        encoder = (encoder_factory or Qwen25VisionEncoder)(args.model, device=args.device, dtype=args.dtype)
        load_seconds = time.perf_counter() - run_started
        write_json(output / 'encoder.json', encoder.metadata)
        managers = {(method, b): TokenMemory(method, b, seed=args.seed)
                    for method in METHODS for b in args.budgets}
        archive = None if args.no_history_archive else HistoryWriter(output / 'history')
    except BaseException as exc:
        write_json(output / 'status.json', {'status': 'failed', 'frames_seen': 0,
                                          'error_type': type(exc).__name__, 'error': str(exc)[:1000]})
        raise
    snapshots = []
    pending = sorted(set(args.snapshot_seconds))
    totals = Counter()
    shapes = Counter()
    frame_count = total_tokens = total_bytes = 0
    dtype_measurements = None
    decoder_stats = {}
    last_timestamp = -1.0
    runtime_started = time.perf_counter()
    stream = iter_stream(inventory, max_seconds=args.max_seconds, stats=decoder_stats)

    def take_snapshot(timestamp):
        if not frame_count:
            return
        start = time.perf_counter()
        key = f'{timestamp:014.3f}s'
        rows = []
        for (method, budget), manager in managers.items():
            row = save_memory_snapshot(manager, output / 'snapshots' / key / f'{method}_B{budget}', timestamp)
            row.update(method=method, budget=budget)
            rows.append(row)
        record = {'timestamp': timestamp, 'total_frames_seen': frame_count,
                  'last_retained_input_timestamp': last_timestamp, 'rows': rows,
                  'full_history_feature_bytes_seen': total_bytes,
                  'history_serialization': archive.sizes() if archive else None}
        write_json(output / 'snapshots' / key / 'summary.json', record)
        snapshots.append(record)
        totals['snapshot_seconds'] += time.perf_counter() - start
        emit('snapshot', timestamp=timestamp, frames_seen=frame_count)

    try:
        with (output / 'frames.jsonl').open('x') as log:
            for item in stream:
                timestamp = float(item['timestamp'])
                if timestamp <= last_timestamp:
                    raise ValueError('Stream timestamps must be strictly increasing')
                while pending and pending[0] <= timestamp:
                    take_snapshot(pending.pop(0))
                features, detail = encoder.encode(item['frame'])
                if features.ndim != 2 or features.device.type != 'cpu' or not features.is_contiguous():
                    raise ValueError('Encoder contract requires a contiguous CPU [tokens,hidden] tensor')
                if features.dtype != DTYPES[args.dtype]:
                    raise ValueError('Encoder storage dtype differs from requested dtype')
                nbytes = features.numel() * features.element_size()
                if not frame_count:
                    dtype_measurements = dtype_probes(features, output / 'dtype_probes')
                    write_json(output / 'dtype_probes.json', dtype_measurements)
                    emit('first_feature', shape=list(features.shape), dtype=str(features.dtype), bytes=nbytes)
                update_start = time.perf_counter()
                for manager in managers.values():
                    manager.update(features, frame_id=item['frame_id'], timestamp=timestamp)
                update_seconds = time.perf_counter() - update_start
                serialize_start = time.perf_counter()
                if archive:
                    archive.append(features, item['frame_id'], timestamp, detail)
                serialization_seconds = time.perf_counter() - serialize_start
                record = {'frame_id': item['frame_id'], 'timestamp': timestamp,
                          'day': item['day'], 'source_timestamp': item['source_timestamp'],
                          'tensor_shape': list(features.shape), 'dtype': args.dtype,
                          'tokens_per_frame': features.shape[0], 'hidden_dimension': features.shape[1],
                          'bytes_per_token': features.shape[1] * features.element_size(),
                          'feature_bytes': nbytes, 'decode_seconds': item['decode_seconds'],
                          'memory_update_seconds': update_seconds,
                          'serialization_seconds': serialization_seconds, 'encoder': detail}
                log.write(json.dumps(record, allow_nan=False) + '\n')
                for k in ('preprocess_seconds', 'encode_seconds', 'h2d_seconds', 'd2h_seconds', 'feature_validation_seconds'):
                    totals[k] += detail.get(k, 0)
                totals['decode_seconds'] += item['decode_seconds']
                totals['memory_update_seconds'] += update_seconds
                totals['serialization_seconds'] += serialization_seconds
                frame_count += 1
                total_tokens += features.shape[0]
                total_bytes += nbytes
                last_timestamp = timestamp
                shapes[str(list(features.shape))] += 1
                if frame_count % 30 == 0:
                    log.flush()
                    write_json(output / 'status.json', {'status': 'running', 'frames_seen': frame_count,
                               'last_timestamp': timestamp, 'payload_bytes': total_bytes,
                               'elapsed_seconds': time.perf_counter() - runtime_started})
                    emit('progress', frames=frame_count, timestamp=timestamp,
                         elapsed_seconds=round(time.perf_counter() - runtime_started, 2))
                del features
        if not frame_count:
            raise ValueError('No frames observed')
        end = min(float(inventory['total_duration_seconds']), args.max_seconds or float('inf'))
        while pending and pending[0] <= end:
            take_snapshot(pending.pop(0))
        if not snapshots or snapshots[-1]['timestamp'] != end:
            take_snapshot(end)
        if archive:
            archive.close()
        elapsed = time.perf_counter() - runtime_started
        final_rows = snapshots[-1]['rows']
        mean_bytes, mean_tokens = total_bytes / frame_count, total_tokens / frame_count
        estimates = {}
        for name, probe in dtype_measurements.items():
            scale = probe['element_size'] / features_element_size(args.dtype)
            estimates[name] = {
                'mean_feature_bytes_per_frame': mean_bytes * scale,
                'nominal_60h_frames': 216000,
                'nominal_60h_feature_bytes': mean_bytes * scale * 216000,
                'dataset_frames': inventory['total_1fps_frames'],
                'dataset_feature_bytes': mean_bytes * scale * inventory['total_1fps_frames'],
                'feature_budget_bytes': {str(b): b * probe['bytes_per_token'] for b in args.budgets}}
        result = {'status': 'completed', 'started_at_utc': started,
                  'scope': 'vision features only; no LLM, QA, audio, or saved raw images',
                  'processed_duration_seconds': end, 'total_frames_seen': frame_count,
                  'total_visual_tokens_seen': total_tokens, 'tensor_shapes': dict(shapes),
                  'mean_tokens_per_frame': mean_tokens, 'mean_feature_bytes_per_frame': mean_bytes,
                  'full_history_observed_tensor_bytes': total_bytes,
                  'full_history_observed_serialization': archive.sizes() if archive else None,
                  'dtype_measurements': dtype_measurements, 'full_history_projection': estimates,
                  'projection_basis': 'observed feature bytes/frame extrapolated to 1fps frame counts; unprocessed frames and full-history disk size are NOT measured',
                  'inventory_counts_provisional': inventory.get('counts_provisional', False),
                  'timing_seconds': {**dict(totals), 'model_load_seconds': load_seconds,
                                     'observation_wall_seconds': elapsed},
                  'observed_frames_per_wall_second': frame_count / elapsed,
                  'decoder_stats': decoder_stats, 'final_summary': final_rows,
                  'snapshots': [{'timestamp': s['timestamp'], 'frames_seen': s['total_frames_seen']} for s in snapshots],
                  'requested_snapshots_not_reached': pending,
                  'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else 0,
                  'accounting': {
                      'actual_memory_bytes': 'live policy tensor allocations plus deterministic metadata/control payload; excludes frozen encoder, transient workspace, Python allocator overhead, and inaccessible measurement archive',
                      'compression_ratio': 'full observed feature payload / retained feature payload',
                      'summary_table_compression_ratio': 'full observed feature payload / actual live bank bytes, including unused capacity and metadata',
                      'snapshot_serialized_bytes': 'actual safetensors plus metadata.json file lengths, separately reported',
                      'MB': 1000000, 'GB': 1000000000, 'MiB': 1048576, 'GiB': 1073741824},
                  'uniform_note': 'causal dyadic-stride uniform thinning, B is an upper bound; not exact moving linspace of B samples',
                  'versions': {'python': platform.python_version(), 'torch': torch.__version__}}
        result['cpu_threads'] = cpu_threads
        write_json(output / 'summary.json', result)
        (output / 'summary.txt').write_text(summary_table(final_rows))
        write_json(output / 'status.json', {'status': 'completed', 'frames_seen': frame_count,
                                          'summary': str(output / 'summary.json')})
        print(summary_table(final_rows), flush=True)
        return result
    except BaseException as exc:
        write_json(output / 'status.json', {'status': 'failed', 'frames_seen': frame_count,
                                          'error_type': type(exc).__name__, 'error': str(exc)[:1000]})
        raise
    finally:
        stream.close()
        if archive:
            archive.close()


def features_element_size(dtype):
    return torch.empty((), dtype=DTYPES[dtype]).element_size()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('inspect', 'run', 'plan', 'build', 'archive'):
        p = sub.add_parser(name)
        p.add_argument('--videos', nargs='+', required=True, type=Path)
        p.add_argument('--dataset-status', type=Path)
        p.add_argument('--timeline', type=Path)
        p.add_argument('--output', type=Path, required=True)
        if name == 'run':
            p.add_argument('--model', type=Path, required=True)
            p.add_argument('--device', default='cuda:0')
            p.add_argument('--dtype', choices=DTYPES, default='bfloat16')
            p.add_argument('--budgets', nargs='+', type=int, default=[512, 1024, 2048, 4096])
            p.add_argument('--snapshot-seconds', nargs='+', type=float,
                           default=[3600, 18000, 36000, 72000, 144000, 216000])
            p.add_argument('--max-seconds', type=float)
            p.add_argument('--seed', type=int, default=20261111)
            p.add_argument('--cpu-threads', type=int, default=1)
            p.add_argument('--no-history-archive', action='store_true',
                           help='omit full-history binary archive; bounded snapshots are still serialized')
        if name in ('plan', 'build', 'archive'):
            p.add_argument('--model', type=Path, required=True)
            p.add_argument('--max-seconds', type=float)
            p.add_argument('--ratios', nargs='+', type=float, default=[0.05, 0.10, 0.25, 0.50])
        if name in ('build', 'archive'):
            p.add_argument('--mode', choices=['sequential', 'parallel', 'A', 'B'], default='sequential')
            p.add_argument('--devices', nargs='+', default=['0'])
            p.add_argument('--stream-id', default='day1_7_continuous')
            p.add_argument('--chunk-frames', type=int, default=8)
            p.add_argument('--max-inflight-chunks', type=int, default=8)
            p.add_argument('--max-queue-bytes', type=int, default=2 * 1024**3)
            p.add_argument('--shard-bytes', type=int, default=64 * 1024**2)
            p.add_argument('--cpu-threads', type=int, default=1)
            p.add_argument('--progress-frames', type=int, default=30)
            p.add_argument('--flux-workers', type=int, default=1,
                           help='1 for serial reference; up to 4 independent FluxMem state updates concurrently, with a per-frame barrier')
            p.add_argument('--seed', type=int, default=20261111)
            p.add_argument('--snapshot-seconds', nargs='+', type=float,
                           default=[18000, 36000, 72000, 144000, 216000])
        if name == 'archive':
            p.add_argument('--archive-frames-per-shard', type=int, default=100)
            p.add_argument('--seed-run', type=Path)
            p.add_argument('--seed-label', default='5pct')
    p = sub.add_parser('consume')
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-frames', type=int)
    p.add_argument('--batch-frames', type=int, default=100)
    p.add_argument('--cpu-threads', type=int, default=1)
    p.add_argument('--shard-bytes', type=int, default=256 * 1024**2)
    p.add_argument('--seed', type=int, default=20261111)
    p.add_argument('--stream-id', default='day1_7_continuous')
    p.add_argument('--snapshot-seconds', nargs='+', type=float,
                   default=[18000, 36000, 72000, 144000, 216000])
    args = parser.parse_args()
    if args.command == 'inspect':
        from .video import inspect_sources
        inventory = inspect_sources(args.videos, status_path=args.dataset_status, timeline_path=args.timeline)
        if args.output.exists():
            raise FileExistsError(args.output)
        write_json(args.output, inventory)
        emit('inventory', frames=inventory['total_1fps_frames'], seconds=inventory['total_duration_seconds'],
             all_verified=inventory['all_verified'], provisional=inventory['counts_provisional'])
    elif args.command == 'plan':
        from .video import inspect_sources
        from .planning import make_plan, plan_table
        inventory = inspect_sources(args.videos, status_path=args.dataset_status, timeline_path=args.timeline)
        plan = make_plan(inventory, args.model, max_seconds=args.max_seconds, ratios=args.ratios)
        if args.output.exists():
            raise FileExistsError(args.output)
        write_json(args.output, plan)
        print(plan_table(plan), flush=True)
    elif args.command == 'build':
        from .construction import build
        build(args)
    elif args.command == 'archive':
        from .archive import build_archive
        build_archive(args)
    elif args.command == 'consume':
        from .staged import consume_archive
        consume_archive(args)
    else:
        execute(args)
