"""Bounded parallel *feature extraction*, followed by a single ordered consumer.

Workers never own a memory policy, query or history archive. Chunks may finish
out of order, but the iterator yields every input frame exactly once in source
order. All dispatched or completed-but-unconsumed chunks count against both
the chunk cap and a conservative transport-payload byte reservation.
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import resource
import time
import traceback


class ParallelEncodingError(RuntimeError):
    pass


def _device_spec(requested):
    """Requested numeric indices are logical indices inside the existing mask."""
    value = str(requested)
    if value == 'cpu' or value.startswith('cpu:'):
        return 'cpu', None
    number = value.removeprefix('cuda:')
    if not number.isdigit():
        raise ValueError('devices must contain logical CUDA indices or cpu[:worker]')
    index = int(number)
    mask = os.environ.get('CUDA_VISIBLE_DEVICES')
    if mask is None:
        return 'cuda:0', str(index)
    available = [part.strip() for part in mask.split(',') if part.strip()]
    if index >= len(available) or available[index] == '-1':
        raise ValueError(f'Logical CUDA device {index} is outside CUDA_VISIBLE_DEVICES')
    return 'cuda:0', available[index]


def _native_feature_size(model_path, dtype):
    """Predict exact image-path feature bytes without running the encoder."""
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    path = Path(model_path)
    vision = json.loads((path / 'config.json').read_text())['vision_config']
    processor = json.loads((path / 'preprocessor_config.json').read_text())
    element_size = {'float16': 2, 'bfloat16': 2, 'float32': 4,
                    'fp16': 2, 'bf16': 2, 'fp32': 4}[dtype]
    patch = processor.get('patch_size', vision['patch_size'])
    merge = processor.get('merge_size', vision['spatial_merge_size'])

    def size(image):
        height, width = image.height, image.width
        if processor.get('do_resize', True):
            height, width = smart_resize(height, width, factor=patch * merge,
                                        min_pixels=processor.get('min_pixels', 56 * 56),
                                        max_pixels=processor.get('max_pixels', 28 * 28 * 1280))
        return (height // patch) * (width // patch) // (merge * merge) * vision['out_hidden_size'] * element_size
    return size


def _memory_metrics(device):
    import torch
    gpu = device.startswith('cuda')
    try:
        current_rss = int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
    except (OSError, ValueError, IndexError):
        current_rss = None
    return {
        'peak_cpu_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'current_cpu_rss_bytes': current_rss,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device) if gpu else 0,
        'current_cuda_allocated_bytes': torch.cuda.memory_allocated(device) if gpu else 0,
        'current_cuda_reserved_bytes': torch.cuda.memory_reserved(device) if gpu else 0,
    }


def _worker(worker_id, requested, model_path, dtype, cpu_threads, tasks, results, encoder_factory):
    """Spawn target: no stateful memory method is imported or constructed."""
    device = 'cpu'
    calls = encoded = 0
    started = time.perf_counter()
    try:
        device, mask = _device_spec(requested)
        if mask is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = mask
        import torch
        from PIL import Image
        torch.set_num_threads(cpu_threads)
        if encoder_factory is None:
            from .encoder import Qwen25VisionEncoder
            encoder_factory = Qwen25VisionEncoder
        encoder = encoder_factory(model_path, device=device, dtype=dtype)
        results.put({'kind': 'ready', 'worker_id': worker_id, 'device': device,
                     'requested_device': str(requested), 'visible_device_token': mask,
                     'pid': os.getpid(), 'encoder_metadata': encoder.metadata,
                     'load_seconds': time.perf_counter() - started,
                     **_memory_metrics(device)})
        while True:
            task = tasks.get()
            if task is None:
                break
            outputs = []
            chunk_started = time.perf_counter()
            for frame in task['frames']:
                with Image.frombytes('RGB', tuple(frame['size']), frame['rgb']) as image:
                    calls += 1
                    features, detail = encoder.encode(image)
                if (features.device.type != 'cpu' or features.ndim != 2
                        or not features.is_contiguous()):
                    raise ValueError('Encoder must return contiguous CPU [tokens,hidden] features')
                actual = features.numel() * features.element_size()
                if actual != frame['expected_feature_bytes']:
                    raise ValueError(f'Feature byte estimate mismatch: actual={actual}, expected={frame["expected_feature_bytes"]}')
                encoded += 1
                outputs.append({**frame['metadata'], 'features': features, 'encoder': detail})
                del features, detail
            results.put({'kind': 'chunk', 'worker_id': worker_id, 'chunk_id': task['chunk_id'],
                         'outputs': outputs, 'encoder_calls': calls,
                         'frames_encoded': encoded,
                         'last_frame_id': task['frames'][-1]['metadata']['frame_id'],
                         'last_timestamp': task['frames'][-1]['metadata']['timestamp'],
                         'chunk_wall_seconds': time.perf_counter() - chunk_started,
                         **_memory_metrics(device)})
            # Queue's feeder owns the outgoing message until its IPC transfer
            # completes. Do not retain a second reference while waiting idle.
            del outputs, frame, task
        results.put({'kind': 'done', 'worker_id': worker_id, 'encoder_calls': calls,
                     'frames_encoded': encoded, 'worker_wall_seconds': time.perf_counter() - started,
                     **_memory_metrics(device)})
    except BaseException as exc:
        results.put({'kind': 'error', 'worker_id': worker_id, 'encoder_calls': calls,
                     'frames_encoded': encoded, 'error_type': type(exc).__name__,
                     'error': str(exc)[:2000], 'traceback': traceback.format_exc()[-5000:]})


def iter_encoded(stream, model_path, mode='sequential', devices=('0',), chunk_frames=8,
                 max_inflight_chunks=8, dtype='bfloat16', *, stats=None,
                 max_queue_bytes=2 * 1024 ** 3, cpu_threads=1,
                 worker_timeout_seconds=600, encoder_factory=None,
                 feature_size_estimator=None):
    """Yield original frame metadata + ``features`` + ``encoder`` in order.

    ``stream`` yields the existing ``video.iter_stream`` envelopes. Sequential
    mode is the reference and does not copy image pixels or use worker queues.
    Parallel mode uses spawn, one vision encoder per selected GPU, and a single
    decoder in this process. Devices are logical indices inside any existing
    CUDA_VISIBLE_DEVICES mask. CPU workers/injected encoders support tests.

    Queue cap is a conservative *payload reservation*, not a claim about process
    RSS: 3*(RGB bytes + expected feature bytes + JSON metadata bytes)+512 per
    frame covers transport copies. Frozen models, codec buffers, one currently
    decoded frame, Python/allocator overhead and consumer state are separate.
    No temporary feature file or full-history archive is created.
    """
    aliases = {'A': 'sequential', 'B': 'parallel', 'sequential-safe': 'sequential'}
    mode = aliases.get(mode, mode)
    if mode not in ('sequential', 'parallel'):
        raise ValueError('mode must be sequential/A or parallel/B')
    if isinstance(devices, str):
        devices = [part.strip() for part in devices.split(',') if part.strip()]
    devices = tuple(str(value) for value in devices)
    if not devices or len(set(devices)) != len(devices):
        raise ValueError('devices must be a nonempty unique list')
    for value in devices:
        _device_spec(value)
    for name, value in [('chunk_frames', chunk_frames), ('max_inflight_chunks', max_inflight_chunks),
                        ('max_queue_bytes', max_queue_bytes), ('cpu_threads', cpu_threads)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if not math.isfinite(worker_timeout_seconds) or worker_timeout_seconds <= 0:
        raise ValueError('worker_timeout_seconds must be finite and positive')
    if stats is None:
        stats = {}
    stats.update({'mode': mode, 'devices': list(devices), 'chunk_frames': chunk_frames,
                  'max_inflight_chunks': max_inflight_chunks, 'max_queue_bytes': max_queue_bytes,
                  'frames_decoded': 0, 'frames_yielded': 0, 'num_encoder_calls': 0,
                  'num_frames_encoded': 0, 'chunks_dispatched': 0, 'chunks_completed': 0,
                  'out_of_order_chunks': 0,
                  'peak_inflight_chunks': 0, 'current_inflight_chunks': 0, 'peak_queue_reserved_bytes': 0,
                  'queue_reserved_bytes': 0, 'peak_ready_feature_bytes': 0,
                  'ready_feature_bytes': 0, 'backpressure_wait_seconds': 0.0,
                  'byte_backpressure_events': 0, 'workers': {},
                  'decode_owner': 'parent_single_pass', 'feature_files_written': 0,
                  'state': 'starting', 'queue_byte_measure': 'conservative transport payload reservation; not RSS'})
    iterator = iter(stream)
    previous_id = None
    previous_timestamp = None
    start = time.perf_counter()

    def validate_item(item):
        nonlocal previous_id, previous_timestamp
        fid, timestamp = item['frame_id'], item['timestamp']
        if not isinstance(fid, int) or isinstance(fid, bool) or fid < 0:
            raise ValueError('frame_id must be a nonnegative integer')
        if (not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool)
                or not math.isfinite(timestamp) or timestamp < 0):
            raise ValueError('timestamp must be finite and nonnegative')
        if previous_id is not None and (fid <= previous_id or timestamp <= previous_timestamp):
            raise ValueError('Input frame IDs and timestamps must strictly increase')
        previous_id, previous_timestamp = fid, timestamp
        stats['frames_decoded'] += 1

    if mode == 'sequential':
        import torch
        torch.set_num_threads(cpu_threads)
        requested = devices[0]
        device = 'cpu' if requested.startswith('cpu') else ('cuda:' + requested.removeprefix('cuda:'))
        try:
            if encoder_factory is None:
                from .encoder import Qwen25VisionEncoder
                encoder_factory = Qwen25VisionEncoder
            encoder = encoder_factory(model_path, device=device, dtype=dtype)
            stats['workers']['0'] = {'device': device, 'encoder_metadata': encoder.metadata}
            stats['state'] = 'running'
            for item in iterator:
                validate_item(item)
                stats['num_encoder_calls'] += 1
                features, metadata = encoder.encode(item['frame'])
                stats['num_frames_encoded'] += 1
                stats['frames_yielded'] += 1
                stats['workers']['0'].update(last_frame_id=item['frame_id'], last_timestamp=item['timestamp'])
                yield {**{k: v for k, v in item.items() if k != 'frame'},
                       'features': features, 'encoder': metadata,
                       'encoder_worker_id': 0, 'encoder_device': device}
                del features, metadata
            stats['workers']['0'].update(encoder_calls=stats['num_encoder_calls'],
                                        frames_encoded=stats['num_frames_encoded'],
                                        **_memory_metrics(device))
            stats['state'] = 'completed'
        except GeneratorExit:
            stats['state'] = 'closed_early'
            raise
        except BaseException:
            stats['state'] = 'failed'
            raise
        finally:
            if hasattr(iterator, 'close'):
                iterator.close()
            stats['wall_seconds'] = time.perf_counter() - start
        return

    estimate_size = feature_size_estimator or _native_feature_size(model_path, dtype)
    context = mp.get_context('spawn')
    results = context.Queue(maxsize=max(len(devices), max_inflight_chunks))
    tasks = [context.Queue(maxsize=1) for _ in devices]
    processes = [context.Process(target=_worker,
                 args=(index, device, str(model_path), dtype, cpu_threads, tasks[index], results, encoder_factory),
                 name=f'feature-encoder-{index}') for index, device in enumerate(devices)]
    inflight, ready = {}, {}
    finished = set()
    idle = set()
    next_chunk_id = next_yield_id = 0
    pending_item = None
    exhausted = False
    normal = False
    last_response = time.perf_counter()

    def receive():
        nonlocal last_response
        waiting = time.perf_counter()
        while True:
            try:
                message = results.get(timeout=min(0.5, worker_timeout_seconds))
                last_response = time.perf_counter()
                stats['backpressure_wait_seconds'] += last_response - waiting
                if message['kind'] == 'error':
                    worker = stats['workers'].setdefault(str(message['worker_id']), {})
                    worker.update(message)
                    raise ParallelEncodingError(f"Worker {message['worker_id']}: {message['error_type']}: {message['error']}")
                return message
            except queue.Empty:
                dead = [(i, p.exitcode) for i, p in enumerate(processes)
                        if p.exitcode is not None and i not in finished]
                if dead:
                    raise ParallelEncodingError(f'Encoder worker exited before completion: {dead}')
                # Consumer memory updates/snapshot writes can legitimately take
                # much longer than the worker timeout while this generator is
                # paused at yield. Charge only time spent awaiting a response.
                if time.perf_counter() - waiting > worker_timeout_seconds:
                    raise ParallelEncodingError('Encoder workers exceeded response timeout')

    def update_workers(message):
        row = stats['workers'].setdefault(str(message['worker_id']), {})
        row.update({k: v for k, v in message.items() if k not in ('outputs', 'kind')})
        if message['kind'] == 'chunk':
            row['completed_chunks'] = row.get('completed_chunks', 0) + 1
            for name in ('current_cuda_allocated_bytes', 'current_cpu_rss_bytes'):
                value = message.get(name)
                if value is not None:
                    row['min_after_chunk_' + name] = min(row.get('min_after_chunk_' + name, value), value)
                    row['max_after_chunk_' + name] = max(row.get('max_after_chunk_' + name, value), value)
        stats['num_encoder_calls'] = sum(worker.get('encoder_calls', 0) for worker in stats['workers'].values())
        stats['num_frames_encoded'] = sum(worker.get('frames_encoded', 0) for worker in stats['workers'].values())

    def make_chunk():
        nonlocal pending_item, exhausted
        frames, reserve = [], 0
        while len(frames) < chunk_frames:
            if pending_item is None:
                try:
                    pending_item = next(iterator)
                    validate_item(pending_item)
                except StopIteration:
                    exhausted = True
                    break
            image = pending_item['frame']
            if image.mode != 'RGB':
                raise ValueError('Input image must be RGB')
            expected_bytes = int(estimate_size(image))
            if expected_bytes <= 0:
                raise ValueError('Expected feature bytes must be positive')
            metadata = {key: value for key, value in pending_item.items() if key != 'frame'}
            raw_bytes = image.width * image.height * 3
            metadata_bytes = len(json.dumps(metadata, allow_nan=False).encode('utf-8'))
            required = 3 * (raw_bytes + expected_bytes + metadata_bytes) + 512
            if required > max_queue_bytes:
                raise ValueError(f'One frame needs {required} queue-reserved bytes, exceeding max_queue_bytes={max_queue_bytes}')
            if stats['queue_reserved_bytes'] + reserve + required > max_queue_bytes:
                stats['byte_backpressure_events'] += 1
                break
            frames.append({'size': image.size, 'rgb': image.tobytes(),
                           'metadata': metadata, 'expected_feature_bytes': expected_bytes})
            reserve += required
            pending_item = None
        return frames, reserve

    try:
        for process in processes:
            process.start()
        while len(idle) < len(processes):
            message = receive()
            if message['kind'] != 'ready' or message['worker_id'] in idle:
                raise ParallelEncodingError('Invalid worker initialization message')
            idle.add(message['worker_id'])
            update_workers(message)
        stats['state'] = 'running'
        while True:
            while idle and not exhausted and len(inflight) < max_inflight_chunks:
                frames, reserved = make_chunk()
                if not frames:
                    break
                worker_id = min(idle)
                idle.remove(worker_id)
                expected = [(f['metadata']['frame_id'], f['metadata']['timestamp'], f['expected_feature_bytes']) for f in frames]
                inflight[next_chunk_id] = {'worker_id': worker_id, 'reserved': reserved,
                                          'expected': expected}
                stats['current_inflight_chunks'] = len(inflight)
                stats['queue_reserved_bytes'] += reserved
                stats['peak_queue_reserved_bytes'] = max(stats['peak_queue_reserved_bytes'], stats['queue_reserved_bytes'])
                stats['peak_inflight_chunks'] = max(stats['peak_inflight_chunks'], len(inflight))
                tasks[worker_id].put({'chunk_id': next_chunk_id, 'frames': frames})
                stats['chunks_dispatched'] += 1
                next_chunk_id += 1
                del frames
            if next_yield_id in ready:
                message = ready.pop(next_yield_id)
                allocation = inflight[next_yield_id]
                outputs = message.pop('outputs')
                for output in outputs:
                    stats['frames_yielded'] += 1
                    worker_id = allocation['worker_id']
                    requested = devices[worker_id]
                    output['encoder_worker_id'] = worker_id
                    output['encoder_device'] = ('cpu' if requested.startswith('cpu')
                                                else 'cuda:' + requested.removeprefix('cuda:'))
                    yield output
                feature_bytes = sum(size for _, _, size in allocation['expected'])
                stats['ready_feature_bytes'] -= feature_bytes
                stats['queue_reserved_bytes'] -= allocation['reserved']
                del inflight[next_yield_id]
                stats['current_inflight_chunks'] = len(inflight)
                del output, outputs, allocation, message
                next_yield_id += 1
                continue
            if exhausted and not inflight:
                break
            if not inflight:
                raise ParallelEncodingError('Dispatch made no progress despite an empty queue')
            message = receive()
            if message['kind'] != 'chunk':
                raise ParallelEncodingError('Unexpected worker message during encoding')
            chunk_id = message['chunk_id']
            allocation = inflight.get(chunk_id)
            if not allocation or chunk_id in ready or allocation['worker_id'] != message['worker_id']:
                raise ParallelEncodingError('Duplicate or unknown chunk result')
            outputs = message['outputs']
            if len(outputs) != len(allocation['expected']):
                raise ParallelEncodingError('Chunk frame count changed in transit')
            for output, (fid, timestamp, size) in zip(outputs, allocation['expected']):
                tensor = output['features']
                if (output['frame_id'] != fid or output['timestamp'] != timestamp
                        or tensor.numel() * tensor.element_size() != size):
                    raise ParallelEncodingError('Chunk frame identity or feature size changed in transit')
            ready[chunk_id] = message
            stats['out_of_order_chunks'] += int(chunk_id != next_yield_id)
            stats['ready_feature_bytes'] += sum(size for _, _, size in allocation['expected'])
            stats['peak_ready_feature_bytes'] = max(stats['peak_ready_feature_bytes'], stats['ready_feature_bytes'])
            stats['chunks_completed'] += 1
            idle.add(message['worker_id'])
            update_workers(message)
            del outputs, output, tensor, allocation, message
        for task_queue in tasks:
            task_queue.put(None)
        while len(finished) < len(processes):
            message = receive()
            if message['kind'] != 'done' or message['worker_id'] in finished:
                raise ParallelEncodingError('Invalid worker completion message')
            finished.add(message['worker_id'])
            update_workers(message)
        if not (stats['num_encoder_calls'] == stats['num_frames_encoded']
                == stats['frames_decoded'] == stats['frames_yielded']):
            raise ParallelEncodingError('Frame/encoder counters disagree; duplicate or missing encoding')
        normal = True
        stats['state'] = 'completed'
    except GeneratorExit:
        stats['state'] = 'closed_early'
        raise
    except BaseException:
        stats['state'] = 'failed'
        raise
    finally:
        if hasattr(iterator, 'close'):
            iterator.close()
        pending_item = None
        ready.clear()
        inflight.clear()
        if not normal:
            for process in processes:
                if process.is_alive():
                    process.terminate()
        for process in processes:
            if process.pid is not None:
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
        for task_queue in tasks:
            if not normal:
                task_queue.cancel_join_thread()
            task_queue.close()
            if normal:
                task_queue.join_thread()
        results.close()
        stats['queue_reserved_bytes'] = 0
        stats['ready_feature_bytes'] = 0
        stats['current_inflight_chunks'] = 0
        stats['worker_processes_alive_after_close'] = sum(process.is_alive() for process in processes)
        stats['wall_seconds'] = time.perf_counter() - start
