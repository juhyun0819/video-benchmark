"""Five ordered CPU consumers over one reusable shared feature buffer.

Four processes own one FluxMem budget each; a fifth owns all twelve lightweight
states. Only batch descriptors and bounded metadata cross Pipes. Workers map the
single parent-owned POSIX shared-memory buffer read-only, update existing memory
managers, and acknowledge before the parent can reuse it. No worker opens an
archive, decodes frames, loads an encoder, or observes an uncommitted later batch.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
import mmap
import multiprocessing as mp
from multiprocessing import shared_memory
import os
from pathlib import Path
import re
import resource
import time
import traceback
import warnings

import torch

from .storage import write_json


_METHODS = ('fifo', 'uniform', 'reservoir', 'fluxmem')


def _peak_ram():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def _status(path, worker, phase, **values):
    write_json(path, {'worker': worker, 'pid': os.getpid(), 'phase': phase,
                      'updated_at_unix': time.time(), 'peak_cpu_ram_bytes': _peak_ram(), **values})


def _make_managers(plan, output, assignments, seed, shard_bytes):
    from .byte_memory import ByteBudgetMemory
    from .fluxmem import FluxMemByteMemory
    result = []
    for method, setting in assignments:
        common = {'directory': Path(output) / 'live' / method / setting['label'],
                  'hard_budget_bytes': setting['hard_budget_bytes'],
                  'retention_ratio': setting['retention_ratio'],
                  'hidden_dim': int(plan['hidden_dim']),
                  'tokens_per_frame': int(plan['tokens_per_frame']),
                  'seed': seed, 'dtype': plan.get('dtype', 'bfloat16'),
                  'shard_bytes': shard_bytes}
        if method == 'fluxmem':
            manager = FluxMemByteMemory(height=plan['post_merger_grid_hw'][0],
                                       width=plan['post_merger_grid_hw'][1], **common)
        else:
            manager = ByteBudgetMemory(method=method, **common)
        result.append((method, setting, manager))
    return result


def _rows(plan, managers, timings, frames_seen, tokens_seen, last_timestamp):
    rows = []
    payload = tokens_seen * int(plan['hidden_dim']) * 2
    for method, setting, manager in managers:
        value = manager.stats()
        if value['total_frames_seen'] != frames_seen or value['total_visual_tokens_seen'] != tokens_seen:
            raise RuntimeError('Consumer counters disagree with committed shared stream')
        actual, cap = value['actual_memory_bytes'], setting['hard_budget_bytes']
        if actual > cap:
            raise RuntimeError(f'{method}/{setting["label"]} exceeded cap by {actual-cap} bytes at {last_timestamp}')
        row = {**value, 'method': method, 'label': setting['label'],
               'retention_ratio': setting['retention_ratio'], 'hard_budget_bytes': cap,
               'memory_update_time': timings[(method, setting['label'])],
               'fixed_reference_feature_bytes': plan['full_feature_bytes'],
               'effective_retention_ratio': actual / payload if payload else 0.0,
               'effective_visual_retention_ratio': value['retained_visual_tokens'] * int(plan['hidden_dim']) * 2 / payload
                                                    if payload else 0.0,
               'budget_utilization': actual / cap,
               'persistent_disk_bytes': manager.store.stats()['persistent_disk_bytes'],
               'worker_pid': os.getpid(), 'worker_peak_cpu_ram_bytes': _peak_ram()}
        rows.append(row)
    return rows


def _readonly_tensor(name, shape, dtype_name):
    """OS-protected shared pages; Torch itself has no read-only tensor flag."""
    if os.name != 'posix' or not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('A safe Linux POSIX shared-memory name is required')
    # Attach by read-only descriptor rather than SharedMemory(create=False):
    # workers then never register ownership with Python's resource_tracker and
    # cannot unlink the parent's reusable segment when they exit.
    descriptor = os.open('/dev/shm/' + name, os.O_RDONLY)
    mapped = None
    try:
        size = math.prod(shape) * torch.empty((), dtype=getattr(torch, dtype_name)).element_size()
        if size > os.fstat(descriptor).st_size:
            raise ValueError('A sufficiently large POSIX shared-memory segment is required')
        mapped = mmap.mmap(descriptor, size, access=mmap.ACCESS_READ)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='The given buffer is not writable')
            tensor = torch.frombuffer(mapped, dtype=getattr(torch, dtype_name)).reshape(shape)
        return mapped, tensor
    except BaseException:
        if mapped is not None:
            mapped.close()
        raise
    finally:
        os.close(descriptor)


def _worker(connection, worker, plan, output, assignments, seed, shard_bytes, cpu_threads):
    torch.set_num_threads(cpu_threads)
    status_path = Path(output) / 'worker_status' / (worker + '.json')
    managers = []
    frames_seen = tokens_seen = 0
    last_timestamp = -1.0
    timings = {(method, setting['label']): 0.0 for method, setting in assignments}
    request_id = None
    try:
        _status(status_path, worker, 'initializing', frames_seen=0)
        managers = _make_managers(plan, output, assignments, seed, shard_bytes)
        _status(status_path, worker, 'ready', frames_seen=0, states=len(managers))
        connection.send({'kind': 'ready', 'worker': worker, 'pid': os.getpid(),
                         'rows': _rows(plan, managers, timings, 0, 0, last_timestamp)})
        while True:
            message = connection.recv()
            request_id = message['request_id']
            command = message['command']
            started = time.perf_counter()
            if command == 'update':
                shape = tuple(message['shape'])
                frames = message['frames']
                if (len(shape) != 3 or shape[0] != len(frames) or not 0 < shape[0] <= 100
                        or shape[1:] != (int(plan['tokens_per_frame']), int(plan['hidden_dim']))
                        or message['dtype'] != 'bfloat16'):
                    raise ValueError('Invalid shared feature batch descriptor')
                # Validate the entire command before any consumer mutates state.
                previous = last_timestamp
                for offset, row in enumerate(frames):
                    stamp = float(row['timestamp'])
                    if row['frame_id'] != frames_seen + offset or not math.isfinite(stamp) or stamp <= previous:
                        raise ValueError('Worker rejected noncausal frame identities/timestamps')
                    previous = stamp
                _status(status_path, worker, 'updating', frames_seen=frames_seen,
                        batch_frames=len(frames), first_frame=frames[0]['frame_id'])
                mapped, tensor = _readonly_tensor(message['shm_name'], shape, message['dtype'])
                elapsed = {}
                try:
                    for offset, row in enumerate(frames):
                        feature = tensor[offset]
                        for method, setting, manager in managers:
                            begin = time.perf_counter()
                            manager.update(feature, row['frame_id'], float(row['timestamp']))
                            key = method, setting['label']
                            seconds = time.perf_counter() - begin
                            timings[key] += seconds
                            elapsed[f'{method}/{setting["label"]}'] = elapsed.get(f'{method}/{setting["label"]}', 0.0) + seconds
                        del feature
                        frames_seen += 1
                        tokens_seen += shape[1]
                        last_timestamp = float(row['timestamp'])
                        if (offset + 1) % 20 == 0:
                            _status(status_path, worker, 'updating', frames_seen=frames_seen,
                                    timestamp=last_timestamp, batch_frames=len(frames), batch_done=offset+1)
                finally:
                    # No manager may retain a view: existing manager contracts
                    # clone live tensors or synchronously write owned records.
                    if 'feature' in locals():
                        del feature
                    del tensor
                    mapped.close()
                rows = _rows(plan, managers, timings, frames_seen, tokens_seen, last_timestamp)
                _status(status_path, worker, 'ready', frames_seen=frames_seen,
                        timestamp=last_timestamp, last_batch_seconds=time.perf_counter()-started)
                connection.send({'kind': 'ok', 'request_id': request_id, 'worker': worker,
                                 'rows': rows, 'seconds': time.perf_counter()-started,
                                 'update_seconds_by_state': elapsed, 'frames_seen': frames_seen})
            elif command == 'stats':
                connection.send({'kind': 'ok', 'request_id': request_id, 'worker': worker,
                                 'rows': _rows(plan, managers, timings, frames_seen, tokens_seen, last_timestamp),
                                 'seconds': time.perf_counter()-started})
            elif command == 'snapshot':
                from .construction import verify_snapshot
                timestamp, key = float(message['timestamp']), message['key']
                if not frames_seen or not math.isfinite(timestamp) or timestamp <= last_timestamp:
                    raise ValueError('Snapshot cutoff must be strictly after all committed frames')
                states = []
                for index, (method, setting, manager) in enumerate(managers):
                    _status(status_path, worker, 'snapshot', frames_seen=frames_seen, timestamp=timestamp,
                            method=method, label=setting['label'], state_index=index, states=len(managers),
                            stage='save')
                    destination = Path(output) / 'snapshots' / method / setting['label'] / key
                    saved = manager.snapshot(destination, timestamp)
                    value = manager.stats()
                    _status(status_path, worker, 'snapshot', frames_seen=frames_seen, timestamp=timestamp,
                            method=method, label=setting['label'], state_index=index, states=len(managers),
                            stage='verify')
                    verified = verify_snapshot(destination, value['retained_visual_tokens'], fluxmem=method == 'fluxmem')
                    if method == 'fluxmem' and saved['core_file_sha256'] != verified['fluxmem_tiers_sha256']:
                        raise RuntimeError('Serialized FluxMem tiers changed during read-back verification')
                    if value['actual_memory_bytes'] > setting['hard_budget_bytes']:
                        raise RuntimeError('Snapshot flush exceeded byte cap')
                    states.append({'method': method, 'label': setting['label'],
                        'ratio': setting['retention_ratio'], 'retention_ratio': setting['retention_ratio'],
                        'path': str(destination), 'hard_budget_bytes': setting['hard_budget_bytes'],
                        'actual_memory_bytes': value['actual_memory_bytes'],
                        'serialized_bytes': saved['snapshot_serialized_bytes'],
                        'snapshot_serialized_bytes': saved['snapshot_serialized_bytes'],
                        'validation': verified, 'verified': True})
                rows = _rows(plan, managers, timings, frames_seen, tokens_seen, last_timestamp)
                _status(status_path, worker, 'ready', frames_seen=frames_seen, timestamp=last_timestamp,
                        last_snapshot_seconds=time.perf_counter()-started)
                connection.send({'kind': 'ok', 'request_id': request_id, 'worker': worker,
                                 'rows': rows, 'states': states, 'seconds': time.perf_counter()-started})
            elif command == 'close':
                for _, _, manager in managers:
                    manager.close()
                managers = []
                _status(status_path, worker, 'closed', frames_seen=frames_seen, timestamp=last_timestamp)
                connection.send({'kind': 'ok', 'request_id': request_id, 'worker': worker,
                                 'seconds': time.perf_counter()-started})
                return
            else:
                raise ValueError('Unknown memory worker command')
    except BaseException as exc:
        error = {'kind': 'error', 'request_id': request_id, 'worker': worker,
                 'error_type': type(exc).__name__, 'error': str(exc),
                 'traceback': traceback.format_exc(limit=12), 'frames_seen': frames_seen}
        try:
            _status(status_path, worker, 'failed', **{k: v for k, v in error.items() if k != 'worker'})
            connection.send(error)
        except BaseException:
            pass
    finally:
        for _, _, manager in managers:
            try:
                manager.close()
            except BaseException:
                pass
        connection.close()


class MemoryFarm:
    """Synchronous ordered batch API backed by five independent CPU processes."""
    def __init__(self, plan, output, seed=0, shard_bytes=64 * 1024 * 1024, cpu_threads=1,
                 max_batch_frames=100, startup_timeout=180.0, command_timeout=None):
        if type(max_batch_frames) is not int or not 1 <= max_batch_frames <= 100:
            raise ValueError('max_batch_frames must be between 1 and 100')
        if type(cpu_threads) is not int or cpu_threads < 1:
            raise ValueError('cpu_threads must be positive')
        if plan.get('dtype', 'bfloat16') != 'bfloat16':
            raise ValueError('The process construction contract currently uses bf16')
        settings = plan.get('settings', [])
        if len(settings) != 4 or len({x['label'] for x in settings}) != 4:
            raise ValueError('MemoryFarm needs four distinct fixed budget settings')
        if set(plan.get('methods', _METHODS)) != set(_METHODS):
            raise ValueError('MemoryFarm builds FIFO/uniform/reservoir/FluxMem')
        if any(not re.fullmatch(r'[A-Za-z0-9_-]+', x['label']) for x in settings):
            raise ValueError('Unsafe budget label')
        self.plan, self.output = deepcopy(plan), Path(output)
        self.max_batch_frames, self.command_timeout = max_batch_frames, command_timeout
        self.frames_seen, self.last_timestamp = 0, -1.0
        self._closed, self._failed, self._busy = False, False, False
        self._request_id = 0
        self._handles = {}
        self._rows = []
        self._shm = None
        self._close_receipt = None
        self.shared_memory_bytes = max_batch_frames * int(plan['tokens_per_frame']) * int(plan['hidden_dim']) * 2
        if self.shared_memory_bytes <= 0:
            raise ValueError('Shared representation dimensions must be positive')
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / 'worker_status').mkdir(exist_ok=True)
        if os.name == 'posix' and Path('/dev/shm').is_dir():
            available = os.statvfs('/dev/shm')
            if self.shared_memory_bytes > available.f_bavail * available.f_frsize:
                raise ValueError('Insufficient /dev/shm capacity for the configured batch buffer')
        context = mp.get_context('spawn')
        assignments = {'lightweight': [(method, setting) for method in _METHODS[:3] for setting in settings]}
        assignments.update({f'fluxmem_{setting["label"]}': [('fluxmem', setting)] for setting in settings})
        try:
            self._shm = shared_memory.SharedMemory(create=True, size=self.shared_memory_bytes)
            self.shared_memory_name = self._shm.name
            for name, owned in assignments.items():
                parent, child = context.Pipe(duplex=True)
                process = context.Process(target=_worker,
                    args=(child, name, self.plan, str(self.output), owned, seed, shard_bytes, cpu_threads),
                    name='feature-memory-' + name)
                process.start()
                child.close()
                self._handles[name] = {'connection': parent, 'process': process}
            ready = self._collect(set(self._handles), request_id=None, timeout=startup_timeout, ready=True)
            self._rows = self._sort_rows([row for message in ready.values() for row in message['rows']])
        except BaseException:
            self._abort()
            raise

    @property
    def worker_pids(self):
        return {name: entry['process'].pid for name, entry in self._handles.items()}

    def _sort_rows(self, rows):
        labels = {item['label']: i for i, item in enumerate(self.plan['settings'])}
        return sorted(rows, key=lambda row: (_METHODS.index(row['method']), labels[row['label']]))

    def _collect(self, pending, request_id, timeout=None, ready=False):
        started = time.monotonic()
        responses = {}
        pending = set(pending)
        while pending:
            for name in tuple(pending):
                entry = self._handles[name]
                connection, process = entry['connection'], entry['process']
                if connection.poll():
                    try:
                        response = connection.recv()
                    except EOFError as exc:
                        raise RuntimeError(f'Memory worker {name} disconnected (exit={process.exitcode})') from exc
                    if response.get('kind') == 'error':
                        raise RuntimeError(f'Memory worker {name}: {response["error_type"]}: {response["error"]}\n'
                                           + response.get('traceback', ''))
                    if response.get('worker') != name or response.get('kind') != ('ready' if ready else 'ok'):
                        raise RuntimeError('Unexpected memory worker response')
                    if not ready and response.get('request_id') != request_id:
                        raise RuntimeError('Memory worker acknowledged a different batch')
                    responses[name] = response
                    pending.remove(name)
                elif not process.is_alive():
                    raise RuntimeError(f'Memory worker {name} exited without acknowledgement (exit={process.exitcode})')
            if pending:
                if timeout is not None and time.monotonic() - started > timeout:
                    raise TimeoutError(f'Memory workers did not acknowledge: {sorted(pending)}')
                time.sleep(0.05)
        return responses

    def _command(self, command, timeout=None, **values):
        if self._closed or self._failed or self._busy:
            raise RuntimeError('MemoryFarm is closed, failed, or already processing a command')
        self._busy = True
        self._request_id += 1
        try:
            for entry in self._handles.values():
                entry['connection'].send({'command': command, 'request_id': self._request_id, **values})
            return self._collect(set(self._handles), self._request_id,
                                 timeout=self.command_timeout if timeout is None else timeout)
        except BaseException:
            self._abort()
            raise
        finally:
            self._busy = False

    def update_batch(self, features, frames):
        if self._closed or self._failed or self._busy:
            raise RuntimeError('MemoryFarm is not available for another batch')
        expected = (len(frames), int(self.plan['tokens_per_frame']), int(self.plan['hidden_dim']))
        if (not isinstance(features, torch.Tensor) or features.device.type != 'cpu'
                or features.dtype != torch.bfloat16 or tuple(features.shape) != expected
                or not features.is_contiguous() or not 0 < len(frames) <= self.max_batch_frames):
            raise ValueError('Expected a bounded contiguous CPU bf16 [frames,tokens,hidden] batch')
        metadata = []
        previous = self.last_timestamp
        for offset, row in enumerate(frames):
            stamp = float(row['timestamp'])
            if (type(row['frame_id']) is not int or row['frame_id'] != self.frames_seen + offset
                    or not math.isfinite(stamp) or stamp <= previous):
                raise ValueError('Frame IDs must be consecutive and timestamps strictly increasing')
            metadata.append({'frame_id': row['frame_id'], 'timestamp': stamp})
            previous = stamp
        started = time.perf_counter()
        source = memoryview(features.view(torch.uint8).numpy()).cast('B')
        size = len(source)
        before = hashlib.sha256(source).hexdigest()
        self._shm.buf[:size] = source
        copy_seconds = time.perf_counter() - started
        del source
        begin = time.perf_counter()
        replies = self._command('update', shm_name=self.shared_memory_name, shape=list(features.shape),
                                dtype='bfloat16', frames=metadata)
        update_wall = time.perf_counter() - begin
        try:
            if hashlib.sha256(self._shm.buf[:size]).hexdigest() != before:
                raise RuntimeError('A memory consumer mutated the shared encoder feature buffer')
            self.frames_seen += len(metadata)
            self.last_timestamp = metadata[-1]['timestamp']
            self._rows = self._sort_rows([row for reply in replies.values() for row in reply['rows']])
            if len(self._rows) != 16 or any(row['total_frames_seen'] != self.frames_seen for row in self._rows):
                raise RuntimeError('Not all sixteen states committed the same feature prefix')
            return {'rows': deepcopy(self._rows), 'worker_timings': {name: {
                        'seconds': reply['seconds'], 'update_seconds_by_state': reply['update_seconds_by_state']}
                        for name, reply in replies.items()},
                    'seconds': time.perf_counter()-started, 'memory_update_wall_seconds': update_wall,
                    'shared_memory_copy_and_hash_seconds': copy_seconds,
                    'shared_feature_sha256': before, 'shared_buffer_bytes': self.shared_memory_bytes,
                    'batch_feature_bytes': size, 'queue_depth': 0,
                    'frames_seen': self.frames_seen, 'last_timestamp': self.last_timestamp,
                    'all_workers_acknowledged_before_buffer_reuse': True}
        except BaseException:
            self._abort()
            raise

    def stats(self):
        replies = self._command('stats')
        self._rows = self._sort_rows([row for reply in replies.values() for row in reply['rows']])
        return deepcopy(self._rows)

    def snapshot(self, timestamp, key):
        if (not self.frames_seen or not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool)
                or not math.isfinite(timestamp) or timestamp <= self.last_timestamp):
            raise ValueError('Snapshot cutoff must be strictly after the committed frame prefix; split batches at cutoffs')
        if not isinstance(key, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', key) or key in ('.', '..'):
            raise ValueError('Unsafe snapshot key')
        started = time.perf_counter()
        replies = self._command('snapshot', timestamp=float(timestamp), key=key)
        self._rows = self._sort_rows([row for reply in replies.values() for row in reply['rows']])
        states = self._sort_rows([state for reply in replies.values() for state in reply['states']])
        return {'rows': deepcopy(self._rows), 'states': states, 'seconds': time.perf_counter()-started,
                'worker_timings': {name: reply['seconds'] for name, reply in replies.items()},
                'timestamp': float(timestamp), 'frames_seen': self.frames_seen}

    def _release_buffer(self):
        if self._shm is not None:
            self._shm.close()
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass
            self._shm = None
        return True

    def _cleanup_receipt(self):
        alive = sum(entry['process'].is_alive() for entry in self._handles.values())
        name = getattr(self, 'shared_memory_name', None)
        unlinked = name is None or not Path('/dev/shm', name).exists()
        result = {'worker_processes_alive_after_close': alive,
                  'shared_memory_unlinked': unlinked, 'failed': self._failed}
        self._close_receipt = result
        return dict(result)

    def _abort(self):
        self._failed = True
        for entry in self._handles.values():
            process = entry['process']
            if process.is_alive():
                process.terminate()
        for entry in self._handles.values():
            process = entry['process']
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
            entry['connection'].close()
        self._release_buffer()
        self._closed = True
        self._cleanup_receipt()

    def close(self):
        if self._closed:
            return self._cleanup_receipt()
        try:
            self._command('close', timeout=60.0)
        except BaseException:
            self._abort()
            raise
        finally:
            for entry in self._handles.values():
                entry['process'].join(timeout=2)
                if entry['process'].is_alive():
                    entry['process'].terminate()
                    entry['process'].join(timeout=2)
                if entry['process'].is_alive():
                    entry['process'].kill()
                    entry['process'].join(timeout=2)
                entry['connection'].close()
            self._release_buffer()
            self._closed = True
        result = self._cleanup_receipt()
        if result['worker_processes_alive_after_close'] or not result['shared_memory_unlinked']:
            raise RuntimeError('MemoryFarm cleanup did not release all processes/shared memory')
        return result

    def __enter__(self):
        return self

    def __exit__(self, kind, value, tb):
        if kind is not None:
            self._abort()
        else:
            self.close()
