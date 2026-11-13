"""One pending immutable CPU block, committed by a spawned CPU-only process.

The parent owns each submitted tensor reference until wait() acknowledges it.
It must not mutate submitted tensors or any aliases during that interval. It
can build the next block independently, but must acknowledge this block before
submitting another. Shared CPU storage avoids another IPC serialization copy.
The existing KVArchiveWriter still clones, hashes, fsyncs and reads back files.
"""
from __future__ import annotations

import copy
import ctypes
import hashlib
import math
import os
from pathlib import Path
import resource
import signal
import threading
import time
import traceback

import torch
import torch.multiprocessing as multiprocessing

from .archive import KVArchiveWriter, _json, _normalize_config


class WriterProcessError(RuntimeError):
    """A writer startup, commit, protocol, or lifecycle failure."""


def _parent_guard(parent_pid):
    """Cover both abrupt parent death and the race before installing prctl."""
    method = 'parent_pid_watchdog'
    if os.name == 'posix' and Path('/proc').is_dir():
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            if libc.prctl(1, signal.SIGTERM, 0, 0, 0) == 0:  # PR_SET_PDEATHSIG
                method = 'SIGTERM_parent_death_signal_and_pid_watchdog'
        except (AttributeError, OSError):
            pass
    if os.getppid() != parent_pid:
        os._exit(128 + signal.SIGTERM)
    stop = threading.Event()

    def watch():
        while not stop.wait(0.5):
            if os.getppid() != parent_pid:
                os._exit(128 + signal.SIGTERM)

    thread = threading.Thread(target=watch, name='writer-parent-watch', daemon=True)
    thread.start()
    return stop, thread, method


def _rss_bytes():
    return int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')


def _writer_main(connection, parent_pid, root, config, cpu_threads, writer_factory):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    guard_stop = guard_thread = None
    chunk_id = None
    try:
        guard_stop, guard_thread, guard_method = _parent_guard(parent_pid)
        if torch.cuda.is_initialized():
            raise RuntimeError('CPU writer inherited or initialized a CUDA context')
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(1)
        writer = writer_factory(root, config)
        connection.send(dict(kind='ready', writer_pid=os.getpid(), config_id=writer.config_id,
            cpu_threads=torch.get_num_threads(), cuda_initialized=torch.cuda.is_initialized(),
            parent_death_guard=guard_method, start_method=multiprocessing.get_start_method()))
        while True:
            command = connection.recv()
            if command.get('kind') == 'close':
                connection.send(dict(kind='closed', writer_pid=os.getpid(),
                                     cuda_initialized=torch.cuda.is_initialized()))
                break
            if command.get('kind') != 'commit':
                raise ValueError('Unknown writer command')
            chunk_id, tensors = command['chunk_id'], command['tensors']
            finite_start = time.perf_counter()
            for name, tensor in tensors.items():
                if (not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu' or
                        tensor.layout != torch.strided or not tensor.is_contiguous() or
                        (tensor.numel() and not tensor.is_shared())):
                    raise ValueError(f'Writer tensor is not contiguous shared CPU storage: {name}')
                if (tensor.is_floating_point() or tensor.is_complex()) and not bool(torch.isfinite(tensor).all()):
                    raise ValueError(f'Nonfinite payload tensor rejected before commit: {name}')
            del tensor  # Do not retain the final tensor alias past its ACK.
            finite_seconds = time.perf_counter() - finite_start
            if torch.cuda.is_initialized():
                raise RuntimeError('CPU writer unexpectedly initialized CUDA')
            started_time, tick = time.time(), time.perf_counter()
            receipt = writer.commit(chunk_id, tensors, command['metadata'])
            atomic_seconds = time.perf_counter() - tick
            finished_time = time.time()
            if torch.cuda.is_initialized():
                raise RuntimeError('CPU writer unexpectedly initialized CUDA during commit')
            result = dict(kind='committed', receipt=receipt, chunk_id=chunk_id,
                started_time=started_time, finished_time=finished_time, seconds=atomic_seconds,
                atomic_nas_seconds=atomic_seconds, finite_validation_seconds=finite_seconds,
                writer_pid=os.getpid(), writer_rss_bytes=_rss_bytes(),
                writer_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                cpu_threads=torch.get_num_threads(), cuda_initialized=False)
            # Release the child references before the parent may reuse storage.
            del tensors, command
            connection.send(result)
            chunk_id = None
    except BaseException as error:
        try:
            connection.send(dict(kind='error', chunk_id=chunk_id, writer_pid=os.getpid(),
                error_type=type(error).__name__, error=str(error),
                traceback=traceback.format_exc()[-8192:], cuda_initialized=torch.cuda.is_initialized()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()
        if guard_stop is not None:
            guard_stop.set()
            guard_thread.join(timeout=1)


class BoundedKVWriter:
    """Spawn one writer and allow exactly one pending block.

    ``submit`` explicitly shares CPU storage and retains the submitted tensor
    objects until ``wait`` returns. Callers must treat those objects and aliases
    as immutable until acknowledgment. Metadata is copied. A second submission
    raises immediately rather than building an unbounded queue.

    ``close`` drains and joins on success. Use ``abort`` on parent failure to
    stop the child immediately; ready receipts remain resumable and interrupted
    writes remain uncommitted. Context-manager exit aborts on exceptions.
    """

    def __init__(self, root, config, *, startup_timeout=60, commit_timeout=1800,
                 shutdown_timeout=10, cpu_threads=4, _writer_factory=KVArchiveWriter):
        for name, value in [('startup_timeout',startup_timeout), ('commit_timeout',commit_timeout),
                            ('shutdown_timeout',shutdown_timeout)]:
            if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if type(cpu_threads) is not int or not 1 <= cpu_threads <= 4:
            raise ValueError('CPU writer thread count must be in 1..4')
        config = _normalize_config(config)
        expected_id = hashlib.sha256(_json(config)).hexdigest()
        self.commit_timeout, self.shutdown_timeout = commit_timeout, shutdown_timeout
        self._pending = None
        self._closed = self._aborted = False
        self._failure = None
        self._config_id = None
        context = multiprocessing.get_context('spawn')
        self._connection, child_connection = context.Pipe(duplex=True)
        self._process = context.Process(target=_writer_main,
            args=(child_connection, os.getpid(), str(root), config, cpu_threads, _writer_factory),
            name='rekv-cpu-archive-writer', daemon=True)
        try:
            self._process.start()
            child_connection.close()
            ready = self._receive(startup_timeout)
            if (ready.get('kind') != 'ready' or ready.get('config_id') != expected_id or
                    ready.get('writer_pid') != self.pid or ready.get('cuda_initialized') is not False or
                    ready.get('cpu_threads') != cpu_threads):
                raise WriterProcessError('Invalid CPU writer startup handshake')
            self._config_id, self.startup_info = ready['config_id'], ready
        except BaseException as error:
            child_connection.close()
            self._failure = error
            self.abort()
            raise

    @property
    def pid(self):
        return self._process.pid

    @property
    def config_id(self):
        return self._config_id

    @property
    def is_pending(self):
        return self._pending is not None

    @property
    def exitcode(self):
        return self._process.exitcode

    def _receive(self, timeout):
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f'CPU writer {self.pid} did not acknowledge within {timeout}s')
            if self._connection.poll(min(0.1, remaining)):
                try:
                    message = self._connection.recv()
                except (EOFError,OSError) as error:
                    raise WriterProcessError(f'CPU writer {self.pid} disconnected (exitcode={self.exitcode})') from error
                if message.get('kind') == 'error':
                    failure = WriterProcessError(f"CPU writer {self.pid} {message['error_type']}: {message['error']}")
                    failure.remote_traceback = message.get('traceback')
                    failure.remote_error_type = message['error_type']
                    raise failure
                return message
            if not self._process.is_alive():
                # Drain a final message emitted immediately before child exit.
                if self._connection.poll():
                    continue
                raise WriterProcessError(f'CPU writer {self.pid} exited without acknowledgment (exitcode={self.exitcode})')

    def _usable(self, *, check_alive=True):
        if self._failure is not None:
            raise WriterProcessError(f'CPU writer previously failed: {self._failure}')
        if self._closed or self._aborted:
            raise WriterProcessError('CPU writer is closed or aborted')
        if check_alive and not self._process.is_alive():
            error = WriterProcessError(f'CPU writer {self.pid} is not alive (exitcode={self.exitcode})')
            self._failure = error
            self.abort()
            raise error

    def submit(self, chunk_id, tensors, metadata):
        self._usable()
        if self.is_pending:
            raise RuntimeError('One block is already pending; wait for its acknowledgment before submitting another')
        if type(chunk_id) is not int or chunk_id < 0:
            raise ValueError('chunk_id must be a nonnegative integer')
        if not isinstance(tensors, dict) or not tensors:
            raise ValueError('Expected a nonempty tensor dictionary')
        for name, tensor in tensors.items():
            if (not isinstance(name,str) or not name or not isinstance(tensor,torch.Tensor) or
                    tensor.device.type != 'cpu' or tensor.layout != torch.strided or not tensor.is_contiguous()):
                raise ValueError('Only named contiguous CPU tensors can be submitted')
        _json(metadata)  # Reject non-JSON metadata before IPC (including any CUDA tensors).
        owned = dict(tensors)
        self._pending = dict(chunk_id=chunk_id,tensors=owned,submitted_at=time.monotonic())
        try:
            for tensor in owned.values():
                tensor.share_memory_()
            self._connection.send(dict(kind='commit',chunk_id=chunk_id,tensors=owned,
                                       metadata=copy.deepcopy(metadata)))
        except BaseException as error:
            self._failure = error
            self.abort()
            raise

    def wait(self, timeout=None):
        # A child can send a final error and exit before wait() is called.
        # Receive that message before treating process death as an empty reply.
        self._usable(check_alive=False)
        if not self.is_pending:
            raise RuntimeError('No submitted block is awaiting acknowledgment')
        if timeout is None:
            timeout = self.commit_timeout
        if isinstance(timeout,bool) or not isinstance(timeout,(int,float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('wait timeout must be positive and finite')
        try:
            result = self._receive(timeout)
            if (result.get('kind') != 'committed' or result.get('chunk_id') != self._pending['chunk_id'] or
                    result.get('writer_pid') != self.pid or result.get('cuda_initialized') is not False or
                    result.get('receipt',{}).get('config_id') != self.config_id or
                    result['receipt'].get('chunk_id') != self._pending['chunk_id']):
                raise WriterProcessError('Invalid CPU writer commit acknowledgment')
            self._pending = None
            return {key:value for key,value in result.items() if key != 'kind'}
        except BaseException as error:
            self._failure = error
            self.abort()
            raise

    def close(self):
        if self._failure is not None:
            raise WriterProcessError(f'Cannot close successfully after writer failure: {self._failure}')
        if self._aborted:
            raise WriterProcessError('Cannot close successfully after writer abort')
        if self._closed:
            return None
        try:
            result = self.wait() if self.is_pending else None
            self._usable()
            self._connection.send(dict(kind='close'))
            closed = self._receive(self.shutdown_timeout)
            if (closed.get('kind') != 'closed' or closed.get('writer_pid') != self.pid or
                    closed.get('cuda_initialized') is not False):
                raise WriterProcessError('Invalid CPU writer shutdown acknowledgment')
            self._process.join(self.shutdown_timeout)
            if self._process.is_alive() or self.exitcode != 0:
                raise WriterProcessError(f'CPU writer did not exit cleanly (exitcode={self.exitcode})')
            self._connection.close()
            self._closed = True
            return result
        except BaseException as error:
            self._failure = error
            self.abort()
            raise

    def abort(self):
        """Stop only this owned child; do not wait for a pending archive commit."""
        self._aborted = True
        if self._process.pid is not None:
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(self.shutdown_timeout)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(self.shutdown_timeout)
            if self._process.is_alive():
                raise WriterProcessError(f'Unable to stop CPU writer {self.pid}')
        self._pending = None
        self._connection.close()
        self._closed = True

    def __enter__(self):
        self._usable()
        return self

    def __exit__(self, exception_type, exception, traceback_value):
        if exception_type is None:
            self.close()
        else:
            self.abort()
        return False
