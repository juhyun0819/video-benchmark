"""Frozen observation protocol with one pending CPU writer process.

CLI arguments match ``rekv_qwen25.runner``. Config additionally requires
writer_mode='cpu_process_overlap_v1', max_pending_blocks=1 and source hashes
for async_runner.py and bounded_writer.py. Model calls, absolute positions,
raw KV, block metadata and exact resume use the frozen runner helpers.

The parent submits immutable block A, computes block B using new buffers,
acknowledges A, then submits B. Only durable acknowledgements advance committed
progress. Completion requires the final acknowledgement and a clean child exit.
No question, retrieval, training or full-run coordination is performed here.
"""
from __future__ import annotations

import copy
import fcntl
import json
import math
import os
from pathlib import Path
import re
import threading
import time

import torch

from . import runner as base


WRITER_MODE = 'cpu_process_overlap_v1'


def create_writer(root, config, *, startup_timeout, commit_timeout):
    from .bounded_writer import BoundedKVWriter
    return BoundedKVWriter(root, config, startup_timeout=startup_timeout, commit_timeout=commit_timeout)


def _load_canonical(args):
    config = json.loads(args.config.read_text())
    if config.get('writer_mode') != WRITER_MODE or type(config.get('max_pending_blocks')) is not int or config['max_pending_blocks'] != 1:
        raise ValueError('Async runner requires cpu_process_overlap_v1 with exactly one pending block')
    sources = config.get('extraction_source_sha256', {})
    if not {'async_runner.py', 'bounded_writer.py'}.issubset(sources):
        raise ValueError('Async configuration must bind both new process-writer source modules')
    tensors, metadata = base.load_block(args.initial_state_root, 0, config.get('initial_state_config_id'))
    initial = base.unflatten_state(tensors, metadata['resume_state'])
    receipt = json.loads((args.initial_state_root/'receipts/chunk_00000000.ready.json').read_text())
    binding = {key: receipt[key] for key in ('config_id', 'payload_sha256', 'index_sha256')}
    halo, attention_hash = base._validate_config(config, initial, binding)
    base._check_range(args.start_frame, args.end_frame, config['total_frames'])
    return config, initial, binding, halo, attention_hash


class CommitPipeline:
    """One queued immutable block with explicit observed/submitted/durable order."""

    def __init__(self, writer, config_id, start_frame, counters, intervals, status):
        self.writer, self.config_id = writer, config_id
        self.committed_next_frame = self.submitted_next_frame = start_frame
        self.counters, self.intervals, self.status = counters, intervals, status
        self.pending = None
        self.writer_pid = None
        self.peak_writer_rss_bytes = 0
        self.max_pending_observed = 0
        self.max_inflight_blocks = 0
        self.max_tracked_observation_intervals = 0
        self._observation_intervals = []

    def record_observation(self, frame, started_time, finished_time):
        """Track only the next block while the previous writer block is pending."""
        if self.pending is None:
            return
        if not self.pending['next_frame_id'] <= frame < self.pending['next_frame_id'] + base.FRAMES_PER_BLOCK:
            raise ValueError('Observed more than one block beyond the pending writer block')
        if not (math.isfinite(started_time) and math.isfinite(finished_time) and finished_time >= started_time):
            raise ValueError('Invalid observation wall-clock interval')
        if len(self._observation_intervals) >= base.FRAMES_PER_BLOCK:
            raise RuntimeError('Observation overlap telemetry exceeded its bounded block capacity')
        self._observation_intervals.append((started_time, finished_time))
        self.max_tracked_observation_intervals = max(self.max_tracked_observation_intervals, len(self._observation_intervals))
        self.max_inflight_blocks = max(self.max_inflight_blocks, 2)

    def submit(self, payload, metadata):
        """Snapshot this block's boundary before waiting on an earlier block."""
        snapshot = copy.deepcopy(metadata)
        ids = snapshot['frame_ids']
        first, next_frame = ids[0], snapshot['next_frame_id']
        if (first != self.submitted_next_frame or ids != list(range(first, next_frame))
                or not 0 < len(ids) <= base.FRAMES_PER_BLOCK
                or snapshot.get('total_seen') != 17 + next_frame*base.TOKENS_PER_FRAME
                or snapshot.get('config_id') != self.config_id):
            raise ValueError('Submitted block does not follow the exact observation boundary')
        if self.pending is not None:
            self.wait()
        chunk_id = first // base.FRAMES_PER_BLOCK
        began = time.perf_counter()
        self.writer.submit(chunk_id, payload, snapshot)
        self.counters['enqueue_seconds'] += time.perf_counter()-began
        self.pending = dict(chunk_id=chunk_id, first_frame_id=first, last_frame_id=next_frame-1,
                            frame_count=len(ids), next_frame_id=next_frame)
        self.submitted_next_frame = next_frame
        self.max_pending_observed = max(self.max_pending_observed, 1)
        self.max_inflight_blocks = max(self.max_inflight_blocks, 1)
        self.status('block_enqueued', enqueued_chunk_id=chunk_id)

    def wait(self):
        if self.pending is None:
            return None
        pending = self.pending
        self.status('waiting_for_writer', pending_chunk_id=pending['chunk_id'])
        began = time.perf_counter()
        envelope = self.writer.wait()
        self.counters['writer_wait_seconds'] += time.perf_counter()-began
        receipt = envelope['receipt']
        if receipt.get('config_id') != self.config_id or any(receipt.get(key) != pending[key]
                for key in ('chunk_id', 'first_frame_id', 'last_frame_id', 'frame_count')):
            raise ValueError('Writer acknowledgement does not identify the pending immutable block')
        if pending['first_frame_id'] != self.committed_next_frame:
            raise ValueError('Writer acknowledgement skips the durable frame cursor')
        if (envelope.get('cuda_initialized') is not False or type(envelope.get('writer_pid')) is not int
                or envelope['writer_pid'] <= 0 or envelope['writer_pid'] == os.getpid()):
            raise ValueError('Archive writer did not attest a separate CPU-only process')
        if self.writer_pid is not None and envelope['writer_pid'] != self.writer_pid:
            raise ValueError('Archive writer process changed during this worker run')
        for field in ('finite_validation_seconds', 'atomic_nas_seconds', 'started_time', 'finished_time', 'seconds'):
            value = envelope.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'Invalid writer timing: {field}')
        if envelope['finished_time'] <= envelope['started_time']:
            raise ValueError('Invalid writer commit wall-clock interval')
        size, rss = receipt.get('serialized_bytes'), envelope.get('writer_rss_bytes')
        peak_rss = envelope.get('writer_peak_rss_bytes', rss)
        if (type(size) is not int or size < 0 or type(rss) is not int or rss < 0
                or type(peak_rss) is not int or peak_rss < 0):
            raise ValueError('Invalid writer payload or memory byte count')
        self.writer_pid = envelope['writer_pid']
        self.peak_writer_rss_bytes = max(self.peak_writer_rss_bytes, rss, peak_rss)
        self.counters['finite_validation_seconds'] += envelope['finite_validation_seconds']
        self.counters['atomic_nas_seconds'] += envelope['atomic_nas_seconds']
        self.counters['serialized_bytes'] += size
        self.counters['committed_frames'] += pending['frame_count']
        overlaps = [max(0.0, min(end, envelope['finished_time'])-max(start, envelope['started_time']))
                    for start, end in self._observation_intervals]
        overlap_seconds = sum(overlaps)
        overlapped_calls = sum(value > 0 for value in overlaps)
        self.counters['compute_write_overlap_seconds'] += overlap_seconds
        self.counters['overlapped_observation_calls'] += overlapped_calls
        self.intervals.append(dict(chunk_id=pending['chunk_id'], started_time=envelope['started_time'],
            finished_time=envelope['finished_time'], seconds=envelope['seconds'], serialized_bytes=size,
            writer_pid=self.writer_pid, writer_rss_bytes=rss, writer_peak_rss_bytes=peak_rss,
            cuda_initialized=False,
            compute_write_overlap_seconds=overlap_seconds, overlapped_observation_calls=overlapped_calls))
        self.committed_next_frame = pending['next_frame_id']
        self.pending = None
        self._observation_intervals.clear()
        self.status('block_committed', chunk_id=receipt['chunk_id'])
        return envelope

    def drain(self):
        self.wait()
        if self.pending is not None or self.committed_next_frame != self.submitted_next_frame:
            raise RuntimeError('Writer pipeline failed to drain its submitted coverage')


def _parser():
    parser = base._parser()
    parser.description = __doc__
    parser.add_argument('--writer-startup-timeout-seconds', type=float, default=60)
    parser.add_argument('--writer-commit-timeout-seconds', type=float, default=1800)
    return parser


@torch.inference_mode()
def _run(args):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.worker_id):
        raise ValueError('Invalid worker-id')
    for name in ('barrier_timeout_seconds', 'feature_wait_timeout_seconds',
                 'writer_startup_timeout_seconds', 'writer_commit_timeout_seconds'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f'{name} must be positive and finite')
    config, initial, binding, halo, attention_hash = _load_canonical(args)
    archive = base.KVArchiveWriter(args.archive_root, config)
    logs, stem = args.archive_root/'logs', f'worker_{args.worker_id}'
    status_path, report_path = logs/f'{stem}.status.json', logs/f'{stem}.report.json'
    lock = (logs/f'{stem}.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError(f'Worker {args.worker_id} is already running') from None
    session = writer = pipeline = telemetry = None
    stop = threading.Event()
    started = time.perf_counter()
    counters = dict(read_seconds=0.0, infer_seconds=0.0, atomic_nas_seconds=0.0,
        capture_seconds=0.0, finite_validation_seconds=0.0, serialized_bytes=0, committed_frames=0,
        warmup_frames=0, warmup_seconds=0.0, warmup_read_seconds=0.0,
        warmup_infer_seconds=0.0, resume_seconds=0.0, feature_readiness_seconds=0.0,
        enqueue_seconds=0.0, writer_wait_seconds=0.0, writer_startup_seconds=0.0,
        compute_write_overlap_seconds=0.0, overlapped_observation_calls=0)
    observed_next = resumed_next = args.start_frame
    intervals = []
    writer_closed = False

    def status(state, **extra):
        committed_next = pipeline.committed_next_frame if pipeline is not None else resumed_next
        value = dict(state=state, worker_id=args.worker_id, pid=os.getpid(),
            writer_pid=getattr(writer, 'pid', None), writer_mode=WRITER_MODE, max_pending_blocks=1,
            start_frame=args.start_frame, end_frame=args.end_frame, next_frame_id=committed_next,
            committed_next_frame=committed_next, observed_next_frame=observed_next,
            submitted_next_frame=pipeline.submitted_next_frame if pipeline is not None else resumed_next,
            pending_chunk_id=pipeline.pending['chunk_id'] if pipeline is not None and pipeline.pending else None,
            config_id=archive.config_id, elapsed_seconds=time.perf_counter()-started, **counters)
        value.update(extra)
        base.write_json(status_path, value)
        print(json.dumps(value), flush=True)

    try:
        entries = base.committed_blocks(archive.root, expected_config_id=archive.config_id)
        observed_next, owned = base._owned_prefix(entries, args.start_frame, args.end_frame)
        resumed_next = observed_next
        if args.no_resume and owned:
            raise ValueError('--no-resume refuses already committed frames in this worker range')
        resumed = observed_next-args.start_frame
        if observed_next == args.end_frame:
            report = dict(status='complete', worker_id=args.worker_id, start_frame=args.start_frame,
                parent_pid=os.getpid(),
                end_frame=args.end_frame, config_id=archive.config_id, resumed_frames=resumed,
                measured_seconds=0.0, processing_fps=None, already_complete=True, peak_vram_bytes=0,
                writer_mode=WRITER_MODE, max_pending_blocks=1, writer_drained=True, writer_exitcode=0,
                writer_pid=None, peak_writer_rss_bytes=0, writer_cuda_initialized=False,
                max_pending_observed=0, max_inflight_blocks=0, max_tracked_observation_intervals=0,
                observed_next_frame=observed_next, committed_next_frame=observed_next,
                commit_intervals=[], **counters)
            base.write_json(report_path, report)
            status('complete', resumed_frames=resumed)
            return report
        warm_start = observed_next if owned else base._warmup_start(observed_next, halo, 17, 15000)
        preflight_end = min(args.end_frame, observed_next+base.FRAMES_PER_BLOCK) if args.wait_for_features else args.end_frame
        feature_id, seconds = base._feature_range(args.features, warm_start, preflight_end, config,
            wait=args.wait_for_features, timeout=args.feature_wait_timeout_seconds, status=status)
        counters['feature_readiness_seconds'] += seconds
        status('validating_checkpoint', resumed_frames=resumed)
        identity = base.validate_checkpoint(args.model, args.ledger)
        torch.set_num_threads(4)
        torch.cuda.set_device(args.gpu)
        telemetry = threading.Thread(target=base.monitor,
            args=(stop, logs/f'{stem}.telemetry.jsonl', args.gpu), daemon=True)
        telemetry.start()
        status('loading_model')
        model = base.Qwen2_5_VLForConditionalGeneration.from_pretrained(str(args.model), local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation='sdpa',
            device_map={'': f'cuda:{args.gpu}'}, use_safetensors=True)
        model.eval().requires_grad_(False)
        processor = base.AutoProcessor.from_pretrained(str(args.model), local_files_only=True, use_fast=False)
        stream = base.StreamingInputs(model, processor)
        if stream.n_init != 17 or stream.stride != 28 or len(model.model.language_model.layers) != 28:
            raise ValueError('Loaded model/input protocol differs from validated production geometry')
        session = base.ReKVSession(model, n_init=17, n_local=15000)
        session.import_state(initial)
        sinks = [{kind: state[kind] for kind in ('init_key', 'init_value')} for state in initial['states']]
        sink_tensors = {f'layer_{index:02d}.{kind}': state[f'init_{kind}']
            for index, state in enumerate(initial['states']) for kind in ('key', 'value')}
        torch.cuda.reset_peak_memory_stats()
        if owned:
            tick = time.perf_counter()
            last = max(owned, key=lambda item: item['chunk_id'])
            tensors, metadata = base.load_block(archive.root, last['chunk_id'], archive.config_id)
            if (metadata.get('next_frame_id'), metadata.get('total_seen'), metadata.get('initial_state_config_id')) != (
                    observed_next, 17+observed_next*base.TOKENS_PER_FRAME, binding['config_id']):
                raise ValueError('Last committed block has incompatible continuation metadata')
            restored = base.reconstruct_long_state(initial, tensors, sink_tensors, 17+observed_next*base.TOKENS_PER_FRAME)
            session.import_state(restored)
            del tensors, restored
            counters['resume_seconds'] = time.perf_counter()-tick
        else:
            tick = time.perf_counter()
            if warm_start:
                base.zero_boundary(session, stream, warm_start, sinks)
            for batch_start in range(warm_start, observed_next, base.FRAMES_PER_BLOCK):
                read_tick = time.perf_counter()
                features, _ = base.read_feature_frames(args.features, batch_start,
                    min(base.FRAMES_PER_BLOCK, observed_next-batch_start))
                counters['warmup_read_seconds'] += time.perf_counter()-read_tick
                for offset, feature in enumerate(features):
                    frame = batch_start+offset
                    infer_tick = time.perf_counter()
                    base.observe(model, session, stream.build(feature.unsqueeze(0), frame))
                    torch.cuda.synchronize()
                    counters['warmup_infer_seconds'] += time.perf_counter()-infer_tick
                    counters['warmup_frames'] += 1
                    if (frame+1) % 10 == 0:
                        status('warming', warmup_next_frame=frame+1)
                del features
            counters['warmup_seconds'] = time.perf_counter()-tick
        if session.total_seen != 17+observed_next*base.TOKENS_PER_FRAME:
            raise RuntimeError('Warmup/resume ended at the wrong global token boundary')
        tick = time.perf_counter()
        writer = create_writer(archive.root, config, startup_timeout=args.writer_startup_timeout_seconds,
                               commit_timeout=args.writer_commit_timeout_seconds)
        counters['writer_startup_seconds'] = time.perf_counter()-tick
        pipeline = CommitPipeline(writer, archive.config_id, observed_next, counters, intervals, status)
        ready = dict(status='ready', worker_id=args.worker_id, pid=os.getpid(), writer_pid=writer.pid,
            start_frame=args.start_frame, end_frame=args.end_frame, next_frame_id=observed_next,
            config_id=archive.config_id, warmup_frames=counters['warmup_frames'],
            warmup_seconds=counters['warmup_seconds'], timestamp=time.time())
        if args.barrier_dir:
            base._wait_barrier(args.barrier_dir, args.worker_id, ready, args.barrier_timeout_seconds, status)
        measured_start, measurement_started_time = time.perf_counter(), time.time()
        while observed_next < args.end_frame:
            block_start = observed_next
            count = min(base.FRAMES_PER_BLOCK, args.end_frame-block_start)
            current_feature_id, seconds = base._feature_range(args.features, block_start, block_start+count, config,
                wait=args.wait_for_features, timeout=args.feature_wait_timeout_seconds, status=status)
            counters['feature_readiness_seconds'] += seconds
            if current_feature_id != feature_id:
                raise ValueError('Visual archive identity changed during extraction')
            tick = time.perf_counter()
            features, rows = base.read_feature_frames(args.features, block_start, count)
            counters['read_seconds'] += time.perf_counter()-tick
            payload = base._allocate_block(session, count)
            for offset, feature in enumerate(features):
                frame = block_start+offset
                if session.total_seen != 17+frame*base.TOKENS_PER_FRAME:
                    raise RuntimeError('Observation cursor does not match its absolute frame ID')
                tick = time.perf_counter()
                observation_started_time = time.time()
                inputs = stream.build(feature.unsqueeze(0), frame)
                base.observe(model, session, inputs)
                torch.cuda.synchronize()
                counters['infer_seconds'] += time.perf_counter()-tick
                pipeline.record_observation(frame, observation_started_time, time.time())
                tick = time.perf_counter()
                base._append_frame(payload, session, inputs, offset)
                counters['capture_seconds'] += time.perf_counter()-tick
                observed_next = frame+1
                if observed_next % 10 == 0:
                    status('observing')
            metadata = base._block_metadata(rows, session, block_start, archive.config_id, binding['config_id'], feature_id)
            pipeline.submit(payload, metadata)
            # Storage ownership passed to the child; every subsequent block
            # allocates independent tensors. Never mutate or reuse this payload.
            del payload, metadata, features, rows, inputs
        status('draining_writer')
        pipeline.drain()
        writer.close()
        writer_closed = True
        if writer.exitcode != 0 or pipeline.committed_next_frame != args.end_frame:
            raise RuntimeError('Writer did not exit cleanly with complete durable coverage')
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-measured_start
        report = dict(status='complete', worker_id=args.worker_id, gpu=args.gpu,
            parent_pid=os.getpid(),
            start_frame=args.start_frame, end_frame=args.end_frame, config_id=archive.config_id,
            checkpoint=identity, initial_state_config_id=binding['config_id'], feature_archive_id=feature_id,
            initial_state_binding=binding, attention_sha256=attention_hash, halo_frames=halo,
            prefix_protocol=base.PREFIX_PROTOCOL, resumed_frames=resumed, measured_seconds=elapsed,
            processing_fps=counters['committed_frames']/elapsed,
            measurement_started_time=measurement_started_time, measurement_finished_time=time.time(),
            peak_vram_bytes=torch.cuda.max_memory_allocated(), total_elapsed_seconds=time.perf_counter()-started,
            writer_mode=WRITER_MODE, max_pending_blocks=1, max_pending_observed=pipeline.max_pending_observed,
            max_inflight_blocks=pipeline.max_inflight_blocks,
            max_tracked_observation_intervals=pipeline.max_tracked_observation_intervals,
            writer_pid=pipeline.writer_pid, peak_writer_rss_bytes=pipeline.peak_writer_rss_bytes,
            writer_drained=True, writer_exitcode=writer.exitcode, writer_cuda_initialized=False,
            observed_next_frame=observed_next, committed_next_frame=pipeline.committed_next_frame,
            commit_intervals=intervals, scope='benchmark' if args.benchmark else 'production_worker_range',
            query_independent=True, whole_archive_complete=False, **counters)
        base.write_json(report_path, report)
        status('complete', resumed_frames=resumed, measured_seconds=elapsed, writer_drained=True)
        return report
    except BaseException as exc:
        status('failed', error=str(exc), error_type=type(exc).__name__)
        raise
    finally:
        try:
            if writer is not None and not writer_closed:
                writer.abort()
        finally:
            if session is not None:
                session.restore()
            stop.set()
            if telemetry is not None:
                telemetry.join(timeout=15)
            lock.close()


def main():
    _run(_parser().parse_args())


if __name__ == '__main__':
    main()
