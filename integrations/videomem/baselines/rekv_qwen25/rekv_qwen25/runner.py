"""One frozen-model worker for a disjoint, query-independent ReKV archive range.

Run with ``python -m rekv_qwen25.runner --archive-root ROOT --config CONFIG
--initial-state-root INITIAL --start-frame A --end-frame B --gpu G
--worker-id ID``. Ranges are half-open, aligned to 100 global frames except
the dataset's final tail. Every model call observes exactly one 678-token
frame after the separate canonical 17-token prefix.

CONFIG must bind n_local=15000, n_init=17, frames_per_block=100, total_frames,
prefix_protocol='separate_fixed_prefix_then_one_frame', and halo_validation
with status='passed', report_path/report_sha256, halo_frames=620 and the current
attention_sha256. transformers_version must be '4.57.3'. The initial archive's
block zero contains a flattened session export described by metadata['resume_state'].
Its initial_state_config_id and initial_state_payload_sha256 must match CONFIG
and the proof's initial_state_binding. Optional initial_state_index_sha256,
feature_archive_id, policy and checkpoint/backbone identity fields are checked.

The worker never creates that initial artifact or approves its own halo proof.
It can resume from its committed raw blocks without duplicating the bounded
GPU cache on disk. A fresh later range replays a validated finite halo before
publishing. No question, retrieval or generation is performed.

For coordinated timing, --barrier-dir emits ID.ready.json after replay/resume,
then waits for start.json. Outputs are logs/worker_ID.{status,report}.json and
logs/worker_ID.telemetry.jsonl. --benchmark changes report scope only; it still
writes and verifies real archive commits. --no-resume rejects existing coverage.
--wait-for-features waits only for absent source ready receipts; committed source
corruption always fails. No source payload is opened before its ready receipt.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time

import torch
import transformers
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .archive import KVArchiveWriter, committed_blocks, load_block
from .attention import ReKVSession
from .halo_smoke import zero_boundary
from .inputs import REPO_ID, REVISION, read_feature_frames, validate_checkpoint
from .smoke import FEATURES, LEDGER, MODEL, monitor, write_json
from .state_io import reconstruct_long_state, unflatten_state
from .stream import StreamingInputs, observe


TOKENS_PER_FRAME = 678
FRAMES_PER_BLOCK = 100
PREFIX_PROTOCOL = 'separate_fixed_prefix_then_one_frame'


def _sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(part)
    return result.hexdigest()


def _validate_config(config, initial, initial_binding):
    expected = {'n_local': 15000, 'n_init': 17, 'frames_per_block': FRAMES_PER_BLOCK,
                'prefix_protocol': PREFIX_PROTOCOL}
    for field, value in expected.items():
        if config.get(field) != value:
            raise ValueError(f'Unsupported production config {field}: expected {value!r}')
    if type(config.get('total_frames')) is not int or config['total_frames'] < 1:
        raise ValueError('config.total_frames must be a positive integer')
    if (initial.get('n_local'), initial.get('n_init'), initial.get('layers'),
            initial.get('total_seen')) != (15000, 17, 28, 17):
        raise ValueError('Canonical state must be the separate 17-token prefix for all 28 layers')
    if initial.get('native_post_rope_cache') is None:
        raise ValueError('Canonical prefix lacks its exact native post-RoPE cache')
    if config.get('transformers_version') != '4.57.3' or transformers.__version__ != '4.57.3':
        raise ValueError('Production requires the validated transformers 4.57.3 runtime')
    if config.get('torch_version', str(torch.__version__)) != str(torch.__version__):
        raise ValueError('Controlled torch version differs from this runtime')
    if (config.get('initial_state_config_id'), config.get('initial_state_payload_sha256')) != (
            initial_binding['config_id'], initial_binding['payload_sha256']):
        raise ValueError('Canonical initial artifact identity/hash differs from controlled config')
    if config.get('initial_state_index_sha256', initial_binding['index_sha256']) != initial_binding['index_sha256']:
        raise ValueError('Canonical initial index differs from controlled config')
    if 'policy' in config and config['policy'] != initial.get('policy'):
        raise ValueError('Canonical initial attention policy differs from controlled config')
    for field in ('backbone', 'checkpoint'):
        identity = config.get(field)
        if isinstance(identity, dict) and (identity.get('repo_id'), identity.get('revision')) != (REPO_ID, REVISION):
            raise ValueError(f'config.{field} does not identify the pinned checkpoint')
    halo = math.ceil(initial['layers'] * initial['n_local'] / TOKENS_PER_FRAME)
    proof = config.get('halo_validation', {})
    if proof.get('status') != 'passed' or proof.get('halo_frames') != halo:
        raise ValueError('Production requires the passed full-model finite-halo proof')
    attention_sha = _sha256(Path(__file__).with_name('attention.py'))
    if proof.get('attention_sha256') != attention_sha:
        raise ValueError('Attention source changed after finite-halo validation')
    report_path = proof.get('report_path')
    if not report_path or _sha256(report_path) != proof.get('report_sha256'):
        raise ValueError('Finite-halo report is missing or its content hash changed')
    report = json.loads(Path(report_path).read_text())
    if any(report.get('initial_state_binding', {}).get(key) != value for key, value in initial_binding.items()):
        raise ValueError('Canonical initial artifact is not bound to the successful halo proof')
    if (report.get('status'), report.get('halo_frames'), report.get('adaptation_sha256')) != (
            'passed', halo, attention_sha):
        raise ValueError('Finite-halo report does not prove this attention implementation')
    if report.get('verification_frames', 0) < 1:
        raise ValueError('Finite-halo report contains no verified output frames')
    if (report.get('checkpoint', {}).get('repo_id'), report.get('checkpoint', {}).get('revision')) != (REPO_ID, REVISION):
        raise ValueError('Finite-halo report used a different checkpoint revision')
    if report.get('n_local') != 15000 or report.get('n_init') != 17:
        raise ValueError('Finite-halo report used a different window or sink length')
    if report.get('policy') != initial.get('policy') or report.get('upstream') != initial.get('upstream'):
        raise ValueError('Finite-halo proof and canonical prefix use different attention policies')
    if (report.get('short_backend'), report.get('chunk_frames'),
            report.get('fixed_original_per_layer_sinks'), report.get('absolute_positions')) != (
            'sdpa', 1, True, True):
        raise ValueError('Finite-halo proof used a different observation protocol')
    if (report.get('torch_version'), report.get('cuda_version')) != (str(torch.__version__), torch.version.cuda):
        raise ValueError('Runtime torch/CUDA version differs from the finite-halo proof')
    if report.get('halo_tokens') != halo * TOKENS_PER_FRAME or report.get('conservative_receptive_bound_tokens') != 28 * 15000:
        raise ValueError('Finite-halo report has the wrong receptive-field bound')
    boundary, checks = report.get('boundary_frame'), report.get('checks')
    if type(boundary) is not int or boundary < 0 or not isinstance(checks, list):
        raise ValueError('Finite-halo report lacks actual boundary comparisons')
    verified = [row for row in checks if row.get('frame_id', -1) >= boundary + halo]
    if len(verified) != report['verification_frames'] or not all(
            row.get('all_kv_exact') is True and row.get('logits', {}).get('exact') is True
            and len(row.get('exact_layers', [])) == 28 and all(value is True for value in row['exact_layers'])
            for row in verified):
        raise ValueError('Finite-halo report does not establish all-layer KV/logit equality')
    if not any(row.get('frame_id') == boundary + halo and row.get('bounded_state_exact') is True for row in verified):
        raise ValueError('Finite-halo proof lacks exact full bounded continuation-state comparison')
    if not any(row.get('frame_id') == boundary and (
            row.get('all_kv_exact') is False or row.get('logits', {}).get('exact') is False) for row in checks):
        raise ValueError('Finite-halo report lacks a nontrivial initial perturbation')
    for filename, expected_hash in config.get('extraction_source_sha256', {}).items():
        if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith('.py'):
            raise ValueError('Invalid controlled extraction source filename')
        if _sha256(Path(__file__).with_name(filename)) != expected_hash:
            raise ValueError(f'Controlled extraction source changed: {filename}')
    return halo, attention_sha


def _check_range(start, end, total):
    if not 0 <= start < end <= total:
        raise ValueError('Require 0 <= start_frame < end_frame <= total_frames')
    if start % FRAMES_PER_BLOCK or (end % FRAMES_PER_BLOCK and end != total):
        raise ValueError('Worker bounds must align to 100-frame blocks except the final dataset tail')


def _owned_prefix(entries, start, end):
    """Find a worker's committed prefix; reject gaps/foreign partial blocks."""
    owned = [entry for entry in entries
             if entry['first_frame_id'] < end and entry['last_frame_id'] >= start]
    cursor = start
    for entry in sorted(owned, key=lambda item: item['first_frame_id']):
        expected_end = min(cursor + FRAMES_PER_BLOCK, end)
        if (entry['chunk_id'], entry['first_frame_id'], entry['last_frame_id'], entry['frame_count']) != (
                cursor // FRAMES_PER_BLOCK, cursor, expected_end - 1, expected_end - cursor):
            raise ValueError('Existing worker coverage has a gap, overlap or wrong global block boundaries')
        cursor = expected_end
    return cursor, owned


def _warmup_start(start, halo, n_init, window):
    replay_start = max(0, start - halo)
    # A short prefix requires HF's native post-RoPE cache. Replay it exactly
    # from zero instead of inventing an intermediate short-prefix state.
    if n_init + replay_start * TOKENS_PER_FRAME <= window:
        replay_start = 0
    return replay_start


def _require_feature_receipts(root, start, end, config):
    """Preflight complete range readiness; reads later verify every payload."""
    root = Path(root)
    header = json.loads((root / 'archive.json').read_text())
    if header['inventory']['total_1fps_frames'] != config['total_frames']:
        raise ValueError('Controlled frame count differs from visual archive inventory')
    if config.get('feature_archive_id', header['archive_id']) != header['archive_id']:
        raise ValueError('Controlled visual archive ID differs from source')
    size = header['frames_per_shard']
    if type(size) is not int or size <= 0:
        raise ValueError('Invalid visual archive shard size')
    for shard in range(start // size, (end - 1) // size + 1):
        path = root / f'ready_{shard:08d}.json'
        receipt = json.loads(path.read_text())
        if (receipt.get('archive_id'), receipt.get('shard_id')) != (header['archive_id'], shard):
            raise ValueError('Visual range includes a mismatched ready receipt')
        for field, prefix, suffix in [('feature_file', 'features_', '.bin'), ('index_file', 'index_', '.json')]:
            filename = f'{prefix}{shard:08d}{suffix}'
            if receipt.get(field) != filename or not (root / filename).is_file():
                raise ValueError(f'Visual ready receipt lacks its committed {field}')
    return header['archive_id']


def _feature_range(root, start, end, config, *, wait, timeout, status):
    """Wait exclusively for missing ready markers, never corrupted commits."""
    began, deadline = time.monotonic(), time.monotonic() + timeout
    while True:
        try:
            return _require_feature_receipts(root, start, end, config), time.monotonic() - began
        except FileNotFoundError as exc:
            missing = Path(exc.filename) if exc.filename else None
            if not wait or missing is None or not re.fullmatch(r'ready_[0-9]{8}\.json', missing.name):
                raise
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Source feature range {start}:{end} did not become ready') from exc
            status('waiting_for_features', feature_start_frame=start, feature_end_frame=end,
                   missing_receipt=missing.name, feature_wait_elapsed_seconds=time.monotonic()-began)
            time.sleep(min(5, max(0, deadline-time.monotonic())))


def _allocate_block(session, count):
    length = count * TOKENS_PER_FRAME
    result = {'position_ids': torch.empty((3, 1, length), dtype=torch.int64),
              'input_ids': torch.empty((1, length), dtype=torch.int64)}
    for layer, attention in enumerate(session.attentions):
        shape = (1, attention.num_key_value_heads, length, attention.head_dim)
        for kind in ('key', 'value'):
            result[f'layer_{layer:02d}.{kind}'] = torch.empty(shape, dtype=torch.bfloat16)
    return result


def _append_frame(payload, session, inputs, offset):
    if set(session.latest_kv) != set(payload) - {'position_ids', 'input_ids'}:
        raise ValueError('A frame is missing one or more layer raw KV tensors')
    lo, hi = offset * TOKENS_PER_FRAME, (offset + 1) * TOKENS_PER_FRAME
    payload['position_ids'][:, :, lo:hi].copy_(inputs['position_ids'].cpu())
    payload['input_ids'][:, lo:hi].copy_(inputs['input_ids'].cpu())
    for name, tensor in session.latest_kv.items():
        expected = payload[name][:, :, lo:hi]
        if tensor.shape != expected.shape or tensor.dtype != torch.bfloat16:
            raise ValueError(f'Unexpected per-frame raw KV geometry: {name}')
        expected.copy_(tensor)


def _require_finite_payload(payload):
    for name, tensor in payload.items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
            raise ValueError(f'Refusing to archive nonfinite raw KV: {name}')


def _block_metadata(rows, session, start, config_id, initial_config_id, feature_archive_id):
    enriched = []
    for offset, row in enumerate(rows):
        current = dict(row)
        current['token_start'] = offset * TOKENS_PER_FRAME
        current['token_end'] = (offset + 1) * TOKENS_PER_FRAME
        enriched.append(current)
    ids = [row['frame_id'] for row in rows]
    if ids != list(range(start, start + len(rows))):
        raise ValueError('Feature rows differ from the requested causal frame sequence')
    return dict(frame_ids=ids, timestamps=[row['timestamp_seconds'] for row in rows],
        frames=enriched, token_count=len(rows) * TOKENS_PER_FRAME,
        tokens_per_frame=TOKENS_PER_FRAME, total_seen=session.total_seen,
        next_frame_id=start + len(rows), config_id=config_id,
        initial_state_config_id=initial_config_id, feature_archive_id=feature_archive_id,
        prefix_protocol=PREFIX_PROTOCOL, key_representation='pre_rope',
        position_convention='native_three_axis_global_frame_stride_28',
        causal_order='global_frame_id', resume_strategy='last_raw_kv_window_plus_canonical_init',
        query_independent=True)


def _wait_barrier(directory, worker_id, ready, timeout, status):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / f'{worker_id}.ready.json', ready)
    deadline = time.monotonic() + timeout
    heartbeat = 0.0
    while not (directory / 'start.json').is_file():
        now = time.monotonic()
        if now >= deadline:
            raise TimeoutError('Coordinator did not release the warmed worker barrier')
        if now >= heartbeat:
            status('waiting_for_barrier')
            heartbeat = now + 10
        time.sleep(0.25)
    release = json.loads((directory / 'start.json').read_text())
    if not isinstance(release, dict) or ('workers' in release and worker_id not in release['workers']):
        raise ValueError('Barrier release does not authorize this worker')


def _parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--archive-root', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--initial-state-root', required=True, type=Path)
    parser.add_argument('--start-frame', required=True, type=int)
    parser.add_argument('--end-frame', required=True, type=int)
    parser.add_argument('--gpu', required=True, type=int)
    parser.add_argument('--worker-id', required=True)
    parser.add_argument('--barrier-dir', type=Path)
    parser.add_argument('--barrier-timeout-seconds', type=float, default=7200)
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--wait-for-features', action='store_true')
    parser.add_argument('--feature-wait-timeout-seconds', type=float, default=86400)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--features', type=Path, default=FEATURES)
    parser.add_argument('--ledger', type=Path, default=LEDGER)
    return parser


@torch.inference_mode()
def _run(args):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.worker_id):
        raise ValueError('worker-id must contain only letters, digits, underscore or hyphen')
    if not math.isfinite(args.barrier_timeout_seconds) or args.barrier_timeout_seconds <= 0:
        raise ValueError('Barrier timeout must be positive and finite')
    if not math.isfinite(args.feature_wait_timeout_seconds) or args.feature_wait_timeout_seconds <= 0:
        raise ValueError('Feature wait timeout must be positive and finite')
    config = json.loads(args.config.read_text())
    initial_tensors, initial_metadata = load_block(args.initial_state_root, 0,
        config.get('initial_state_config_id'))
    initial = unflatten_state(initial_tensors, initial_metadata['resume_state'])
    del initial_tensors
    initial_receipt = json.loads((args.initial_state_root / 'receipts/chunk_00000000.ready.json').read_text())
    initial_binding = {key: initial_receipt[key] for key in ('config_id', 'payload_sha256', 'index_sha256')}
    halo, attention_sha = _validate_config(config, initial, initial_binding)
    _check_range(args.start_frame, args.end_frame, config['total_frames'])
    initial_config_id = initial_binding['config_id']
    writer = KVArchiveWriter(args.archive_root, config)
    logs = args.archive_root / 'logs'
    stem = f'worker_{args.worker_id}'
    status_path, report_path = logs / f'{stem}.status.json', logs / f'{stem}.report.json'
    lock = (logs / f'{stem}.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError(f'Worker {args.worker_id} is already running') from None
    session = None
    stop, telemetry = threading.Event(), None
    started = time.perf_counter()
    counters = dict(read_seconds=0.0, infer_seconds=0.0, atomic_nas_seconds=0.0,
        capture_seconds=0.0, finite_validation_seconds=0.0, serialized_bytes=0, committed_frames=0,
        warmup_frames=0, warmup_seconds=0.0, warmup_read_seconds=0.0,
        warmup_infer_seconds=0.0, resume_seconds=0.0, feature_readiness_seconds=0.0)
    cursor = args.start_frame
    commit_intervals = []

    def status(state, **extra):
        value = dict(state=state, worker_id=args.worker_id, pid=os.getpid(),
            start_frame=args.start_frame, end_frame=args.end_frame, next_frame_id=cursor,
            config_id=writer.config_id, elapsed_seconds=time.perf_counter()-started,
            **counters, **extra)
        write_json(status_path, value)
        print(json.dumps(value), flush=True)

    try:
        entries = committed_blocks(writer.root, expected_config_id=writer.config_id)
        cursor, owned = _owned_prefix(entries, args.start_frame, args.end_frame)
        if args.no_resume and owned:
            raise ValueError('--no-resume refuses already committed frames in this worker range')
        resumed = cursor - args.start_frame
        if cursor == args.end_frame:
            report = dict(status='complete', worker_id=args.worker_id, start_frame=args.start_frame,
                end_frame=args.end_frame, config_id=writer.config_id, resumed_frames=resumed,
                measured_seconds=0.0, processing_fps=None, already_complete=True,
                peak_vram_bytes=0, commit_intervals=[], **counters)
            write_json(report_path, report)
            status('complete', resumed_frames=resumed)
            return report
        warm_start = cursor if owned else _warmup_start(cursor, halo, 17, 15000)
        preflight_end = min(args.end_frame, cursor+FRAMES_PER_BLOCK) if args.wait_for_features else args.end_frame
        feature_id, readiness_seconds = _feature_range(args.features, warm_start, preflight_end, config,
            wait=args.wait_for_features, timeout=args.feature_wait_timeout_seconds, status=status)
        counters['feature_readiness_seconds'] += readiness_seconds
        status('validating_checkpoint', resumed_frames=resumed)
        identity = validate_checkpoint(args.model, args.ledger)
        torch.set_num_threads(4)
        torch.cuda.set_device(args.gpu)
        telemetry = threading.Thread(target=monitor,
            args=(stop, logs / f'{stem}.telemetry.jsonl', args.gpu), daemon=True)
        telemetry.start()
        status('loading_model')
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(args.model), local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation='sdpa',
            device_map={'': f'cuda:{args.gpu}'}, use_safetensors=True)
        model.eval().requires_grad_(False)
        processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True, use_fast=False)
        stream = StreamingInputs(model, processor)
        if stream.n_init != 17 or stream.stride != 28 or len(model.model.language_model.layers) != 28:
            raise ValueError('Loaded model/input protocol differs from the validated production geometry')
        session = ReKVSession(model, n_init=17, n_local=15000)
        # import_state binds every layer's full geometry, policy and upstream hashes.
        session.import_state(initial)
        sinks = [{kind: state[kind] for kind in ('init_key', 'init_value')} for state in initial['states']]
        sink_tensors = {f'layer_{index:02d}.{kind}': state[f'init_{kind}']
            for index, state in enumerate(initial['states']) for kind in ('key', 'value')}
        torch.cuda.reset_peak_memory_stats()
        if owned:
            tick = time.perf_counter()
            last = max(owned, key=lambda entry: entry['chunk_id'])
            tensors, metadata = load_block(writer.root, last['chunk_id'], writer.config_id)
            if (metadata.get('next_frame_id'), metadata.get('total_seen'), metadata.get('initial_state_config_id')) != (
                    cursor, 17 + cursor * TOKENS_PER_FRAME, initial_config_id):
                raise ValueError('Last committed block has incompatible continuation metadata')
            restored = reconstruct_long_state(initial, tensors, sink_tensors, 17 + cursor * TOKENS_PER_FRAME)
            session.import_state(restored)
            del tensors, restored
            counters['resume_seconds'] = time.perf_counter() - tick
        else:
            tick = time.perf_counter()
            if warm_start:
                zero_boundary(session, stream, warm_start, sinks)
            for batch_start in range(warm_start, cursor, FRAMES_PER_BLOCK):
                read_tick = time.perf_counter()
                features, _ = read_feature_frames(args.features, batch_start,
                    min(FRAMES_PER_BLOCK, cursor-batch_start))
                counters['warmup_read_seconds'] += time.perf_counter() - read_tick
                for offset, feature in enumerate(features):
                    frame = batch_start + offset
                    infer_tick = time.perf_counter()
                    observe(model, session, stream.build(feature.unsqueeze(0), frame))
                    torch.cuda.synchronize()
                    counters['warmup_infer_seconds'] += time.perf_counter() - infer_tick
                    counters['warmup_frames'] += 1
                    if (frame + 1) % 10 == 0:
                        status('warming', warmup_next_frame=frame+1)
                del features
            counters['warmup_seconds'] = time.perf_counter() - tick
        if session.total_seen != 17 + cursor * TOKENS_PER_FRAME:
            raise RuntimeError('Warmup/resume ended at the wrong global token boundary')
        ready = dict(status='ready', worker_id=args.worker_id, pid=os.getpid(),
            start_frame=args.start_frame, end_frame=args.end_frame, next_frame_id=cursor,
            config_id=writer.config_id, warmup_frames=counters['warmup_frames'],
            warmup_seconds=counters['warmup_seconds'], timestamp=time.time())
        if args.barrier_dir:
            _wait_barrier(args.barrier_dir, args.worker_id, ready, args.barrier_timeout_seconds, status)
        measured_start = time.perf_counter()
        measurement_started_time = time.time()
        while cursor < args.end_frame:
            block_start = cursor
            count = min(FRAMES_PER_BLOCK, args.end_frame-cursor)
            current_feature_id, readiness_seconds = _feature_range(args.features, cursor, cursor+count, config,
                wait=args.wait_for_features, timeout=args.feature_wait_timeout_seconds, status=status)
            counters['feature_readiness_seconds'] += readiness_seconds
            if current_feature_id != feature_id:
                raise ValueError('Visual archive identity changed while extraction was running')
            tick = time.perf_counter()
            features, rows = read_feature_frames(args.features, cursor, count)
            counters['read_seconds'] += time.perf_counter() - tick
            payload = _allocate_block(session, count)
            for offset, feature in enumerate(features):
                frame = block_start + offset
                if session.total_seen != 17 + frame * TOKENS_PER_FRAME:
                    raise RuntimeError('Observation cursor does not match its absolute frame ID')
                tick = time.perf_counter()
                inputs = stream.build(feature.unsqueeze(0), frame)
                observe(model, session, inputs)
                torch.cuda.synchronize()
                counters['infer_seconds'] += time.perf_counter() - tick
                tick = time.perf_counter()
                _append_frame(payload, session, inputs, offset)
                counters['capture_seconds'] += time.perf_counter() - tick
                if (frame+1) % 10 == 0:
                    status('observing', observed_next_frame=frame+1)
            metadata = _block_metadata(rows, session, block_start, writer.config_id, initial_config_id, feature_id)
            tick = time.perf_counter()
            _require_finite_payload(payload)
            counters['finite_validation_seconds'] += time.perf_counter() - tick
            tick = time.perf_counter()
            commit_started_time = time.time()
            receipt = writer.commit(block_start // FRAMES_PER_BLOCK, payload, metadata)
            commit_seconds = time.perf_counter() - tick
            commit_intervals.append(dict(chunk_id=block_start // FRAMES_PER_BLOCK,
                started_time=commit_started_time, finished_time=time.time(),
                serialized_bytes=receipt['serialized_bytes'], seconds=commit_seconds))
            counters['atomic_nas_seconds'] += commit_seconds
            counters['serialized_bytes'] += receipt['serialized_bytes']
            counters['committed_frames'] += count
            cursor += count
            del payload, features, rows, inputs
            status('block_committed', chunk_id=block_start // FRAMES_PER_BLOCK)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - measured_start
        report = dict(status='complete', worker_id=args.worker_id, gpu=args.gpu,
            start_frame=args.start_frame, end_frame=args.end_frame, config_id=writer.config_id,
            checkpoint=identity, initial_state_config_id=initial_config_id, feature_archive_id=feature_id,
            initial_state_binding=initial_binding,
            attention_sha256=attention_sha, halo_frames=halo, prefix_protocol=PREFIX_PROTOCOL,
            resumed_frames=resumed, measured_seconds=elapsed,
            processing_fps=counters['committed_frames']/elapsed,
            measurement_started_time=measurement_started_time, measurement_finished_time=time.time(),
            peak_vram_bytes=torch.cuda.max_memory_allocated(),
            total_elapsed_seconds=time.perf_counter()-started,
            scope='benchmark' if args.benchmark else 'production_worker_range',
            query_independent=True, whole_archive_complete=False, **counters)
        report['commit_intervals'] = commit_intervals
        write_json(report_path, report)
        status('complete', resumed_frames=resumed, measured_seconds=elapsed)
        return report
    except Exception as exc:
        status('failed', error=str(exc), error_type=type(exc).__name__)
        raise
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
