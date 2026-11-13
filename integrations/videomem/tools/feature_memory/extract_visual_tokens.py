#!/usr/bin/env python3
"""Encode videos/**/*.mp4 once into independent visual-only archives.

One persistent vision-only worker per selected GPU shares the source queue. Each
source keeps its own 1fps time axis and native image grid; clips.csv is retained
as a mapping artifact, never interpreted as encoder input. No memory method,
captioner, QA model, or new video file is constructed.
"""
from __future__ import annotations

import argparse
from collections import deque
from fractions import Fraction
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path, PurePosixPath
import queue
import shutil
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def file_state(path):
    stat = Path(path).stat()
    return {'device': stat.st_dev, 'inode': stat.st_ino, 'size': stat.st_size,
            'mtime_ns': stat.st_mtime_ns, 'ctime_ns': stat.st_ctime_ns}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_id(relative_path):
    return hashlib.sha256(relative_path.encode('utf-8')).hexdigest()[:20]


def discover_sources(root, relative_paths=None):
    root = Path(root).resolve(strict=True)
    if relative_paths:
        paths = [(root / value).resolve(strict=True) for value in relative_paths]
    else:
        paths = sorted((root / 'videos').rglob('*.mp4'))
    sources, identities = [], set()
    for path in paths:
        path = path.resolve(strict=True)
        if not path.is_file() or not path.is_relative_to(root / 'videos'):
            raise ValueError('Each source must be a video file under dataset-root/videos')
        relative = path.relative_to(root).as_posix()
        state = file_state(path)
        identity = (state['device'], state['inode'])
        if identity in identities:
            raise ValueError('Duplicate path/hardlink would encode the same source twice: ' + relative)
        identities.add(identity)
        sources.append({'source_id': source_id(relative), 'relative_path': relative,
                        'path': str(path), 'source_file_state': state,
                        'archive_relative_path': 'sources/' + source_id(relative)})
    if not sources:
        raise ValueError('No .mp4 sources found')
    if len({x['source_id'] for x in sources}) != len(sources):
        raise ValueError('Source ID collision')
    return sources


def load_source_map(path):
    """Bind logical source names to explicitly selected external read-only files.

    The map must account for every original source, including omissions. Matching
    original size/mtime/probe data is provenance evidence, not a historical hash.
    """
    path = Path(path).resolve(strict=True)
    value = json.loads(path.read_text())
    if value.get('schema') != 'cook_source_map_v1':
        raise ValueError('Expected cook_source_map_v1')
    selected, omitted = value.get('sources', []), value.get('omitted_sources', [])
    if (not selected or value.get('original_source_count') != len(selected) + len(omitted)
            or value.get('selected_source_count') != len(selected)):
        raise ValueError('Source map must explicitly account for selected and omitted original sources')
    logical_names, identities, sources = set(), set(), []
    for item in selected + omitted:
        relative = item['relative_path']
        logical = PurePosixPath(relative)
        if (logical.is_absolute() or len(logical.parts) < 3 or logical.parts[0] != 'videos'
                or '..' in logical.parts or logical.as_posix() != relative
                or logical.suffix.lower() != '.mp4'):
            raise ValueError('Invalid logical source path: ' + relative)
        if relative in logical_names:
            raise ValueError('Duplicate logical source path: ' + relative)
        logical_names.add(relative)
    for item in selected:
        actual = Path(item['path'])
        if not actual.is_absolute():
            raise ValueError('Mapped actual source path must be absolute')
        actual = actual.resolve(strict=True)
        if not actual.is_file() or actual.suffix.lower() != '.mp4':
            raise ValueError('Mapped source must be an existing MP4 file')
        state = file_state(actual)
        identity = (state['device'], state['inode'])
        if identity in identities:
            raise ValueError('Duplicate path/hardlink would encode the same mapped source twice')
        identities.add(identity)
        expected = item['original_source_file_state']
        if any(state[key] != expected[key] for key in ('size', 'mtime_ns')):
            raise ValueError('Mapped source differs from original size/mtime: ' + item['relative_path'])
        probe = item['expected_probe']
        if not all(key in probe for key in ('width', 'height', 'duration_fraction', 'expected_1fps_frames')):
            raise ValueError('Mapped source requires validated geometry and exact duration')
        relative = item['relative_path']
        provenance = {'original_path': item.get('original_path'),
                      'original_source_file_state': expected, 'expected_probe': probe,
                      'original_size_matches': True, 'original_mtime_matches': True,
                      'original_sha256_verified': False,
                      'identity_basis': 'original size, mtime and video probe match; original SHA256 unavailable; current file freshly hashed during extraction'}
        sources.append({'source_id': source_id(relative), 'relative_path': relative,
                        'path': str(actual), 'source_file_state': state,
                        'archive_relative_path': 'sources/' + source_id(relative),
                        'recovery_provenance': provenance})
    if any(not item.get('reason') for item in omitted):
        raise ValueError('Every omitted source requires an explicit reason')
    scope = {'mode': 'explicit_mapped_subset', 'original_source_count': value['original_source_count'],
             'selected_source_count': len(sources), 'omitted_source_count': len(omitted),
             'omitted_sources': omitted, 'all_original_sources_selected': not omitted,
             'source_map_path': str(path), 'source_map_sha256': file_sha256(path),
             'original_sha256_verified': False}
    return sources, scope


def validate_mapped_probe(source, inventory):
    provenance = source.get('recovery_provenance')
    if not provenance:
        return
    expected, actual = provenance['expected_probe'], inventory['sources'][0]
    if (any(actual[key] != expected[key] for key in ('width', 'height', 'expected_1fps_frames'))
            or Fraction(actual['duration_fraction']) != Fraction(expected['duration_fraction'])):
        raise ValueError('Mapped source probe changed: ' + source['relative_path'])


def model_identity(path):
    path = Path(path).resolve(strict=True)
    config = json.loads((path / 'config.json').read_text())
    if config.get('model_type') != 'qwen2_5_vl' or config['vision_config']['out_hidden_size'] != 3584:
        raise ValueError('Expected the existing Qwen2.5-VL-7B image-feature checkpoint')
    index = json.loads((path / 'model.safetensors.index.json').read_text())
    # Snapshot identities detect mutation/replacement during this job. A remote
    # content verification, when available, is separately recorded by encoder.
    return {'path': str(path), 'config_sha256': file_sha256(path / 'config.json'),
            'preprocessor_sha256': file_sha256(path / 'preprocessor_config.json'),
            'index_sha256': file_sha256(path / 'model.safetensors.index.json'),
            'weight_file_states': {name: file_state(path / name) for name in sorted(set(index['weight_map'].values()))},
            'identity_basis': 'local metadata hashes and checkpoint file identities; encoder records remote verification when available'}


def validate_completed(directory, source, model, max_frames, deep=False):
    """Validate a previously committed source; never silently reuse a partial.

    Default trusts write-time hashes only while every recorded file identity is
    unchanged. --verify-existing additionally hashes complete feature/index
    bytes, using bounded streaming reads.
    """
    directory = Path(directory)
    receipt_path = directory / 'source_receipt.json'
    if not receipt_path.is_file():
        raise ValueError('Source is incomplete; preserve it and choose a new output root: ' + str(directory))
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('status') != 'completed' or receipt['source'] != source:
        raise ValueError('Completed source identity differs: ' + str(directory))
    if file_state(source['path']) != source['source_file_state']:
        raise ValueError('Source video changed since extraction: ' + source['path'])
    if receipt['model_identity'] != model or receipt['max_frames'] != max_frames:
        raise ValueError('Completed source model/smoke scope differs')
    expected_names = set(receipt['file_states']) | {'source_receipt.json'}
    actual_names = {p.name for p in directory.iterdir() if p.is_file()}
    if actual_names != expected_names:
        raise ValueError('Completed source file set changed')
    for name, expected in receipt['file_states'].items():
        if file_state(directory / name) != expected:
            raise ValueError('Completed archive file changed: ' + name)
    complete = json.loads((directory / 'complete.json').read_text())
    header = json.loads((directory / 'archive.json').read_text())
    if (complete['total_frames'] != receipt['frames'] or complete['payload_bytes'] != receipt['payload_bytes']
            or complete['archive_id'] != header['archive_id']
            or complete['archive_header_sha256'] != file_sha256(directory / 'archive.json')):
        raise ValueError('Completed source count/header validation failed')
    if deep:
        for ready_path in sorted(directory.glob('ready_*.json')):
            ready = json.loads(ready_path.read_text())
            if file_sha256(directory / ready['feature_file']) != ready['feature_sha256']:
                raise ValueError('Completed source feature SHA256 mismatch')
            if file_sha256(directory / ready['index_file']) != ready['index_sha256']:
                raise ValueError('Completed source index SHA256 mismatch')
    return receipt


def extract_source(source, encoder, output_root, model, max_frames=None, frames_per_shard=16,
                   on_progress=None):
    from baselines.feature_memory.video import inspect_sources, iter_stream
    from baselines.feature_memory.archive import ArchiveWriter
    started = time.perf_counter()
    if file_state(source['path']) != source['source_file_state']:
        raise ValueError('Source changed after discovery')
    inventory = inspect_sources([source['path']])
    validate_mapped_probe(source, inventory)
    inventory_seconds = time.perf_counter() - started
    expected = inventory['total_1fps_frames']
    target = min(expected, max_frames) if max_frames is not None else expected
    directory = Path(output_root) / source['archive_relative_path']
    if directory.exists():
        raise FileExistsError(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    stream, writer, frames, payload = None, None, 0, 0
    decoder_stats, timings = {}, {'encode_seconds': 0.0, 'preprocess_seconds': 0.0,
                                'h2d_seconds': 0.0, 'd2h_seconds': 0.0}
    archive_write_seconds = 0.0
    try:
        stream = iter_stream(inventory, stats=decoder_stats)
        for item in stream:
            features, detail = encoder.encode(item['frame'])
            if writer is None:
                writer = ArchiveWriter(directory, {
                    'frame_shape': list(features.shape), 'source': source,
                    'inventory': inventory, 'model_identity': model,
                    'encoder_metadata': encoder.metadata,
                    'planned_frames': target, 'full_source_1fps_frames': expected,
                    'source_duration_seconds': inventory['total_duration_seconds'],
                    'source_sha256': inventory['sources'][0]['source_fingerprint'],
                    'time_basis': 'source-local video playback seconds; no concatenated day timeline',
                    'independent_image_encoding': True, 'fps': 1.0,
                    'qa_used': False, 'memory_construction': False,
                    'smoke_prefix': max_frames is not None and target < expected},
                    frames_per_shard=frames_per_shard)
            metadata = {key: value for key, value in item.items() if key != 'frame'}
            metadata.update(source_relative_path=source['relative_path'], source_video_id=source['source_id'],
                            encoder=detail, feature_origin='new_encoder_forward',
                            source_local_frame_id=item['frame_id'], source_local_timestamp=item['timestamp'])
            write_started = time.perf_counter()
            writer.append(features, metadata)
            archive_write_seconds += time.perf_counter() - write_started
            frames += 1
            payload += features.numel() * features.element_size()
            for name in timings:
                timings[name] += float(detail.get(name, 0.0))
            if on_progress and (frames == 1 or frames % frames_per_shard == 0):
                on_progress({'phase': 'encoding', 'source_id': source['source_id'],
                             'relative_path': source['relative_path'], 'frames': frames,
                             'target_frames': target, 'payload_bytes': payload,
                             'source_inventory_seconds': inventory_seconds,
                             'decode_seconds': decoder_stats.get('decode_seconds', 0.0),
                             'archive_write_seconds': archive_write_seconds,
                             'encoder_timings': dict(timings),
                             'elapsed_seconds': time.perf_counter() - started})
            del features, item
            if frames == target and target < expected:
                break
        if stream is not None:
            stream.close()
            stream = None
        if frames != target or writer is None:
            raise ValueError(f'Source ended before expected 1fps count: {frames}/{target}')
        if file_state(source['path']) != source['source_file_state']:
            raise ValueError('Source video changed during extraction')
        write_started = time.perf_counter()
        complete = writer.complete(target, num_encoder_calls=frames,
                                   full_source_encoded=frames == expected,
                                   source_total_1fps_frames=expected)
        archive_write_seconds += time.perf_counter() - write_started
        writer.close()
        writer = None
        receipt = {'status': 'completed', 'source': source, 'model_identity': model,
                   'max_frames': max_frames, 'frames': frames, 'payload_bytes': payload,
                   'expected_full_source_frames': expected, 'full_source_encoded': frames == expected,
                   'archive_id': complete['archive_id'], 'new_encoder_calls': frames,
                   'raw_frames_saved': False, 'audio_decoded': False, 'qa_used': False,
                   'memory_construction': False, 'elapsed_seconds': time.perf_counter() - started,
                   'source_inventory_seconds': inventory_seconds,
                   'decoder_stats': decoder_stats, 'timings': {**timings, 'archive_write_seconds': archive_write_seconds},
                   'file_states': {p.name: file_state(p) for p in sorted(directory.iterdir()) if p.is_file()},
                   'resume_validation': 'write-time SHA256 receipts plus unchanged source/archive file identities; optional deep rehash'}
        atomic_json(directory / 'source_receipt.json', receipt)
        return receipt
    finally:
        if stream is not None:
            stream.close()
        if writer is not None:
            writer.close()


def worker(number, requested_gpu, args, tasks, results, signature):
    from baselines.feature_memory.parallel import _device_spec, _memory_metrics
    device, mask = _device_spec(requested_gpu)
    if mask is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = mask
    import torch
    from baselines.feature_memory.encoder import Qwen25VisionEncoder
    torch.set_num_threads(args.cpu_threads)
    current = None
    try:
        started = time.perf_counter()
        encoder = Qwen25VisionEncoder(args.model, device=device, dtype='bfloat16')
        results.put({'kind': 'ready', 'worker': number, 'pid': os.getpid(),
                     'physical_device_token': mask, 'model_load_seconds': time.perf_counter() - started,
                     'encoder_metadata': encoder.metadata, **_memory_metrics(device)})
        while True:
            current = tasks.get()
            if current is None:
                break
            results.put({'kind': 'progress', 'worker': number, 'phase': 'source_inventory',
                         'source_id': current['source_id'], 'relative_path': current['relative_path'],
                         'frames': 0, 'payload_bytes': 0, 'target_frames': None, 'elapsed_seconds': 0.0,
                         'source_inventory_seconds': 0.0, 'decode_seconds': 0.0,
                         'archive_write_seconds': 0.0, 'encoder_timings': {}})
            def report(value):
                results.put({'kind': 'progress', 'worker': number, **value, **_memory_metrics(device)})
            receipt = extract_source(current, encoder, args.output, signature,
                                     max_frames=args.max_frames_per_source,
                                     frames_per_shard=args.frames_per_shard, on_progress=report)
            results.put({'kind': 'source_complete', 'worker': number,
                         'source_id': current['source_id'], 'relative_path': current['relative_path'],
                         'frames': receipt['frames'], 'payload_bytes': receipt['payload_bytes'],
                         'elapsed_seconds': receipt['elapsed_seconds'],
                         'source_inventory_seconds': receipt['source_inventory_seconds'],
                         'decode_seconds': receipt['decoder_stats'].get('decode_seconds', 0.0),
                         'archive_write_seconds': receipt['timings']['archive_write_seconds'],
                         'encoder_timings': receipt['timings'],
                         'archive_relative_path': current['archive_relative_path'],
                         'source_receipt_sha256': file_sha256(Path(args.output) / current['archive_relative_path'] / 'source_receipt.json'),
                         **_memory_metrics(device)})
        del encoder
        torch.cuda.empty_cache()
        results.put({'kind': 'worker_done', 'worker': number, 'pid': os.getpid(), **_memory_metrics(device)})
    except BaseException as exc:
        results.put({'kind': 'error', 'worker': number, 'source': current,
                     'error_type': type(exc).__name__, 'error': str(exc),
                     'traceback': traceback.format_exc()})


def validate_device_selection(devices):
    """Reject aliases of the same GPU before writing output or spawning workers."""
    from baselines.feature_memory.parallel import _device_spec
    selected = []
    for requested in devices:
        device, token = _device_spec(requested)
        if not device.startswith('cuda') or token is None:
            raise ValueError('This entrypoint requires CUDA GPU devices')
        canonical = str(int(token)) if token.isdigit() else token
        if canonical in selected:
            raise ValueError('Multiple device arguments resolve to the same CUDA GPU')
        selected.append(canonical)
    return selected


def run(args):
    validate_device_selection(args.devices)
    root = Path(args.dataset_root).resolve(strict=True)
    output = Path(args.output).resolve()
    source_map = getattr(args, 'source_map', None)
    if source_map and args.source_relative:
        raise ValueError('--source-map and --source-relative cannot be combined; omissions must be explicit in the map')
    sources, scope = load_source_map(source_map) if source_map else (discover_sources(root, args.source_relative), None)
    signature = model_identity(args.model)
    run_identity = {'schema': 'cook_full_source_visual_archive_v1', 'dataset_root': str(root),
                    'sources': sources, 'model_identity': signature,
                    'max_frames_per_source': args.max_frames_per_source,
                    'frames_per_shard': args.frames_per_shard,
                    'time_basis': 'independent source-local timestamps; clips.csv maps sources to fictional day timeline',
                    'fps': 1.0, 'dtype': 'bfloat16', 'new_memory_construction': False, 'qa_inference': False}
    if scope is not None:
        run_identity['declared_scope'] = scope
    if output.exists():
        if not args.resume:
            raise FileExistsError('Choose a new output or explicitly --resume: ' + str(output))
        existing = json.loads((output / 'sources.json').read_text())
        if existing != run_identity:
            raise ValueError('Resume source list/model/settings differ from the original extraction')
    else:
        output.mkdir(parents=True)
        atomic_json(output / 'sources.json', run_identity)
        if source_map:
            shutil.copy2(source_map, output / 'source_map.json')
        mapping = output / 'dataset_mapping'
        mapping.mkdir()
        mapped = []
        for relative in ('clips.csv', 'summary.json', 'README.md', 'timeline/manifest.json'):
            source = root / relative
            if source.is_file():
                target = mapping / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                mapped.append({'path': relative, 'sha256': file_sha256(target),
                               'use': 'mapping/audit artifact only; never sent to visual encoder'})
        atomic_json(mapping / 'manifest.json', {'files': mapped})
    pending, completed = deque(), {}
    for source in sources:
        directory = output / source['archive_relative_path']
        if directory.exists():
            receipt = validate_completed(directory, source, signature, args.max_frames_per_source,
                                         deep=args.verify_existing)
            completed[source['source_id']] = {'source_id': source['source_id'], 'relative_path': source['relative_path'],
                'frames': receipt['frames'], 'payload_bytes': receipt['payload_bytes'],
                'archive_relative_path': source['archive_relative_path'], 'reused_completed_source': True,
                'source_receipt_sha256': file_sha256(directory / 'source_receipt.json')}
        else:
            pending.append(source)
    started = time.perf_counter()
    workers, ready, finished, processes = {}, set(), set(), []
    context = mp.get_context('spawn')
    tasks, results = context.Queue(maxsize=max(2, len(args.devices)*2)), context.Queue(maxsize=64)
    stop_sent, phase = False, 'starting'
    def publish(status='running', **extra):
        active_frames = sum(value.get('frames', 0) for value in workers.values()
                            if value.get('source_id') not in completed)
        active_bytes = sum(value.get('payload_bytes', 0) for value in workers.values()
                           if value.get('source_id') not in completed)
        record = {'status': status, 'phase': phase, 'pid': os.getpid(), 'devices': args.devices,
                  'source_count': len(sources), 'sources_completed': len(completed),
                  'frames_completed_sources': sum(x['frames'] for x in completed.values()),
                  'frames_in_progress_sources': active_frames,
                  'payload_bytes_completed_sources': sum(x['payload_bytes'] for x in completed.values()),
                  'payload_bytes_in_progress_sources': active_bytes,
                  'elapsed_seconds': time.perf_counter()-started, 'workers': workers,
                  'qa_inference': False, 'memory_construction': False,
                  'raw_images_written': False, 'audio_decoded': False, **extra}
        if scope is not None:
            record['declared_scope'] = scope
        atomic_json(output / 'status.json', record)
    try:
        if pending:
            for index, gpu in enumerate(args.devices):
                process = context.Process(target=worker, args=(index,gpu,args,tasks,results,signature),
                                          name='visual-token-' + str(gpu))
                process.start()
                processes.append(process)
            phase = 'loading_vision_models'
        publish()
        last_publish = 0.0
        while len(completed) < len(sources) or len(finished) < len(processes):
            while pending:
                try:
                    tasks.put_nowait(pending[0])
                    pending.popleft()
                except queue.Full:
                    break
            if len(completed) == len(sources) and not stop_sent:
                for _ in processes:
                    tasks.put(None)
                stop_sent = True
            try:
                message = results.get(timeout=1)
            except queue.Empty:
                dead = [(i,p.exitcode) for i,p in enumerate(processes) if p.exitcode is not None and i not in finished]
                if dead:
                    raise RuntimeError('Vision worker exited without completion: ' + str(dead))
                if time.perf_counter()-last_publish > 5:
                    publish(); last_publish=time.perf_counter()
                continue
            kind, number = message['kind'], str(message['worker'])
            workers[number] = {**workers.get(number, {}), **message}
            if kind == 'error':
                raise RuntimeError(f'Worker {number}: {message["error_type"]}: {message["error"]}')
            if kind == 'ready':
                ready.add(message['worker'])
                phase = 'extracting_source_features'
            elif kind == 'source_complete':
                completed[message['source_id']] = message
                atomic_json(output / 'completed_sources.json', {'sources': list(completed.values())})
                print(json.dumps({'event': 'source_complete', 'sources_completed': len(completed),
                                  'source_count': len(sources), 'relative_path': message['relative_path'],
                                  'frames': message['frames'], 'payload_bytes': message['payload_bytes']}), flush=True)
            elif kind == 'worker_done':
                finished.add(message['worker'])
            if kind != 'progress' or time.perf_counter()-last_publish > 3:
                publish(); last_publish=time.perf_counter()
        for process in processes:
            process.join(timeout=30)
            if process.exitcode != 0:
                raise RuntimeError('Vision worker did not exit cleanly')
        phase = 'completed'
        total = {'schema': 'cook_full_source_archive_complete_v1', 'sources': [completed[x['source_id']] for x in sources],
                 'source_count': len(sources), 'total_frames': sum(x['frames'] for x in completed.values()),
                 'payload_bytes': sum(x['payload_bytes'] for x in completed.values()),
                 'max_frames_per_source': args.max_frames_per_source,
                 'all_requested_sources_complete': True, 'qa_inference': False, 'memory_construction': False,
                 'elapsed_seconds': time.perf_counter()-started,
                 'sources_index_sha256': file_sha256(output / 'sources.json')}
        if scope is not None:
            total['declared_scope'] = scope
        atomic_json(output / 'complete.json', total)
        publish('completed')
        return total
    except BaseException as exc:
        phase = 'failed'
        publish('failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill(); process.join(timeout=10)
        tasks.close(); results.close()
        tasks.cancel_join_thread(); results.cancel_join_thread()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True,
                        help='Dataset directory containing videos/**/*.mp4; no fallback source search')
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--devices', nargs='+', default=['0'],
                        help='Logical CUDA indices within inherited CUDA_VISIBLE_DEVICES, or physical indices when unset')
    parser.add_argument('--cpu-threads', type=int, default=1)
    parser.add_argument('--frames-per-shard', type=int, default=16)
    parser.add_argument('--source-relative', nargs='+', help='Optional source subset, relative to dataset root, for smoke tests')
    parser.add_argument('--source-map', help='Explicit logical-to-physical source JSON; dataset-root supplies mapping artifacts only')
    parser.add_argument('--max-frames-per-source', type=int, help='Smoke prefix only; omit for every full source')
    parser.add_argument('--resume', action='store_true', help='Reuse verified completed sources; partial sources are never overwritten')
    parser.add_argument('--verify-existing', action='store_true', help='Also rehash every existing feature/index byte on resume')
    args = parser.parse_args()
    if (args.cpu_threads < 1 or args.frames_per_shard < 1 or not args.devices
            or len(set(args.devices)) != len(args.devices)
            or (args.max_frames_per_source is not None and args.max_frames_per_source < 1)):
        parser.error('Threads, shard and smoke frame counts must be positive; devices must be unique')
    run(args)


if __name__ == '__main__':
    main()
