"""Causal archive replay, bounded snapshots, and offline evidence retrieval.

The reader never initializes an encoder. Audit logs are not reachable by the
memory method, and a resumed run skips previously committed frames rather than
updating the memory twice.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from decimal import Decimal, ROUND_FLOOR
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

MODEL_ID = 'Qwen/Qwen2.5-VL-7B-Instruct'
LEGACY_RATIOS = ('0.05', '0.10', '0.25', '0.50')


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write(path, value):
    path = Path(path)
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(data, encoding='utf-8')
    temporary.replace(path)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def inspect_archive(directory, model_path=None):
    """Validate the representation/provenance header without loading weights.

    Shard and per-frame bytes are verified by the existing iter_archive reader.
    An optional local model makes the header/config comparison independent of
    the archive's own declared metadata.
    """
    directory = Path(directory).resolve(strict=True)
    header_path = directory / 'archive.json'
    header = _json(header_path)
    if header.get('schema') != 'feature_memory_archive_v1':
        raise ValueError('Unsupported feature archive schema')
    if header.get('dtype') != 'bfloat16' or header.get('byte_order') != 'little':
        raise ValueError('Routine archive input must use little-endian bf16 features')
    if header.get('frame_shape') != [676, 3584]:
        raise ValueError('This archive CLI expects existing Qwen2.5-VL [676,3584] features')
    identity = header.get('model_identity', {})
    if identity.get('repo_id') != MODEL_ID:
        raise ValueError('Archive must identify Qwen/Qwen2.5-VL-7B-Instruct')
    if (not isinstance(identity.get('revision'), str) or len(identity['revision']) != 40
            or any(c not in '0123456789abcdef' for c in identity['revision'])):
        raise ValueError('Archive must record a pinned model revision')
    for field in ('config_sha256', 'preprocessor_sha256'):
        if not _valid_sha(identity.get(field)):
            raise ValueError('Missing or invalid archive model hash: ' + field)
    plan = header.get('plan', {})
    if (plan.get('hidden_dim') != 3584 or plan.get('tokens_per_frame') != 676
            or plan.get('dtype') != 'bfloat16'):
        raise ValueError('Archive plan differs from declared frame representation')
    inventory = header.get('inventory', {})
    if inventory.get('fps') != 1.0:
        raise ValueError('Routine archive input must declare the 1fps sampling rate')
    total = inventory.get('total_1fps_frames')
    if type(total) is not int or total <= 0:
        raise ValueError('Archive must declare the full inventory frame count')
    reference = total * 676 * 3584 * 2
    if plan.get('full_feature_bytes') != reference:
        raise ValueError('Archive full-history byte reference disagrees with frame geometry')
    model_check = False
    if model_path is not None:
        model_path = Path(model_path).resolve(strict=True)
        for field, filename in [('config_sha256', 'config.json'),
                                ('preprocessor_sha256', 'preprocessor_config.json')]:
            if identity[field] != _sha256_file(model_path / filename):
                raise ValueError('Local model and archive identity differ: ' + field)
        model_config = _json(model_path / 'config.json')
        if model_config.get('model_type') != 'qwen2_5_vl':
            raise ValueError('Local model is not Qwen2.5-VL')
        verification = model_path.parent / '.downloads' / model_path.name / 'verification.json'
        if verification.exists():
            recorded = _json(verification)
            for field in ('repo_id', 'revision'):
                if recorded.get(field) != identity[field]:
                    raise ValueError('Local verified checkpoint revision differs from archive')
        model_check = True
    return header, {'archive': str(directory), 'archive_id': header['archive_id'],
                    'archive_header_sha256': _sha256_file(header_path),
                    'model_identity': identity, 'frame_shape': [676, 3584], 'dtype': 'bfloat16',
                    'full_reference_frames': total, 'full_reference_feature_bytes': reference,
                    'local_model_identity_checked': model_check,
                    'validation': 'header geometry/model provenance; ready shard and per-frame SHA256 on read',
                    'new_encoder_calls': 0, 'audio_used': False, 'questions_used': False}


def make_config(path=None, budget_bytes=None, budget_ratio=None, full_reference_bytes=None):
    from .config import RoutineConfig
    settings = _json(path) if path else {}
    if not isinstance(settings, dict):
        raise ValueError('Configuration must be a JSON object')
    if budget_bytes is not None:
        settings['budget_bytes'] = budget_bytes
    if budget_ratio is not None:
        if full_reference_bytes is None:
            raise ValueError('Legacy ratio budgets require a full archive reference')
        settings['budget_bytes'] = int((Decimal(str(budget_ratio)) * full_reference_bytes).to_integral_value(
            rounding=ROUND_FLOOR))
    return RoutineConfig(**settings)


def _progress(memory, started, processed_this_run, update_seconds, read_seconds):
    import psutil
    stats = memory.stats()
    if stats['actual_memory_bytes'] > memory.config.budget_bytes:
        raise AssertionError('Committed routine state exceeds the fixed hard budget')
    elapsed = time.perf_counter() - started
    return {**stats, 'frames_processed_this_run': processed_this_run,
            'elapsed_seconds': elapsed, 'memory_update_seconds': update_seconds,
            'archive_read_seconds': read_seconds,
            'throughput_frames_per_second': processed_this_run / elapsed if elapsed else 0.0,
            'current_process_rss_bytes': psutil.Process().memory_info().rss,
            'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'rss_scope': 'process RSS includes input batch and temporary workspace; separate from retained budget',
            'new_encoder_calls': 0}


def _checkpoint(memory, directory, source, cutoff):
    from .checkpoint import save_checkpoint
    started = time.perf_counter()
    manifest = save_checkpoint(memory, directory)
    return {'path': str(directory), 'timestamp': cutoff,
            'total_frames_seen': memory.stats()['total_frames_seen'],
            'last_timestamp': memory.stats()['last_timestamp'],
            'serialization_seconds': time.perf_counter() - started,
            'serialized_directory_bytes': sum(p.stat().st_size for p in Path(directory).rglob('*') if p.is_file()),
            'checkpoint_manifest': manifest}


def build(args):
    import torch
    from baselines.feature_memory.archive import iter_archive
    from .checkpoint import load_checkpoint
    from .memory import RoutineAwareMemory
    torch.set_num_threads(args.cpu_threads)
    header, source = inspect_archive(args.archive, args.model)
    target = (source['full_reference_frames'] if args.full else
              (args.max_frames if args.max_frames is not None else min(300, source['full_reference_frames'])))
    if not 0 < target <= source['full_reference_frames']:
        raise ValueError('Requested frame target must be within the source inventory')
    config = make_config(args.config, args.budget_bytes, args.budget_ratio,
                         source['full_reference_feature_bytes'])
    identity = {key: source[key] for key in ('archive_id', 'archive_header_sha256', 'frame_shape', 'dtype')}
    identity['model_identity'] = {key: source['model_identity'][key] for key in (
        'repo_id', 'revision', 'config_sha256', 'preprocessor_sha256')}
    if args.resume_checkpoint:
        checkpoint_path = Path(args.resume_checkpoint)
        memory = load_checkpoint(checkpoint_path)
        if memory.control.get('runner_source') != identity:
            raise ValueError('Resume checkpoint belongs to another archive or has no charged source identity')
        if (args.config or args.budget_bytes is not None or args.budget_ratio is not None) and asdict(config) != asdict(memory.config):
            raise ValueError('Resume cannot change the saved routine configuration')
        config = memory.config
        seen = memory.stats()['total_frames_seen']
        if target <= seen:
            raise ValueError('Resume target must be greater than the checkpoint frame count')
    else:
        memory = RoutineAwareMemory(config=config)
        memory.set_source_context(identity)
        seen = 0
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    snapshots, processed, read_seconds, update_seconds = [], 0, 0.0, 0.0
    cuts = sorted(set(args.snapshot_seconds or []))
    if any(not math.isfinite(cut) or cut <= 0 for cut in cuts):
        raise ValueError('Snapshot seconds must be finite and positive')
    cuts = [cut for cut in cuts if memory.stats()['last_timestamp'] is None or cut > memory.stats()['last_timestamp']]
    _write(output / 'run_config.json', {'schema': 'routine_memory_run_v1', 'config': asdict(config),
        'source': source, 'target_frames': target, 'resume_frames': seen,
        'resume_checkpoint': str(args.resume_checkpoint) if args.resume_checkpoint else None,
        'snapshot_seconds': cuts, 'full_stream_explicitly_requested': args.full,
        'full_history_output_created': False, 'audit_logs_retrievable_by_memory': False,
        'input_mode': 'existing independent-image 1fps Qwen2.5-VL post-merger archive'})
    try:
        with (output / 'decisions.jsonl').open('w', encoding='utf-8') as decisions, (
                output / 'metrics.jsonl').open('w', encoding='utf-8') as metrics:
            for batch in iter_archive(args.archive, wait=False, max_frames=target):
                read_seconds += batch['read_seconds']
                for feature, frame in zip(batch['features'], batch['frames']):
                    fid, stamp = frame['frame_id'], float(frame['timestamp'])
                    if fid < seen:
                        continue
                    if fid != seen + processed:
                        raise ValueError('Archive cursor is not contiguous at resume/update boundary')
                    while cuts and stamp >= cuts[0]:
                        cutoff = cuts.pop(0)
                        snapshots.append(_checkpoint(memory, output / 'snapshots' / f't_{cutoff:g}', source, cutoff))
                    before = time.perf_counter()
                    decision = memory.update(feature, frame_id=fid, timestamp=stamp,
                                             metadata={'source_id': frame.get('source_id'),
                                                       'day': frame.get('day'),
                                                       'feature_sha256': frame['shared_feature_sha256']})
                    elapsed = time.perf_counter() - before
                    update_seconds += elapsed
                    processed += 1
                    decisions.write(json.dumps({'frame_id': fid, 'timestamp': stamp, 'update_seconds': elapsed,
                                                'decision': decision}, sort_keys=True, allow_nan=False) + '\n')
                    report = _progress(memory, started, processed, update_seconds, read_seconds)
                    metrics.write(json.dumps(report, sort_keys=True, allow_nan=False) + '\n')
                    if processed == 1 or processed % args.progress_frames == 0:
                        decisions.flush()
                        metrics.flush()
                        _write(output / 'status.json', {'status': 'running', **report})
                        print(json.dumps({'event': 'routine_progress', **report}, sort_keys=True), flush=True)
            if memory.stats()['total_frames_seen'] != target:
                raise ValueError(f'Archive prefix unavailable: {memory.stats()["total_frames_seen"]}/{target}')
        last_timestamp = memory.stats()['last_timestamp']
        final_cutoff = min(last_timestamp + 1.0, header['inventory'].get('total_duration_seconds', last_timestamp + 1.0))
        for cutoff in cuts:
            if cutoff <= final_cutoff:
                snapshots.append(_checkpoint(memory, output / 'snapshots' / f't_{cutoff:g}', source, cutoff))
        final_checkpoint = output / 'checkpoint_final'
        snapshots.append(_checkpoint(memory, final_checkpoint, source, final_cutoff))
        summary = {'status': 'completed', 'schema': 'routine_memory_summary_v1',
                   **_progress(memory, started, processed, update_seconds, read_seconds),
                   'source': source, 'config': asdict(config), 'resume_frames': seen,
                   'snapshots': snapshots, 'final_checkpoint': str(final_checkpoint),
                   'full_history_output_created': False, 'qa_inference': False,
                   'archive_frames_reused': processed,
                   'checkpoint_artifacts_counted_separately_from_live_memory': True}
        _write(output / 'summary.json', summary)
        _write(output / 'status.json', summary)
        return summary
    except BaseException as exc:
        _write(output / 'status.json', {'status': 'failed', 'error_type': type(exc).__name__,
                                       'error': str(exc), 'total_frames_seen': memory.stats()['total_frames_seen'],
                                       'frames_processed_this_run': processed})
        raise


def query(args):
    import torch
    from safetensors.torch import load_file, save_file
    from .checkpoint import RoutineSnapshotReader
    if args.features:
        tensors = load_file(str(args.features), device='cpu')
        features = tensors[args.feature_key]
        provenance = {'features': str(args.features), 'key': args.feature_key,
                      'file_sha256': _sha256_file(args.features)}
    else:
        from baselines.feature_memory.archive import iter_archive
        _, provenance = inspect_archive(args.archive)
        features = None
        for batch in iter_archive(args.archive, wait=False, max_frames=args.frame_id + 1):
            for tensor, frame in zip(batch['features'], batch['frames']):
                if frame['frame_id'] == args.frame_id:
                    features = tensor.clone()
        if features is None:
            raise ValueError('Query frame is not available in the committed archive')
        provenance = {**provenance, 'frame_id': args.frame_id}
    if features.ndim != 2 or not features.is_floating_point() or not torch.isfinite(features).all():
        raise ValueError('Query features must be a finite floating [tokens,hidden] tensor')
    reader = RoutineSnapshotReader(args.checkpoint)
    matches = reader.query(features, top_k=args.top_k, as_of=args.as_of)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    selected, rows = {}, []
    for index, item in enumerate(matches):
        row = {}
        for key, value in item.items():
            if isinstance(value, torch.Tensor):
                name = f'match_{index}.{key}'
                selected[name] = value.detach().cpu().contiguous().clone()
                row[key] = {'tensor_key': name, 'shape': list(value.shape), 'dtype': str(value.dtype)}
            else:
                row[key] = value
        rows.append(row)
    if selected:
        save_file(selected, str(output / 'retrieved_evidence.safetensors'))
    result = {'checkpoint': str(args.checkpoint), 'query_source': provenance,
              'as_of': args.as_of, 'top_k': args.top_k, 'matches': rows,
              'saved_memory_stats': reader.stats(), 'qa_inference': False,
              'retrieval_scope': 'selected visual evidence; no answer generation or raw-frame recovery'}
    _write(output / 'retrieval.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('build', 'synthetic'):
        cmd = commands.add_parser(name)
        cmd.add_argument('--output', type=Path, required=True)
        cmd.add_argument('--config', type=Path)
        budgets = cmd.add_mutually_exclusive_group()
        budgets.add_argument('--budget-bytes', type=int)
        if name == 'build':
            budgets.add_argument('--budget-ratio', choices=LEGACY_RATIOS)
        cmd.add_argument('--cpu-threads', type=int, default=1)
    cmd = commands.choices['build']
    cmd.add_argument('--archive', type=Path, required=True)
    cmd.add_argument('--model', type=Path, help='Optional local config/processor identity cross-check; no model loading')
    scope = cmd.add_mutually_exclusive_group()
    scope.add_argument('--max-frames', type=int, help='Total prefix target, including frames already in a resumed checkpoint; default300')
    scope.add_argument('--full', action='store_true', help='Explicitly process the entire archive instead of the default300-frame smoke')
    cmd.add_argument('--resume-checkpoint', type=Path)
    cmd.add_argument('--snapshot-seconds', type=float, nargs='+')
    cmd.add_argument('--progress-frames', type=int, default=30)
    cmd = commands.add_parser('query')
    cmd.add_argument('--checkpoint', type=Path, required=True)
    inputs = cmd.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--features', type=Path, help='Safe safetensors file containing a [tokens,hidden] query')
    inputs.add_argument('--archive', type=Path)
    cmd.add_argument('--feature-key', default='features')
    cmd.add_argument('--frame-id', type=int, default=0)
    cmd.add_argument('--top-k', type=int, default=5)
    cmd.add_argument('--as-of', type=float)
    cmd.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command in ('build', 'synthetic'):
        if args.cpu_threads < 1 or (args.budget_bytes is not None and args.budget_bytes < 1):
            parser.error('CPU threads and budget bytes must be positive')
    if args.command == 'build':
        if (args.max_frames is not None and args.max_frames < 1) or args.progress_frames < 1:
            parser.error('Frame counts must be positive')
        return build(args)
    if args.command == 'synthetic':
        from .synthetic import run_synthetic
        return run_synthetic(args)
    if args.frame_id < 0 or args.top_k < 1:
        parser.error('frame-id must be nonnegative and top-k positive')
    return query(args)


if __name__ == '__main__':
    main()
