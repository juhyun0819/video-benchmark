"""Question-free observation and independent QA from cumulative day states."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import resource
import time

from .checkpoint import read_checkpoint, write_checkpoint, write_json
from .method import UnsupportedBudget
from .scoring import score_answer


def capture_rng():
    import numpy as np
    import torch
    state = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if 'cuda' in state:
        if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Checkpoint CUDA RNG topology differs from this runtime')
        torch.cuda.set_rng_state_all(state['cuda'])


def _identity(method, config):
    return {'adapter': type(method).__module__ + '.' + type(method).__qualname__,
            'method_config_sha256': hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest(),
            'memory_budget_bytes': method.memory_budget_bytes}


def _log(path, record):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')


def _resources():
    import torch
    result = {'peak_cpu_ram_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              'peak_gpu_allocated_bytes': 0, 'peak_gpu_reserved_bytes': 0}
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        result.update(peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
    return result


def _synchronize():
    import torch
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


def observe_stream(method, chunks, output_dir, *, stream_id, manifest_digest, method_config,
                   resume_checkpoint=None, stop_after_day=None):
    """Consume bounded `(frames, frame_records)` chunks, each within one day.

    The caller must skip already checkpointed records on resume. This runner
    validates the next frame ID/time/day and rejects replay. Observations never
    receive questions. Audit files and archived checkpoints are not retrievable
    by an adapter, hence are not counted as its memory.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f'Use a new output directory: {output}')
    identity = _identity(method, method_config)
    method_name = method_config.get('method', type(method).__name__)
    days, frames_seen, last_id, last_time = [], 0, -1, None
    if resume_checkpoint:
        checkpoint = read_checkpoint(resume_checkpoint)
        if checkpoint.get('identity') != identity:
            raise ValueError('Checkpoint method/configuration/budget identity mismatch')
        meta = checkpoint['metadata']
        if meta['stream_id'] != stream_id or meta['manifest_sha256'] != manifest_digest:
            raise ValueError('Checkpoint stream/manifest identity mismatch')
        method.initialize()
        method.restore(checkpoint['snapshot'])
        restore_rng(checkpoint['snapshot']['protocol']['rng'])
        days = list(meta['days_observed'])
        frames_seen, last_id, last_time = meta['frames_seen_total'], meta['last_global_frame_id'], meta['last_global_timestamp']
    else:
        method.reset()  # An independent experiment, never a day transition.
    started = time.monotonic()
    active_day = None
    initial = minimum = maximum = 0
    sizes, size_sum = 0, 0
    checkpoints = []

    def protocol():
        method.protocol_state = {'stream_id': stream_id, 'days_observed': list(days),
                                 'frames_seen_total': frames_seen, 'last_global_frame_id': last_id,
                                 'last_global_timestamp': last_time, 'manifest_sha256': manifest_digest,
                                 'rng': capture_rng()}

    def boundary(day):
        nonlocal maximum
        days.append(day)
        protocol()
        measured = method.enforce_budget()
        maximum = max(maximum, measured['persistent_total_bytes'])
        name = 'checkpoint_D1' if day == 1 else f'checkpoint_D1_D{day}'
        meta = {'method': method_name, 'stream_id': stream_id, 'days_observed': list(days),
                'frames_seen_total': frames_seen, 'last_global_frame_id': last_id,
                'last_global_timestamp': last_time, 'memory_budget_bytes': method.memory_budget_bytes,
                'persistent_memory_bytes': measured['persistent_total_bytes'], 'memory_components': measured,
                'manifest_sha256': manifest_digest}
        payload = {'schema_version': 1, 'identity': identity, 'metadata': meta, 'snapshot': method.snapshot()}
        file_size = write_checkpoint(output/name, payload)
        record = dict(meta, checkpoint=name, checkpoint_file_size=file_size,
                      compression_count=getattr(method, 'compression_count', 0),
                      eviction_count=getattr(method, 'eviction_count', 0))
        boundary_events = getattr(method, 'drain_events', lambda: [])()
        if boundary_events:
            record['method_events'] = boundary_events
        _log(output/'days.jsonl', record)
        checkpoints.append(record)

    try:
        protocol()
        initial = method.enforce_budget()['persistent_total_bytes']
        minimum = max(initial, method.minimum_memory_bytes or 0)
        maximum = initial
        for frames, timestamps in chunks:
            if not frames or len(frames) != len(timestamps):
                raise ValueError('Each chunk needs matching nonempty frames/timestamps')
            day = timestamps[0].get('day')
            if type(day) is not int or day < 1 or any(row.get('day') != day for row in timestamps):
                raise ValueError('Chunks must contain one positive integer day')
            if active_day != day:
                if active_day is not None:
                    boundary(active_day)
                    if stop_after_day is not None and active_day >= stop_after_day:
                        active_day = None
                        break
                expected_day = (days[-1] + 1) if days else 1
                if day != expected_day:
                    raise ValueError(f'Expected day {expected_day}, got {day}; no resets/replays/gaps allowed')
                active_day = day
            previous_id, previous_time = last_id, last_time
            public_fields = {'stream_id', 'day', 'global_frame_id', 'source_timestamp',
                             'global_timestamp', 'source_id', 'source_frame_index'}
            for record in timestamps:
                if not isinstance(record, dict) or set(record) != public_fields:
                    raise ValueError('Observation accepts only the public frame timestamp fields')
                fid, timestamp = record.get('global_frame_id'), record.get('global_timestamp')
                if record.get('stream_id') != stream_id or type(fid) is not int or fid != previous_id + 1:
                    raise ValueError('Frame manifest must have consecutive IDs and one stream identity')
                if (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
                        or not math.isfinite(timestamp) or timestamp < 0
                        or (previous_time is not None and timestamp <= previous_time)):
                    raise ValueError('Frame timestamps must be finite and strictly chronological')
                previous_id, previous_time = fid, timestamp
            _synchronize()
            tic = time.monotonic()
            method.observe(frames, timestamps)
            method.finalize_chunk()
            _synchronize()
            last_id, last_time = previous_id, previous_time
            frames_seen += len(timestamps)
            protocol()
            measured = method.enforce_budget()
            minimum = max(minimum, method.minimum_memory_bytes or 0)
            current = measured['persistent_total_bytes']
            maximum, size_sum, sizes = max(maximum, current), size_sum + current, sizes + 1
            event = {'method': method_name, 'stream_id': stream_id, 'day': day, 'timestamp': last_time,
                     'frames_seen_total': frames_seen, 'persistent_memory_bytes': current, 'memory_components': measured,
                     'processing_latency_seconds': time.monotonic()-tic,
                     'compression_count': getattr(method, 'compression_count', 0),
                     'eviction_count': getattr(method, 'eviction_count', 0), **_resources()}
            details = getattr(method, 'drain_events', lambda: [])()
            if details:
                event['method_events'] = details
            _log(output/'chunks.jsonl', event)
            _log(output/'observed_frames.jsonl', {'frame_ids': [r['global_frame_id'] for r in timestamps],
                                                  'timestamps': [r['global_timestamp'] for r in timestamps], 'day': day})
        if active_day is not None:
            boundary(active_day)
        if not days:
            raise ValueError('Cannot finish an empty stream')
        report = {'status': 'complete', 'method': method_name, 'days_observed': days,
                  'frames_seen_total': frames_seen, 'memory_budget_bytes': method.memory_budget_bytes,
                  'minimum_memory_bytes': minimum, 'maximum_persistent_memory_bytes': maximum,
                  'average_persistent_memory_bytes': size_sum/sizes if sizes else initial,
                  'chunk_count': sizes, 'day_checkpoints': checkpoints,
                  'wall_seconds': time.monotonic()-started, **_resources()}
        write_json(output/'summary.json', report)
        return report
    except UnsupportedBudget as exc:
        write_json(output/'summary.json', dict(exc.as_dict(), method=method_name, frames_seen_total=frames_seen,
                                              days_observed=days, maximum_persistent_memory_bytes=maximum))
        raise
    except Exception as exc:
        write_json(output/'summary.json', {'status': 'failed', 'error': str(exc), 'days_observed': days,
                                           'frames_seen_total': frames_seen})
        raise
    finally:
        close = getattr(chunks, 'close', None)
        if close:
            close()


def evaluate_checkpoint(method, checkpoint_path, questions, *, output=None, method_config=None):
    """Read labels only after answer(). Each question starts from the same memory/RNG."""
    checkpoint = read_checkpoint(checkpoint_path)
    identity = checkpoint['identity']
    if identity['adapter'] != type(method).__module__ + '.' + type(method).__qualname__ or identity['memory_budget_bytes'] != method.memory_budget_bytes:
        raise ValueError('QA adapter/budget differs from checkpoint')
    if method_config is not None and identity != _identity(method, method_config):
        raise ValueError('QA method configuration differs from checkpoint')
    original, original_rng = method.snapshot(), capture_rng()
    rows, seen = [], set()
    meta = checkpoint['metadata']
    try:
        for q in questions:
            qid = q.get('question_id')
            if not isinstance(qid, str) or not qid or qid in seen:
                raise ValueError('Each question needs a distinct nonempty question_id')
            seen.add(qid)
            if not isinstance(q.get('question'), str) or not q['question'].strip():
                raise ValueError('Question must be nonempty text')
            source_day = q.get('source_day')
            if type(source_day) is not int or source_day not in meta['days_observed']:
                raise ValueError('source_day must be a day observed by this checkpoint')
            choices = q.get('choices', [])
            if not isinstance(choices, list) or any(not isinstance(c, str) for c in choices):
                raise ValueError('choices must be a list of strings')
            method.restore(checkpoint['snapshot'])
            restore_rng(checkpoint['snapshot']['protocol']['rng'])
            before = method.enforce_budget()
            tic = time.monotonic()
            response = method.answer(q['question'], choices=choices)
            if isinstance(response, str):
                response = {'answer': response}
            if not isinstance(response, dict) or not isinstance(response.get('answer'), str):
                raise ValueError('Adapter answer must include answer text')
            _synchronize()
            answer = response['answer']
            correct, scoring = score_answer(answer, q)
            row = {'question_id': qid, 'source_day': source_day,
                   'evaluation_checkpoint': Path(checkpoint_path).name,
                   'days_after_event': max(meta['days_observed'])-source_day,
                   'answer': answer, 'status': response.get('status', 'complete'), 'correct': correct, 'scoring': scoring,
                   'memory_budget_bytes': method.memory_budget_bytes,
                   'persistent_memory_bytes': before['persistent_total_bytes'],
                   'query_read_bytes': response.get('query_read_bytes'),
                   'num_retrieved_items': response.get('num_retrieved_items'),
                   'num_visual_tokens_used': response.get('num_visual_tokens_used'),
                   'num_kv_positions_used': response.get('num_kv_positions_used'),
                   'num_text_items_used': response.get('num_text_items_used'),
                   'retrieved_items': response.get('retrieved_items', []),
                   'retrieved_frame_ids': response.get('retrieved_frame_ids'),
                   'retrieved_attention_bytes': response.get('retrieved_attention_bytes'),
                   'query_read_bytes_is_upper_bound': response.get('query_read_bytes_is_upper_bound', False),
                   'read_accounting': response.get('read_accounting'),
                   'qa_temporal_padding_frames': response.get('qa_temporal_padding_frames', 0),
                   'retrieval_scan_key_bytes': response.get('retrieval_scan_key_bytes'),
                   'query_selected_kv_bytes': response.get('query_selected_kv_bytes'),
                   'query_representative_scan_bytes': response.get('query_representative_scan_bytes'),
                   'qa_latency_seconds': time.monotonic()-tic, **_resources()}
            rows.append(row)
            if output:
                _log(output, row)
            method.restore(checkpoint['snapshot'])
    finally:
        method.restore(original)
        restore_rng(original_rng)
    return rows
