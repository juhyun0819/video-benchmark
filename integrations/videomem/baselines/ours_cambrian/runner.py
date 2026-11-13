"""One causal stream, independent QA branches, and strict checkpoint resumption.

The orchestration is dependency-light so it can be tested without loading a VLM.
Native model imports live in the CLI, never in the test implementation.
"""
import bisect
import hashlib
import json
import math
import time
from pathlib import Path

from baselines.ours_offline.inputs import sha256, load_inputs as load_canonical_inputs, validate_sources
from .stream_frames import decode_window, sample_plan
from baselines.streambench.scoring import score_answer


def state_digest(value):
    """Hash tensor values and continuation metadata, including physical device tags."""
    h = hashlib.sha256()
    def visit(item):
        if hasattr(item, 'detach') and hasattr(item, 'numel'):
            h.update(str((tuple(item.shape), item.dtype)).encode())
            h.update(item.detach().contiguous().reshape(-1).view(__import__('torch').uint8).cpu().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                h.update(repr(key).encode()); visit(item[key])
        elif isinstance(item, (list, tuple)):
            h.update(type(item).__name__.encode())
            for entry in item: visit(entry)
        else:
            h.update(repr(item).encode())
    visit(value)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def load_inputs(root, manifest_path, sources_path, days=None):
    """Keep the complete source prefix, even when QA selects a later subset."""
    if manifest_path is None:
        questions = []
        sources = validate_sources(root, json.loads(Path(sources_path).read_text())['sources'])
    else:
        _, sources, questions = load_canonical_inputs(root, manifest_path, sources_path, days)
    scorer = Path(__file__).parents[1] / 'streambench/scoring.py'
    identity = {'manifest_sha256': None if manifest_path is None else sha256(manifest_path),
                'sources_sha256': sha256(sources_path), 'scorer_sha256': sha256(scorer),
                'selected_days': None if days is None else sorted(set(days))}
    return questions, sources, identity


def run_stream(method, *, root, questions, sources, identity, output, fps=1.0,
               chunk_frames=8, checkpoint_every=10, max_questions=None, resume=None,
               decoder=decode_window, synchronize=lambda: None, memory_stats=lambda: {},
               clock=time.perf_counter):
    """Run a complete chronological prefix, preserving all state at every QA.

    ``decoder``/``method`` injection exists for CPU orchestration tests; the CLI
    always selects the real native method and raw decoder.
    """
    if not questions:
        raise ValueError('QA is required for a causal run; use plan-only preflight while QA is pending')
    plan = sample_plan(sources, fps=fps)
    if type(chunk_frames) is not int or chunk_frames < 1:
        raise ValueError('chunk_frames must be a positive integer')
    if type(checkpoint_every) is not int or checkpoint_every < 1 or (max_questions is not None and (type(max_questions) is not int or max_questions < 1)):
        raise ValueError('checkpoint_every and max_questions must be positive')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    identity = dict(identity, fps=fps, sample_policy='source-local grid quantized to original CFR frames; overlapping tail owned by next clip')
    times = [r['requested_global_timestamp'] for r in plan]
    if not plan:
        raise ValueError('No frames in source timeline')
    if questions != sorted(questions, key=lambda q: (q['query_time'], str(q['qa_id']))):
        raise ValueError('Questions must be chronological')
    if len({q['qa_id'] for q in questions}) != len(questions):
        raise ValueError('Duplicate question IDs')
    status = {'status': 'initializing', 'completed_qa': 0, 'expected_qa': len(questions),
              'stream_frames': 0, 'errors': 0, 'method': 'Cambrian-S-7B-LFP', 'day_boundary_reset': False}
    predictions = output / 'predictions.jsonl'
    status_path = output / 'status.json'
    cursor, timestamps, completed, rows = 0, [], [], []
    update_seconds = decode_seconds = 0.0
    started = clock()
    active_question = None
    def publish():
        status.update(completed_qa=len(completed), stream_frames=cursor,
                      update_seconds=update_seconds, decode_seconds=decode_seconds,
                      elapsed_seconds=clock() - started)
        atomic_json(status_path, status)
    publish()
    try:
        method.initialize()
        if resume is not None:
            resume = Path(resume)
            method.load_checkpoint(resume)
            protocol = method.protocol_state
            if protocol.get('input_identity') != identity:
                raise ValueError('Checkpoint manifest/source/config/fps identity differs')
            cursor, timestamps, completed = protocol['cursor'], list(protocol['actual_timestamps']), list(protocol['completed_qa_ids'])
            if not 0 <= cursor <= len(plan) or len(timestamps) != cursor:
                raise ValueError('Checkpoint cursor and timestamp history disagree')
            if completed != [q['qa_id'] for q in questions[:len(completed)]]:
                raise ValueError('Checkpoint QA IDs are not the exact chronological prefix')
            if len(timestamps) > 1 and any(a >= b for a, b in zip(timestamps, timestamps[1:])):
                raise ValueError('Checkpoint timestamps are not strictly increasing')
            for actual, requested in zip(timestamps, times):
                if not math.isfinite(actual) or actual > requested + 1e-8:
                    raise ValueError('Checkpoint contains a future observation')
            if getattr(getattr(method, 'memory', None), 'frames_seen', cursor) != cursor:
                raise ValueError('Native frame counter differs from checkpoint cursor')
            last = getattr(method, 'last_timestamp', None)
            if cursor and last is not None and (last['global_frame_id'] != cursor - 1 or last['global_timestamp'] != timestamps[-1]):
                raise ValueError('Native last timestamp differs from checkpoint cursor')
            all_rows = [json.loads(line) for line in (resume.parent / 'predictions.jsonl').read_text().splitlines() if line.strip()]
            rows = [r for r in all_rows if r['qa_id'] in completed and r['status'] == 'ok']
            if [r['qa_id'] for r in rows] != completed:
                raise ValueError('Checkpoint predictions do not match completed QA prefix')
            for row, question in zip(rows, questions):
                cutoff = question['query_time']
                expected_count = bisect.bisect_right(times, cutoff + 1e-8)
                expected_gt = question.get('answer', question.get('correct_answer'))
                expected_score, expected_kind = score_answer(row['prediction'], question)
                if (row['query_time'] != cutoff or row['actual_frame_timestamps'] != timestamps[:expected_count]
                        or row['GT'] != expected_gt or row['correct'] != expected_score
                        or row.get('score_kind') != expected_kind):
                    raise ValueError('Checkpoint prediction row differs from its canonical cutoff, observed prefix or score')
            update_seconds, decode_seconds = protocol['update_seconds'], protocol['decode_seconds']
            atomic_json(output / 'resume-audit.json', {'status': 'passed', 'checkpoint': str(resume),
                        'cursor': cursor, 'completed_qa_ids': completed, 'identity': identity})
        predictions.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        status['status'] = 'streaming'; publish()

        def advance(end):
            nonlocal cursor, update_seconds, decode_seconds
            while cursor < end:
                stop = min(end, cursor + chunk_frames)
                tick = clock(); frames, metadata = decoder(root, plan[cursor:stop]); dt = clock() - tick
                if len(frames) != stop - cursor or len(metadata) != len(frames):
                    raise ValueError('Decoded frame count mismatch')
                actual = [r['global_timestamp'] for r in metadata]
                if any(not math.isfinite(a) or a > req + 1e-8 for a, req in zip(actual, times[cursor:stop])):
                    raise ValueError('Decoder returned a future frame')
                if any(a >= b for a, b in zip(actual, actual[1:])) or (timestamps and actual[0] <= timestamps[-1]):
                    raise ValueError('Decoded frames are not strictly chronological')
                if [r['global_frame_id'] for r in metadata] != list(range(cursor, stop)):
                    raise ValueError('Decoded frame IDs changed source ordering')
                synchronize(); tick = clock()
                method.observe(frames, metadata); method.finalize_chunk()
                synchronize(); ut = clock() - tick
                memory = method.enforce_budget()
                timestamps.extend(actual); update_seconds += ut; decode_seconds += dt
                event = {'start_frame': cursor, 'end_frame_exclusive': stop, 'actual_frame_timestamps': actual,
                         'decode_seconds': dt, 'update_seconds': ut, 'persistent_memory_bytes': memory['persistent_total_bytes'],
                         **memory_stats()}
                if hasattr(method, 'drain_events'):event['memory_events'] = method.drain_events()
                with (output / 'updates.jsonl').open('a') as stream:stream.write(json.dumps(event) + '\n')
                cursor = stop; status.update(status='streaming', **memory_stats()); publish()

        def checkpoint(name):
            method.protocol_state = {'input_identity': identity, 'cursor': cursor,
                'actual_timestamps': list(timestamps), 'completed_qa_ids': list(completed),
                'update_seconds': update_seconds, 'decode_seconds': decode_seconds, 'day_boundary_reset': False}
            path = output / name
            method.save_checkpoint(path)
            status['checkpoint'] = str(path); publish()

        processed = 0
        for question in questions:
            if question['qa_id'] in completed:continue
            active_question = question
            advance(bisect.bisect_right(times, question['query_time'] + 1e-8))
            if not timestamps or timestamps[-1] > question['query_time'] + 1e-8:
                raise ValueError('QA has no causal observed frames')
            synchronize(); query_started = clock()
            saved = method.snapshot()
            before = state_digest(saved['method'])
            synchronize(); tick = clock()
            try:
                # Ground truth, target time, rationale and diagnostic flags are not passed.
                response = method.answer(question['question'], question['choices'])
                synchronize(); native_latency = clock() - tick
                native_unchanged = before == state_digest(method.state_dict())
            finally:
                method.restore(saved)
                if before != state_digest(method.state_dict()):
                    raise RuntimeError('QA branch could not restore all continuation state')
            synchronize(); latency = clock() - query_started
            answer = response['answer']
            correct, kind = score_answer(answer, question)
            row = {'method': 'Cambrian-S-7B-LFP', 'qa_id': question['qa_id'],
                'query_time': question['query_time'], 'original_query_time': question.get('original_query_time'),
                'actual_frame_timestamps': list(timestamps), 'prediction': answer, 'raw_output': answer,
                'GT': question.get('answer', question.get('correct_answer')), 'correct': correct, 'score_kind': kind,
                'answer_latency': latency, 'native_answer_latency': native_latency,
                'answer_latency_scope': 'snapshot, native answer, state audits and restoration',
                'stream_update_time': update_seconds, 'stream_decode_time': decode_seconds,
                'status': 'ok', 'error': None, 'qa_state_restored': True,
                'native_answer_state_unchanged': native_unchanged, 'retrieval_metadata': {k:v for k,v in response.items() if k != 'answer'},
                'need_audio': question.get('need_audio'), 'need_name': question.get('need_name'), **memory_stats()}
            with predictions.open('a') as stream:stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            rows.append(row); completed.append(question['qa_id']); processed += 1
            active_question = None
            status.update(last_qa_id=question['qa_id'], last_answer_latency=latency); publish()
            if max_questions is not None and processed >= max_questions:break
            if len(completed) % checkpoint_every == 0:checkpoint(f'checkpoint_qa{len(completed):03d}')
        if len(completed) == len(questions):advance(len(plan))
        checkpoint('checkpoint')
        status.update(status='complete' if len(completed) == len(questions) else 'partial_checkpointed',
                      correct=sum(r['correct'] is True for r in rows), **memory_stats())
        publish()
        return status
    except BaseException as error:
        if active_question is not None and active_question['qa_id'] not in completed:
            row = {'method': 'Cambrian-S-7B-LFP', 'qa_id': active_question['qa_id'], 'query_time': active_question['query_time'],
                   'actual_frame_timestamps': list(timestamps), 'prediction': '', 'raw_output': None,
                   'GT': active_question.get('answer'), 'correct': False, 'answer_latency': None,
                   'stream_update_time': update_seconds, 'stream_decode_time': decode_seconds,
                   'status': 'error', 'error': f'{type(error).__name__}: {error}'}
            with predictions.open('a') as stream:stream.write(json.dumps(row) + '\n')
        status.update(status='failed', errors=1, error=f'{type(error).__name__}: {error}')
        publish()
        raise
