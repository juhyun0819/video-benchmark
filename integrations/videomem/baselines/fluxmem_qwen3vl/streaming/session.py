"""Causal query/arrival scheduling, independent of model and device."""
from copy import deepcopy
import math
import time

from local_videomem.data import public_question
from streaming.video import frame_key, time_key

PROMPT_CONTRACT = 'streaming_observed_video_v1'


def question_prompt(question):
    q = public_question(question)
    text = 'Use only the observed video. Question: ' + q['question']
    if q.get('choices'):
        text += '\n' + '\n'.join(f'{chr(65+i)}. {value}' for i, value in enumerate(q['choices']))
        text += '\nAnswer with the option letter only.'
    else:
        text += '\nAnswer in at most three sentences; do not invent unseen details.'
    return text


def run_session(*, pairs, questions, bank, encode, answer, emit, fps=1,
                temporal_tokens_per_second=2, proactive=True, time_id_mode='scaled'):
    """Stream once; answer from only complete temporal groups before each cutoff.

    A pair whose last timestamp equals a query cutoff is excluded (half-open).
    Arrival-associated queries precede cascading compression. Between arrivals,
    queries use the previously completed bank. CPU lookahead is never encoded
    until all earlier queries have been answered. Explicit queries take priority
    over a proactive trigger on the same arrival. The most recent successfully
    answered explicit question remains the proactive task; this lifecycle is a
    local policy, not a released upstream API.
    """
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be positive and finite')
    if time_id_mode not in ('scaled', 'ordinal'):
        raise ValueError('Invalid bank time ID policy')
    qs = [public_question(q) for q in questions]
    if len({q['question_id'] for q in qs}) != len(qs):
        raise ValueError('Duplicate question IDs')
    qs.sort(key=time_key)
    if not qs:
        raise ValueError('At least one explicit question is required')
    cursor = 0
    active = None
    answers = []
    encoded = 0
    previous_pair = -1
    previous_time_id = -1
    previous_available = None

    def respond(q, kind, observed_time, phase):
        nonlocal active
        started = time.monotonic()
        snapshot = bank.snapshot()
        # The source/frame metadata is audit-only and never included in prompt.
        cutoff = time_key(q) if kind == 'explicit' else None
        for record in snapshot['records']:
            for frame in record['metadata'].get('frame_metadata', []):
                if cutoff is not None and frame_key(frame) >= cutoff:
                    raise RuntimeError('Future frame reached explicit-query memory')
        prompt = question_prompt(q)
        row = {'kind': kind, 'question': deepcopy(q), 'observed_time': observed_time,
               'phase': phase, 'encoder_calls': encoded, 'prompt': prompt,
               'memory_records': snapshot['records'], 'memory_stats': snapshot['stats']}
        if snapshot['stats']['total_tokens'] == 0:
            row.update(status='insufficient_observation', prediction=None)
        else:
            result = answer(snapshot, prompt)
            row.update(result)
            row['status'] = ('complete' if result.get('stopped_on_eos') and
                             result.get('prediction', '').strip() else 'incomplete_generation')
            if row['status'] == 'complete':
                bank.mark_response()
                if kind == 'explicit':
                    active = deepcopy(q)
        row['wall_seconds'] = time.monotonic() - started
        answers.append(row)
        emit({'event': 'answer', **row})

    try:
        for pair in pairs:
            if pair['pair_index'] <= previous_pair:
                raise ValueError('Pair indices must increase')
            available = time_key(pair['available_time'])
            frontier = time_key(pair['frontier_time'])
            if previous_available is not None and available <= previous_available:
                raise ValueError('Pair availability must increase')
            if frontier <= available:
                raise ValueError('Pair frontier must follow availability')
            while cursor < len(qs) and time_key(qs[cursor]) <= available:
                respond(qs[cursor], 'explicit', None, 'between_arrivals')
                cursor += 1
            if cursor == len(qs):
                break  # Never encode after the last explicit observation horizon.
            tic = time.monotonic()
            projection = encode(pair['frames'])
            encoded += 1
            # Original Day/file timestamps remain separate from native compact
            # video coordinates. Reject an FPS that collapses temporal IDs.
            time_id = (pair['pair_index'] if time_id_mode == 'ordinal' else
                       int(pair['pair_index'] * 2 / fps * temporal_tokens_per_second))
            if time_id <= previous_time_id:
                raise ValueError('FPS collapses native temporal coordinates')
            metadata = deepcopy(pair['metadata'])
            metadata.update(frame_metadata=deepcopy(pair['frame_metadata']),
                            memory_id=f"pair_{pair['pair_index']:08d}",
                            native_timestamp_seconds=(pair['pair_index'] * 2 + 0.5) / fps)
            extras = {'deepstack': projection['deepstack']} if 'deepstack' in projection else {}
            observation = bank.ingest(projection['embeddings'], projection['height'],
                                      projection['width'], time_id,
                                      pair['pair_index'] * 2 / fps, metadata, **extras)
            explicit_this_arrival = False
            while cursor < len(qs) and time_key(qs[cursor]) <= frontier:
                respond(qs[cursor], 'explicit', pair['available_time'], 'before_cascade')
                cursor += 1
                explicit_this_arrival = True
            automatic = bool(proactive and not explicit_this_arrival and active is not None
                             and bank.should_respond())
            if automatic:
                respond(active, 'proactive', pair['available_time'], 'before_cascade')
            emit({'event': 'arrival', 'pair_index': pair['pair_index'],
                  'available_time': pair['available_time'], 'frontier_time': pair['frontier_time'],
                  'frame_metadata': pair['frame_metadata'], 'observation': observation,
                  'proactive_answered': automatic, 'explicit_answered': explicit_this_arrival,
                  'encode_and_query_seconds': time.monotonic() - tic,
                  'after_advance': bank.advance(), 'encoder_calls': encoded})
            previous_pair, previous_time_id = pair['pair_index'], time_id
            previous_available = available
        while cursor < len(qs):
            respond(qs[cursor], 'explicit', None, 'after_stream')
            cursor += 1
    finally:
        close = getattr(pairs, 'close', None)
        if close:
            close()
    return {'answers': answers, 'encoder_calls': encoded, 'bank_stats': bank.stats(),
            'explicit_questions': len(qs),
            'proactive_answers': sum(row['kind'] == 'proactive' for row in answers),
            'complete_answers': sum(row['status'] == 'complete' for row in answers)}
