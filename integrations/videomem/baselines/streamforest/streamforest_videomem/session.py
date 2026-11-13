"""Explicit causal questions over persistent single-frame StreamForest memory."""
from copy import deepcopy
import math
import time

from .shared import load_shared
from .video import frame_key, time_key

load_shared()
from local_videomem.data import public_question

PROMPT_CONTRACT = 'streaming_observed_video_v1'


def question_prompt(question):
    public = public_question(question)
    text = 'Use only the observed video. Question: ' + public['question']
    if public['choices']:
        text += '\n' + '\n'.join(f'{chr(65 + index)}. {choice}'
                                  for index, choice in enumerate(public['choices']))
        text += '\nAnswer with the option letter only.'
    else:
        text += '\nAnswer in at most three sentences; do not invent unseen details.'
    return text


def run_session(*, frames, questions, bank, encode, answer, emit, fps=1):
    """Encode each available frame once, answer only from t < query cutoff.

    CPU lookahead may decode the next frame, but questions at or before its
    original presentation time are answered before encoding/inserting it.
    Compressed memory has no per-frame records, so causality is audited at the
    ingestion frontier. This entrypoint expects an initially empty bank; a bank
    checkpoint alone cannot restore its input cursor or observation frontier.
    """
    encoded = 0
    answers = []
    cursor = 0
    previous_index = -1
    last_original_time = None
    last_time_object = None
    try:
        if (isinstance(fps, bool) or not isinstance(fps, (int, float))
                or not math.isfinite(fps) or fps <= 0):
            raise ValueError('fps must be positive and finite')
        qs = [public_question(question) for question in questions]
        if not qs:
            raise ValueError('At least one explicit question is required')
        if len({question['question_id'] for question in qs}) != len(qs):
            raise ValueError('Duplicate question IDs')
        qs.sort(key=time_key)
        if bank.stats()['total_tokens']:
            raise ValueError('run_session requires an empty bank and a fresh observation frontier')

        def respond(question, phase):
            started = time.monotonic()
            cutoff = time_key(question)
            if last_original_time is not None and last_original_time >= cutoff:
                raise RuntimeError('Future frame reached explicit-query memory')
            snapshot = bank.snapshot()
            prompt = question_prompt(question)
            row = {'kind': 'explicit', 'question': deepcopy(question),
                   'observation_cutoff': deepcopy(question['query_time']),
                   'observed_time': deepcopy(last_time_object), 'phase': phase,
                   'encoder_calls': encoded, 'prompt': prompt,
                   'memory_stats': deepcopy(snapshot['stats'])}
            if snapshot['stats']['total_tokens'] == 0:
                row.update(status='insufficient_observation', prediction=None)
            else:
                result = answer(snapshot, prompt)
                row.update(result)
                prediction = result.get('prediction', '')
                row['status'] = ('complete' if result.get('stopped_on_eos') and
                                 isinstance(prediction, str) and prediction.strip()
                                 else 'incomplete_generation')
            row['wall_seconds'] = time.monotonic() - started
            answers.append(row)
            emit({'event': 'answer', **row})

        for item in frames:
            index = item['frame_index']
            if type(index) is not int or index <= previous_index:
                raise ValueError('Frame indices must be nonnegative and strictly increase')
            original_time = frame_key(item)
            if last_original_time is not None and original_time <= last_original_time:
                raise ValueError('Frame presentation times must strictly increase')
            while cursor < len(qs) and time_key(qs[cursor]) <= original_time:
                respond(qs[cursor], 'before_arrival')
                cursor += 1
            if cursor == len(qs):
                break
            started = time.monotonic()
            features = encode(item['frame'])
            encoded += 1
            metadata = deepcopy(item['metadata'])
            metadata.update(memory_id=f'frame_{index:08d}', native_timestamp_seconds=index / fps)
            observation = bank.ingest(features, time_id=index, timestamp_seconds=index / fps,
                                      metadata=metadata)
            previous_index = index
            last_original_time = original_time
            last_time_object = deepcopy(item['time'])
            emit({'event': 'arrival', 'frame_index': index, 'time': deepcopy(item['time']),
                  'metadata': metadata, 'observation': observation,
                  'bank': bank.stats(), 'encoder_calls': encoded,
                  'encode_and_ingest_seconds': time.monotonic() - started})
        while cursor < len(qs):
            respond(qs[cursor], 'after_stream')
            cursor += 1
        return {'answers': answers, 'encoder_calls': encoded, 'bank_stats': bank.stats(),
                'explicit_questions': len(qs), 'proactive_answers': 0,
                'complete_answers': sum(row['status'] == 'complete' for row in answers)}
    finally:
        close = getattr(frames, 'close', None)
        if close is not None:
            close()
