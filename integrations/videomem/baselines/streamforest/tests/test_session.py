"""Single-frame scheduling and question independence of persistent memory."""
from copy import deepcopy

import pytest
import torch

from streamforest_videomem.session import question_prompt, run_session


def cutoff(seconds, day_id='Day1', day_order=1):
    return {'day_id': day_id, 'day_order': day_order, 'seconds': seconds}


def question(identifier, seconds, **private):
    return {'question_id': identifier, 'question': 'What is visible?',
            'choices': ['sitting', 'standing'], 'query_time': cutoff(seconds), **private}


class Frames:
    def __init__(self, count, trace=None, times=None):
        self.count, self.index = count, 0
        self.trace = trace if trace is not None else []
        self.times = times
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.count:
            raise StopIteration
        index = self.index
        self.index += 1
        self.trace.append(('peek', index))
        time = self.times[index] if self.times is not None else cutoff(index)
        return {'frame': index, 'frame_index': index, 'time': deepcopy(time),
                'metadata': {'day_id': time['day_id'], 'day_order': time['day_order'],
                             'day_seconds': time['seconds'], 'day_seconds_fraction': str(time['seconds']),
                             'source_id': 'synthetic', 'private_note': 'PRIVATE_SOURCE'}}

    def close(self):
        self.closed = True
        self.trace.append(('close', None))


class Bank:
    def __init__(self):
        self.arrivals = []
        self.features = []

    def ingest(self, features, *, time_id, timestamp_seconds, metadata):
        self.arrivals.append((time_id, timestamp_seconds, deepcopy(metadata)))
        self.features.append(features.detach().clone())
        self.features = self.features[-2:]
        return {'time_id': time_id}

    def stats(self):
        return {'total_tokens': 4 * len(self.features), 'total_ingested': len(self.arrivals)}

    def snapshot(self):
        return {'features': torch.cat(self.features).unsqueeze(0) if self.features
                else torch.empty((1, 0, 8)), 'stats': self.stats()}


class Model:
    def __init__(self, trace=None, result=None):
        self.trace = trace if trace is not None else []
        self.encoded, self.answers = [], []
        self.result = result if result is not None else {'prediction': 'A', 'stopped_on_eos': True}

    def encode(self, image):
        self.trace.append(('encode', image))
        self.encoded.append(image)
        return torch.full((4, 8), float(image))

    def answer(self, snapshot, prompt):
        self.trace.append(('answer', prompt))
        self.answers.append((deepcopy(snapshot), prompt))
        return dict(self.result)


def execute(questions, *, count=6, bank=None, times=None, result=None, fps=1):
    trace, events = [], []
    frames = Frames(count, trace, times)
    bank = Bank() if bank is None else bank
    model = Model(trace, result)
    outcome = run_session(frames=frames, questions=questions, bank=bank, encode=model.encode,
                          answer=model.answer, emit=events.append, fps=fps)
    return outcome, bank, model, frames, trace, events


def test_equal_cutoff_and_future_lookahead_never_reach_question_memory():
    result, bank, model, frames, trace, _ = execute([question('early', 2), question('last', 5)])
    first_answer = next(index for index, row in enumerate(trace) if row[0] == 'answer')
    assert trace.index(('peek', 2)) < first_answer < trace.index(('encode', 2))
    assert model.encoded == [0, 1, 2, 3, 4]
    first = model.answers[0][0]['features']
    assert set(first.flatten().tolist()) == {0, 1}
    assert set(model.answers[1][0]['features'].flatten().tolist()) == {3, 4}
    assert [row['encoder_calls'] for row in result['answers']] == [2, 5]
    assert result['answers'][0]['observed_time'] == cutoff(1)
    assert result['answers'][0]['observation_cutoff'] == cutoff(2)
    assert result['proactive_answers'] == 0 and frames.closed


def test_repeated_queries_reuse_memory_and_do_not_reencode_or_mutate_bank():
    result, bank, model, frames, _, _ = execute(
        [question('same1', 1.5), question('same2', 1.5), question('later', 3.5)])
    assert model.encoded == [0, 1, 2, 3]
    assert len(bank.arrivals) == 4
    first, second = [snapshot for snapshot, _ in model.answers[:2]]
    torch.testing.assert_close(first['features'], second['features'], rtol=0, atol=0)
    assert first['stats'] == second['stats']
    assert [row['encoder_calls'] for row in result['answers']] == [2, 2, 4]
    assert frames.closed


def test_zero_cutoff_and_empty_video_skip_answer_model():
    result, bank, model, frames, _, _ = execute([question('zero', 0)])
    assert result['answers'][0]['status'] == 'insufficient_observation'
    assert model.encoded == [] and model.answers == [] and not bank.arrivals
    assert frames.closed
    empty, _, model, frames, _, _ = execute([question('no_frames', 5)], count=0)
    assert empty['answers'][0]['prediction'] is None
    assert empty['answers'][0]['phase'] == 'after_stream'
    assert model.answers == [] and frames.closed


def test_private_labels_stay_out_of_public_questions_and_prompts():
    private = {'answer': 'PRIVATE_ANSWER', 'evidence': 'PRIVATE_EVIDENCE',
               'metadata': {'future': 'PRIVATE_FUTURE'}}
    result, _, model, _, _, _ = execute([question('q', 1.5, **private)])
    row = result['answers'][0]
    assert set(row['question']) == {'question_id', 'question', 'choices', 'query_time'}
    assert 'PRIVATE_' not in row['prompt']
    assert 'A. sitting' in model.answers[0][1]
    assert 'PRIVATE_' not in question_prompt(question('q', 1.5, **private))


@pytest.mark.parametrize('result,status', [
    ({'prediction': 'A', 'stopped_on_eos': True}, 'complete'),
    ({'prediction': 'A', 'stopped_on_eos': False}, 'incomplete_generation'),
    ({'prediction': ' ', 'stopped_on_eos': True}, 'incomplete_generation'),
    ({'prediction': None, 'stopped_on_eos': True}, 'incomplete_generation'),
])
def test_only_nonempty_eos_answers_are_complete(result, status):
    outcome, _, _, _, _, _ = execute([question('q', 1)], result=result)
    assert outcome['answers'][0]['status'] == status
    assert outcome['complete_answers'] == int(status == 'complete')


def test_day_transition_preserves_source_time_but_compacts_bank_time():
    times = [cutoff(8), cutoff(9), cutoff(0, 'Day2', 2), cutoff(1, 'Day2', 2)]
    questions = [question('day1', 10),
                 {**question('day2', 1.5), 'query_time': cutoff(1.5, 'Day2', 2)}]
    result, bank, model, _, _, _ = execute(questions, count=4, times=times, fps=2)
    assert [row[1] for row in bank.arrivals] == [0, 0.5, 1, 1.5]
    assert [row[2]['day_seconds'] for row in bank.arrivals] == [8, 9, 0, 1]
    assert model.encoded == [0, 1, 2, 3]
    assert result['answers'][0]['encoder_calls'] == 2
    assert result['answers'][1]['observed_time'] == cutoff(1, 'Day2', 2)


@pytest.mark.parametrize('stage', ['encode', 'ingest', 'answer', 'emit'])
def test_error_always_closes_iterator(stage):
    frames, model, bank = Frames(4), Model(), Bank()

    def fail(*args, **kwargs):
        raise RuntimeError('injected ' + stage)

    if stage == 'ingest':
        bank.ingest = fail
    with pytest.raises(RuntimeError, match='injected ' + stage):
        run_session(frames=frames, questions=[question('q', 1.5)], bank=bank,
                    encode=fail if stage == 'encode' else model.encode,
                    answer=fail if stage == 'answer' else model.answer,
                    emit=fail if stage == 'emit' else lambda event: None)
    assert frames.closed


@pytest.mark.parametrize('questions,fps', [([], 1), ([question('q', 1)], 0),
    ([question('q', 1)], True), ([question('q', 1), question('q', 2)], 1)])
def test_invalid_input_closes_iterator_before_encoding(questions, fps):
    frames, model = Frames(4), Model()
    with pytest.raises(ValueError):
        run_session(frames=frames, questions=questions, bank=Bank(), encode=model.encode,
                    answer=model.answer, emit=lambda event: None, fps=fps)
    assert frames.closed and not model.encoded


def test_nonempty_bank_requires_restored_session_frontier():
    bank = Bank()
    bank.ingest(torch.zeros(4, 8), time_id=7, timestamp_seconds=7, metadata={})
    frames = Frames(2)
    with pytest.raises(ValueError, match='empty bank'):
        run_session(frames=frames, questions=[question('q', 1)], bank=bank,
                    encode=lambda frame: pytest.fail('must not encode'),
                    answer=lambda *args: pytest.fail('must not answer'), emit=lambda row: None)
    assert frames.closed


def test_out_of_order_source_frames_rejected_and_closed():
    frames, model = Frames(3, times=[cutoff(1), cutoff(0), cutoff(2)]), Model()
    with pytest.raises(ValueError, match='presentation times'):
        run_session(frames=frames, questions=[question('q', 3)], bank=Bank(),
                    encode=model.encode, answer=model.answer, emit=lambda event: None)
    assert model.encoded == [0] and frames.closed


def test_actual_streamforest_memory_final_state_is_independent_of_question_count():
    from streamforest_videomem.memory import StreamForestMemory
    config = {'st_memory_windows': [2, 2], 'st_memory_tokens': [4, 2],
              'event_split_window': 2, 'long_memory_tokens_per_frame': 1,
              'long_memory_tokens_quota': 4, 'sim_weight_g': 1.0,
              'time_weight_a': 1.0, 'merge_weight_b': 1.0}
    banks = [StreamForestMemory(8, 2, config=config) for _ in range(2)]
    # Nonzero distinguishable vectors exercise compression without invalid norms.
    generator = torch.Generator().manual_seed(17)
    features = [torch.randn((4, 8), generator=generator) + 1 for _ in range(12)]
    question_sets = [[question('last', 12)],
                     [question('early', 3.5), question('repeat', 3.5), question('last', 12)]]
    finals = []
    for bank, questions in zip(banks, question_sets):
        frames = Frames(12)
        result = run_session(frames=frames, questions=questions, bank=bank,
            encode=lambda index: features[index], answer=lambda snapshot, prompt: {
                'prediction': 'A', 'stopped_on_eos': True}, emit=lambda event: None)
        assert result['encoder_calls'] == 12 and frames.closed
        finals.append(bank.snapshot())
    torch.testing.assert_close(finals[0]['features'], finals[1]['features'], rtol=0, atol=0)
    assert finals[0]['stats'] == finals[1]['stats']
