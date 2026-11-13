"""Causal session scheduling with fake model calls and real persistent memory."""
from copy import deepcopy

import pytest
import torch

from streaming.memory import PersistentMemory
from streaming.session import question_prompt, run_session


torch.set_num_threads(2)


def cutoff(seconds, day_id="Day1", day_order=1):
    return {"day_id": day_id, "day_order": day_order, "seconds": seconds}


def question(identifier, seconds, **private):
    return {"question_id": identifier, "question": f"What is visible for {identifier}?",
            "query_time": cutoff(seconds), "choices": ["sitting", "standing"], **private}


def pair(index):
    return {"pair_index": index, "frames": [index, index],
            "available_time": cutoff(index * 2 + 1),
            "frontier_time": cutoff(index * 2 + 2),
            "metadata": {"source_id": "fake_video", "secret_label": "PRIVATE_SOURCE_LABEL"},
            "frame_metadata": [
                {"day_id": "Day1", "day_order": 1, "day_seconds": index * 2 + offset,
                 "day_seconds_fraction": str(index * 2 + offset)} for offset in (0, 1)
            ]}


def projection_sequence():
    # The second grid has exactly 2/4 backward Otsu innovations, r=0.5.
    base = torch.eye(4, 8)
    distances = torch.tensor([0.0, 0.1, 0.8, 0.9])
    changed = base * (1 - distances).unsqueeze(1)
    changed[:, 4:] = torch.diag(torch.sqrt(1 - (1 - distances) ** 2))
    return [base, changed, changed.clone(), base.clone(), base.clone()]


class TrackingPairs:
    def __init__(self, count, trace):
        self.items = iter(pair(index) for index in range(count))
        self.trace = trace
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self.items)
        self.trace.append(("peek", item["pair_index"]))
        return item

    def close(self):
        self.closed = True
        self.trace.append(("close", None))


class FakeModel:
    def __init__(self, trace, results=None):
        self.trace = trace
        self.results = iter(results) if results is not None else None
        self.sequence = projection_sequence()
        self.encoded = []
        self.queries = []

    def encode(self, images):
        index = images[0]
        self.trace.append(("encode", index))
        self.encoded.append(index)
        return {"embeddings": self.sequence[index], "height": 1, "width": 4}

    def answer(self, snapshot, prompt):
        self.trace.append(("answer", prompt))
        self.queries.append((deepcopy(snapshot), prompt))
        return (next(self.results) if self.results is not None else
                {"prediction": "A", "stopped_on_eos": True})


def execute(questions, *, count=4, results=None, proactive=True, bank=None):
    trace, events = [], []
    pairs = TrackingPairs(count, trace)
    model = FakeModel(trace, results)
    if bank is None:
        bank = PersistentMemory(short_frames=2, medium_frames=2, gamma=0.4)
    outcome = run_session(pairs=pairs, questions=questions, bank=bank,
                          encode=model.encode, answer=model.answer, emit=events.append,
                          proactive=proactive)
    return outcome, bank, model, pairs, trace, events


def test_future_lookahead_is_not_encoded_before_earlier_query():
    outcome, _, model, pairs, trace, _ = execute(
        [question("between", 2.5), question("last", 5.5)], proactive=False)
    first_answer = next(i for i, event in enumerate(trace) if event[0] == "answer")
    assert trace.index(("peek", 1)) < first_answer < trace.index(("encode", 1))
    first_snapshot = model.queries[0][0]
    assert [r["metadata"]["memory_id"] for r in first_snapshot["records"]] == ["pair_00000000"]
    assert outcome["answers"][0]["encoder_calls"] == 1
    assert outcome["answers"][0]["phase"] == "between_arrivals"
    assert model.encoded == [0, 1, 2]
    assert 3 not in model.encoded
    assert pairs.closed


def test_pair_at_equal_cutoff_is_excluded_but_frontier_cutoff_includes_prior_pair():
    outcome, _, model, _, _, _ = execute(
        [question("equal_first", 1), question("first_frontier", 2),
         question("equal_second", 3), question("second_frontier", 4)], proactive=False)
    rows = {row["question"]["question_id"]: row for row in outcome["answers"]}
    assert rows["equal_first"]["status"] == "insufficient_observation"
    assert rows["equal_first"]["encoder_calls"] == 0
    assert rows["first_frontier"]["encoder_calls"] == 1
    assert rows["first_frontier"]["phase"] == "before_cascade"
    assert rows["equal_second"]["encoder_calls"] == 1
    assert [r["time_id"] for r in rows["equal_second"]["memory_records"]] == [0]
    assert rows["second_frontier"]["encoder_calls"] == 2
    assert rows["second_frontier"]["phase"] == "before_cascade"
    assert model.encoded == [0, 1]
    for row in rows.values():
        for record in row["memory_records"]:
            assert all(frame["day_seconds"] < row["question"]["query_time"]["seconds"]
                       for frame in record["metadata"]["frame_metadata"])


def test_questions_at_same_cutoff_share_identical_snapshot_without_reencoding():
    outcome, bank, model, _, _, _ = execute(
        [question("q1", 1.5), question("q2", 1.5)], proactive=False)
    assert model.encoded == [0]
    assert outcome["encoder_calls"] == 1
    assert len(model.queries) == 2
    first, second = [item[0] for item in model.queries]
    torch.testing.assert_close(first["embeddings"], second["embeddings"], rtol=0, atol=0)
    torch.testing.assert_close(first["positions"], second["positions"], rtol=0, atol=0)
    assert first["records"] == second["records"]
    assert all(row["phase"] == "before_cascade" for row in outcome["answers"])
    assert bank.stats()["response_count"] == 2
    assert bank.stats()["total_ingested"] == 1


def test_private_labels_and_unknown_fields_never_reach_prompt_or_public_question():
    private = {"answer": "PRIVATE_CORRECT_ANSWER", "evidence": "PRIVATE_EVIDENCE",
               "future_scene": "PRIVATE_FUTURE", "metadata": {"label": "PRIVATE_NESTED"}}
    outcome, _, model, _, _, _ = execute([question("public", 1.5, **private)], proactive=False)
    prompt = model.queries[0][1]
    assert "What is visible for public?" in prompt
    assert "A. sitting" in prompt and "B. standing" in prompt
    assert "PRIVATE_" not in prompt
    assert set(outcome["answers"][0]["question"]) == {"question_id", "question", "query_time", "choices"}
    assert "PRIVATE_" not in question_prompt(question("public", 1.5, **private))


@pytest.mark.parametrize("first_result,expected_proactive", [
    ({"prediction": "A", "stopped_on_eos": True}, 1),
    ({"prediction": "A", "stopped_on_eos": False}, 0),
    ({"prediction": "   ", "stopped_on_eos": True}, 0),
])
def test_proactive_runs_before_cascade_only_after_successful_answer(first_result, expected_proactive):
    successful = {"prediction": "B", "stopped_on_eos": True}
    outcome, bank, model, _, _, events = execute(
        [question("active", 1.5), question("later", 5.5)],
        results=[first_result, successful, successful], count=3)
    automatic = [row for row in outcome["answers"] if row["kind"] == "proactive"]
    assert len(automatic) == expected_proactive
    assert outcome["proactive_answers"] == expected_proactive
    second_arrival = next(event for event in events if event.get("pair_index") == 1)
    assert second_arrival["observation"]["backward_score"] == 0.5
    assert second_arrival["proactive_answered"] is bool(expected_proactive)
    assert second_arrival["observation"]["has_prior_response"] is bool(expected_proactive)
    if automatic:
        row = automatic[0]
        assert row["question"]["question_id"] == "active"
        assert row["phase"] == "before_cascade"
        assert row["memory_stats"]["pending_advance"]
        assert row["memory_stats"]["tier_grids"]["short"] == 2
        assert row["memory_stats"]["total_tokens"] == 8
        assert second_arrival["after_advance"]["tier_grids"]["short"] == 1
        assert model.queries[1][1] == model.queries[0][1]
    assert bank.stats()["total_ingested"] == 3


def test_explicit_query_takes_priority_over_proactive_on_same_arrival():
    outcome, _, model, _, _, events = execute(
        [question("first", 1.5), question("priority", 3.5), question("last", 5.5)], count=3)
    assert len(outcome["answers"]) == 3
    assert all(row["kind"] == "explicit" for row in outcome["answers"])
    second_arrival = next(event for event in events if event.get("pair_index") == 1)
    assert second_arrival["observation"]["triggered"]
    assert second_arrival["explicit_answered"]
    assert not second_arrival["proactive_answered"]
    assert model.queries[1][0]["stats"]["tier_grids"]["short"] == 2


@pytest.mark.parametrize("failing_stage", ["encode", "answer", "emit"])
def test_runtime_exception_always_closes_pair_iterator(failing_stage):
    trace = []
    pairs = TrackingPairs(3, trace)
    model = FakeModel(trace)

    def fail(*args, **kwargs):
        raise RuntimeError(f"injected {failing_stage} failure")

    with pytest.raises(RuntimeError, match=f"injected {failing_stage} failure"):
        run_session(pairs=pairs, questions=[question("q", 1.5)], bank=PersistentMemory(),
                    encode=fail if failing_stage == "encode" else model.encode,
                    answer=fail if failing_stage == "answer" else model.answer,
                    emit=fail if failing_stage == "emit" else lambda event: None)
    assert pairs.closed
    assert trace[-1] == ("close", None)


def test_early_queries_and_empty_stream_report_insufficient_observation_without_model_call():
    outcome, bank, model, pairs, _, _ = execute(
        [question("zero", 0), question("before_pair", 0.5)], count=2)
    assert model.encoded == [] and model.queries == []
    assert all(row["status"] == "insufficient_observation" for row in outcome["answers"])
    assert all(row["prediction"] is None for row in outcome["answers"])
    assert bank.stats()["response_count"] == 0
    assert pairs.closed
    empty, _, empty_model, empty_pairs, _, _ = execute([question("no_frames", 5)], count=0)
    assert empty["answers"][0]["status"] == "insufficient_observation"
    assert empty["answers"][0]["phase"] == "after_stream"
    assert empty_model.queries == [] and empty_pairs.closed


def test_valid_unknown_prediction_is_preserved_without_inventing_an_answer():
    outcome, _, _, _, _, _ = execute(
        [question("unanswerable", 1.5)],
        results=[{"prediction": "Unknown from the observed video.", "stopped_on_eos": True}])
    assert outcome["answers"][0]["prediction"] == "Unknown from the observed video."
    assert outcome["answers"][0]["status"] == "complete"


def test_unordered_questions_are_answered_in_cutoff_order():
    outcome, _, _, _, _, _ = execute(
        [question("later", 3.5), question("earlier", 1.5)], proactive=False)
    assert [row["question"]["question_id"] for row in outcome["answers"]] == ["earlier", "later"]


def test_duplicate_question_id_is_rejected_before_encoding():
    trace = []
    pairs = TrackingPairs(2, trace)
    model = FakeModel(trace)
    with pytest.raises(ValueError, match="Duplicate question IDs"):
        run_session(pairs=pairs, questions=[question("duplicate", 1.5), question("duplicate", 3.5)],
                    bank=PersistentMemory(), encode=model.encode, answer=model.answer,
                    emit=lambda event: None)
    assert model.encoded == []
