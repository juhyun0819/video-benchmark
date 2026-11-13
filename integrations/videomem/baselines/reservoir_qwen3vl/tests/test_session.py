"""Seeded reservoir retention through the shared causal question scheduler."""
from copy import deepcopy

import pytest
import torch

from reservoir_videomem.memory import Qwen3ReservoirMemory
from reservoir_videomem.shared import load_shared

load_shared()
from streaming.session import run_session


def cutoff(seconds):
    return {"day_id": "Day1", "day_order": 1, "seconds": seconds}


def question(identifier, seconds, **private):
    return {"question_id": identifier, "question": "What is visible?",
            "choices": ["A scene", "Another scene"],
            "query_time": cutoff(seconds), **private}


class Pairs:
    def __init__(self, count):
        self.index = 0
        self.count = count
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.count:
            raise StopIteration
        index = self.index
        self.index += 1
        return {
            "pair_index": index, "frames": [index, index],
            "available_time": cutoff(index * 2 + 1),
            "frontier_time": cutoff(index * 2 + 2),
            "metadata": {"source_id": "synthetic", "private_note": "PRIVATE_SOURCE"},
            "frame_metadata": [
                {"day_id": "Day1", "day_order": 1, "day_seconds": index * 2 + offset,
                 "day_seconds_fraction": str(index * 2 + offset)}
                for offset in (0, 1)
            ],
        }

    def close(self):
        self.closed = True


class Model:
    def __init__(self):
        self.encoded = []
        self.answers = []

    def encode(self, frames):
        index = frames[0]
        self.encoded.append(index)
        main = torch.arange(24, dtype=torch.float32).reshape(4, 6) + index * 100
        return {"embeddings": main, "height": 2, "width": 2,
                "deepstack": [main + (level + 1) * 1000 for level in range(3)]}

    def answer(self, snapshot, prompt):
        self.answers.append((deepcopy(snapshot), prompt))
        return {"prediction": "A", "stopped_on_eos": True}


def execute(questions, *, count=5, proactive=False, bank=None):
    bank = Qwen3ReservoirMemory(memory_frames=4, seed=0) if bank is None else bank
    pairs, model, events = Pairs(count), Model(), []
    result = run_session(
        pairs=pairs, questions=questions, bank=bank, encode=model.encode,
        answer=model.answer, emit=events.append, proactive=proactive,
        time_id_mode="ordinal",
    )
    return result, bank, pairs, model, events


def ids(records):
    return [record["time_id"] for record in records]


@pytest.mark.parametrize("budget", [{"memory_frames": 4}, {"memory_tokens": 9}])
def test_causal_cutoffs_strict_bound_and_rejected_arrival_horizon(budget):
    result, bank, pairs, model, events = execute([
        question("equal_first", 1), question("first_frontier", 2),
        question("equal_second", 3), question("second_frontier", 4),
        question("third_frontier", 6), question("fourth_frontier", 8),
    ], bank=Qwen3ReservoirMemory(seed=0, **budget))
    rows = {row["question"]["question_id"]: row for row in result["answers"]}
    # Independent concrete Algorithm R path for random.Random(0), k=2:
    # draw 1 from [0,3), replacing slot 1 with grid 2; draw 3 from [0,4),
    # rejecting grid 3. A recent-only queue would instead return [2,3].
    expected = {"equal_first": [], "first_frontier": [0], "equal_second": [0],
                "second_frontier": [0, 1], "third_frontier": [0, 2],
                "fourth_frontier": [0, 2]}
    for name, retained in expected.items():
        row = rows[name]
        assert ids(row["memory_records"]) == retained
        assert row["memory_stats"]["total_tokens"] == 4 * len(retained)
        for record in row["memory_records"]:
            assert all(frame["day_seconds"] < row["question"]["query_time"]["seconds"]
                       for frame in record["metadata"]["frame_metadata"])
    assert rows["equal_first"]["status"] == "insufficient_observation"
    # The latest observation was processed even though it did not enter memory.
    last = rows["fourth_frontier"]
    assert last["phase"] == "before_cascade"
    assert last["observed_time"] == cutoff(7)
    assert last["encoder_calls"] == last["memory_stats"]["total_ingested"] == 4
    assert last["memory_records"][-1]["time_id"] == 2
    assert all(event["after_advance"]["total_tokens"] <= 8
               for event in events if event["event"] == "arrival")
    assert bank.snapshot()["memory_truncated"] is True
    assert model.encoded == [0, 1, 2, 3]
    assert pairs.closed


def test_repeated_questions_do_not_consume_rng_or_leak_private_labels():
    private = {"answer": "PRIVATE_ANSWER", "evidence": "PRIVATE_EVIDENCE",
               "future_description": "PRIVATE_FUTURE"}
    result, bank, pairs, model, _ = execute([
        question("early", 4, **private),
        question("one", 5.5, **private), question("two", 5.5, **private),
        question("after_rejection", 8, **private), question("final", 12, **private),
    ], count=6)
    control, control_bank, _, control_model, _ = execute([question("final", 12)], count=6)
    assert model.encoded == control_model.encoded == list(range(6))
    assert result["encoder_calls"] == control["encoder_calls"] == 6
    first, second = [snapshot for snapshot, _ in model.answers[1:3]]
    assert ids(first["records"]) == [0, 2]
    assert first["records"] == second["records"]
    for key in ("embeddings", "positions"):
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    for left, right in zip(first["deepstack"], second["deepstack"]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    # Later reservoir draws must be unchanged by answer count and snapshotting.
    final, control_final = bank.snapshot(), control_bank.snapshot()
    assert ids(final["records"]) == ids(control_final["records"]) == [2, 4]
    torch.testing.assert_close(final["embeddings"], control_final["embeddings"], rtol=0, atol=0)
    for row, (_, prompt) in zip(result["answers"], model.answers):
        assert set(row["question"]) == {"question_id", "question", "choices", "query_time"}
        assert "PRIVATE_" not in prompt
    assert bank.stats()["response_count"] == 5
    assert pairs.closed


def test_rejecting_newest_marks_truncation_without_eviction_or_automatic_response():
    result, bank, _, model, _ = execute([
        question("early", 1.5), question("late", 5.5),
    ], count=3, bank=Qwen3ReservoirMemory(memory_frames=4, seed=5), proactive=True)
    # random.Random(5) draws slot 2 at n=3: the old reservoir is unchanged.
    latest = model.answers[-1][0]
    assert ids(latest["records"]) == [0, 1]
    assert latest["stats"]["evicted_grids"] == 0
    assert latest["memory_truncated"] is True
    assert latest["stats"]["total_ingested"] == 3
    assert bank.stats()["response_count"] == 2
    assert result["proactive_answers"] == 0
    assert [row["kind"] for row in result["answers"]] == ["explicit", "explicit"]


@pytest.mark.parametrize("stage", ["encode", "answer", "emit"])
def test_model_or_logging_failure_closes_video_iterator(stage):
    pairs, model = Pairs(3), Model()

    def fail(*args, **kwargs):
        raise RuntimeError(f"injected {stage} failure")

    with pytest.raises(RuntimeError, match=f"injected {stage} failure"):
        run_session(
            pairs=pairs, questions=[question("one", 1.5)],
            bank=Qwen3ReservoirMemory(memory_frames=2, seed=0),
            encode=fail if stage == "encode" else model.encode,
            answer=fail if stage == "answer" else model.answer,
            emit=fail if stage == "emit" else lambda event: None,
            proactive=False, time_id_mode="ordinal",
        )
    assert pairs.closed


def test_empty_stream_reports_insufficient_observation_without_model_calls():
    result, _, pairs, model, _ = execute([question("empty", 5)], count=0)
    assert model.encoded == model.answers == []
    assert result["answers"][0]["status"] == "insufficient_observation"
    assert result["answers"][0]["prediction"] is None
    assert pairs.closed
