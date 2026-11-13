"""FIFO retention exercised through the shared causal video/question scheduler."""
from copy import deepcopy

import pytest
import torch

from fifo_videomem.memory import Qwen3FIFOMemory
from fifo_videomem.shared import load_shared

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
    bank = Qwen3FIFOMemory(memory_frames=4) if bank is None else bank
    pairs, model, events = Pairs(count), Model(), []
    result = run_session(
        pairs=pairs, questions=questions, bank=bank, encode=model.encode,
        answer=model.answer, emit=events.append, proactive=proactive,
        time_id_mode="ordinal",
    )
    return result, bank, pairs, model, events


def test_query_cutoffs_and_fifo_bound_hold_before_and_between_arrivals():
    result, bank, pairs, model, events = execute([
        question("equal_first", 1), question("first_frontier", 2),
        question("equal_second", 3), question("second_frontier", 4),
        question("third_frontier", 6), question("fourth_frontier", 8),
    ])
    rows = {row["question"]["question_id"]: row for row in result["answers"]}
    expected = {"equal_first": [], "first_frontier": [0], "equal_second": [0],
                "second_frontier": [0, 1], "third_frontier": [1, 2],
                "fourth_frontier": [2, 3]}
    for name, ids in expected.items():
        row = rows[name]
        assert [record["time_id"] for record in row["memory_records"]] == ids
        assert row["memory_stats"]["total_tokens"] == 4 * len(ids)
        for record in row["memory_records"]:
            assert all(frame["day_seconds"] < row["question"]["query_time"]["seconds"]
                       for frame in record["metadata"]["frame_metadata"])
    assert rows["equal_first"]["status"] == "insufficient_observation"
    # In the shared scheduler these answers precede advance(): FIFO must evict
    # during ingest rather than granting a hidden extra temporal grid here.
    assert rows["third_frontier"]["phase"] == "before_cascade"
    assert rows["third_frontier"]["memory_stats"]["evicted_grids"] == 1
    assert all(event["after_advance"]["total_tokens"] <= 8
               for event in events if event["event"] == "arrival")
    assert bank.stats()["evicted_grids"] == 2
    assert model.encoded == [0, 1, 2, 3]
    assert pairs.closed


def test_repeated_questions_reuse_latest_bank_without_reencoding_or_label_leakage():
    private = {"answer": "PRIVATE_ANSWER", "evidence": "PRIVATE_EVIDENCE",
               "future_description": "PRIVATE_FUTURE"}
    result, bank, pairs, model, _ = execute([
        question("one", 5.5, **private), question("two", 5.5, **private),
    ])
    assert model.encoded == [0, 1, 2]
    assert result["encoder_calls"] == 3
    assert len(model.answers) == 2
    first, second = [snapshot for snapshot, _ in model.answers]
    assert [record["time_id"] for record in first["records"]] == [1, 2]
    assert first["records"] == second["records"]
    for key in ("embeddings", "positions"):
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    for left, right in zip(first["deepstack"], second["deepstack"]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for row, (_, prompt) in zip(result["answers"], model.answers):
        assert set(row["question"]) == {"question_id", "question", "choices", "query_time"}
        assert "PRIVATE_" not in prompt
    assert bank.stats()["response_count"] == 2
    assert pairs.closed


def test_token_budget_and_no_automatic_responses_survive_shared_scheduler():
    result, _, _, model, _ = execute([
        question("early", 1.5), question("late", 7.5),
    ], bank=Qwen3FIFOMemory(memory_tokens=7), proactive=True)
    # One full four-token grid fits; a second must be removed as a unit.
    assert all(snapshot["stats"]["total_tokens"] == 4 for snapshot, _ in model.answers)
    assert [record["time_id"] for record in model.answers[-1][0]["records"]] == [3]
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
            bank=Qwen3FIFOMemory(memory_frames=2),
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
