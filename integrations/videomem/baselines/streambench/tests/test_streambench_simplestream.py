"""CPU protocol tests with an explicitly injected synthetic encoder/backend."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from baselines.streambench.adapters.simplestream import (
    MODEL_ID, MODEL_REVISION, SimpleStreamMethod,
)
from baselines.streambench.method import UnsupportedBudget


class FixtureBackend:
    identity = {"fixture": "simplestream-protocol-v1", "synthetic": True}

    def __init__(self):
        self.encoded_values = []
        self.questions = []
        self.fail_encode = False
        self.fail_answer = False
        self.mutate_query = False

    def encode_frames(self, frames):
        if self.fail_encode:
            raise RuntimeError("fixture encoding failure")
        self.encoded_values.extend(frames)
        return [(torch.full((1, 3584), float(value), dtype=torch.bfloat16),
                 torch.tensor([1, 2, 2], dtype=torch.long)) for value in frames]

    def answer(self, frames, question, choices):
        self.questions.append((question, list(choices)))
        used = [frame.frame_id for frame in frames]
        if self.mutate_query:
            frames[0].features.fill_(999)
            frames[0].metadata["stream_id"] = "query-private-mutation"
        if self.fail_answer:
            raise RuntimeError("fixture answering failure")
        return {"answer": "A" if choices else str(used), "used_frame_ids": used,
                "visual_tokens": sum(frame.features.shape[0] for frame in frames),
                "token_ids": [1]}


def make_method(*, backend=None, budget=1_000_000, recent_frames=4, **options):
    backend = backend or FixtureBackend()
    method = SimpleStreamMethod({"method": "simplestream", "memory_budget_bytes": budget,
        "recent_frames": recent_frames, "synthetic_test_mode": True,
        "encoder_id": "protocol-fixture-v1", **options}, backend=backend)
    method.initialize()
    return method


def records(start, count, day=1):
    return [{"stream_id": "fixture-stream", "day": day, "global_frame_id": i,
             "source_timestamp": float(i), "global_timestamp": float(i),
             "source_id": f"day{day}", "source_frame_index": i}
            for i in range(start, start + count)]


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and left.shape == right.shape
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


def test_recent_window_continues_across_day_and_query_is_private():
    backend = FixtureBackend()
    method = make_method(backend=backend)
    method.observe([0, 1, 2], records(0, 3))
    method.finalize_chunk()
    method.observe([3, 4, 5], records(3, 3, day=2))
    before = method.snapshot()
    assert backend.encoded_values == list(range(6))
    assert method.eviction_count == 2
    backend.mutate_query = True
    result = method.answer("synthetic question", ["one", "two"])
    assert result["retrieved_frame_ids"] == [2, 3, 4, 5]
    assert result["num_visual_tokens_used"] == 4
    assert result["query_read_bytes"] == 4 * (3584 * 2 + 3 * 8)
    assert_same(before, method.snapshot())
    backend.fail_answer = True
    with pytest.raises(RuntimeError, match="answering failure"):
        method.answer("another synthetic question")
    assert_same(before, method.snapshot())
    state_text = json.dumps(method.persistent_state()["metadata_bytes"], default=str)
    assert "synthetic question" not in state_text


def test_checkpoint_restore_preserves_window_and_avoids_reencoding(tmp_path):
    method = make_method(recent_frames=3)
    method.observe(list(range(5)), records(0, 5))
    path = tmp_path / "checkpoint"
    method.save_checkpoint(path)
    resumed = make_method(recent_frames=3)
    resumed.load_checkpoint(path)
    assert resumed._backend.encoded_values == []
    for item in (method, resumed):
        item.observe([5, 6], records(5, 2, day=2))
    assert_same(method.snapshot(), resumed.snapshot())
    assert method.answer("fixture") == resumed.answer("fixture")


def test_budget_overflow_rolls_back_without_shrinking_window():
    method = make_method(budget=22_000, recent_frames=4)
    method.observe([0], records(0, 1))
    before = method.snapshot()
    with pytest.raises(UnsupportedBudget):
        method.observe([1, 2, 3], records(1, 3))
    assert method.recent_frames == 4
    assert_same(before, method.snapshot())


def test_encoder_failure_and_wrong_count_leave_state_unchanged():
    method = make_method()
    method.observe([0], records(0, 1))
    before = method.snapshot()
    method._backend.fail_encode = True
    with pytest.raises(RuntimeError, match="encoding failure"):
        method.observe([1], records(1, 1))
    assert_same(before, method.snapshot())
    method._backend.encode_frames = lambda frames: []
    with pytest.raises(ValueError, match="exactly one"):
        method.observe([1], records(1, 1))
    assert_same(before, method.snapshot())


@pytest.mark.parametrize("feature,grid", [
    (torch.zeros(1, 3584), torch.tensor([1, 2, 2])),
    (torch.full((1, 3584), float("nan"), dtype=torch.bfloat16), torch.tensor([1, 2, 2])),
    (torch.zeros(1, 128, dtype=torch.bfloat16), torch.tensor([1, 2, 2])),
    (torch.zeros(1, 3584, dtype=torch.bfloat16), torch.tensor([1, 4, 4])),
    (torch.zeros(2, 3584, dtype=torch.bfloat16), torch.tensor([2, 2, 2])),
])
def test_incompatible_projection_is_rejected_without_observation(feature, grid):
    method = make_method()
    method._backend.encode_frames = lambda frames: [(feature, grid)]
    before = method.snapshot()
    with pytest.raises(ValueError, match="Encoder"):
        method.observe([0], records(0, 1))
    assert_same(before, method.snapshot())


@pytest.mark.parametrize("change", [
    {"question": "private"}, {"day": "Day1"}, {"global_timestamp": float("nan")},
    {"global_frame_id": 4}, {"source_frame_index": -1},
])
def test_invalid_or_private_timestamp_rejected_before_encoding(change):
    method = make_method()
    row = records(0, 1)[0]
    row.update(change)
    with pytest.raises(ValueError):
        method.observe([0], [row])
    assert method._backend.encoded_values == []


def test_nonconsecutive_stream_rejected_before_encoding():
    method = make_method()
    method.observe([0], records(0, 1))
    wrong = records(1, 1)
    wrong[0]["stream_id"] = "different-stream"
    with pytest.raises(ValueError, match="one stream"):
        method.observe([1], wrong)
    assert method._backend.encoded_values == [0]


def test_accounting_exposes_all_live_features_and_zero_persistent_kv():
    method = make_method()
    method.observe([0, 1, 2, 3, 4], records(0, 5))
    report = method.memory_bytes()
    assert report["visual_memory_bytes"] == 4 * 3584 * 2
    assert report["metadata_bytes"] >= 4 * 3 * 8
    assert report["kv_memory_bytes"] == 0
    assert report["persistent_total_bytes"] == report["visual_memory_bytes"] + report["metadata_bytes"]


def test_changed_checkpoint_identity_is_rejected():
    method = make_method()
    method.observe([0], records(0, 1))
    saved = method.snapshot()
    other = make_method(recent_frames=8)
    with pytest.raises(ValueError, match="identity"):
        other.restore(saved)
    backend = FixtureBackend()
    backend.identity = {"fixture": "different-checkpoint"}
    other = make_method(backend=backend)
    with pytest.raises(ValueError, match="identity"):
        other.restore(saved)


def test_memory_cursor_tampering_is_rejected_before_installing_state():
    method = make_method()
    method.observe([0], records(0, 1))
    original = method.snapshot()
    state = deepcopy(original["method"])
    state["frames_seen"] = 2
    state["last_record"]["global_frame_id"] = 1
    with pytest.raises(ValueError, match="window/cursor"):
        method.load_state_dict(state)
    assert_same(original, method.snapshot())


def test_only_explicit_fixture_injection_and_pinned_7b_are_allowed():
    with pytest.raises(ValueError, match="explicit synthetic"):
        SimpleStreamMethod({"memory_budget_bytes": 100_000}, backend=FixtureBackend())
    with pytest.raises(ValueError, match="encoder_id"):
        SimpleStreamMethod({"memory_budget_bytes": 100_000, "synthetic_test_mode": True}, backend=FixtureBackend())
    with pytest.raises(ValueError, match="pinned"):
        make_method(model_id="SimpleStream-8B")
    with pytest.raises(ValueError, match="pinned"):
        make_method(model_revision="mutable-main")
    assert make_method(model_id=MODEL_ID, model_revision=MODEL_REVISION).recent_frames == 4


def test_registry_and_example_identify_7b_adaptation_without_model_load():
    from baselines.streambench.registry import available_methods, create_method
    assert available_methods()["simplestream"]["status"] == "adapter_available"
    config = json.loads((Path(__file__).parents[1] / "configs/simplestream.example.json").read_text())
    method = create_method(config)
    assert isinstance(method, SimpleStreamMethod)
    assert method._backend is None
    assert config["model_id"] == MODEL_ID and config["model_revision"] == MODEL_REVISION


def test_common_runner_day_resume_and_independent_scoring(tmp_path):
    from baselines.streambench.runner import evaluate_checkpoint, observe_stream
    first = make_method(recent_frames=2)
    observe_stream(first, [([0, 1, 2], records(0, 3))], tmp_path / "first",
                   stream_id="fixture-stream", manifest_digest="fixture-manifest",
                   method_config=first.config)
    resumed = make_method(recent_frames=2)
    observe_stream(resumed, [([3, 4], records(3, 2, day=2))], tmp_path / "resumed",
                   stream_id="fixture-stream", manifest_digest="fixture-manifest",
                   method_config=resumed.config, resume_checkpoint=tmp_path / "first/checkpoint_D1")
    assert resumed._backend.encoded_values == [3, 4]
    before = resumed.snapshot()
    queries = [{"question_id": str(i), "source_day": 1, "question": "fixture question",
                "choices": ["one", "two"], "correct_idx": 0} for i in range(2)]
    rows = evaluate_checkpoint(resumed, tmp_path / "resumed/checkpoint_D1_D2", queries,
                               method_config=resumed.config)
    assert all(row["correct"] and row["retrieved_frame_ids"] == [3, 4] for row in rows)
    assert all(row["days_after_event"] == 1 for row in rows)
    assert resumed._backend.questions == [("fixture question", ["one", "two"])] * 2
    assert_same(before, resumed.snapshot())


def test_reset_is_only_explicit_new_stream_and_empty_answer_is_abstention():
    method = make_method()
    assert method.answer("fixture")["status"] == "insufficient_observation"
    method.observe([0, 1], records(0, 2))
    method.reset()
    assert method.state_dict()["frames_seen"] == 0
    assert method.eviction_count == 0
    method.observe([0], records(0, 1))
    assert method.answer("fixture")["retrieved_frame_ids"] == [0]
