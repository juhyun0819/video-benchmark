import copy
import hashlib

import pytest

from day1_compare.prepare import evaluation_labels, public_question, sampling_plan, select_questions


def test_selection_is_frozen_by_id_and_salt_only():
    rows = [{"qa_id": f"q{i}", "correct_idx": i % 5, "question": "old"} for i in range(30)]
    expected = sorted(rows, key=lambda row: hashlib.sha256(("day1-pilot-v1" + row["qa_id"]).encode()).hexdigest())[:16]
    chosen = select_questions(rows, 16)
    assert [r["qa_id"] for r in chosen] == [r["qa_id"] for r in expected]
    changed = [{**row, "correct_idx": 99, "question": "changed"} for row in reversed(rows)]
    assert [r["qa_id"] for r in select_questions(changed, 16)] == [r["qa_id"] for r in chosen]
    with pytest.raises(ValueError, match="unique"):
        select_questions([rows[0], rows[0]], 1)


def test_midpoints_and_offsets_include_boundary_in_next_clip():
    segments = [
        {"clip_key": "first", "concat_start_seconds": 0, "concat_end_seconds": 3, "video_duration_seconds": 3},
        {"clip_key": "second", "concat_start_seconds": 3, "concat_end_seconds": 12, "video_duration_seconds": 9},
    ]
    result = sampling_plan(segments, 2)
    assert [(r["requested_global_seconds"], r["requested_source_seconds"], r["source_clip_key"]) for r in result] == [(3.0, 0.0, "second"), (9.0, 6.0, "second")]
    broken = copy.deepcopy(segments)
    broken[1]["concat_start_seconds"] = 4
    with pytest.raises(ValueError, match="contiguous"):
        sampling_plan(broken, 2)


def test_decimal_offsets_do_not_accumulate_float_drift():
    segments = [
        {"clip_key": "a", "concat_start_seconds": 0, "concat_end_seconds": 0.1, "video_duration_seconds": 0.1},
        {"clip_key": "b", "concat_start_seconds": 0.1, "concat_end_seconds": 0.3, "video_duration_seconds": 0.2},
    ]
    result = sampling_plan(segments, 3)
    assert [r["requested_global_seconds"] for r in result] == [0.05, 0.15, 0.25]
    assert [r["requested_source_seconds"] for r in result] == [0.05, 0.05, 0.15]


def test_public_allowlist_keeps_gold_and_annotations_out_and_nested_choices_intact():
    row = {
        "qa_id": "qa_original_7", "correct_idx": 4, "correct_answer": "SECRET_GOLD_METADATA",
        "original_qa": {"hidden": "SECRET_ORIGINAL"}, "annotation": "SECRET_ANNOTATION",
        "day_time_view": {"question": "What occurred?", "choices": [["a", "b"], "c", "d", "e", "f"], "correct_idx": 4, "correct_answer": "SECRET_VIEW_GOLD"},
        "mapped_inputs": {"video 1": {"original_source_video_id": "source", "source_start_seconds": 1, "source_end_seconds": 3, "day_start_seconds": 11, "day_end_seconds": 13, "other_secret": "SECRET_SCOPE"}},
    }
    frames = [{"actual_global_seconds": t} for t in [10.999, 11, 12.999, 13]]
    public = public_question(row, 100, frames)
    assert set(public) == {"question_id", "question", "choices", "query_time_seconds", "input_aliases"}
    assert "SECRET" not in str(public)
    assert public["choices"][0] == ["a", "b"]
    assert public["input_aliases"]["video 1"]["sampled_frame_count"] == 2
    assert evaluation_labels([row]) == [{"question_id": "qa_original_7", "correct_idx": 4}]
    assert public_question(row, 100, [])["input_aliases"]["video 1"]["sampled_frame_count"] == 0
