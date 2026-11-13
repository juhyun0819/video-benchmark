"""CPU fixtures exercise freezing, reuse eligibility and label isolation."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from day1_compare import prepare_full as full
from day1_compare.check_pair import MODEL_FILE
from day1_compare.prepare import evaluation_labels, public_question, select_questions, sha256_file
from day1_compare.run import public_prompt


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(full, "EXPECTED_POPULATION", 7)
    monkeypatch.setattr(full, "EXPECTED_REUSED", 2)
    monkeypatch.setattr(full, "EXPECTED_FRAMES", 4)
    monkeypatch.setattr(full, "EXPECTED_DAY_SECONDS", 8)
    base, on, off = (tmp_path / name for name in ("pilot", "on", "off"))
    sources = {}
    for filename in ("day1_compare/run.py", "day1_compare/run_v2.py", "day1_compare/run_multi_gpu.py", MODEL_FILE, "local_videomem/run.py"):
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen source\n")
        sources[filename] = sha256_file(path)
    rows = []
    for i in range(7):
        row = {key: False for key in (
            "actual_render_alignment_verified", "answer_reverified_on_rendered_video", "one_hz_visual_evidence_reverified")}
        row.update({key: "metadata" for key in (
            "original_qa_sha256", "source_revision", "source_file", "source_url", "status",
            "timeline_basis", "edit_alignment_status", "input_scope_policy")})
        row.update(qa_id=f"q{i}", correct_idx=i % 3,
                   day_time_view={"question": f"Question {i}", "choices": ["red", "blue", "green"]},
                   ground_truth="PRIVATE_GOLD_SENTINEL", evidence="PRIVATE_EVIDENCE_SENTINEL",
                   mapped_inputs={"video 1": {"original_source_video_id": "clip",
                                             "source_start_seconds": 0, "source_end_seconds": 3,
                                             "day_start_seconds": 0, "day_end_seconds": 3}})
        rows.append(row)
    qa, timeline = tmp_path / "qa.json", tmp_path / "timeline.json"
    save(qa, rows)
    save(timeline, {"segments": []})
    ranked = select_questions(rows, len(rows))
    selected = ranked[:2]
    protocol = {
        "source_qa_path": str(qa), "source_qa_sha256": sha256_file(qa),
        "source_timeline_path": str(timeline), "source_timeline_sha256": sha256_file(timeline),
        "selection": {"count": 2, "selected_ids": [row["qa_id"] for row in selected],
                      "salt": full.SELECTION_SALT, "population_count": 7,
                      "population_ids_sha256": full.object_sha(sorted(row["qa_id"] for row in rows))},
        "query_time_seconds": 8,
        "sampling": {"frame_count": 4, "image_width": 224, "image_height": 224},
    }
    save(base / "protocol.json", protocol)
    frames = []
    for i in range(4):
        path = base / f"frame_{i:04d}.png"
        path.write_bytes(bytes([i]))
        frames.append({"sample_index": i, "actual_global_seconds": i * 2 + 1,
                       "requested_global_seconds": i * 2 + 1, "width": 224, "height": 224,
                       "path": str(path), "sha256": sha256_file(path)})
    manifest = {"protocol_sha256": sha256_file(base / "protocol.json"), "frames": frames,
                "total_frames": 4, "day_duration_seconds": 8}
    questions = [public_question(row, 8, frames) for row in selected]
    save(base / "source_frames_manifest.json", manifest)
    save(base / "public_questions.json", questions)
    save(base / "eval_labels.json", evaluation_labels(selected))
    save(base / "completed.json", {"status": "complete", "questions": 2, "frames": 4,
         "files": {name: sha256_file(base / name) for name in full.PACKAGE_FILES}})
    video = {"hashes": {"pixel_values_videos": "a" * 64, "video_grid_thw": "b" * 64},
             "shapes": {"pixel_values_videos": [512, 1176], "video_grid_thw": [1, 3]}, "effective_fps": 0.5}
    verification = tmp_path / "verification.json"
    save(verification, {"revision": "original"})
    for arm, directory in (("on", on), ("off", off)):
        save(directory / "status.json", {"phase": "finished", "total": 2, "attempted": 2, "completed": 2})
        save(directory / "processed_video.json", video)
        save(directory / "attention_runtime.json", {"checkpoint_use_sliding_window": False,
             "checkpoint_sliding_window": 32768, "effective_sliding_window": None})
        save(directory / "provenance.json", {
            "settings": {"input": str(base), "output": str(directory), "gpu": 4 if arm == "on" else 5,
                         "fluxmem": arm, "model": str(tmp_path / "model"), "verification": str(verification),
                         "max_new_tokens": 16, "short_frames": 8, "medium_frames": 512, "video_chunk_t": 1},
            "source_hashes": sources, "training": False, "no_label_file_opened": True,
            "public_questions_sha256": sha256_file(base / "public_questions.json"),
            "frames_manifest_sha256": sha256_file(base / "source_frames_manifest.json"),
            "upstream_commit": "upstream", "model_revision": "original", "torch": "test", "transformers": "test",
            "dtype": "bfloat16", "attention": "sdpa", "decoder": "greedy", "video_chunk_t": 1,
            "frames": 4, "effective_fps": 0.5, "checkpoint": "original", "temporal_positions": "native"})
        predictions = []
        for i, question in enumerate(questions):
            prompt = public_prompt(question, duration=8, frames=4)
            save(directory / f"wire_{i:04d}.json", {
                "question_id": question["question_id"], "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "input_ids_sha256": str(i) * 64,
                "processed_video_hashes": video["hashes"], "second_per_grid_ts": [4.0], "input_tokens": 140})
            event = {"input_tokens": 140, "retained_tokens": 100, "dropped_tokens": 40,
                     "visual_tokens_before": 128, "visual_tokens_after": 88, "temporal_calls": 1, "spatial_calls": 1}
            predictions.append({"question_id": question["question_id"], "status": "complete", "prediction": "A",
                                "output_token_ids": [7, 2], "output_tokens": 2, "input_tokens": 140,
                                "memory_events": [event] if arm == "on" else []})
        save(directory / "predictions.json", predictions)
    sidecar = {"transformers_version": "test", "effective_generation_config": {"eos_token_id": [2]}}
    monkeypatch.setattr("day1_compare.check_pair.generation_sidecar", lambda *args: copy.deepcopy(sidecar))
    return {"root": tmp_path, "pilot": base, "on": on, "off": off, "output": tmp_path / "full"}


def test_full_and_remaining_bind_same_frames_and_exact_reuse_map(frozen):
    before = {str(path): sha256_file(path) for path in frozen["root"].rglob("*") if path.is_file()}
    result = full.prepare_full(**frozen)
    assert result["counts"] == {"all": 7, "reused": 2, "remaining": 5}
    assert result["question_ids"]["all"] == result["question_ids"]["reused"] + result["question_ids"]["remaining"]
    pilot_frames = full.read_json(frozen["pilot"] / "source_frames_manifest.json")["frames"]
    all_questions = full.read_json(frozen["output"] / "all/public_questions.json")
    assert all_questions[:2] == full.read_json(frozen["pilot"] / "public_questions.json")
    assert all_questions[2:] == full.read_json(frozen["output"] / "remaining/public_questions.json")
    shards = result["shards"]
    assert [s["gpu"] for s in shards] == [1, 2, 3, 4, 5]
    assert [qid for s in shards for qid in s["question_ids"]] == result["question_ids"]["remaining"]
    assert [s["arm_order"] for s in shards] == [["on", "off"], ["off", "on"], ["on", "off"], ["off", "on"], ["on", "off"]]
    for spec in shards:
        directory = Path(spec["package"]["path"])
        assert [q["question_id"] for q in full.read_json(directory / "public_questions.json")] == spec["question_ids"]
        assert full.read_json(directory / "source_frames_manifest.json")["frames"] == pilot_frames
    for name in ("all", "remaining"):
        directory = frozen["output"] / name
        manifest = full.read_json(directory / "source_frames_manifest.json")
        assert manifest["frames"] == pilot_frames
        assert manifest["protocol_sha256"] == sha256_file(directory / "protocol.json")
        assert manifest["protocol_sha256"] != sha256_file(frozen["pilot"] / "protocol.json")
        assert full.read_json(directory / "completed.json")["files"] == result["packages"][name]["files"]
    reuse = full.read_json(frozen["output"] / "reuse_manifest.json")
    for arm in ("on", "off"):
        rows = full.read_json(frozen[arm] / "predictions.json")
        assert [r["question_id"] for r in reuse["arms"][arm]["rows"]] == result["question_ids"]["reused"]
        for ordinal, mapping in enumerate(reuse["arms"][arm]["rows"]):
            assert mapping["full_ordinal"] == mapping["pilot_ordinal"] == ordinal
            assert mapping["prediction_row_sha256"] == full.object_sha(rows[ordinal])
            assert mapping["wire_sha256"] == sha256_file(frozen[arm] / mapping["wire_filename"])
    assert {path: sha256_file(Path(path)) for path in before} == before
    assert not list(frozen["output"].rglob("*.png"))


def test_runtime_inputs_do_not_contain_gold_or_evidence(frozen):
    result = full.prepare_full(**frozen)
    for name in ("all", "remaining"):
        for filename in result["runtime_inputs"]:
            text = (frozen["output"] / name / filename).read_text()
            assert "correct_idx" not in text
            assert "PRIVATE_GOLD_SENTINEL" not in text
            assert "PRIVATE_EVIDENCE_SENTINEL" not in text
        labels = full.read_json(frozen["output"] / name / "eval_labels.json")
        assert all(set(row) == {"question_id", "correct_idx"} for row in labels)


@pytest.mark.parametrize("change", [{"status": "truncated"}, {"status": "execution_error"}, {"prediction": "I think A because red"}])
def test_ineligible_pilot_prediction_rejects_entire_reuse(frozen, change):
    path = frozen["on"] / "predictions.json"
    rows = full.read_json(path)
    rows[0].update(change)
    save(path, rows)
    with pytest.raises(ValueError, match="audit did not pass|complete and parse-valid"):
        full.prepare_full(**frozen)
    assert not frozen["output"].exists()


def test_reordered_predictions_rejected_before_publish(frozen):
    path = frozen["off"] / "predictions.json"
    save(path, list(reversed(full.read_json(path))))
    with pytest.raises(ValueError, match="audit did not pass"):
        full.prepare_full(**frozen)
    assert not frozen["output"].exists()


def test_source_qa_change_rejected_before_publish(frozen):
    path = frozen["root"] / "qa.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="Original source qa hash changed"):
        full.prepare_full(**frozen)
    assert not frozen["output"].exists()


def test_frame_corruption_rejected_before_publish(frozen):
    (frozen["pilot"] / "frame_0000.png").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="cached_frame_hashes"):
        full.prepare_full(**frozen)
    assert not frozen["output"].exists()


def test_no_overwrite_even_empty_directory(frozen):
    frozen["output"].mkdir()
    with pytest.raises(FileExistsError, match="Refusing"):
        full.prepare_full(**frozen)
    assert list(frozen["output"].iterdir()) == []


def test_completed_preparation_cannot_be_replaced(frozen):
    full.prepare_full(**frozen)
    before = sha256_file(frozen["output"] / "run_manifest.json")
    with pytest.raises(FileExistsError, match="Refusing"):
        full.prepare_full(**frozen)
    assert sha256_file(frozen["output"] / "run_manifest.json") == before


def test_runtime_transition_permits_only_exact_authorized_edits(frozen):
    current = frozen["root"] / "day1_compare/run.py"
    old_text = "\n".join(old for old, _ in full.GPU_REPLACEMENTS) + "\n"
    original = frozen["root"] / "reports/day1_full_v2_runtime/run_pilot_v2.py.txt"
    original.parent.mkdir(parents=True)
    original.write_text(old_text)
    (original.parent / "gpu_allowlist_only.patch").write_text("reviewable allowlist diff")
    new_text = old_text
    for old, new in full.GPU_REPLACEMENTS:
        new_text = new_text.replace(old, new)
    current.write_text(new_text)
    provenance = {"source_hashes": {"day1_compare/run.py": sha256_file(original)}}
    transition = full.runtime_transition(provenance, frozen["root"])
    assert transition["old_sha256"] == sha256_file(original)
    assert transition["new_sha256"] == sha256_file(current)
    current.write_text(new_text + "# another unauthorized change\n")
    with pytest.raises(ValueError, match="exceeds authorized"):
        full.runtime_transition(provenance, frozen["root"])
