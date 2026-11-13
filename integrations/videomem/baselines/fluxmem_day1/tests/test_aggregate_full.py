"""Synthetic postprocessing artifacts; no models or GPUs are loaded."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from day1_compare.aggregate_full import CORE, ValidationError, aggregate, canonical_bytes, digest
from day1_compare.run import public_prompt


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def checksum(path):
    return digest(path.read_bytes())


class FullAggregationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.package = self.root / "package"
        self.new_root = self.root / "new"
        self.output = self.root / "combined"
        self.ids = ["q0", "q1", "q2", "q3"]
        self.pilot_ids = ["q2"]
        self.questions = [{"question_id": qid, "query_time_seconds": 8, "question": "Question " + qid,
                           "choices": ["one", "two"], "input_aliases": []} for qid in self.ids]
        self.labels = [{"question_id": qid, "correct_idx": 0} for qid in self.ids]
        self.frames = []
        for i in range(4):
            path = self.root / f"frame{i}.png"
            path.write_bytes(bytes([i]))
            self.frames.append({"sample_index": i, "path": str(path), "sha256": checksum(path),
                                "actual_global_seconds": 2 * i + 1, "requested_global_seconds": 2 * i + 1,
                                "width": 224, "height": 224})
        self.sources = {}
        original = "def main():\n    p.add_argument('--gpu', type=int, choices=(4,5), required=True)\n"
        for name in CORE | {"day1_compare/run_multi_gpu.py"}:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(original.replace("def main():", "def main(*, allowed_gpus=(4, 5)):").replace("choices=(4,5), required=True", "choices=allowed_gpus, required=True") if name == "day1_compare/run.py" else "# frozen source\n")
            self.sources[name] = checksum(path)
        old_source = self.root / "old_run.py.txt"
        old_source.write_text(original)
        (self.root / "gpu.patch").write_text("test patch\n")
        self.old_hash = checksum(old_source)
        self.settings = {"model": str(self.root / "model"), "verification": str(self.root / "verification.json"),
                         "max_new_tokens": 16, "short_frames": 8, "medium_frames": 512, "video_chunk_t": 1}
        self.marker = {"repo_id": "Qwen/Qwen2.5-VL-7B-Instruct", "revision": "original", "verified_files": []}
        save(self.root / "verification.json", self.marker)
        self.paired = {"upstream_commit": "upstream", "model_revision": "original", "torch": "test", "transformers": "test",
                       "dtype": "bfloat16", "attention": "sdpa", "decoder": "greedy", "video_chunk_t": 1,
                       "frames": 4, "effective_fps": 0.5, "checkpoint": "official", "temporal_positions": "native"}
        self.sidecar = {"transformers_version": "test", "effective_generation_config": {"eos_token_id": [2]}}
        self.video = {"hashes": {"pixel_values_videos": "a" * 64, "video_grid_thw": "b" * 64},
                      "shapes": {"pixel_values_videos": [512, 1176], "video_grid_thw": [1, 3]}, "effective_fps": 0.5}
        all_spec = self.make_package(self.package / "all", self.ids)
        remaining = [qid for qid in self.ids if qid not in self.pilot_ids]
        remain_spec = self.make_package(self.package / "remaining", remaining)
        pilot_spec = self.make_package(self.root / "pilot_input", self.pilot_ids)
        self.pilot_dirs = {arm: self.root / f"pilot_{arm}" for arm in ("on", "off")}
        reuse = {"question_ids": self.pilot_ids, "arms": {}}
        for arm, directory in self.pilot_dirs.items():
            self.make_run(directory, pilot_spec, self.pilot_ids, arm, 4 if arm == "on" else 5, pilot=True)
            rows = json.loads((directory / "predictions.json").read_text())
            reuse["arms"][arm] = {"directory": str(directory), "files": {p.name: checksum(p) for p in directory.iterdir()},
                "rows": [{"question_id": row["question_id"], "full_ordinal": self.ids.index(row["question_id"]),
                          "pilot_ordinal": i, "prediction_row_sha256": digest(canonical_bytes(row)),
                          "wire_filename": f"wire_{i:04d}.json", "wire_sha256": checksum(directory / f"wire_{i:04d}.json")}
                         for i, row in enumerate(rows)]}
        save(self.package / "reuse_manifest.json", reuse)
        shards = []
        for number, selected in enumerate((remaining[:2], remaining[2:]), 1):
            name = f"shard_{number:02d}"
            spec = self.make_package(self.package / name, selected)
            outputs = {arm: str(self.new_root / name / arm) for arm in ("on", "off")}
            for arm, directory in outputs.items():
                self.make_run(Path(directory), spec, selected, arm, number)
            start = 0 if number == 1 else 2
            shards.append({"name": name, "package": spec, "question_ids": selected, "remaining_start": start,
                           "remaining_stop": start + len(selected), "gpu": number, "arm_order": ["on", "off"], "outputs": outputs})
        for name, value in (("pilot_audit.json", {"status": "passed", "valid_for_paired_accuracy": True}),
                            ("generation.json", self.sidecar), ("qa.json", {}), ("timeline.json", {})):
            save(self.root / name, value)
        source = lambda path: {"path": str(path), "sha256": checksum(path)}
        self.manifest = {"status": "prepared", "counts": {"all": 4, "reused": 1, "remaining": 3},
            "question_ids": {"all": self.ids, "reused": self.pilot_ids, "remaining": remaining},
            "packages": {"all": all_spec, "remaining": remain_spec}, "pilot_package": pilot_spec, "shards": shards,
            "frame_content_sha256": digest(canonical_bytes(self.frames)),
            "source_files": {"qa": source(self.root / "qa.json"), "timeline": source(self.root / "timeline.json")},
            "runner": {"module": "day1_compare.run_multi_gpu", "source_hashes": self.sources},
            "runtime_source_transition": {"old_source": source(old_source), "diff": source(self.root / "gpu.patch"), "old_sha256": self.old_hash, "new_sha256": self.sources["day1_compare/run.py"]},
            "reuse_manifest": source(self.package / "reuse_manifest.json"),
            "pilot_pair_audit": source(self.root / "pilot_audit.json"), "generation_config": source(self.root / "generation.json"),
            "checkpoint_verification": source(self.root / "verification.json"),
            "paired_provenance": self.paired, "inference_settings": self.settings}
        save(self.package / "run_manifest.json", self.manifest)
        self.addCleanup(patch.stopall)
        patch("day1_compare.aggregate_full.generation_sidecar", return_value=self.sidecar).start()
        self.checkpoint_check = patch("day1_compare.aggregate_full.verify_weights", return_value=self.marker).start()

    def make_package(self, base, ids):
        protocol = {"selection": {"count": len(ids), "selected_ids": ids}, "query_time_seconds": 8,
                    "sampling": {"frame_count": 4, "image_width": 224, "image_height": 224}}
        save(base / "protocol.json", protocol)
        save(base / "public_questions.json", [q for qid in ids for q in self.questions if q["question_id"] == qid])
        save(base / "eval_labels.json", [q for qid in ids for q in self.labels if q["question_id"] == qid])
        save(base / "source_frames_manifest.json", {"protocol_sha256": checksum(base / "protocol.json"),
             "frames": self.frames, "total_frames": 4, "day_duration_seconds": 8})
        files = {name: checksum(base / name) for name in ("protocol.json", "public_questions.json", "eval_labels.json", "source_frames_manifest.json")}
        save(base / "completed.json", {"status": "complete", "questions": len(ids), "frames": 4, "files": files})
        return {"path": str(base), "completed_sha256": checksum(base / "completed.json"), "files": files}

    def make_run(self, directory, spec, ids, arm, gpu, pilot=False):
        sources = dict(self.sources)
        if pilot:
            sources["day1_compare/run.py"] = self.old_hash
            sources.pop("day1_compare/run_multi_gpu.py")
        save(directory / "provenance.json", {**self.paired, "settings": {**self.settings, "input": spec["path"], "output": str(directory), "gpu": gpu, "fluxmem": arm},
            "gpu": {"physical_index": gpu, "uuid": f"GPU-{gpu}", "name": "Test GPU"},
            "training": False, "no_label_file_opened": True, "source_hashes": sources,
            "public_questions_sha256": spec["files"]["public_questions.json"], "frames_manifest_sha256": spec["files"]["source_frames_manifest.json"]})
        save(directory / "status.json", {"phase": "finished", "arm": arm, "total": len(ids), "attempted": len(ids), "completed": len(ids)})
        save(directory / "processed_video.json", self.video)
        save(directory / "attention_runtime.json", {"checkpoint_use_sliding_window": False, "effective_sliding_window": None})
        rows = []
        for i, qid in enumerate(ids):
            question = next(q for q in self.questions if q["question_id"] == qid)
            prompt = public_prompt(question, duration=8, frames=4)
            save(directory / f"wire_{i:04d}.json", {"question_id": qid, "prompt": prompt, "prompt_sha256": digest(prompt.encode()),
                "input_ids_sha256": digest(qid.encode()), "processed_video_hashes": self.video["hashes"], "second_per_grid_ts": [4.0], "input_tokens": 140})
            event = {"input_tokens": 140, "retained_tokens": 100, "dropped_tokens": 40, "visual_tokens_before": 128,
                     "visual_tokens_after": 88, "temporal_calls": 1, "spatial_calls": 1}
            rows.append({"question_id": qid, "status": "complete", "prediction": "A", "input_tokens": 140,
                         "output_token_ids": [7, 2], "output_tokens": 2, "memory_events": [event] if arm == "on" else []})
        save(directory / "predictions.json", rows)

    def mutate(self, path, callback):
        value = json.loads(path.read_text())
        callback(value)
        save(path, value)

    def combine(self):
        return aggregate(self.package, self.pilot_dirs["on"], self.pilot_dirs["off"], self.new_root,
                         self.output, root=self.root, expected_questions=4, expected_reused=1)

    def reject(self, pattern):
        with self.assertRaisesRegex(ValidationError, pattern):
            self.combine()
        self.assertFalse(self.output.exists())

    def test_exact_union_preserves_original_records_and_provenance(self):
        summary = self.combine()
        self.assertTrue(summary["technical_audit"]["valid_for_paired_accuracy"])
        self.assertEqual(summary["selected_questions"], 4)
        self.assertEqual(summary["arms"]["on"]["correct"], 4)
        rows = json.loads((self.output / "on/predictions.json").read_text())
        self.assertEqual([row["question_id"] for row in rows], self.ids)
        pilot_raw = json.loads((self.pilot_dirs["on"] / "predictions.json").read_text())[0]
        self.assertEqual(rows[2], pilot_raw)
        self.assertEqual((self.output / "sources/pilot_on/provenance.json").read_bytes(), (self.pilot_dirs["on"] / "provenance.json").read_bytes())
        origins = json.loads((self.output / "row_origins.json").read_text())
        self.assertEqual(len(origins), 8)
        self.assertEqual(next(row for row in origins if row["question_id"] == "q2" and row["arm"] == "on")["source_ordinal"], 0)
        self.assertTrue((self.output / "comparison.csv").is_file())
        self.assertTrue((self.output / "completed.json").is_file())
        self.checkpoint_check.assert_called_once()

    def test_duplicate_and_missing_predictions_fail_closed(self):
        path = self.new_root / "shard_01/on/predictions.json"
        original = path.read_bytes()
        self.mutate(path, lambda rows: rows.append(rows[0]))
        self.reject("duplicate question ID")
        path.write_bytes(original)
        self.mutate(path, lambda rows: rows.pop())
        self.reject("missing, extra, or out of order")

    def test_unfinished_jobs_are_pending_and_do_not_create_outputs(self):
        self.mutate(self.new_root / "shard_01/on/status.json", lambda data: data.update(phase="answering"))
        with self.assertRaises(ValidationError) as raised:
            self.combine()
        self.assertTrue(raised.exception.pending)
        self.assertFalse(self.output.exists())

    def test_cached_frame_mutation_is_rejected(self):
        (self.root / "frame0.png").write_bytes(b"changed")
        self.reject("Cached frame changed")

    def test_text_or_tensor_wire_mutation_is_rejected(self):
        path = self.new_root / "shard_01/off/wire_0000.json"
        self.mutate(path, lambda wire: wire.update(input_ids_sha256="f" * 64))
        self.reject("ON/OFF question text/tensor/frame wires differ")

    def test_matching_forged_text_wires_are_rejected_against_public_question(self):
        for arm in ("on", "off"):
            self.mutate(self.new_root / f"shard_01/{arm}/wire_0000.json", lambda wire: wire.update(prompt="forged", prompt_sha256=digest(b"forged")))
        self.reject("prompt mismatch")

    def test_settings_or_native_switch_drift_cannot_pass(self):
        path = self.new_root / "shard_01/off/provenance.json"
        original = path.read_bytes()
        self.mutate(path, lambda data: data["settings"].update(max_new_tokens=32))
        self.reject("inference settings differ")
        path.write_bytes(original)
        self.mutate(self.new_root / "shard_01/off/predictions.json", lambda rows: rows[0].update(memory_events=[{}]))
        self.reject("native FluxMem switch mismatch")

    def test_paired_hardware_is_checked(self):
        self.mutate(self.new_root / "shard_01/off/provenance.json", lambda data: data["gpu"].update(uuid="GPU-impostor"))
        self.reject("ON/OFF physical GPUs differ")

    def test_pilot_raw_record_mutation_is_rejected(self):
        self.mutate(self.pilot_dirs["on"] / "predictions.json", lambda rows: rows[0].update(prediction="B"))
        self.reject("frozen pilot artifact changed")

    def test_generation_failure_retained_but_not_technical_pass(self):
        self.mutate(self.new_root / "shard_01/on/predictions.json", lambda rows: rows[0].update(status="truncated", prediction="A", output_token_ids=[7, 7]))
        self.mutate(self.new_root / "shard_01/on/status.json", lambda data: data.update(completed=1))
        summary = self.combine()
        self.assertEqual(summary["technical_audit"]["status"], "failed")
        self.assertFalse(summary["technical_audit"]["valid_for_paired_accuracy"])
        self.assertEqual(summary["arms"]["on"]["status_counts"]["truncated"], 1)
        self.assertEqual(summary["arms"]["on"]["accuracy_all_selected"], 0.75)

    def test_existing_empty_output_directory_is_never_reused(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.combine()
        self.assertEqual(list(self.output.iterdir()), [])

    def test_only_exact_gpu_allowlist_source_transition_is_allowed(self):
        path = self.root / "day1_compare/run.py"
        path.write_text(path.read_text() + "# extra inference edit\n")
        self.manifest["runner"]["source_hashes"]["day1_compare/run.py"] = checksum(path)
        self.manifest["runtime_source_transition"]["new_sha256"] = checksum(path)
        save(self.package / "run_manifest.json", self.manifest)
        self.reject("exceeds the authorized GPU argument change")


if __name__ == "__main__":
    unittest.main()
