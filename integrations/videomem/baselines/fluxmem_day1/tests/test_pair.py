"""Independent saved-artifact checks; fixtures contain no real inference."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from day1_compare.check_pair import MODEL_FILE, check_pair, generation_sidecar, sha


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class PairChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base = self.root / "input"
        self.on, self.off = self.root / "on", self.root / "off"
        self.ids = ["q0", "q1"]
        sources = {}
        for name in ("day1_compare/run.py", MODEL_FILE, "local_videomem/run.py"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# frozen source\n")
            sources[name] = sha(path)
        protocol = {"selection": {"count": 2, "selected_ids": self.ids}, "query_time_seconds": 8,
                    "sampling": {"frame_count": 4, "image_width": 224, "image_height": 224}}
        save(self.base / "protocol.json", protocol)
        frames = []
        for i in range(4):
            path = self.base / f"frame{i}.png"
            path.write_bytes(bytes([i]))
            frames.append({"sample_index": i, "actual_global_seconds": 2 * i + 1,
                           "requested_global_seconds": 2 * i + 1,
                           "path": str(path), "sha256": sha(path), "width": 224, "height": 224})
        save(self.base / "source_frames_manifest.json", {
            "protocol_sha256": sha(self.base / "protocol.json"), "frames": frames,
            "total_frames": 4, "day_duration_seconds": 8})
        save(self.base / "public_questions.json", [{"question_id": q, "query_time_seconds": 8} for q in self.ids])
        save(self.base / "eval_labels.json", [{"question_id": q, "correct_idx": 0} for q in self.ids])
        save(self.base / "completed.json", {"status": "complete", "questions": 2, "frames": 4,
             "files": {name: sha(self.base / name) for name in ("protocol.json", "public_questions.json", "source_frames_manifest.json", "eval_labels.json")}})
        video = {"hashes": {"pixel_values_videos": "f" * 64, "video_grid_thw": "e" * 64},
                 "shapes": {"pixel_values_videos": [512, 1176], "video_grid_thw": [1, 3]}, "effective_fps": 0.5}
        for name, directory in (("on", self.on), ("off", self.off)):
            save(directory / "status.json", {"phase": "finished", "completed": 2, "attempted": 2, "total": 2})
            save(directory / "processed_video.json", video)
            save(directory / "attention_runtime.json", {"checkpoint_use_sliding_window": False,
                                                       "checkpoint_sliding_window": 32768, "effective_sliding_window": None})
            save(directory / "provenance.json", {
                "settings": {"input": str(self.base), "output": str(directory), "gpu": 4 if name == "on" else 5,
                             "fluxmem": name, "model": "model", "max_new_tokens": 16},
                "source_hashes": sources, "training": False, "no_label_file_opened": True,
                "frames": 4, "effective_fps": 0.5, "public_questions_sha256": sha(self.base / "public_questions.json"),
                "frames_manifest_sha256": sha(self.base / "source_frames_manifest.json"),
                "upstream_commit": "upstream", "model_revision": "original", "torch": "test",
                "transformers": "test", "dtype": "bfloat16", "attention": "sdpa", "decoder": "greedy", "video_chunk_t": 1})
            rows = []
            for i, qid in enumerate(self.ids):
                text = "Question " + qid
                save(directory / f"wire_{i:04d}.json", {"question_id": qid, "prompt": text,
                     "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(), "input_ids_sha256": str(i) * 64,
                     "processed_video_hashes": video["hashes"], "second_per_grid_ts": [4.0], "input_tokens": 140})
                event = {"input_tokens": 140, "retained_tokens": 100, "dropped_tokens": 40,
                         "visual_tokens_before": 128, "visual_tokens_after": 88,
                         "temporal_calls": 1, "spatial_calls": 1}
                rows.append({"question_id": qid, "status": "complete", "prediction": "A", "input_tokens": 140,
                             "output_token_ids": [7, 2], "output_tokens": 2, "memory_events": [event] if name == "on" else []})
            save(directory / "predictions.json", rows)
        self.sidecar = {"transformers_version": "test", "effective_generation_config": {"eos_token_id": [2]}}

    def mutate(self, directory, name, callback):
        path = directory / name
        value = json.loads(path.read_text())
        callback(value)
        save(path, value)

    def check(self):
        with patch("day1_compare.check_pair.generation_sidecar", return_value=copy.deepcopy(self.sidecar)):
            return check_pair(self.base, self.on, self.off, root=self.root)[0]

    def test_complete_matching_pair_passes_and_new_extra_source_is_allowed(self):
        self.mutate(self.on, "provenance.json", lambda v: v["source_hashes"].update({"day1_compare/check_pair.py": "new-file"}))
        report = self.check()
        self.assertEqual(report["status"], "passed", report["failures"])
        self.assertTrue(report["valid_for_paired_accuracy"])

    def test_in_progress_matching_prefix_is_pending(self):
        for directory in (self.on, self.off):
            save(directory / "status.json", {"phase": "answering", "completed": 1, "total": 2})
            self.mutate(directory, "predictions.json", lambda rows: rows.pop())
            (directory / "wire_0001.json").unlink()
        report = self.check()
        self.assertEqual(report["status"], "pending", report["failures"])
        self.assertFalse(report["valid_for_paired_accuracy"])

    def test_truncated_or_execution_error_is_never_valid(self):
        for status in ("truncated", "execution_error"):
            self.mutate(self.on, "predictions.json", lambda rows: rows[0].update(status=status))
            report = self.check()
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["valid_for_paired_accuracy"])

    def test_token_prompt_or_native_switch_difference_fails(self):
        self.mutate(self.off, "wire_0000.json", lambda v: v.update(input_ids_sha256="changed"))
        self.mutate(self.off, "predictions.json", lambda rows: rows[0].update(memory_events=[{}]))
        report = self.check()
        failed = {item["check"] for item in report["failures"]}
        self.assertIn("paired_wire:0", failed)
        self.assertIn("off:row:0:native_memory_switch", failed)

    def test_hash_corruption_or_runner_change_fails(self):
        (self.base / "frame0.png").write_bytes(b"changed")
        (self.root / "day1_compare/run.py").write_text("changed")
        report = self.check()
        failed = {item["check"] for item in report["failures"]}
        self.assertIn("cached_frame_hashes", failed)
        self.assertIn("on:frozen_runner_model_source", failed)

    def test_sliding_attention_and_false_complete_eos_fail(self):
        self.mutate(self.on, "attention_runtime.json", lambda v: v.update(effective_sliding_window=32768))
        self.mutate(self.off, "predictions.json", lambda rows: rows[0].update(output_token_ids=[7, 8]))
        report = self.check()
        failed = {item["check"] for item in report["failures"]}
        self.assertIn("on:full_attention", failed)
        self.assertIn("off:row:0:complete_eos", failed)

    def test_wrong_question_order_is_detected(self):
        self.mutate(self.off, "predictions.json", lambda rows: rows.reverse())
        report = self.check()
        self.assertIn("off:prediction_order", {item["check"] for item in report["failures"]})


class GenerationSidecarTests(unittest.TestCase):
    def test_checkpoint_penalty_and_eos_preserved_while_sampling_is_overridden(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "model/generation_config.json"
            save(source, {"do_sample": True, "eos_token_id": [2, 3], "pad_token_id": 3,
                          "repetition_penalty": 1.05, "temperature": 0.000001})
            sidecar = generation_sidecar({"settings": {"model": "model", "max_new_tokens": 16}}, root)
            config = sidecar["effective_generation_config"]
            self.assertFalse(config["do_sample"])
            self.assertEqual(config["max_new_tokens"], 16)
            self.assertEqual(config["repetition_penalty"], 1.05)
            self.assertEqual(config["eos_token_id"], [2, 3])
            self.assertEqual(config["num_beams"], 1)
            self.assertEqual(sidecar["checkpoint_sha256"], sha(source))


if __name__ == "__main__":
    unittest.main()
