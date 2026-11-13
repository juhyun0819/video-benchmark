"""Daily composition clocks and source provenance; no NAS or FFmpeg required."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import mock
import unittest

import daily_video_captions as captions
import render_daily_videos as daily
import timestamp_timeline as timeline


def clip(identifier, day, start, source_start, source_end, scope="whole_original"):
    return {"id": identifier, "day": day, "label": identifier,
            "start_seconds": start, "expected_duration_s": source_end - source_start,
            "input": f"/synthetic-inputs/{identifier}.mp4",
            "source": {"dataset": "ego4d", "video_id": f"source-{identifier}",
                       "start_s": source_start, "end_s": source_end},
            "source_scope": scope, "review_status": "pending_visual_review"}


def plan(clips):
    return {"version": 1, "base_date": "2025-07-07", "timezone": "Asia/Seoul",
            "style": dict(timeline.DEFAULT_STYLE), "clips": clips}


def results(*entries):
    return {identifier: {"output": f"/normalized/{identifier}.mp4",
                         "metadata": {"duration_s": duration, "frames": frames}}
            for identifier, duration, frames in entries}


class AssembleMappingTests(unittest.TestCase):
    def test_output_uses_actual_file_durations_not_schedule_gaps_or_expected_lengths(self):
        # Inputs have independent1fps final-frame rounding. The afternoon gap
        # must affect only the burned wall clock, never output playback length.
        morning = clip("d1-am", 1, 8 * 3600, 0, 4.5)
        afternoon = clip("d1-pm", 1, 14 * 3600, 100, 103.25,
                         "caption_supported_salvage")
        source_plan = plan([afternoon, morning])  # Sorting is by scheduled time.
        before = copy.deepcopy(source_plan)
        mapped = daily.assemble_mapping(source_plan, 1,
            results(("d1-am", 5.0, 5), ("d1-pm", 3.0, 3)))
        first, second = mapped["segments"]
        self.assertEqual([first["slot_id"], second["slot_id"]], ["d1-am", "d1-pm"])
        self.assertEqual((first["output_start_s"], first["output_end_s"]), (0.0, 5.0))
        self.assertEqual((second["output_start_s"], second["output_end_s"]), (5.0, 8.0))
        self.assertEqual((second["output_start_frame"], second["output_end_frame_exclusive"]), (5, 8))
        self.assertEqual((mapped["duration_s"], mapped["frames"]), (8.0, 8))
        self.assertEqual(first["planned_duration_s"], 4.5)
        self.assertEqual(second["planned_duration_s"], 3.25)
        self.assertEqual(first["synthetic_end"], "2025-07-07T08:00:05")
        self.assertEqual(second["synthetic_start"], "2025-07-07T14:00:00")
        self.assertEqual(source_plan, before)

    def test_moved_d3_slot_uses_explicit_day13_instead_of_id_prefix(self):
        moved = clip("d3-s7", 13, 9 * 3600, 0, 9879.3)
        ordinary = clip("d3-s2", 3, 7 * 3600, 0, 20)
        source_plan = plan([ordinary, moved])
        outputs = results(("d3-s7", 9879.0, 9879), ("d3-s2", 20.0, 20))
        mapped = daily.assemble_mapping(source_plan, 13, outputs)
        self.assertEqual(mapped["day"], 13)
        self.assertEqual(mapped["date"], "2025-07-19")
        self.assertEqual([s["slot_id"] for s in mapped["segments"]], ["d3-s7"])
        self.assertEqual(mapped["segments"][0]["synthetic_start"], "2025-07-19T09:00:00")
        self.assertEqual([s["slot_id"] for s in daily.assemble_mapping(source_plan, 3, outputs)["segments"]], ["d3-s2"])

    def test_all28_dates_cover_july7_through_august3(self):
        expected_dates = [f"2025-07-{day:02d}" for day in range(7, 32)] + [
            "2025-08-01", "2025-08-02", "2025-08-03"]
        clips = [clip(f"arbitrary-id-{day}", day, 3600, 0, 2) for day in range(1, 29)]
        outputs = results(*[(c["id"], 2.0, 2) for c in clips])
        source_plan = plan(clips)
        for day, expected in enumerate(expected_dates, 1):
            with self.subTest(day=day):
                mapped = daily.assemble_mapping(source_plan, day, outputs)
                self.assertEqual(mapped["date"], expected)
                self.assertEqual(mapped["segments"][0]["synthetic_start"], expected + "T01:00:00")
                self.assertEqual(mapped["timezone"], "Asia/Seoul")

    def test_source_offset_is_applied_once_to_captions_and_never_added_to_clock(self):
        first = clip("d1-first", 1, 6 * 3600, 0, 5)
        meal = clip("d1-meal", 1, 7 * 3600, 300, 310, "caption_supported_salvage")
        mapped = daily.assemble_mapping(plan([first, meal]), 1,
            results(("d1-first", 5.0, 5), ("d1-meal", 10.0, 10)))
        manifest = {"slots": []}
        for entry in (first, meal):
            manifest["slots"].append({"slot_id": entry["id"], "day": 1,
                "source": entry["source"], "proposed_observation": {
                    "duration_s": entry["expected_duration_s"],
                    "synthetic_start_s": entry["start_seconds"],
                    "synthetic_end_s": entry["start_seconds"] + entry["expected_duration_s"]},
                "evidence": {"captions": [{"timestamp_sec": 305.5,
                    "narration_text": "#C C eats food."}]} if entry is meal else {"captions": []}})
        exported = captions.build_day_captions(mapped, manifest)
        self.assertEqual(len(exported["captions"]), 1)
        row = exported["captions"][0]
        self.assertEqual(row["source_time"], {"kind": "point", "t": 305.5})
        self.assertEqual(row["clip_time"], {"kind": "point", "t": 5.5})
        self.assertEqual(row["output_time"], {"kind": "point", "t": 10.5})
        self.assertEqual(row["synthetic_time"], {"kind": "point", "t": "2025-07-07 07:00:05.500000"})
        self.assertEqual(mapped["segments"][1]["source_start_s"], 300)
        self.assertEqual(mapped["segments"][1]["synthetic_start"], "2025-07-07T07:00:00")

    def test_selected_source_intervals_survive_import_and_mapping_unchanged(self):
        originals = [clip("d1-work", 1, 8 * 3600, 0, 600),
                     clip("d1-meal", 1, 12 * 3600, 768.332, 1227.635,
                          "caption_supported_salvage")]
        manifest = {"slots": []}
        for original in originals:
            start = original["start_seconds"]
            duration = original["expected_duration_s"]
            manifest["slots"].append({"slot_id": original["id"], "day": 1,
                "label": original["label"], "source": {**original["source"], "media_path": original["input"]},
                "source_scope": original["source_scope"], "status": "candidate_needs_review",
                "proposed_observation": {"synthetic_start_s": start, "synthetic_end_s": start + duration,
                                         "duration_s": duration}})
        manifest["slots"].append({"slot_id": "d1-home-prep", "day": 1, "proposed_observation": None})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest))
            imported = timeline.import_manifest(path, "2025-07-07", "Asia/Seoul")
        self.assertEqual(len(imported["clips"]), 2)
        mapped = daily.assemble_mapping(imported, 1,
            results(("d1-work", 600.0, 600), ("d1-meal", 459.0, 459)))
        for segment, original in zip(mapped["segments"], originals):
            self.assertEqual((segment["source_start_s"], segment["source_end_s"]),
                             (original["source"]["start_s"], original["source"]["end_s"]))
            self.assertEqual(segment["source_scope"], original["source_scope"])
            self.assertEqual(segment["planned_duration_s"], original["expected_duration_s"])
        self.assertEqual(mapped["duration_s"], 1059.0)
        self.assertEqual(mapped["segments"][1]["output_start_s"], 600.0)

    def test_missing_selected_result_is_not_silently_omitted(self):
        with self.assertRaises(KeyError):
            daily.assemble_mapping(plan([clip("required", 1, 3600, 0, 10)]), 1, {})


    def test_finish_day_forwards_canvas_size_and_writes_sidecars(self):
        source_plan = plan([clip("only", 1, 3600, 0, 2)])
        outputs = results(("only", 2.0, 2))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(size=640, manifest=root / "manifest.json")
            def concat(paths, output, *, size):
                self.assertEqual(size, 640)
                Path(output).write_bytes(b"fixture")
                return {"expected_frames": 2}
            with mock.patch.object(daily.video, "concat_clips", side_effect=concat), \
                 mock.patch.object(daily.stamp, "probe", return_value={"frames": 2, "duration_s": 2}), \
                 mock.patch.object(daily.captions, "build_day_captions", return_value={"coverage": {}}):
                answer = daily.finish_day(source_plan, 1, outputs, {}, args, root)
            self.assertEqual(answer["status"], "complete")
            for filename in ("video.mp4", "mapping.json", "captions.json", "validation.json"):
                self.assertTrue((root / "day01" / filename).is_file())


if __name__ == "__main__":
    unittest.main()
