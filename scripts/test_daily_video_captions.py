"""Source/clip/output/wall-clock caption regressions, using only the stdlib."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from daily_video_captions import build_day_captions


def fixture(captions):
    slot = {"slot_id": "d3-s7", "day": 13, "status": "ready",
            "source": {"dataset": "ego4d", "video_id": "walk", "start_s": 11610, "end_s": 11620},
            "proposed_observation": {"synthetic_start_s": 34800, "duration_s": 10},
            "evidence": {"captions": captions}}
    mapping = {"day": 13, "date": "2025-07-19", "timezone": "Asia/Seoul", "segments": [{
        "slot_id": "d3-s7", "output_start_s": 600, "output_end_s": 610,
        "synthetic_start": "2025-07-19 09:40:00", "synthetic_start_s": 34800,
        "source_start_s": 11610, "source_end_s": 11620, "input_duration_s": 10}]}
    return mapping, {"slots": [slot]}


def point(t, identifier="point"):
    return {"id": identifier, "text": "#C C walks.", "time": {"kind": "point", "t": t}}


def interval(a, b):
    return {"id": "summary", "text": "Original summary unchanged.",
            "time": {"kind": "interval", "start": a, "end": b}}


class CaptionTests(unittest.TestCase):
    def test_source_offset_is_subtracted_once_for_all_three_destination_clocks(self):
        mapping, manifest = fixture([point(11612.5)])
        row = build_day_captions(mapping, manifest)["captions"][0]
        self.assertEqual(row["source_time"]["t"], 11612.5)
        self.assertEqual(row["clip_time"]["t"], 2.5)
        self.assertEqual(row["output_time"]["t"], 602.5)
        self.assertEqual(row["synthetic_time"]["t"], "2025-07-19 09:40:02.500000")
        self.assertEqual(row["kind"], "point")
        self.assertNotIn("end", row["output_time"])

    def test_concatenated_output_is_contiguous_while_wallclock_jumps_at_next_clip(self):
        mapping, manifest = fixture([point(11612.5)])
        second = copy.deepcopy(manifest["slots"][0])
        second.update(slot_id="lunch")
        second["source"].update(video_id="lunch", start_s=100, end_s=110)
        second["proposed_observation"]["synthetic_start_s"] = 45000
        second["evidence"]["captions"] = [point(103)]
        manifest["slots"].append(second)
        mapping["segments"].append({
            "slot_id": "lunch", "output_start_s": 610, "output_end_s": 620,
            "source_start_s": 100, "source_end_s": 110, "input_duration_s": 10,
            "synthetic_start_s": 45000, "synthetic_start": "2025-07-19 12:30:00",
        })
        rows = build_day_captions(mapping, manifest)["captions"]
        self.assertEqual([row["output_time"]["t"] for row in rows], [602.5, 613])
        self.assertEqual(rows[1]["synthetic_time"]["t"], "2025-07-19 12:30:03")

    def test_point_selection_uses_half_open_source_interval(self):
        mapping, manifest = fixture([point(11609.99, "before"), point(11610, "start"),
                                      point(11619.99, "inside"), point(11620, "end")])
        rows = build_day_captions(mapping, manifest)["captions"]
        self.assertEqual([r["id"] for r in rows], ["d3-s7:start", "d3-s7:inside"])

    def test_intervals_are_clipped_and_marked_without_rewriting_text(self):
        mapping, manifest = fixture([interval(11600, 11630)])
        row = build_day_captions(mapping, manifest)["captions"][0]
        self.assertEqual(row["source_time"], {"kind": "interval", "start": 11600, "end": 11630})
        self.assertEqual(row["selected_source_time"], {"kind": "interval", "start": 11610, "end": 11620})
        self.assertEqual(row["output_time"], {"kind": "interval", "start": 600, "end": 610})
        self.assertEqual(row["boundary_clipped"], {"start": True, "end": True})
        self.assertEqual(row["text"], "Original summary unchanged.")

    def test_captions_cannot_extend_beyond_shorter_actual_output(self):
        mapping, manifest = fixture([interval(11610, 11620), point(11619.8)])
        mapping["segments"][0]["input_duration_s"] = 9.5
        mapping["segments"][0]["output_end_s"] = 609.5
        result = build_day_captions(mapping, manifest)
        self.assertEqual(len(result["captions"]), 1)
        self.assertEqual(result["captions"][0]["output_time"]["end"], 609.5)
        self.assertEqual(result["captions"][0]["boundary_clipped"], {"start": False, "end": True})

    def test_longer_rounded_input_does_not_invent_source_annotation_coverage(self):
        mapping, manifest = fixture([interval(11610, 11621), point(11620.1)])
        mapping["segments"][0]["input_duration_s"] = 11
        mapping["segments"][0]["output_end_s"] = 611
        result = build_day_captions(mapping, manifest)
        self.assertEqual(len(result["captions"]), 1)
        self.assertEqual(result["captions"][0]["output_time"]["end"], 610)
        self.assertEqual(result["coverage"]["segments"][0]["interval_annotation_coverage_s"], 10)

    def test_unknown_raw_alignment_stays_unaligned_even_if_caption_has_a_time(self):
        mapping, manifest = fixture([point(11612)])
        evidence = manifest["slots"][0]["evidence"]
        evidence.update(alignment="video_level_only_raw_timestamp_unknown",
                        text="A processed woodworking excerpt.")
        result = build_day_captions(mapping, manifest)
        self.assertEqual(result["captions"], [])
        self.assertEqual(result["unaligned_evidence"][0]["text"], evidence["text"])
        self.assertEqual(result["coverage"]["unaligned_segment_count"], 1)

    def test_missing_evidence_does_not_turn_a_label_into_a_caption(self):
        mapping, manifest = fixture([])
        manifest["slots"][0]["label"] = "산책"
        manifest["slots"][0]["evidence"] = {}
        result = build_day_captions(mapping, manifest)
        self.assertEqual(result["captions"], [])
        self.assertIsNone(result["unaligned_evidence"][0]["text"])

    def test_take_metadata_is_not_promoted_to_a_timed_caption(self):
        mapping, manifest = fixture([])
        manifest["slots"][0]["evidence"] = {
            "text": "Rock Climbing", "time": {"start": 11610, "end": 11620},
            "evidence_kind": "activity_metadata_only", "original": {"is_narrated": False},
        }
        result = build_day_captions(mapping, manifest)
        self.assertEqual(result["captions"], [])
        self.assertEqual(result["unaligned_evidence"][0]["evidence_kind"], "activity_metadata_only")

    def test_nested_youtube_bank_captions_are_used_instead_of_activity_label(self):
        mapping, manifest = fixture([])
        manifest["slots"][0]["source"]["dataset"] = "youtube"
        manifest["slots"][0]["evidence"] = {
            "text": "generic dishwashing activity",
            "original": {"captions": [interval(11610, 11620)]},
        }
        result = build_day_captions(mapping, manifest)
        self.assertEqual(result["captions"][0]["text"], "Original summary unchanged.")
        self.assertEqual(result["captions"][0]["provenance"]["location"],
                         "manifest.evidence.original.captions")

    def test_overlapping_summary_coverage_is_not_double_counted(self):
        mapping, manifest = fixture([interval(11610, 11618), interval(11615, 11620)])
        result = build_day_captions(mapping, manifest)
        self.assertEqual(result["coverage"]["segments"][0]["interval_annotation_coverage_s"], 10)

    def test_original_range_and_synthetic_clock_must_match_mapping(self):
        mapping, manifest = fixture([point(11612)])
        with self.subTest("source"):
            wrong = copy.deepcopy(mapping)
            wrong["segments"][0]["source_start_s"] += 1
            with self.assertRaisesRegex(ValueError, "Source range differs"):
                build_day_captions(wrong, manifest)
        with self.subTest("clock"):
            wrong = copy.deepcopy(mapping)
            wrong["segments"][0]["synthetic_start"] = "2025-07-19 10:40:00"
            with self.assertRaisesRegex(ValueError, "Synthetic date/time mismatch"):
                build_day_captions(wrong, manifest)

    def test_excluded_observation_is_rejected(self):
        mapping, manifest = fixture([point(11612)])
        manifest["slots"][0]["status"] = "removed_from_video_plan"
        with self.assertRaisesRegex(ValueError, "excluded"):
            build_day_captions(mapping, manifest)

    def test_per_video_ego4d_file_preserves_passes_and_original_point_times(self):
        mapping, manifest = fixture([])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotation = root / "caption-review" / "walk.json"
            annotation.parent.mkdir()
            annotation.write_text(json.dumps({
                "narration_pass_1": {"narrations": [{"timestamp_sec": 11612,
                    "narration_text": "Pass one", "annotation_uid": "one"}], "summaries": []},
                "narration_pass_2": {"narrations": [{"timestamp_sec": 11613,
                    "narration_text": "Pass two", "annotation_uid": "two"}], "summaries": []},
            }))
            manifest["slots"][0]["evidence"] = {"full_caption_file": "caption-review/walk.json"}
            result = build_day_captions(mapping, manifest, manifest_path=root / "manifest.json")
        self.assertEqual([r["text"] for r in result["captions"]], ["Pass one", "Pass two"])
        self.assertEqual([r["output_time"]["t"] for r in result["captions"]], [602, 603])
        self.assertEqual([r["provenance"]["pass_name"] for r in result["captions"]],
                         ["narration_pass_1", "narration_pass_2"])

    def test_egoexo_sibling_annotation_discovery_preserves_visibility(self):
        mapping, manifest = fixture([])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "egoexo4d"
            annotation = root / "synced_annotations" / "annotations" / "walk.jsonl"
            annotation.parent.mkdir(parents=True)
            row = point(11612)
            row["original"] = {"ego_visible": False, "unsure": True}
            annotation.write_text(json.dumps(row) + "\n")
            source = manifest["slots"][0]["source"]
            source.update(dataset="egoexo4d", media_path=str(root / "fps1/walk.mp4"))
            manifest["slots"][0]["evidence"] = {}
            result = build_day_captions(mapping, manifest)
        self.assertEqual(len(result["captions"]), 1)
        self.assertIs(result["captions"][0]["provenance"]["ego_visible"], False)


if __name__ == "__main__":
    unittest.main()
