"""Planner and clock regressions; no FFmpeg, NAS, or third-party packages required.

Run: python3 -m unittest discover -s scripts -p 'test_timestamp_timeline.py' -v
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import timestamp_timeline as timeline


def clip(clip_id="breakfast", *, day=1, start=27300, duration=600,
         input_path="media/breakfast.mp4", video_id="video-breakfast"):
    return {
        "id": clip_id,
        "day": day,
        "start_seconds": start,
        "expected_duration_s": duration,
        "input": input_path,
        "source": {"dataset": "ego4d", "video_id": video_id},
    }


def plan(*clips):
    return {
        "version": 1,
        "base_date": "2025-07-07",
        "timezone": "Asia/Seoul",
        "clips": list(clips),
    }


def manifest_slot():
    # This actual project pattern was moved to Day13 without renaming its ID.
    return {
        "slot_id": "d3-s7",
        "day": 13,
        "label": "산책",
        "status": "ready",
        "source_scope": "salvaged_interval",
        "source": {
            "dataset": "ego4d",
            "video_id": "park-walk",
            "start_s": 11610,
            "end_s": 12210,
            "media_path": "media/walk-1fps.mp4",
        },
        "proposed_observation": {
            "synthetic_start_s": 34800,
            "synthetic_end_s": 35400,
            "duration_s": 600,
        },
    }


class ClockTests(unittest.TestCase):
    def test_july_day1_and_august_day28_are_synthetic_local_dates(self):
        first = clip()
        last = clip(day=28)
        p = plan(first)
        self.assertEqual(timeline.wallclock(p, first), dt.datetime(2025, 7, 7, 7, 35))
        self.assertEqual(timeline.wallclock(p, last), dt.datetime(2025, 8, 3, 7, 35))

    def test_source_offset_is_provenance_not_an_extra_clock_offset(self):
        c = clip(day=13, start=34800)
        c["source"].update(start_s=11610, end_s=12210)
        p = plan(c)
        self.assertEqual(timeline.wallclock(p, c), dt.datetime(2025, 7, 19, 9, 40))
        stamp = dt.datetime.fromtimestamp(timeline.display_epoch(p, c), dt.timezone.utc)
        self.assertEqual(stamp.strftime("%Y-%m-%d %H:%M:%S"), "2025-07-19 09:40:00")

    def test_display_epoch_is_a_wallclock_encoding_not_a_real_utc_conversion(self):
        c = clip(start=8 * 3600)
        p = plan(c)
        expected = dt.datetime(2025, 7, 7, 8, tzinfo=dt.timezone.utc).timestamp()
        self.assertEqual(timeline.display_epoch(p, c), expected)
        p["timezone"] = "UTC"
        self.assertEqual(timeline.display_epoch(p, c), expected)

    def test_elapsed_seconds_cross_midnight_without_resetting_the_date(self):
        c = clip(day=25, start=23 * 3600 + 59 * 60 + 59, duration=3)
        p = plan(c)
        timeline.validate_plan(p)
        epoch = timeline.display_epoch(p, c)
        values = [
            dt.datetime.fromtimestamp(epoch + second, dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            for second in range(3)
        ]
        self.assertEqual(values, [
            "2025-07-31 23:59:59", "2025-08-01 00:00:00", "2025-08-01 00:00:01"
        ])


class ValidationTests(unittest.TestCase):
    def test_overlapping_observations_are_rejected_but_touching_edges_are_allowed(self):
        a = clip("a", start=3600, duration=600, input_path="a.mp4", video_id="a")
        b = clip("b", start=4199, input_path="b.mp4", video_id="b")
        with self.assertRaisesRegex(ValueError, "overlap"):
            timeline.validate_plan(plan(b, a))
        b["start_seconds"] = 4200
        timeline.validate_plan(plan(b, a))

    def test_overlap_detection_accounts_for_a_previous_days_midnight_crossing(self):
        a = clip("night", day=1, start=86390, duration=30, input_path="a.mp4", video_id="a")
        b = clip("next", day=2, start=10, input_path="b.mp4", video_id="b")
        with self.assertRaisesRegex(ValueError, "overlap"):
            timeline.validate_plan(plan(a, b))
        b["start_seconds"] = 20
        timeline.validate_plan(plan(a, b))

    def test_same_source_video_cannot_be_reused_on_another_day(self):
        a = clip("day1", input_path="first.mp4")
        b = clip("day2", day=2, input_path="another-segment.mp4")
        with self.assertRaisesRegex(ValueError, "Repeated source video"):
            timeline.validate_plan(plan(a, b))

    def test_source_ids_are_scoped_to_the_dataset(self):
        a = clip("a", input_path="a.mp4", video_id="same-name")
        b = clip("b", day=2, input_path="b.mp4", video_id="same-name")
        b["source"]["dataset"] = "youtube"
        timeline.validate_plan(plan(a, b))

    def test_same_input_is_rejected_even_without_source_metadata(self):
        a, b = clip("a"), clip("b", day=2)
        a.pop("source")
        b.pop("source")
        with self.assertRaisesRegex(ValueError, "Repeated input video"):
            timeline.validate_plan(plan(a, b))

    def test_duplicate_ids_and_unsafe_filename_ids_are_rejected(self):
        a = clip("same", input_path="a.mp4", video_id="a")
        b = clip("same", day=2, input_path="b.mp4", video_id="b")
        with self.assertRaisesRegex(ValueError, "Duplicate clip id"):
            timeline.validate_plan(plan(a, b))
        for unsafe in ("../escape", "bad/name", "", ".hidden", "has space"):
            with self.subTest(unsafe=unsafe), self.assertRaises(ValueError):
                timeline.validate_plan(plan(clip(unsafe)))

    def test_nonfinite_and_invalid_schedule_values_are_rejected(self):
        cases = [
            ("start_seconds", -1), ("start_seconds", 86400),
            ("start_seconds", float("nan")), ("start_seconds", True),
            ("expected_duration_s", 0), ("expected_duration_s", float("inf")),
            ("day", 0), ("day", 1.5), ("day", True),
        ]
        for key, value in cases:
            c = clip()
            c[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                timeline.validate_plan(plan(c))

    def test_input_fps_must_be_one_and_length_must_not_be_silently_changed(self):
        c = clip(duration=600.4)
        timeline.check_video({"fps": "1/1", "duration_s": 600}, c)
        with self.assertRaisesRegex(ValueError, "already be 1fps"):
            timeline.check_video({"fps": "30/1", "duration_s": 600.4}, c)
        with self.assertRaisesRegex(ValueError, "will not trim or stretch"):
            timeline.check_video({"fps": "1/1", "duration_s": 620}, c)


class ManifestImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "source" / "manifest.json"
        self.manifest.parent.mkdir()

    def import_slots(self, *slots):
        self.manifest.write_text(json.dumps({"schedule_sha256": "schedule-proof", "slots": list(slots)}))
        return timeline.import_manifest(self.manifest, "2025-07-07", "Asia/Seoul")

    def test_explicit_day_takes_precedence_over_slot_id_and_source_offset(self):
        imported = self.import_slots(manifest_slot())
        c = imported["clips"][0]
        self.assertEqual((c["id"], c["day"]), ("d3-s7", 13))
        self.assertEqual(c["start_seconds"], 34800)
        self.assertEqual(c["expected_duration_s"], 600)
        self.assertEqual(c["source"]["start_s"], 11610)
        self.assertEqual(timeline.wallclock(imported, c), dt.datetime(2025, 7, 19, 9, 40))
        self.assertEqual(imported["provenance"]["schedule_sha256"], "schedule-proof")

    def test_manifest_relative_path_is_anchored_to_manifest_not_future_plan(self):
        imported = self.import_slots(manifest_slot())
        c = imported["clips"][0]
        expected = (self.manifest.parent / "media/walk-1fps.mp4").resolve()
        self.assertEqual(Path(c["input"]), expected)
        relocated_plan = self.root / "delivery" / "plan.json"
        self.assertEqual(timeline.resolve_input(c["input"], relocated_plan, []), expected)

    def test_excluded_and_nonobserved_slots_are_omitted(self):
        imported = self.import_slots(
            {"slot_id": "prep", "status": "excluded_by_user"},
            {"slot_id": "removed", "status": "removed_from_video_plan", "proposed_observation": None},
            {"slot_id": "schedule-only", "status": "unmatched"},
            manifest_slot(),
        )
        self.assertEqual([c["id"] for c in imported["clips"]], ["d3-s7"])

    def test_excluded_slot_with_stale_observation_is_not_silently_rendered(self):
        slot = manifest_slot()
        slot["status"] = "removed_from_video_plan"
        with self.assertRaisesRegex(ValueError, "Excluded slot still has an observation"):
            self.import_slots(slot)

    def test_inconsistent_observation_and_source_lengths_are_rejected(self):
        for field, label in (("proposed_observation", "Observation"), ("source", "Source")):
            slot = manifest_slot()
            key = "synthetic_end_s" if field == "proposed_observation" else "end_s"
            slot[field][key] += 30
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, f"{label} length mismatch"):
                self.import_slots(slot)

    def test_missing_media_path_is_kept_pending_not_guessed(self):
        slot = manifest_slot()
        slot["source"]["media_path"] = None
        imported = self.import_slots(slot)
        self.assertIsNone(imported["clips"][0]["input"])


class PathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.plan_path = self.root / "delivery" / "plan.json"

    def test_relative_input_resolves_against_plan_directory(self):
        self.assertEqual(
            timeline.resolve_input("../media/video.mp4", self.plan_path, []),
            self.root / "media" / "video.mp4",
        )
        self.assertIsNone(timeline.resolve_input(None, self.plan_path, []))

    def test_path_map_respects_component_boundaries(self):
        old, new = self.root / "nas", self.root / "mounted"
        mappings = [(old, new)]
        self.assertEqual(timeline.resolve_input(str(old / "clips/a.mp4"), self.plan_path, mappings),
                         new / "clips/a.mp4")
        neighbor = self.root / "nas2" / "clips/a.mp4"
        self.assertEqual(timeline.resolve_input(str(neighbor), self.plan_path, mappings), neighbor)

    def test_longest_path_map_prefix_wins_independent_of_option_order(self):
        old, broad, specific = self.root / "nas", self.root / "broad", self.root / "specific"
        mappings = [(old, broad), (old / "videos", specific)]
        source = str(old / "videos" / "walking.mp4")
        self.assertEqual(timeline.resolve_input(source, self.plan_path, mappings), specific / "walking.mp4")
        self.assertEqual(timeline.resolve_input(source, self.plan_path, mappings[::-1]), specific / "walking.mp4")

    def test_runtime_rejects_aliases_that_resolve_to_the_same_input(self):
        # Textually different input paths must not bypass the no-reuse policy.
        from types import SimpleNamespace
        inputs = [str(self.root / "nas/a.mp4"), str(self.root / "mounted/a.mp4")]
        p = plan(clip("a", input_path=inputs[0], video_id="a"),
                 clip("b", day=2, input_path=inputs[1], video_id="b"))
        self.plan_path.parent.mkdir()
        self.plan_path.write_text(json.dumps(p))
        args = SimpleNamespace(
            plan=str(self.plan_path), output=str(self.root / "outputs"), font=None,
            path_map=[(self.root / "nas", self.root / "mounted")], only=[],
        )
        with mock.patch.object(timeline, "find_font", return_value=self.root / "font.ttf"), \
                self.assertRaisesRegex(ValueError, "multiple clips to the same video"):
            timeline.run_render(args)


if __name__ == "__main__":
    unittest.main()
