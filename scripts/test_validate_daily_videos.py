import copy
import json
from pathlib import Path
import tempfile
import unittest

import validate_daily_videos as validator


def caption_fixture():
    mapping = {"day": 1, "date": "2025-07-07", "timezone": "Asia/Seoul", "duration_s": 4,
               "segments": [
                   {"slot_id": "a", "output_start_s": 0, "output_end_s": 2,
                    "source_start_s": 100, "source_end_s": 102, "synthetic_start": "2025-07-07T08:00:00"},
                   {"slot_id": "b", "output_start_s": 2, "output_end_s": 4,
                    "source_start_s": 0, "source_end_s": 2, "synthetic_start": "2025-07-07T12:00:00"},
               ]}
    document = {"day": 1, "date": "2025-07-07", "timezone": "Asia/Seoul", "captions": [
        {"id": "a-caption-1", "slot_id": "a", "kind": "point",
         "source_time": {"kind": "point", "t": 101},
         "selected_source_time": {"kind": "point", "t": 101},
         "clip_time": {"kind": "point", "t": 1}, "output_time": {"kind": "point", "t": 1},
         "synthetic_time": {"kind": "point", "t": "2025-07-07 08:00:01"}}
    ], "unaligned_evidence": [{"slot_id": "b"}], "coverage": {
        "caption_count": 1, "unaligned_segment_count": 1, "segments": [
            {"slot_id": "a", "status": "aligned_annotations_available", "point_count": 1, "interval_count": 0},
            {"slot_id": "b", "status": "no_aligned_annotations", "point_count": 0, "interval_count": 0},
        ]}}
    return mapping, document


class PacketTests(unittest.TestCase):
    def test_checks_every_presentation_timestamp(self):
        packets = [{"pts_time": str(t), "duration_time": "1.0"} for t in (0, 2, 1)]
        self.assertTrue(validator.check_packets(packets, 3)["all_pts_contiguous_1fps"])
        packets[1]["pts_time"] = "1"
        with self.assertRaisesRegex(ValueError, "irregular PTS"):
            validator.check_packets(packets, 3)

    def test_rejects_packet_count_and_duration_errors(self):
        packets = [{"pts_time": "0", "duration_time": ".5"}]
        with self.assertRaisesRegex(ValueError, "packet duration"):
            validator.check_packets(packets, 1)
        with self.assertRaisesRegex(ValueError, "packet count"):
            validator.check_packets(packets, 2)


class CaptionTests(unittest.TestCase):
    def test_valid_maps_preserve_explicit_caption_gaps(self):
        mapping, document = caption_fixture()
        result = validator.check_captions(document, mapping)
        self.assertTrue(result["time_maps_valid"])
        self.assertFalse(result["all_segments_have_aligned_captions"])
        self.assertEqual(result["unaligned_slot_ids"], ["b"])

    def test_cannot_claim_missing_captions_are_aligned(self):
        mapping, document = caption_fixture()
        document["unaligned_evidence"] = []
        with self.assertRaisesRegex(ValueError, "incorrectly flagged"):
            validator.check_captions(document, mapping)

    def test_rejects_source_offset_and_wrong_wall_clock(self):
        mapping, document = caption_fixture()
        bad = copy.deepcopy(document)
        bad["captions"][0]["selected_source_time"]["t"] = 1
        with self.assertRaisesRegex(ValueError, "source offset"):
            validator.check_captions(bad, mapping)
        bad = copy.deepcopy(document)
        bad["captions"][0]["synthetic_time"]["t"] = "2025-07-07 08:01:41"
        with self.assertRaisesRegex(ValueError, "synthetic caption clock"):
            validator.check_captions(bad, mapping)

    def test_rejects_caption_after_actual_video_end(self):
        mapping, document = caption_fixture()
        document["captions"][0]["clip_time"]["t"] = 3
        with self.assertRaisesRegex(ValueError, "actual clip duration"):
            validator.check_captions(document, mapping)


class PendingTests(unittest.TestCase):
    def test_missing_days_are_pending_and_exit_three(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = {"base_date": "2025-07-07", "timezone": "Asia/Seoul", "clips": [
                {"id": "a", "day": 1, "source": {"dataset": "ego4d", "video_id": "one"}},
                {"id": "b", "day": 2, "source": {"dataset": "ego4d", "video_id": "two"}},
            ]}
            (root / "plan.json").write_text(json.dumps(plan))
            result, code = validator.run(root)
            self.assertEqual(code, 3)
            self.assertEqual(result["pending_days"], 2)
            self.assertFalse(result["all_requested_media_validated"])
            self.assertFalse(result["all_requested_segments_have_aligned_captions"])

    def test_repeated_source_is_not_hidden_by_missing_days(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = {"base_date": "2025-07-07", "timezone": "Asia/Seoul", "clips": [
                {"id": "a", "day": 1, "source": {"dataset": "ego4d", "video_id": "same"}},
                {"id": "b", "day": 2, "source": {"dataset": "ego4d", "video_id": "same"}},
            ]}
            (root / "plan.json").write_text(json.dumps(plan))
            result, code = validator.run(root)
            self.assertEqual(code, 1)
            self.assertEqual(len(result["duplicate_sources"]), 1)


if __name__ == "__main__":
    unittest.main()
