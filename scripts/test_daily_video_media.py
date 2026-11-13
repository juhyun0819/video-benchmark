"""End-to-end checks for aspect normalization, clock placement and concat."""
import datetime as dt
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import daily_video_media as daily
import timestamp_media as stamp


class GeometryTests(unittest.TestCase):
    def test_preserves_display_aspect_and_uses_square_pixels(self):
        landscape = daily.normalized_geometry({"width": 640, "height": 360}, {"sample_aspect_ratio": "1:1"})
        self.assertEqual((landscape["scaled_width"], landscape["scaled_height"], landscape["pad_y"]),
                         (1024, 576, 224))
        portrait = daily.normalized_geometry({"width": 360, "height": 640}, {"sample_aspect_ratio": "1:1"})
        self.assertEqual((portrait["scaled_width"], portrait["scaled_height"], portrait["pad_x"]),
                         (576, 1024, 224))
        anamorphic = daily.normalized_geometry({"width": 640, "height": 640}, {"sample_aspect_ratio": "2:1"})
        self.assertEqual((anamorphic["scaled_width"], anamorphic["scaled_height"]), (1024, 512))

    def test_rejects_invalid_size(self):
        with self.assertRaises(ValueError):
            daily.normalized_geometry({"width": 640, "height": 360}, {}, size=1023)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe") and Path(stamp.DEFAULT_FONT).is_file(),
                     "FFmpeg, ffprobe and DejaVuSansMono-Bold are required")
class RenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="daily-video-test-")
        cls.root = Path(cls.directory.name)
        cls.calibration = stamp.calibrate_font(stamp.DEFAULT_FONT)
        cls.epoch = dt.datetime(2025, 7, 7, 8, 0, 0, tzinfo=dt.timezone.utc).timestamp()
        cls.sources, cls.clips, cls.reports = [], [], []
        for index, dimensions in enumerate(("640x360", "360x640")):
            source = cls.root / f"source-{index}.mp4"
            clip = cls.root / f"clip '{index}.mp4"
            subprocess.run([
                "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                f"color=c=white:s={dimensions}:r=1:d=2", "-f", "lavfi", "-i",
                "sine=frequency=400:duration=2", "-c:v", "libx264", "-threads", "1",
                "-c:a", "aac", str(source),
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            before_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            report = daily.render_clip(source, clip, cls.epoch + 4 * 3600 * index,
                                       stamp.DEFAULT_STYLE, stamp.DEFAULT_FONT, cls.calibration)
            assert hashlib.sha256(source.read_bytes()).hexdigest() == before_hash
            cls.sources.append(source)
            cls.clips.append(clip)
            cls.reports.append(report)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_normalized_files_keep_all_frames_and_duration(self):
        for report in self.reports:
            self.assertEqual((report["output"]["width"], report["output"]["height"]), (1024, 1024))
            self.assertEqual(report["output"]["frames"], report["input"]["frames"])
            self.assertEqual(report["output"]["duration_s"], report["input"]["duration_s"])
            self.assertEqual(report["output"]["fps"], "1/1")
            self.assertTrue(report["input"]["audio"])
            self.assertFalse(report["output"]["audio"])
            self.assertEqual(report["output_codec"]["sample_aspect_ratio"], "1:1")
            self.assertEqual(report["output_codec"]["time_base"], "1/16384")

    def test_old_mask_precedes_resize_and_new_clock_stays_20px(self):
        metadata = stamp.probe(self.sources[0])
        codec = daily._codec_info(self.sources[0])
        filt = daily.make_normalized_filter(metadata, codec, self.epoch,
                                            stamp.DEFAULT_STYLE, stamp.DEFAULT_FONT, self.calibration)
        frame = stamp._run([
            "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
            "color=c=white:s=640x360:r=1", "-vf", filt, "-frames:v", "1",
            "-threads", "1", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
        ]).stdout
        self.assertEqual(len(frame), 1024 * 1024)
        pixel = lambda x, y: frame[y * 1024 + x]
        self.assertEqual(pixel(512, 100), 0)  # top black pad
        self.assertEqual(pixel(512, 300), 255)  # original image
        self.assertLess(pixel(20, 790), 3)  # resized OLD mask within image
        self.assertGreater(pixel(140, 790), 250)  # outside old mask
        left, top, right, bottom = stamp._ink_bbox(frame[980 * 1024:], 1024)
        self.assertEqual((left, top + 980, bottom + 980), (10, 994, 1014))
        self.assertEqual(bottom - top, 20)
        self.assertEqual(1024 - (bottom + 980), 10)

    @staticmethod
    def decoded_gray(path):
        return stamp._run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
                           "-map", "0:v:0", "-an", "-threads", "1", "-f", "rawvideo",
                           "-pix_fmt", "gray", "pipe:1"]).stdout

    def test_concat_preserves_every_frame_in_order_and_clock_jump(self):
        destination = self.root / "day01.mp4"
        result = daily.concat_clips(self.clips, destination)
        self.assertEqual(result["method"], "concat_demuxer_stream_copy")
        self.assertEqual(result["output"]["frames"], 4)
        self.assertEqual(result["output"]["duration_s"], 4)
        self.assertEqual(result["output"]["fps"], "1/1")
        first, second = (self.decoded_gray(path) for path in self.clips)
        self.assertEqual(self.decoded_gray(destination), first + second)
        # Sources were stamped at 08:00 and12:00, so the burned clock pixels
        # change across the cut instead of filling four hours of absent footage.
        row_start, row_end = 990 * 1024, 1015 * 1024
        self.assertNotEqual(first[row_start:row_end], second[row_start:row_end])

    def test_rejects_unprocessed_clips_and_overwrite(self):
        with self.assertRaisesRegex(ValueError, "normalized"):
            daily.concat_clips([self.sources[0]], self.root / "invalid.mp4")
        with self.assertRaises(FileExistsError):
            daily.render_clip(self.sources[0], self.clips[0], self.epoch,
                              stamp.DEFAULT_STYLE, stamp.DEFAULT_FONT, self.calibration)
        with self.assertRaises(ValueError):
            daily.concat_clips([], self.root / "empty.mp4")


if __name__ == "__main__":
    unittest.main()
