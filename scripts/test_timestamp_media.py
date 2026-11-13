"""Small FFmpeg integration checks for the portable timestamp renderer."""
import datetime as dt
import pathlib
import shutil
import subprocess
import tempfile
import unittest

import timestamp_media as media


class GeometryTests(unittest.TestCase):
    def test_only_old_mask_scales(self):
        self.assertEqual(media.geometry(1024, 1024)["old_mask_xyxy"], [0, 995, 200, 1024])
        self.assertEqual(media.geometry(1920, 1080)["old_mask_xyxy"], [0, 1049, 211, 1080])
        small = media.geometry(640, 360)
        self.assertEqual(small["old_mask_xyxy"], [0, 349, 71, 360])
        self.assertEqual((small["text_height_px"], small["text_x_px"], small["bottom_margin_px"]), (20, 10, 10))
        self.assertEqual(small["box_padding_px"], 7)

    def test_rejects_invalid_geometry(self):
        with self.assertRaises(ValueError):
            media.geometry(640, 360, {"original_mask_xyxy": [200, 995, 0, 1024]})
        with self.assertRaises(ValueError):
            media.geometry(640, 20)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe") and pathlib.Path(media.DEFAULT_FONT).is_file(),
                     "FFmpeg, ffprobe and DejaVuSansMono-Bold are required")
class RenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="timestamp-media-test-")
        cls.root = pathlib.Path(cls.directory.name)
        cls.calibration = media.calibrate_font(media.DEFAULT_FONT)
        cls.epoch = dt.datetime(2025, 7, 7, 23, 59, 59, tzinfo=dt.timezone.utc).timestamp()
        cls.source = cls.root / "input ' video.mp4"
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
            "color=c=white:s=640x360:r=1:d=3", "-f", "lavfi", "-i",
            "sine=frequency=400:duration=3", "-c:v", "libx264", "-threads", "1",
            "-c:a", "aac", str(cls.source),
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def filter(self):
        return media.make_filter(640, 360, self.epoch, media.DEFAULT_STYLE, media.DEFAULT_FONT, self.calibration)

    def gray_frames(self, filter_text, count=1, background="black"):
        return media._run([
            "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
            f"color=c={background}:s=640x360:r=1", "-vf", filter_text,
            "-frames:v", str(count), "-threads", "1", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
        ]).stdout

    def test_glyph_and_mask_pixels(self):
        frame = self.gray_frames(self.filter())
        left, top, right, bottom = media._ink_bbox(frame, 640)
        self.assertEqual((left, top, bottom), (10, 330, 350))
        self.assertEqual(bottom - top, 20)
        self.assertEqual(360 - bottom, 10)
        white = self.gray_frames(self.filter(), background="white")
        pixel = lambda x, y: white[y * 640 + x]
        # Old mask: scaled bounds [0,349,71,360]. No newer box covers x=0.
        self.assertEqual(pixel(0, 348), 255)
        self.assertEqual(pixel(0, 349), 0)
        self.assertEqual(pixel(70, 359), 0)
        self.assertEqual(pixel(71, 359), 255)
        # New box: x=3, top=323, giving exactly 7px to the visible glyph.
        self.assertEqual(pixel(2, 323), 255)
        self.assertEqual(pixel(3, 323), 0)
        self.assertEqual(pixel(100, 322), 255)
        self.assertEqual(pixel(100, 323), 0)
        self.assertEqual(pixel(100, 356), 0)
        self.assertEqual(pixel(100, 357), 255)

    def test_clock_advances_and_crosses_midnight(self):
        filt = self.filter()
        frames = self.gray_frames(filt, count=3)
        expected = ["2025-07-07 23:59:59", "2025-07-08 00:00:00", "2025-07-08 00:00:01"]
        expression = f"%{{pts\\:gmtime\\:{self.epoch:.6f}}}"
        self.assertIn(expression, filt)
        for index, timestamp in enumerate(expected):
            # Independent literal rendering avoids relying on gmtime for the oracle.
            literal = filt.replace(expression, timestamp.replace(":", "\\:"))
            frame = frames[index * 640 * 360:(index + 1) * 640 * 360]
            self.assertEqual(frame, self.gray_frames(literal))

    def test_render_preserves_media_and_handles_font_path_metacharacters(self):
        special_font = self.root / "font with ' comma, colon:.ttf"
        shutil.copyfile(media.DEFAULT_FONT, special_font)
        result = media.render(self.source, self.root / "output.mp4", self.epoch,
                              media.DEFAULT_STYLE, special_font, self.calibration)
        before, after = result["input"], result["output"]
        self.assertEqual(after["fps"], "1/1")
        self.assertEqual(after["frames"], 3)
        self.assertEqual((after["width"], after["height"]), (640, 360))
        self.assertEqual(after["duration_s"], before["duration_s"])
        self.assertTrue(after["audio"])

    def test_refuses_overwrite_and_missing_font(self):
        with self.assertRaises(ValueError):
            media.render(self.source, self.source, self.epoch, media.DEFAULT_STYLE,
                         media.DEFAULT_FONT, self.calibration)
        with self.assertRaises(FileNotFoundError):
            media.calibrate_font(self.root / "missing.ttf")

    def test_refuses_non_1fps_input(self):
        source = self.root / "30fps.mp4"
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
            "color=c=black:s=640x360:r=30:d=0.1", "-c:v", "libx264",
            "-threads", "1", str(source),
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with self.assertRaisesRegex(ValueError, "existing CFR 1fps"):
            media.render(source, self.root / "rejected.mp4", self.epoch,
                         media.DEFAULT_STYLE, media.DEFAULT_FONT, self.calibration)
        self.assertFalse((self.root / "rejected.mp4").exists())


if __name__ == "__main__":
    unittest.main()
