"""FFmpeg helpers for timestamping existing 1 fps clips without trimming them.

Only Python's standard library is required. ``wallclock_epoch`` is the UTC
encoding of a *displayed wall clock*, not the real UTC instant of that clock.
Using FFmpeg's gmtime formatter makes rendering independent of the host TZ.
"""
from __future__ import annotations

import datetime as dt
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time


DEFAULT_STYLE = {
    "reference_height_px": 1024,
    "original_mask_xyxy": [0, 995, 200, 1024],
    "text_height_px": 20,
    "text_x_px": 10,
    "bottom_margin_px": 10,
}
DEFAULT_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"


def _run(args: list[str], **kwargs):
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace")[-5000:]
        raise RuntimeError(f"{args[0]} failed ({result.returncode}): {detail}")
    return result


def probe(path: str | Path) -> dict:
    """Read metadata for the first video stream; no full video decoding."""
    data = json.loads(_run([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)
    ], timeout=60).stdout)
    videos = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    if not videos:
        raise ValueError(f"No video stream: {path}")
    video = videos[0]
    duration = video.get("duration")
    if duration in (None, "N/A"):
        duration = data.get("format", {}).get("duration")
    if duration in (None, "N/A") or not math.isfinite(float(duration)) or float(duration) <= 0:
        raise ValueError(f"No positive video duration: {path}")
    frames = video.get("nb_frames")
    rotation = float(video.get("tags", {}).get("rotate", 0))
    for side in video.get("side_data_list", []):
        rotation = float(side.get("rotation", rotation))
    return {
        "width": int(video["width"]), "height": int(video["height"]),
        "duration_s": float(duration), "fps": video.get("avg_frame_rate", "0/0"),
        "nominal_fps": video.get("r_frame_rate", "0/0"),
        "frames": int(frames) if frames not in (None, "N/A") else None,
        "start_time": float(video.get("start_time", 0)),
        "audio": any(s.get("codec_type") == "audio" for s in data.get("streams", [])),
        "rotation_degrees": rotation,
    }


def _safe_font(font_path: str | Path) -> str:
    """Public filter builder accepts safe paths; render/calibration copy any path."""
    value = str(font_path)
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", value):
        raise ValueError("For make_filter, copy the font to a path without filter metacharacters")
    return value


def _gray_frame(filter_text: str, width: int, height: int) -> bytes:
    return _run([
        "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
        f"color=c=black:s={width}x{height}:r=1", "-vf", filter_text,
        "-frames:v", "1", "-threads", "1", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
    ], timeout=30).stdout


def _ink_bbox(frame: bytes, width: int, x0: int = 0, x1: int | None = None) -> tuple[int, int, int, int]:
    x1 = width if x1 is None else x1
    left, top, right, bottom = width, len(frame) // width, -1, -1
    for y in range(len(frame) // width):
        for x in range(x0, x1):
            if frame[y * width + x] > 0:
                left, top, right, bottom = min(left, x), min(top, y), max(right, x), max(bottom, y)
    if right < 0:
        raise ValueError("Font calibration produced no visible text")
    return left, top, right + 1, bottom + 1


def calibrate_font(font_path: str | Path, target_height: int = 20) -> dict:
    """Find a nominal fontsize with the requested actual raster digit height.

    Draws through the installed FFmpeg/FreeType, so a changed font or rendering
    environment cannot silently turn fontsize=20 into only 15 visible pixels.
    """
    if not isinstance(target_height, int) or not 8 <= target_height <= 128:
        raise ValueError("target_height must be an integer from 8 to 128")
    font = Path(font_path)
    if not font.is_file():
        raise FileNotFoundError(f"Font missing: {font}")
    with tempfile.TemporaryDirectory(prefix="timestamp-font-") as directory:
        safe = Path(directory) / "font.ttf"
        shutil.copyfile(font, safe)
        guess = round(target_height * 1.35)
        sizes = sorted(range(max(1, target_height // 2), target_height * 3 + 1),
                       key=lambda n: (abs(n - guess), n))
        width, height = 4096, 512
        for fontsize in sizes:
            filt = (f"drawtext=fontfile={safe}:text=0123456789:"
                    f"fontsize={fontsize}:fontcolor=white:x=16:y=16")
            frame = _gray_frame(filt, width, height)
            bbox = _ink_bbox(frame, width)
            if bbox[3] - bbox[1] == target_height:
                break
        else:
            raise ValueError(f"This font cannot render digits at exactly {target_height}px")

        cell = max(64, fontsize * 2)
        width, height = cell * 10, max(64, fontsize * 3)
        # Even dimensions are required by the lavfi color source's yuv420 format.
        width += width % 2
        height += height % 2
        filt = ",".join(
            f"drawtext=fontfile={safe}:text={n}:fontsize={fontsize}:fontcolor=white:x={n*cell+8}:y=8"
            for n in range(10)
        )
        frame = _gray_frame(filt, width, height)
        bearings, heights = {}, {}
        for n in range(10):
            bounds = _ink_bbox(frame, width, n * cell, (n + 1) * cell)
            bearings[str(n)] = bounds[0] - (n * cell + 8)
            heights[str(n)] = bounds[3] - bounds[1]
        if any(h != target_height for h in heights.values()):
            raise ValueError("This font has unequal digit heights; choose a monospaced numeric font")
        # A fixed box uses the widest ending digit, avoiding a visibly resizing
        # background as seconds tick. Up to one extra pixel remains for narrower
        # digits in the default font; padding is never smaller than requested.
        widest_right = 0
        for digit in range(10):
            sample = f"2000-01-02 03\\:45\\:5{digit}"
            filt = (f"drawtext=fontfile={safe}:text='{sample}':fontsize={fontsize}:"
                    "fontcolor=white:x=16:y=16")
            bounds = _ink_bbox(_gray_frame(filt, 4096, 512), 4096)
            widest_right = max(widest_right, bounds[2] - 16)
    return {
        "fontsize": fontsize, "ink_height": target_height,
        "left_bearing": bearings["2"], "digit_left_bearings": bearings,
        "digit_heights": heights, "font_sha256": hashlib.sha256(font.read_bytes()).hexdigest(),
        "timestamp_max_right_px": widest_right,
    }


def geometry(width: int, height: int, style: dict | None = None) -> dict:
    """Scale only the old mask by input height / reference height."""
    values = {**DEFAULT_STYLE, **(style or {})}
    ref = values["reference_height_px"]
    rect = values["original_mask_xyxy"]
    if not isinstance(ref, (int, float)) or not math.isfinite(ref) or ref <= 0:
        raise ValueError("reference_height_px must be positive")
    if len(rect) != 4 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in rect):
        raise ValueError("original_mask_xyxy must contain four finite coordinates")
    if not (0 <= rect[0] < rect[2] and 0 <= rect[1] < rect[3] <= ref):
        raise ValueError("Invalid old timestamp mask rectangle")
    if min(width, height) <= 0:
        raise ValueError("Invalid video dimensions")
    scale = height / ref
    x0, y0 = math.floor(rect[0] * scale), math.floor(rect[1] * scale)
    x1, y1 = min(width, math.ceil(rect[2] * scale)), min(height, math.ceil(rect[3] * scale))
    if x0 >= x1 or y0 >= y1:
        raise ValueError("Mask falls outside the image")
    text_height, text_x, bottom = (values[key] for key in ("text_height_px", "text_x_px", "bottom_margin_px"))
    padding = values.get("box_padding_px", math.ceil(max(3, text_height / 3)))
    for value in (text_height, text_x, bottom, padding):
        if not isinstance(value, int) or value < 0:
            raise ValueError("Text dimensions and padding must be nonnegative integer pixels")
    if text_height <= 0 or text_height + bottom + padding > height or text_x < padding or bottom < padding:
        raise ValueError("Timestamp or its box does not fit within the image")
    return {"old_mask_xyxy": [x0, y0, x1, y1], "scale": scale,
            "text_height_px": text_height, "text_x_px": text_x,
            "text_y_px": height - bottom - text_height, "bottom_margin_px": bottom,
            "box_padding_px": padding}


def make_filter(width: int, height: int, wallclock_epoch: float, style: dict,
                font_path: str | Path, calibration: dict) -> str:
    """Build a no-trim filter. Input PTS zero is this clip's synthetic start."""
    if not isinstance(wallclock_epoch, (int, float)) or not math.isfinite(wallclock_epoch):
        raise ValueError("wallclock_epoch must be finite")
    layout = geometry(width, height, style)
    if calibration["ink_height"] != layout["text_height_px"]:
        raise ValueError("Font calibration and requested text height differ")
    first_digit = dt.datetime.fromtimestamp(wallclock_epoch, dt.timezone.utc).strftime("%Y")[0]
    bearing = calibration.get("digit_left_bearings", {}).get(first_digit, calibration["left_bearing"])
    x = layout["text_x_px"] - bearing
    x0, y0, x1, y1 = layout["old_mask_xyxy"]
    pad = layout["box_padding_px"]
    box_x, box_y = layout["text_x_px"] - pad, layout["text_y_px"] - pad
    box_width = calibration["timestamp_max_right_px"] - bearing + 2 * pad
    box_height = layout["text_height_px"] + 2 * pad
    if box_x + box_width > width:
        raise ValueError("Timestamp and its padding exceed the frame width")
    # gmtime's default format is YYYY-MM-DD HH:MM:SS. No host-timezone lookup.
    text = f"%{{pts\\:gmtime\\:{wallclock_epoch:.6f}}}"
    return (
        "setpts=PTS-STARTPTS,"
        f"drawbox=x={x0}:y={y0}:w={x1-x0}:h={y1-y0}:color=black:t=fill,"
        f"drawbox=x={box_x}:y={box_y}:w={box_width}:h={box_height}:color=black:t=fill,"
        f"drawtext=fontfile={_safe_font(font_path)}:text='{text}':"
        f"fontsize={calibration['fontsize']}:fontcolor=white:"
        f"x={x}:y=h-th-{layout['bottom_margin_px']}"
    )


def render(input_path: str | Path, output_path: str | Path, wallclock_epoch: float,
           style: dict, font_path: str | Path, calibration: dict,
           threads: int = 2, preset: str = "veryfast", crf: int = 18) -> dict:
    """Render a caller-owned temporary MP4; caller handles locks/atomic publish.

    Never changes source media and never trims or loops. Existing audio, when
    present, is copied without transcoding. Inputs must already be CFR 1 fps.
    """
    source, destination = Path(input_path), Path(output_path)
    if source.resolve() == destination.resolve():
        raise ValueError("Output must differ from the source")
    if destination.exists():
        raise FileExistsError(destination)
    if not isinstance(threads, int) or not 1 <= threads <= 128:
        raise ValueError("threads must be 1..128")
    if preset not in {"ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"}:
        raise ValueError("Invalid x264 preset")
    if not isinstance(crf, int) or not 0 <= crf <= 51:
        raise ValueError("crf must be 0..51")
    before = probe(source)
    try:
        is_one = Fraction(before["fps"]) == 1 and Fraction(before["nominal_fps"]) == 1
    except (ValueError, ZeroDivisionError):
        is_one = False
    if not is_one:
        raise ValueError(f"Expected an existing CFR 1fps file, got {before['fps']}")
    if before["rotation_degrees"] % 360:
        raise ValueError("Normalize rotation before timestamp rendering")
    if before["width"] % 2 or before["height"] % 2:
        raise ValueError("H.264 yuv420p requires even input dimensions")
    if hashlib.sha256(Path(font_path).read_bytes()).hexdigest() != calibration["font_sha256"]:
        raise ValueError("Font changed since calibration")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="timestamp-font-") as directory:
        safe = Path(directory) / "font.ttf"
        shutil.copyfile(font_path, safe)
        filt = make_filter(before["width"], before["height"], wallclock_epoch, style, safe, calibration)
        args = ["ffmpeg", "-nostdin", "-v", "error", "-n", "-threads", str(threads),
                "-i", str(source), "-map", "0:v:0", "-map", "0:a?", "-map_metadata", "-1",
                "-map_chapters", "-1", "-vf", filt, "-vsync", "0", "-c:v", "libx264",
                "-preset", preset, "-crf", str(crf), "-threads", str(threads),
                "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(destination)]
        started = time.monotonic()
        try:
            _run(args)
            elapsed = time.monotonic() - started
            after = probe(destination)
            if (after["width"], after["height"]) != (before["width"], before["height"]):
                raise ValueError("Output dimensions changed")
            if Fraction(after["fps"]) != 1 or abs(after["duration_s"] - before["duration_s"]) > 0.05:
                raise ValueError("Output frame rate or duration changed")
            if before["frames"] is not None and after["frames"] != before["frames"]:
                raise ValueError("Output frame count changed")
            if before["audio"] != after["audio"]:
                raise ValueError("Output audio availability changed")
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
    return {"input": before, "output": after, "elapsed_seconds": elapsed,
            "filter": filt.replace(str(safe), "<fontfile>"),
            "geometry": geometry(before["width"], before["height"], style),
            "font_calibration": calibration, "preset": preset, "crf": crf, "threads": threads}
