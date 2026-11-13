"""Normalize selected 1fps clips, stamp their schedule clocks, and concatenate.

Unlike timestamp_media.render, this module produces square, silent benchmark
clips. The original timestamp mask is applied BEFORE resizing; the synthetic
20px timestamp is applied AFTER resizing and padding. No media interval is cut.
The caller owns cache keys, inter-process locks and atomic publication.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import time

import timestamp_media as stamp


def _codec_info(path: str | Path) -> dict:
    data = json.loads(stamp._run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams",
        "-show_data_hash", "sha256", "-of", "json", str(path),
    ], timeout=60).stdout)
    stream = data["streams"][0]
    keys = ("codec_name", "profile", "level", "width", "height", "pix_fmt", "field_order",
            "sample_aspect_ratio", "time_base", "avg_frame_rate", "r_frame_rate",
            "extradata_hash", "color_range", "color_space", "color_transfer", "color_primaries")
    return {key: stream.get(key) for key in keys}


def _check_input(metadata: dict):
    try:
        one_fps = Fraction(metadata["fps"]) == 1 and Fraction(metadata["nominal_fps"]) == 1
    except (ValueError, ZeroDivisionError):
        one_fps = False
    if not one_fps:
        raise ValueError("Input must already be CFR 1fps; this renderer does not resample frames")
    if metadata["rotation_degrees"] % 360:
        raise ValueError("Normalize rotation before rendering daily clips")


def _display_aspect(metadata: dict, codec: dict) -> float:
    sar = codec.get("sample_aspect_ratio")
    try:
        ratio = Fraction(sar.replace(":", "/")) if sar not in (None, "N/A", "0:1") else Fraction(1)
    except (ValueError, ZeroDivisionError):
        raise ValueError("Invalid sample aspect ratio") from None
    if ratio <= 0:
        raise ValueError("Sample aspect ratio must be positive")
    return metadata["width"] * float(ratio) / metadata["height"]


def normalized_geometry(metadata: dict, codec: dict, size: int = 1024) -> dict:
    if not isinstance(size, int) or size < 320 or size % 2:
        raise ValueError("Canvas size must be an even integer >= 320")
    aspect = _display_aspect(metadata, codec)
    if aspect >= 1:
        scaled_width, scaled_height = size, max(2, math.floor(size / aspect / 2) * 2)
    else:
        scaled_width, scaled_height = max(2, math.floor(size * aspect / 2) * 2), size
    # Explicit chroma-aligned offsets avoid FFmpeg silently rounding odd offsets.
    pad_x = ((size - scaled_width) // 2) // 2 * 2
    pad_y = ((size - scaled_height) // 2) // 2 * 2
    return {"canvas_size": size, "scaled_width": scaled_width, "scaled_height": scaled_height,
            "pad_x": pad_x, "pad_y": pad_y, "input_display_aspect": aspect}


def make_normalized_filter(metadata: dict, codec: dict, wallclock_epoch: float,
                           style: dict, font_path: str | Path, calibration: dict,
                           size: int = 1024) -> str:
    """Build source-mask → aspect-fit/pad → final-size timestamp filter."""
    original = stamp.geometry(metadata["width"], metadata["height"], style)
    x0, y0, x1, y1 = original["old_mask_xyxy"]
    layout = normalized_geometry(metadata, codec, size)
    timestamp = stamp.make_filter(size, size, wallclock_epoch, style, font_path, calibration)
    # Reuse the calibrated new-text box/drawtext only. The first two filters in
    # stamp.make_filter reset PTS and mask original pixels; they belong before
    # scale here, not at the bottom of the final padded canvas.
    pieces = timestamp.split(",", 2)
    if len(pieces) != 3 or pieces[0] != "setpts=PTS-STARTPTS" or not pieces[1].startswith("drawbox="):
        raise RuntimeError("Unexpected timestamp filter layout")
    return (
        "setpts=PTS-STARTPTS,"
        f"drawbox=x={x0}:y={y0}:w={x1-x0}:h={y1-y0}:color=black:t=fill,"
        f"scale={layout['scaled_width']}:{layout['scaled_height']}:flags=bicubic:"
        "in_color_matrix=auto:out_color_matrix=bt709:out_range=tv,"
        f"pad={size}:{size}:{layout['pad_x']}:{layout['pad_y']}:color=black,setsar=1,"
        f"{pieces[2]}"
    )


def render_clip(input_path: str | Path, output_path: str | Path, wallclock_epoch: float,
                style: dict, font_path: str | Path, calibration: dict,
                threads: int = 4, preset: str = "veryfast", crf: int = 18,
                size: int = 1024) -> dict:
    """Encode one complete selected input to a square, silent, stamped MP4."""
    source, destination = Path(input_path), Path(output_path)
    if source.resolve() == destination.resolve():
        raise ValueError("Output must differ from input")
    if destination.exists():
        raise FileExistsError(destination)
    if not isinstance(threads, int) or not 1 <= threads <= 128:
        raise ValueError("threads must be 1..128")
    if preset not in {"ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"}:
        raise ValueError("Invalid x264 preset")
    if not isinstance(crf, int) or not 0 <= crf <= 51:
        raise ValueError("crf must be 0..51")
    before = stamp.probe(source)
    _check_input(before)
    codec = _codec_info(source)
    if hashlib.sha256(Path(font_path).read_bytes()).hexdigest() != calibration["font_sha256"]:
        raise ValueError("Font changed since calibration")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="daily-video-font-") as directory:
        safe = Path(directory) / "font.ttf"
        shutil.copyfile(font_path, safe)
        filt = make_normalized_filter(before, codec, wallclock_epoch, style, safe, calibration, size)
        args = ["ffmpeg", "-nostdin", "-v", "error", "-n", "-threads", str(threads),
                "-i", str(source), "-map", "0:v:0", "-an", "-sn", "-dn", "-map_metadata", "-1",
                "-map_chapters", "-1", "-vf", filt, "-vsync", "0", "-c:v", "libx264",
                "-preset", preset, "-crf", str(crf), "-threads", str(threads),
                "-pix_fmt", "yuv420p", "-profile:v", "high", "-level:v", "4.0",
                "-bf", "0", "-g", "60", "-keyint_min", "60", "-sc_threshold", "0",
                "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709",
                "-color_trc", "bt709", "-video_track_timescale", "16384",
                "-movflags", "+faststart", str(destination)]
        started = time.monotonic()
        try:
            stamp._run(args)
            elapsed = time.monotonic() - started
            after = stamp.probe(destination)
            output_codec = _codec_info(destination)
            _check_input(after)
            if (after["width"], after["height"]) != (size, size) or after["audio"]:
                raise ValueError("Output must be a silent square video")
            if abs(after["duration_s"] - before["duration_s"]) > 0.05:
                raise ValueError("Output duration changed")
            if before["frames"] is not None and before["frames"] != after["frames"]:
                raise ValueError("Output frame count changed")
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
    return {"input": before, "output": after, "output_codec": output_codec,
            "elapsed_seconds": elapsed, "filter": filt.replace(str(safe), "<fontfile>"),
            "normalization": normalized_geometry(before, codec, size),
            "source_mask": stamp.geometry(before["width"], before["height"], style),
            "final_timestamp": stamp.geometry(size, size, style), "font_calibration": calibration,
            "preset": preset, "crf": crf, "threads": threads, "audio_policy": "omit",
            "encoding": {"gop": 60, "b_frames": 0, "track_timescale": 16384}}


def _concat_line(path: Path) -> str:
    text = str(path.resolve())
    if "\n" in text or "\r" in text or "\x00" in text:
        raise ValueError("Concat paths cannot contain line breaks or NUL")
    return "file '" + text.replace("'", "'\\''") + "'"


def concat_clips(paths: list[str | Path], output_path: str | Path, size: int = 1024) -> dict:
    """Copy-concatenate compatible normalized clips with no re-encoding/gaps."""
    if not paths:
        raise ValueError("At least one clip is required")
    inputs, destination = [Path(p) for p in paths], Path(output_path)
    if destination.exists():
        raise FileExistsError(destination)
    if any(p.resolve() == destination.resolve() for p in inputs):
        raise ValueError("Output must differ from every input")
    metadata, codecs = [], []
    for source in inputs:
        info, codec = stamp.probe(source), _codec_info(source)
        _check_input(info)
        if (info["width"], info["height"]) != (size, size) or info["audio"]:
            raise ValueError(f"Clip was not normalized to silent {size}x{size}: {source}")
        if codec["codec_name"] != "h264" or codec["pix_fmt"] != "yuv420p":
            raise ValueError(f"Clip must use H.264 yuv420p: {source}")
        if codecs and codec != codecs[0]:
            differing = [k for k, v in codec.items() if codecs[0].get(k) != v]
            raise ValueError(f"Incompatible stream-copy clips ({', '.join(differing)}): {source}")
        metadata.append(info)
        codecs.append(codec)
    expected_duration = sum(info["duration_s"] for info in metadata)
    expected_frames = (sum(info["frames"] for info in metadata)
                       if all(info["frames"] is not None for info in metadata) else None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="daily-video-concat-") as directory:
        listing = Path(directory) / "inputs.ffconcat"
        lines = ["ffconcat version 1.0"]
        for source, info in zip(inputs, metadata):
            lines.extend([_concat_line(source), f"duration {info['duration_s']:.9f}"])
        listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
        started = time.monotonic()
        try:
            stamp._run(["ffmpeg", "-nostdin", "-v", "error", "-n", "-f", "concat", "-safe", "0",
                        "-i", str(listing), "-map", "0:v:0", "-an", "-sn", "-dn", "-c:v", "copy",
                        "-map_metadata", "-1", "-map_chapters", "-1", "-video_track_timescale", "16384",
                        "-movflags", "+faststart", str(destination)])
            elapsed = time.monotonic() - started
            after = stamp.probe(destination)
            _check_input(after)
            if abs(after["duration_s"] - expected_duration) > 0.05:
                raise ValueError("Concatenation duration differs from the sum of clips")
            if expected_frames is not None and after["frames"] != expected_frames:
                raise ValueError("Concatenation frame count differs from the sum of clips")
            if (after["width"], after["height"]) != (size, size) or after["audio"]:
                raise ValueError("Concatenation output format changed")
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
    return {"input_count": len(inputs), "inputs": metadata, "output": after,
            "expected_duration_s": expected_duration, "expected_frames": expected_frames,
            "elapsed_seconds": elapsed, "method": "concat_demuxer_stream_copy", "codec": codecs[0]}
