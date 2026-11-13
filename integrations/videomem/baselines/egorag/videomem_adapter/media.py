"""Local video manifests and bounded frame extraction; no annotation inputs.

Day times are playback offsets. Calendar times are copied only when supplied by
the source manifest and are never inferred from file names or Day numbers.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from decimal import Decimal, ROUND_CEILING
from typing import Any


SCHEMA_VERSION = "videomem_media_manifest_v1"
_TIMELINE_STATUSES = {"direct_mp4", "provisional_concatenation"}


def _number(value: Any, name: str, *, positive: bool = False) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError(f"{name} must be a finite number")
    result = Decimal(str(value))
    if not result.is_finite() or result < 0 or (positive and result == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'} and finite")
    return result


def _state(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {
        "path": str(path), "device": st.st_dev, "inode": st.st_ino,
        "size": st.st_size, "mtime_ns": st.st_mtime_ns, "ctime_ns": st.st_ctime_ns,
    }


def _content_hash(path: Path, cache_dir: Path | None) -> tuple[str, dict[str, Any]]:
    state = _state(path)
    cache = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(str(path).encode()).hexdigest()
        cache = cache_dir / f"{key}.sha256.json"
        try:
            stored = json.loads(cache.read_text(encoding="utf-8"))
            digest = stored["sha256"]
            if stored["state"] == state and re.fullmatch(r"[a-f0-9]{64}", digest):
                return digest, state
        except (OSError, ValueError, KeyError, TypeError):
            pass
    digestor = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digestor.update(block)
    if _state(path) != state:
        raise ValueError(f"Video changed while hashing: {path}")
    digest = digestor.hexdigest()
    if cache is not None:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=cache.parent,
                                             prefix=".checksum-", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump({"state": state, "sha256": digest}, handle, sort_keys=True)
            os.replace(temporary, cache)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return digest, state


def _probe(path: Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams",
             "-show_format", "-of", "json", str(path)],
            check=True, capture_output=True, text=True, timeout=60,
        )
        data = json.loads(completed.stdout)
        stream = data["streams"][0]
        duration = stream.get("duration")
        if duration in (None, "N/A") and stream.get("duration_ts") is not None:
            numerator, denominator = stream["time_base"].split("/")
            duration = (Decimal(str(stream["duration_ts"])) * Decimal(numerator)
                        / Decimal(denominator))
        if duration in (None, "N/A"):
            duration = data["format"]["duration"]
        duration = _number(Decimal(str(duration)), "video duration", positive=True)
        width, height = int(stream["width"]), int(stream["height"])
        if width <= 0 or height <= 0:
            raise ValueError("Video has invalid dimensions")
        return {"duration_seconds": float(duration), "width": width, "height": height,
                "codec_name": stream.get("codec_name"),
                "avg_frame_rate": stream.get("avg_frame_rate"),
                "video_start_time": stream.get("start_time")}
    except (subprocess.SubprocessError, OSError, KeyError, IndexError, ValueError,
            ArithmeticError, TypeError) as exc:
        raise ValueError(f"Cannot probe readable video: {path}: {exc}") from exc


def prepare_manifest(source: Any, clip_seconds: float = 30, *,
                     fingerprint_cache_dir: str | Path | None = None) -> dict[str, Any]:
    """Probe source videos, validate Day placement, and split into [start, end) clips.

    ``source`` is a JSON file path, a list, or ``{"videos": [...]}``. Each video
    requires ``day_id``, explicit integer ``day_order``, absolute ``path``, and
    ``timeline_status``. ``day_start_seconds`` defaults to zero. Full SHA256 is
    streamed with bounded memory; an optional stat-validated cache speeds reuse.
    """
    clip_length = _number(clip_seconds, "clip_seconds", positive=True)
    if isinstance(source, (str, Path)):
        source = json.loads(Path(source).read_text(encoding="utf-8"))
    if isinstance(source, dict):
        source = source.get("videos")
    if not isinstance(source, list) or not source:
        raise ValueError("source must contain a nonempty videos list")
    cache_dir = Path(fingerprint_cache_dir).expanduser() if fingerprint_cache_dir else None
    videos: list[dict[str, Any]] = []
    day_orders: dict[str, int] = {}
    order_days: dict[int, str] = {}
    for row in source:
        if not isinstance(row, dict):
            raise ValueError("Every video source must be an object")
        day_id, day_order = row.get("day_id"), row.get("day_order")
        if not isinstance(day_id, str) or not day_id.strip():
            raise ValueError("day_id must be a nonempty string")
        if type(day_order) is not int or day_order < 1:
            raise ValueError("day_order must be an explicit positive integer")
        if day_id in day_orders and day_orders[day_id] != day_order:
            raise ValueError(f"Conflicting day_order for {day_id}")
        if day_order in order_days and order_days[day_order] != day_id:
            raise ValueError(f"day_order {day_order} is assigned to multiple days")
        day_orders[day_id], order_days[day_order] = day_order, day_id
        timeline_status = row.get("timeline_status")
        if timeline_status not in _TIMELINE_STATUSES:
            raise ValueError(f"timeline_status must be one of {sorted(_TIMELINE_STATUSES)}")
        raw_path = row.get("path")
        if not isinstance(raw_path, (str, Path)) or not Path(raw_path).is_absolute():
            raise ValueError("Video path must be absolute")
        path = Path(raw_path).resolve(strict=True)
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"Video is missing, empty, or not a file: {path}")
        day_start = _number(row.get("day_start_seconds", 0), "day_start_seconds")
        before = _state(path)
        media = _probe(path)
        fingerprint, state = _content_hash(path, cache_dir)
        if before != state or _state(path) != state:
            raise ValueError(f"Video changed during manifest preparation: {path}")
        day_end = day_start + Decimal(str(media["duration_seconds"]))
        provenance = {
            "timeline_status": timeline_status,
            "source_path": str(path), "source_state": state,
            "virtual_start": row.get("virtual_start"),
            "recording_start": row.get("recording_start"),
            "time_basis": "day_playback_seconds",
            "source_fingerprint_algorithm": "sha256",
        }
        videos.append({
            "day_id": day_id, "day_order": day_order, "path": str(path),
            "day_start_seconds": float(day_start), "day_end_seconds": float(day_end),
            "timeline_status": timeline_status,
            "virtual_start": row.get("virtual_start"),
            "recording_start": row.get("recording_start"),
            "source_fingerprint": fingerprint, "size_bytes": state["size"],
            "provenance": provenance, **media,
        })
    videos.sort(key=lambda item: (item["day_order"], item["day_start_seconds"], item["path"]))
    previous: dict[str, dict[str, Any]] = {}
    for video in videos:
        prior = previous.get(video["day_id"])
        if prior and video["day_start_seconds"] < prior["day_end_seconds"]:
            raise ValueError(f"Overlapping video components in {video['day_id']}")
        previous[video["day_id"]] = video
    clips = []
    for video in videos:
        duration = Decimal(str(video["duration_seconds"]))
        day_start = Decimal(str(video["day_start_seconds"]))
        cursor = Decimal(0)
        while cursor < duration:
            end = min(cursor + clip_length, duration)
            clip = {
                "day_id": video["day_id"], "day_order": video["day_order"],
                "path": video["path"], "file_start_seconds": float(cursor),
                "file_end_seconds": float(end),
                "day_start_seconds": float(day_start + cursor),
                "day_end_seconds": float(day_start + end),
                "source_fingerprint": video["source_fingerprint"],
                "provenance": video["provenance"].copy(),
            }
            identity = {key: value for key, value in clip.items() if key != "provenance"}
            identity["timeline_status"] = video["timeline_status"]
            digest = hashlib.sha256(json.dumps(identity, sort_keys=True,
                                               separators=(",", ":")).encode()).hexdigest()
            clip["clip_id"] = f"clip_{digest[:24]}"
            clips.append(clip)
            cursor = end
    return {"schema_version": SCHEMA_VERSION, "interval_convention": "[start,end)",
            "clip_seconds": float(clip_length), "videos": videos, "clips": clips}


def extract_frames(clip: dict[str, Any], fps: float = 1, max_frames: int = 16,
                   max_side: int = 512) -> list[dict[str, Any]]:
    """Seek to a bounded selection of samples and return JPEGs with actual PTS.

    Samples begin at the clip start. If the fps grid exceeds ``max_frames``,
    evenly spaced grid points cover the entire interval. The decoder may return
    the next frame; frames whose actual timestamps fall outside [start, end)
    are excluded. No complete video or unbounded frame sequence is decoded.
    """
    rate = _number(fps, "fps", positive=True)
    if type(max_frames) is not int or max_frames <= 0:
        raise ValueError("max_frames must be a positive integer")
    if type(max_side) is not int or max_side <= 0:
        raise ValueError("max_side must be a positive integer")
    start = _number(clip.get("file_start_seconds"), "file_start_seconds")
    end = _number(clip.get("file_end_seconds"), "file_end_seconds")
    day_start = _number(clip.get("day_start_seconds"), "day_start_seconds")
    if end <= start:
        raise ValueError("Clip must have a nonempty half-open interval")
    path = Path(clip["path"])
    if not path.is_absolute() or not path.is_file():
        raise ValueError("Clip path must be an existing absolute file path")
    expected_state = clip.get("provenance", {}).get("source_state")
    if expected_state is not None and _state(path) != expected_state:
        raise ValueError(f"Source changed since manifest preparation: {path}")
    count = int(((end - start) * rate).to_integral_value(rounding=ROUND_CEILING))
    if count <= max_frames:
        indices = range(count)
    elif max_frames == 1:
        indices = [0]
    else:
        indices = [round(index * (count - 1) / (max_frames - 1))
                   for index in range(max_frames)]
    frames = []
    seen_times: set[Decimal] = set()
    for index in indices:
        requested = start + Decimal(index) / rate
        # ffmpeg decimal seek precision is microseconds. Rounding upwards would
        # allow a sample to cross an exclusive boundary, so truncate instead.
        seek = requested.quantize(Decimal("0.000001"), rounding="ROUND_DOWN")
        if seek < start or seek >= end:
            continue
        command = [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "info", "-threads", "1",
            "-copyts", "-start_at_zero", "-ss", str(seek), "-i", str(path),
            "-map", "0:v:0", "-an", "-sn", "-dn", "-filter_threads", "1",
            "-vf", f"scale=w='min({max_side},iw)':h='min({max_side},ih)':"
                   "force_original_aspect_ratio=decrease,showinfo",
            "-frames:v", "1", "-vsync", "0", "-c:v", "mjpeg", "-threads", "1",
            "-q:v", "3", "-f", "image2pipe", "pipe:1",
        ]
        try:
            completed = subprocess.run(command, capture_output=True, check=True, timeout=60)
        except (subprocess.SubprocessError, OSError) as exc:
            raise ValueError(f"Frame decoding failed at {seek}s in {path}: {exc}") from exc
        if not completed.stdout:
            continue
        match = re.search(rb"\bpts:\s*(-?\d+)\s+pts_time:", completed.stderr)
        time_base = re.search(rb"config in time_base:\s*(\d+)/(\d+)", completed.stderr)
        if match is None or time_base is None:
            raise ValueError("ffmpeg did not report the decoded frame timestamp")
        # The decimal pts_time display has limited precision. Integer PTS with
        # its rational time base preserves identity and exclusive boundaries.
        actual = (Decimal(match.group(1).decode("ascii"))
                  * Decimal(time_base.group(1).decode("ascii"))
                  / Decimal(time_base.group(2).decode("ascii")))
        if not start <= actual < end or actual in seen_times:
            continue
        seen_times.add(actual)
        jpeg = completed.stdout
        if not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
            raise ValueError("ffmpeg returned an invalid JPEG")
        frames.append({
            "file_seconds": float(actual), "day_seconds": float(day_start + actual - start),
            "jpeg_base64": base64.b64encode(jpeg).decode("ascii"),
            "sha256": hashlib.sha256(jpeg).hexdigest(),
        })
    if expected_state is not None and _state(path) != expected_state:
        raise ValueError(f"Source changed during frame extraction: {path}")
    return frames
