"""Causal local-video inputs for the official FluxMem inference path.

The native video frame list is ordered by explicit Day playback offsets. Feeding
that list at a fixed FPS compacts gaps and Day boundaries into a dense model
clock; frame_metadata preserves the original playback mapping. No recording or
wall-clock alignment is inferred. This module imports no model/GPU libraries.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile


class FrameBudgetExceeded(ValueError):
    """The entire causal prefix exceeds the explicit in-memory sample budget."""


def _seconds(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return Fraction(str(value))


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _day_order(value):
    if type(value) is not int or value < 1:
        raise ValueError("day_order must be a positive integer")
    return value


def _state(path):
    value = path.stat()
    return {"path": str(path), "device": value.st_dev, "inode": value.st_ino,
            "size": value.st_size, "mtime_ns": value.st_mtime_ns, "ctime_ns": value.st_ctime_ns}


def _sha256_file(path, cache_directory=None):
    state = _state(path)
    cache = None
    if cache_directory is not None:
        cache_directory.mkdir(parents=True, exist_ok=True)
        cache = cache_directory / (hashlib.sha256(str(path).encode()).hexdigest() + ".json")
        try:
            saved = json.loads(cache.read_text())
            digest = saved["sha256"]
            if saved["state"] == state and isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest):
                return digest, state
        except (OSError, ValueError, KeyError, TypeError):
            pass
    digestor = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digestor.update(chunk)
    if _state(path) != state:
        raise ValueError(f"Video changed while hashing: {path}")
    digest = digestor.hexdigest()
    if cache is not None:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("w", dir=cache.parent, encoding="utf-8", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump({"state": state, "sha256": digest}, stream)
            os.replace(temporary, cache)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return digest, state


def _probe(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams",
         "-show_format", "-of", "json", str(path)],
        check=True, capture_output=True, text=True, timeout=60,
    )
    try:
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        if stream.get("duration_ts") is not None and stream.get("time_base"):
            duration = Fraction(str(stream["duration_ts"])) * Fraction(stream["time_base"])
        else:
            duration = Fraction(str(stream.get("duration", data["format"].get("duration"))))
        width, height = int(stream["width"]), int(stream["height"])
        if duration <= 0 or min(width, height) <= 0:
            raise ValueError("Invalid video dimensions/duration")
        return {"duration_seconds": float(duration), "width": width, "height": height,
                "codec_name": stream.get("codec_name"), "time_base": stream.get("time_base"),
                "video_start_time": stream.get("start_time"), "avg_frame_rate": stream.get("avg_frame_rate"),
                "duration_fraction": str(duration)}
    except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError) as exc:
        raise ValueError(f"Cannot probe video: {path}") from exc


def prepare_manifest(sources, *, fingerprint_cache_dir=None):
    """Probe/hash explicit local source mappings without cutting or decoding them.

    Sources may be a JSON path, a list, or {"videos": [...]}. Each source needs
    path, day_id, and day_order. day_start_seconds defaults to zero. Unspecified
    timeline_status is explicitly marked provisional_concatenation.
    """
    if isinstance(sources, (str, Path)):
        sources = json.loads(Path(sources).read_text(encoding="utf-8"))
    if isinstance(sources, dict):
        sources = sources.get("videos")
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources must contain a nonempty videos list")
    cache = Path(fingerprint_cache_dir) if fingerprint_cache_dir is not None else None
    videos, day_orders, order_days = [], {}, {}
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Every video source must be an object")
        day = _identifier(source.get("day_id"), "day_id")
        order = _day_order(source.get("day_order"))
        if day in day_orders and day_orders[day] != order or order in order_days and order_days[order] != day:
            raise ValueError("Conflicting Day ID/order mapping")
        day_orders[day], order_days[order] = order, day
        supplied = source.get("path")
        if not isinstance(supplied, (str, Path)) or not Path(supplied).is_absolute():
            raise ValueError("Video path must be absolute")
        path = Path(supplied).resolve(strict=True)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Video must be a readable nonempty file: {path}")
        start = _seconds(source.get("day_start_seconds", 0), "day_start_seconds")
        status = source.get("timeline_status", "provisional_concatenation")
        if status not in {"direct_mp4", "provisional_concatenation"}:
            raise ValueError("timeline_status must be direct_mp4 or provisional_concatenation")
        before = _state(path)
        probe = _probe(path)
        fingerprint, state = _sha256_file(path, cache)
        if before != state or state != _state(path):
            raise ValueError(f"Source changed during manifest preparation: {path}")
        end = start + Fraction(probe["duration_fraction"])
        identity = {"path": str(path), "day_id": day, "day_order": order,
                    "day_start_seconds": float(start), "source_fingerprint": fingerprint}
        video = {**identity, **probe, "day_end_seconds": float(end), "timeline_status": status,
                 "source_id": "source_" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24],
                 "provenance": {"source_state": state, "source_fingerprint_algorithm": "sha256",
                                "timeline_status": status, "time_basis": "day_playback_seconds",
                                "recording_start": None, "virtual_start": None}}
        videos.append(video)
    videos.sort(key=lambda item: (item["day_order"], item["day_start_seconds"], item["path"]))
    previous = {}
    for video in videos:
        prior = previous.get(video["day_id"])
        if prior is not None and video["day_start_seconds"] < prior["day_end_seconds"]:
            raise ValueError("Overlapping components in the same Day")
        previous[video["day_id"]] = video
    return {"schema_version": "fluxmem_videomem_manifest_v1", "videos": videos,
            "time_basis": "day_playback_seconds", "interval_convention": "[start,end)",
            "source_video_reencoded": False}


def _query_time(value):
    if not isinstance(value, dict):
        raise ValueError("query_time must be an explicit object")
    return {"day_id": _identifier(value.get("day_id"), "query_time.day_id"),
            "day_order": _day_order(value.get("day_order")),
            "seconds": float(_seconds(value.get("seconds"), "query_time.seconds"))}


def public_question(row):
    """Copy only fields allowed to reach inference; never infer a query cutoff."""
    if not isinstance(row, dict):
        raise ValueError("Question must be an object")
    choices = row.get("choices", [])
    if not isinstance(choices, list) or any(not isinstance(choice, str) or not choice.strip() for choice in choices):
        raise ValueError("choices must be a list of nonempty strings")
    return {"question_id": _identifier(row.get("question_id"), "question_id"),
            "question": _identifier(row.get("question"), "question"), "choices": list(choices),
            "query_time": _query_time(row.get("query_time"))}


def select_prefix(manifest, query_time):
    """Select every source interval observable before the explicit cutoff.

    A source crossing the cutoff contributes its partial prefix. Gaps remain
    gaps in provenance; neither labels nor evidence ranges affect selection.
    """
    cutoff = _query_time(query_time)
    videos = manifest.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ValueError("Manifest must contain videos")
    day = [v for v in videos if v["day_id"] == cutoff["day_id"] and v["day_order"] == cutoff["day_order"]]
    if not day or any(v["day_id"] == cutoff["day_id"] and v["day_order"] != cutoff["day_order"] for v in videos):
        raise ValueError("query_time Day ID/order does not match the manifest")
    if cutoff["seconds"] > max(v["day_end_seconds"] for v in day):
        raise ValueError("query_time exceeds the available Day playback range")
    selected = []
    for video in sorted(videos, key=lambda v: (v["day_order"], v["day_start_seconds"], v["path"])):
        if video["day_order"] > cutoff["day_order"]:
            continue
        start = Fraction(str(video["day_start_seconds"]))
        duration = Fraction(video.get("duration_fraction", str(video["duration_seconds"])))
        end = duration
        if video["day_order"] == cutoff["day_order"]:
            end = min(end, Fraction(str(cutoff["seconds"])) - start)
        if end <= 0:
            continue
        if _state(Path(video["path"])) != video["provenance"]["source_state"]:
            raise ValueError(f"Source changed since manifest preparation: {video['path']}")
        selected.append({**video, "file_start_seconds": 0.0, "file_end_seconds": float(end),
                         "file_end_fraction": str(end), "day_end_seconds": float(start + end)})
    return selected


def decode_prefix(manifest, question, *, fps=1.0, max_frames, max_side=512):
    """Sequentially decode a bounded sample of the entire visible video prefix.

    Samples use a per-source grid starting at file time zero. Select the first
    actual frame at/after each grid point and deduplicate frames serving multiple
    grid points. Only frames with actual PTS strictly before the cutoff survive.
    Memory is bounded by max_frames RGB images plus the codec's decode buffers;
    budget overflow is an error, never a sliding window or a lower sampling rate.
    """
    rate = _seconds(fps, "fps", positive=True)
    if type(max_frames) is not int or max_frames <= 0:
        raise ValueError("max_frames must be an explicit positive integer")
    if type(max_side) is not int or max_side <= 0:
        raise ValueError("max_side must be a positive integer")
    public = public_question(question)
    segments = select_prefix(manifest, public["query_time"])
    requested = sum(math.ceil(Fraction(s["file_end_fraction"]) * rate) for s in segments)
    if requested > max_frames:
        raise FrameBudgetExceeded(f"Causal prefix requests {requested} sample points, exceeding max_frames={max_frames}; increase the explicit budget")
    import av
    from PIL import Image

    frames, metadata = [], []
    try:
        for segment in segments:
            path = Path(segment["path"])
            end = Fraction(segment["file_end_fraction"])
            next_sample, previous_pts = Fraction(0), None
            with av.open(str(path), mode="r") as container:
                if not container.streams.video:
                    raise ValueError(f"No video stream: {path}")
                stream = container.streams.video[0]
                stream.thread_type = "SLICE"
                stream.codec_context.thread_count = 1
                origin = Fraction(stream.start_time or 0) * Fraction(stream.time_base)
                for frame in container.decode(stream):
                    if frame.pts is None or frame.time_base is None:
                        raise ValueError(f"Video frame has no exact PTS: {path}")
                    raw_pts = Fraction(frame.pts) * Fraction(frame.time_base)
                    actual = raw_pts - origin
                    if previous_pts is not None and actual < previous_pts:
                        raise ValueError(f"Video presentation timestamps are not ordered: {path}")
                    previous_pts = actual
                    if actual >= end:
                        break
                    if actual < next_sample:
                        continue
                    if len(frames) >= max_frames:
                        raise FrameBudgetExceeded("Actual decoded samples exceed the explicit frame budget")
                    image = frame.to_image().convert("RGB")
                    if max(image.size) > max_side:
                        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                    frames.append(image)
                    day_seconds = Fraction(str(segment["day_start_seconds"])) + actual
                    metadata.append({
                        "position": len(frames) - 1, "source_id": segment["source_id"], "path": str(path),
                        "day_id": segment["day_id"], "day_order": segment["day_order"],
                        "file_seconds": float(actual), "day_seconds": float(day_seconds),
                        "file_seconds_fraction": str(actual), "day_seconds_fraction": str(day_seconds),
                        "pts": frame.pts, "time_base": str(frame.time_base), "raw_pts_seconds_fraction": str(raw_pts),
                        "sample_target_seconds": float(next_sample), "source_fingerprint": segment["source_fingerprint"],
                        "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                        "width": image.width, "height": image.height, "mode": "RGB", "decode_device": "cpu",
                    })
                    next_sample = Fraction(math.floor(actual * rate) + 1, 1) / rate
                    if next_sample >= end:
                        break
            if _state(path) != segment["provenance"]["source_state"]:
                raise ValueError(f"Source changed during decoding: {path}")
        return {"question": public, "frames": frames, "frame_metadata": metadata, "segments": segments,
                "fps": float(rate), "sampling": {"grid": "per_source_from_zero", "requested_sample_points": requested,
                "actual_frames": len(frames), "max_frames": max_frames, "max_side": max_side,
                "model_time_basis": "compacted_sampled_playback", "original_time_mapping": "frame_metadata",
                "interval_convention": "[start,cutoff)", "decode_device": "cpu"}}
    except BaseException:
        for image in frames:
            image.close()
        raise
