"""Bounded CPU sampling for persistent FluxMem streams.

An iterator owns its yielded PIL images. Encode a pair before calling next() or
close(): either operation closes the previous pair. Only complete temporal
patches from one source are yielded; an unmatched boundary frame is discarded.
The iterator may be peeked ahead on CPU, but callers must gate encoding and
memory insertion by available_time and the explicit query cutoff.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import math
from pathlib import Path

from local_videomem.data import _query_time, _seconds, _state, select_prefix


def time_key(value):
    """Return an exact chronological key for a query, cutoff, or pair time."""
    if not isinstance(value, dict):
        raise ValueError("Time must be an explicit object")
    value = value.get("query_time", value.get("available_time", value))
    checked = _query_time(value)
    return checked["day_order"], Fraction(str(checked["seconds"]))


def frame_key(value):
    """Return a frame's original Day/PTS key, or a pair's availability key."""
    if "day_seconds" not in value:
        return time_key(value)
    checked = _query_time({"day_id": value["day_id"],
                           "day_order": value["day_order"],
                           "seconds": value["day_seconds"]})
    seconds = Fraction(value.get("day_seconds_fraction", str(checked["seconds"])))
    if seconds < 0:
        raise ValueError("Frame time must be nonnegative")
    return checked["day_order"], seconds


def _check_source(segment):
    """Check the exact source state to which the manifest SHA256 is bound."""
    path = Path(segment["path"])
    fingerprint = segment["source_fingerprint"]
    if (not isinstance(fingerprint, str) or len(fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in fingerprint)):
        raise ValueError(f"Invalid source SHA256 fingerprint: {path}")
    if _state(path) != segment["provenance"]["source_state"]:
        raise ValueError(f"Source changed since manifest preparation or during decoding: {path}")


def iter_pairs(manifest, cutoffdict, *, fps=1, max_side=448, stats=None):
    """Yield complete, causal, source-local two-frame temporal patches.

    Sampling matches local_videomem.decode_prefix: each source has its own
    zero-origin FPS grid, and each selected actual presentation timestamp is
    strictly before the source's cutoff. Source offsets and prior Days retain
    their original chronology. No whole video or unbounded frame list is read.

    stats counts decoder frames consumed (including a possible cutoff boundary
    frame), selected samples (including discarded orphans), yielded pairs, and
    source/final-cutoff orphans. max_live_sampled_frames excludes codec buffers.
    Callers should close the generator in finally, including on early exit.
    """
    rate = _seconds(fps, "fps", positive=True)
    if type(max_side) is not int or max_side <= 0:
        raise ValueError("max_side must be a positive integer")
    if stats is None:
        stats = {}
    if not isinstance(stats, dict):
        raise ValueError("stats must be a dictionary")
    for key in ("decoded_frames", "sampled_frames", "pairs", "orphan_frames",
                "max_live_sampled_frames"):
        stats[key] = 0
    segments = select_prefix(manifest, cutoffdict)
    import av
    from PIL import Image

    pair_index = 0
    for segment in segments:
        _check_source(segment)
        path = Path(segment["path"])
        end = Fraction(segment["file_end_fraction"])
        day_start = Fraction(str(segment["day_start_seconds"]))
        next_sample, previous_pts = Fraction(0), None
        pending_images, pending_metadata = [], []
        try:
            with av.open(str(path), mode="r") as container:
                if not container.streams.video:
                    raise ValueError(f"No video stream: {path}")
                stream = container.streams.video[0]
                stream.thread_type = "SLICE"
                stream.codec_context.thread_count = 1
                origin = Fraction(stream.start_time or 0) * Fraction(stream.time_base)
                for frame in container.decode(stream):
                    stats["decoded_frames"] += 1
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
                    image = frame.to_image()
                    if image.mode != "RGB":
                        original = image
                        try:
                            image = original.convert("RGB")
                        finally:
                            original.close()
                    pending_images.append(image)
                    if max(image.size) > max_side:
                        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                    day_seconds = day_start + actual
                    pending_metadata.append({
                        "position": stats["sampled_frames"],
                        "source_id": segment["source_id"], "path": str(path),
                        "day_id": segment["day_id"], "day_order": segment["day_order"],
                        "file_seconds": float(actual), "day_seconds": float(day_seconds),
                        "file_seconds_fraction": str(actual), "day_seconds_fraction": str(day_seconds),
                        "pts": frame.pts, "time_base": str(frame.time_base),
                        "raw_pts_seconds_fraction": str(raw_pts),
                        "sample_target_seconds": float(next_sample),
                        "source_fingerprint": segment["source_fingerprint"],
                        "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                        "width": image.width, "height": image.height,
                        "mode": "RGB", "decode_device": "cpu",
                    })
                    stats["sampled_frames"] += 1
                    stats["max_live_sampled_frames"] = max(
                        stats["max_live_sampled_frames"], len(pending_images))
                    next_sample = Fraction(math.floor(actual * rate) + 1, 1) / rate
                    if len(pending_images) == 2:
                        _check_source(segment)
                        pair = {
                            "frames": pending_images,
                            "frame_metadata": pending_metadata,
                            "pair_index": pair_index,
                            "available_time": {"day_id": segment["day_id"],
                                               "day_order": segment["day_order"],
                                               "seconds": float(day_seconds)},
                            "frontier_time": {"day_id": segment["day_id"],
                                              "day_order": segment["day_order"],
                                              "seconds": float(day_start + min(next_sample, end))},
                            "metadata": {
                                "source_id": segment["source_id"], "path": str(path),
                                "source_fingerprint": segment["source_fingerprint"],
                                "frame_ids": [f"{row['source_id']}:pts={row['pts']}:tb={row['time_base']}"
                                              for row in pending_metadata],
                                "frame_positions": [row["position"] for row in pending_metadata],
                                "temporal_patch_policy": "two_frames_same_source_discard_boundary_orphan",
                            },
                        }
                        stats["pairs"] += 1
                        try:
                            yield pair
                        finally:
                            for image in pending_images:
                                image.close()
                            pending_images, pending_metadata = [], []
                            _check_source(segment)
                        pair_index += 1
                    if next_sample >= end:
                        break
            if pending_images:
                stats["orphan_frames"] += len(pending_images)
        finally:
            for image in pending_images:
                image.close()
            _check_source(segment)
