#!/usr/bin/env python3
"""Read-only validation of daily MP4s, clip boundaries and caption time maps.

Exit 0: requested media/mapping checks passed (caption gaps remain explicit).
Exit 1: validation failed. Exit 3: some requested days are not published yet.
This checks timing and decoded boundary pixels, not visual scene consistency.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

import timestamp_media as media
import timestamp_timeline as timeline


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value):
    _require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
             "Expected a finite number")
    return float(value)


def _close(actual, expected, message, tolerance=1e-5):
    _require(abs(_number(actual) - _number(expected)) <= tolerance, message)


def _metadata(path):
    info = media.probe(path)
    _require((info["width"], info["height"]) == (1024, 1024), f"Expected 1024x1024: {path}")
    _require(Fraction(info["fps"]) == 1 and Fraction(info["nominal_fps"]) == 1,
             f"Expected CFR 1fps: {path}")
    _require(not info["audio"], f"Expected silent benchmark video: {path}")
    _require(isinstance(info["frames"], int) and info["frames"] > 0, f"Missing frame count: {path}")
    _close(info["start_time"], 0, f"Video does not start at zero: {path}")
    _close(info["duration_s"], info["frames"], f"1fps duration/frame count mismatch: {path}", .05)
    return info


def check_packets(packets, expected_frames):
    """Validate every presentation timestamp and packet duration, not just FPS tags."""
    _require(len(packets) == expected_frames, "Video packet count differs from expected frame count")
    timestamps = sorted(_number(float(packet["pts_time"])) for packet in packets)
    for index, stamp in enumerate(timestamps):
        _close(stamp, index, f"Missing, duplicate or irregular PTS near frame {index}")
    for index, packet in enumerate(packets):
        _close(float(packet["duration_time"]), 1, f"Non-1-second packet duration at packet {index}")
    return {"packets": len(packets), "first_pts_s": timestamps[0], "last_pts_s": timestamps[-1],
            "all_pts_contiguous_1fps": True}


def packet_audit(path, expected_frames):
    document = json.loads(media._run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_packets",
        "-show_entries", "packet=pts_time,duration_time", "-of", "json", str(path),
    ], timeout=600).stdout)
    return check_packets(document["packets"], expected_frames)


def frame_hash(path, position):
    """Accurately decode one displayed RGB frame. No packet-copy seeking/trimming."""
    output = media._run([
        "ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-filter_threads", "1",
        "-ss", f"{position:.9f}", "-i", str(path), "-map", "0:v:0", "-an", "-sn", "-dn",
        "-frames:v", "1", "-vsync", "0", "-pix_fmt", "rgb24", "-threads", "1",
        "-f", "hash", "-hash", "sha256", "pipe:1",
    ], timeout=180).stdout.decode("ascii").strip()
    match = re.fullmatch(r"SHA256=([a-fA-F0-9]{64})", output)
    _require(match is not None, f"Could not hash decoded frame at {position}s: {path}")
    digest = match.group(1).lower()
    _require(digest != hashlib.sha256(b"").hexdigest(), f"No decoded frame at {position}s: {path}")
    return digest


def _time_values(value, kind):
    _require(value.get("kind") == kind, "Caption time kinds differ")
    if kind == "point":
        return [_number(value["t"])]
    _require(kind == "interval", f"Unknown caption kind: {kind}")
    values = [_number(value["start"]), _number(value["end"])]
    _require(values[0] < values[1], "Caption interval must have positive duration")
    return values


def check_captions(document, mapping):
    _require(document["day"] == mapping["day"] and document["date"] == mapping["date"],
             "Caption day/date differs from mapping")
    _require(document["timezone"] == mapping["timezone"], "Caption timezone differs from mapping")
    segments = {s["slot_id"]: s for s in mapping["segments"]}
    counts = {identifier: {"point": 0, "interval": 0} for identifier in segments}
    identifiers = set()
    for caption in document["captions"]:
        identifier = caption["id"]
        _require(identifier not in identifiers, f"Duplicate caption ID: {identifier}")
        identifiers.add(identifier)
        slot_id = caption["slot_id"]
        _require(slot_id in segments, f"Caption refers to unknown slot: {slot_id}")
        segment, kind = segments[slot_id], caption["kind"]
        local = _time_values(caption["clip_time"], kind)
        output = _time_values(caption["output_time"], kind)
        selected = _time_values(caption["selected_source_time"], kind)
        original = _time_values(caption["source_time"], kind)
        duration = segment["output_end_s"] - segment["output_start_s"]
        _require(local[0] >= -1e-6 and local[-1] <= duration + 1e-6,
                 f"Caption extends outside actual clip duration: {identifier}")
        if kind == "point":
            _require(local[0] < duration, f"Point caption falls at/after clip end: {identifier}")
        clock = dt.datetime.fromisoformat(segment["synthetic_start"])
        synthetic = caption["synthetic_time"]
        _require(synthetic.get("kind") == kind, f"Synthetic caption kind mismatch: {identifier}")
        clocks = ([synthetic["t"]] if kind == "point" else [synthetic["start"], synthetic["end"]])
        for index, relative in enumerate(local):
            _close(output[index], segment["output_start_s"] + relative, f"Wrong output offset: {identifier}")
            _close(selected[index], segment["source_start_s"] + relative, f"Wrong source offset: {identifier}")
            expected_clock = clock + dt.timedelta(seconds=relative)
            _require(abs((dt.datetime.fromisoformat(clocks[index]) - expected_clock).total_seconds()) <= 1e-5,
                     f"Wrong synthetic caption clock: {identifier}")
        _require(output[0] >= -1e-6 and output[-1] <= mapping["duration_s"] + 1e-6,
                 f"Caption extends outside daily video: {identifier}")
        effective_end = min(segment["source_end_s"], segment["source_start_s"] + duration)
        _require(selected[0] >= segment["source_start_s"] - 1e-6 and selected[-1] <= effective_end + 1e-6,
                 f"Caption extends outside selected source interval: {identifier}")
        _require(selected[0] >= original[0] - 1e-6 and selected[-1] <= original[-1] + 1e-6,
                 f"Selected caption interval extends beyond its source annotation: {identifier}")
        counts[slot_id][kind] += 1
    unaligned_rows = document.get("unaligned_evidence", [])
    unaligned = {row["slot_id"] for row in unaligned_rows}
    _require(len(unaligned) == len(unaligned_rows), "Duplicate unaligned-evidence slot")
    expected_unaligned = {identifier for identifier, count in counts.items() if sum(count.values()) == 0}
    _require(unaligned == expected_unaligned, "Unaligned caption slots are missing or incorrectly flagged")
    coverage = document["coverage"]
    coverage_rows = {row["slot_id"]: row for row in coverage["segments"]}
    _require(set(coverage_rows) == set(segments) and len(coverage_rows) == len(coverage["segments"]),
             "Caption coverage rows do not match segments")
    for identifier, count in counts.items():
        row = coverage_rows[identifier]
        _require(row["point_count"] == count["point"] and row["interval_count"] == count["interval"],
                 f"Caption coverage counts differ: {identifier}")
        expected_status = "no_aligned_annotations" if identifier in unaligned else "aligned_annotations_available"
        _require(row["status"] == expected_status, f"Wrong alignment status: {identifier}")
    _require(coverage["caption_count"] == len(identifiers), "Caption summary count differs")
    _require(coverage["unaligned_segment_count"] == len(unaligned), "Unaligned segment count differs")
    return {"caption_count": len(identifiers), "time_maps_valid": True,
            "aligned_segments": len(segments) - len(unaligned), "unaligned_segments": len(unaligned),
            "unaligned_slot_ids": sorted(unaligned), "all_segments_have_aligned_captions": not unaligned,
            "source_warning_count": len(document.get("warnings", [])),
            "limits": "Aligned annotations do not imply continuous coverage or visually correct activity labels."}


def validate_day(root, day, plan):
    started = time.monotonic()
    directory = root / f"day{day:02d}"
    paths = {name: directory / name for name in ("video.mp4", "mapping.json", "captions.json", "validation.json")}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        return {"day": day, "status": "pending", "missing": missing}
    fingerprints = {name: timeline.fingerprint(path) for name, path in paths.items()}
    report = {"day": day, "status": "failed", "video": str(paths["video.mp4"])}
    try:
        mapping = json.loads(paths["mapping.json"].read_text())
        saved = json.loads(paths["validation.json"].read_text())
        expected_date = dt.date.fromisoformat(plan["base_date"]) + dt.timedelta(days=day - 1)
        _require(mapping["day"] == day and mapping["date"] == expected_date.isoformat(), "Wrong synthetic day/date")
        _require(mapping["timezone"] == plan["timezone"], "Wrong synthetic timezone")
        _require(mapping["canvas"] == [1024, 1024], "Expected a 1024x1024 mapping")
        _require(saved["mapping_sha256"] == timeline.digest(mapping), "Published mapping hash is stale")
        _require(saved["output_fingerprint"] == fingerprints["video.mp4"], "Published video fingerprint is stale")
        selected = sorted((c for c in plan["clips"] if c["day"] == day), key=lambda c: c["start_seconds"])
        _require([s["slot_id"] for s in mapping["segments"]] == [c["id"] for c in selected],
                 "Daily mapping does not match scheduled clip order")
        actual = _metadata(paths["video.mp4"])
        _close(actual["duration_s"], mapping["duration_s"], "Daily duration differs from mapping", .05)
        _require(actual["frames"] == mapping["frames"], "Daily frame count differs from mapping")
        cursor, frame_cursor, previous_end = 0.0, 0, None
        boundaries = []
        for segment, planned in zip(mapping["segments"], selected):
            identifier = segment["slot_id"]
            _require(segment["day"] == day, f"Segment day differs: {identifier}")
            source = planned.get("source") or {}
            _require(segment["source_dataset"] == source.get("dataset") and
                     segment["source_video_id"] == source.get("video_id"),
                     f"Source identity differs from plan: {identifier}")
            _close(segment["source_start_s"], source.get("start_s", 0), f"Source start differs from plan: {identifier}")
            _close(segment["source_end_s"], source.get("end_s", planned["expected_duration_s"]),
                   f"Source end differs from plan: {identifier}")
            _close(segment["output_start_s"], cursor, f"Gap or overlap in daily video: {identifier}")
            _require(segment["output_start_frame"] == frame_cursor, f"Wrong starting frame: {identifier}")
            normalized = Path(segment["normalized_file"])
            if not normalized.is_absolute():
                normalized = root / normalized
            segment_fingerprint = timeline.fingerprint(normalized)
            segment_sidecar = json.loads(normalized.with_suffix(".json").read_text())
            recipe = segment_sidecar["recipe"]
            _require(segment_sidecar["output_fingerprint"] == segment_fingerprint,
                     f"Normalized video fingerprint is stale: {identifier}")
            _require(recipe["clip"] == planned and recipe["base_date"] == plan["base_date"],
                     f"Normalized video recipe differs from its scheduled clip: {identifier}")
            _require(segment_sidecar["recipe_sha256"] == timeline.digest(recipe),
                     f"Normalized video recipe hash is stale: {identifier}")
            clip_info = _metadata(normalized)
            duration = clip_info["duration_s"]
            _close(segment["input_duration_s"], duration, f"Normalized clip length differs: {identifier}")
            _close(segment["output_end_s"] - cursor, duration, f"Mapped length differs: {identifier}")
            _close(segment["synthetic_start_s"], planned["start_seconds"], f"Synthetic start differs from plan: {identifier}")
            clock = dt.datetime.fromisoformat(segment["synthetic_start"])
            expected_clock = dt.datetime.combine(expected_date, dt.time()) + dt.timedelta(seconds=planned["start_seconds"])
            _require(clock == expected_clock, f"Synthetic timestamp differs from plan: {identifier}")
            clock_end = dt.datetime.fromisoformat(segment["synthetic_end"])
            _close((clock_end - clock).total_seconds(), duration, f"Synthetic duration differs: {identifier}")
            _require(clock.date() == expected_date and clock_end <= dt.datetime.combine(expected_date + dt.timedelta(days=1), dt.time()),
                     f"Clip extends beyond its synthetic day: {identifier}")
            _require(previous_end is None or clock >= previous_end, f"Overlapping synthetic intervals: {identifier}")
            previous_end = clock_end
            for endpoint, relative in (("first", 0.0), ("last", duration - 1.0)):
                daily_time = cursor + relative
                daily_hash = frame_hash(paths["video.mp4"], daily_time)
                clip_hash = frame_hash(normalized, relative)
                _require(daily_hash == clip_hash, f"Boundary frame mismatch: {identifier} {endpoint}")
                boundaries.append({"slot_id": identifier, "endpoint": endpoint, "daily_time_s": daily_time,
                                   "clip_time_s": relative, "sha256_rgb24": daily_hash, "matches": True})
            _require(timeline.fingerprint(normalized) == segment_fingerprint,
                     f"Normalized video changed during validation: {identifier}")
            cursor += duration
            frame_cursor += clip_info["frames"]
            _require(segment["output_end_frame_exclusive"] == frame_cursor, f"Wrong ending frame: {identifier}")
        _close(cursor, actual["duration_s"], "Sum of normalized clip durations differs", .05)
        _require(frame_cursor == actual["frames"], "Sum of normalized clip frames differs")
        report["metadata"] = actual
        report["packets"] = packet_audit(paths["video.mp4"], frame_cursor)
        report["boundary_checks"] = boundaries
        report["captions"] = check_captions(json.loads(paths["captions.json"].read_text()), mapping)
        report.update(status="passed", segment_count=len(mapping["segments"]), date=mapping["date"])
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    if any(not path.is_file() or timeline.fingerprint(path) != fingerprints[name] for name, path in paths.items()):
        report.update(status="pending", error="Output changed during validation; retry after publication finishes")
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return report


def run(root, days=None, workers=4):
    root = Path(root).resolve()
    plan = json.loads((root / "plan.json").read_text())
    available_days = sorted({c["day"] for c in plan["clips"]})
    requested = sorted(set(days or available_days))
    _require(set(requested) <= set(available_days), "Requested day is absent from the plan")
    sources, duplicate_sources = {}, []
    for clip in plan["clips"]:
        if clip["day"] not in requested:
            continue
        source = clip.get("source", {})
        key = (source.get("dataset"), source.get("video_id"))
        if key[1] is None:
            continue
        if key in sources:
            duplicate_sources.append({"dataset": key[0], "video_id": key[1], "slots": [sources[key], clip["id"]]})
        sources[key] = clip["id"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(lambda day: validate_day(root, day, plan), requested))
    failed = sum(r["status"] == "failed" for r in results)
    pending = sum(r["status"] == "pending" for r in results)
    passed = sum(r["status"] == "passed" for r in results)
    gaps = sum(r.get("captions", {}).get("unaligned_segments", 0) for r in results)
    exit_code = 1 if failed or duplicate_sources else (3 if pending else 0)
    status = "failed" if exit_code == 1 else ("pending" if pending else ("passed_with_caption_gaps" if gaps else "passed"))
    return {"checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "root": str(root), "status": status, "requested_days": requested,
            "passed_days": passed, "failed_days": failed, "pending_days": pending,
            "unaligned_segments_in_checked_days": gaps, "duplicate_sources": duplicate_sources,
            "all_requested_media_validated": exit_code == 0,
            "all_requested_segments_have_aligned_captions": exit_code == 0 and gaps == 0,
            "limits": "Checks all packet timestamps and every segment's first/last decoded pixels. Does not decode every frame or certify visual activity/home/day-night consistency or continuous caption coverage.",
            "days": results}, exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--day", type=int, action="append")
    parser.add_argument("--report", help="Optional explicit JSON report path; otherwise prints JSON to stdout")
    args = parser.parse_args(argv)
    try:
        report, code = run(args.root, args.day, args.workers)
        if args.report:
            timeline.write_json(args.report, report)
            print(json.dumps({"report": str(Path(args.report).resolve()), "status": report["status"],
                              "passed_days": report["passed_days"], "pending_days": report["pending_days"]}))
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return code
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f"daily-validation: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
