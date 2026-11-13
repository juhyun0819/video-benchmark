#!/usr/bin/env python3
"""Burn scheduled wall clocks onto complete 1fps clips; never trim to a slot."""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import time
from zoneinfo import ZoneInfo

import timestamp_media as media

DEFAULT_STYLE = {
    "reference_height_px": 1024,
    "original_mask_xyxy": [0, 995, 200, 1024],
    "text_height_px": 20,
    "text_x_px": 10,
    "bottom_margin_px": 10,
    "box_padding_px": 7,
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def positive(value, name, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"Invalid {name}: {value}")
    return value


def wallclock(plan, clip):
    """Naive synthetic local clock, independent of machine timezone and source offset."""
    return dt.datetime.combine(dt.date.fromisoformat(plan["base_date"]), dt.time()) + dt.timedelta(
        days=clip["day"] - 1, seconds=clip["start_seconds"])


def display_epoch(plan, clip):
    # gmtime is only a deterministic formatter here, not a claim about real capture UTC.
    return wallclock(plan, clip).replace(tzinfo=dt.timezone.utc).timestamp()


def validate_plan(plan):
    if plan.get("version") != 1:
        raise ValueError("Expected timestamp plan version 1")
    dt.date.fromisoformat(plan["base_date"])
    ZoneInfo(plan["timezone"])
    style = {**DEFAULT_STYLE, **plan.get("style", {})}
    for key in ("reference_height_px", "text_height_px"):
        positive(style[key], key)
    for key in ("text_x_px", "bottom_margin_px", "box_padding_px"):
        positive(style[key], key, zero=True)
    for key in ("text_height_px", "text_x_px", "bottom_margin_px", "box_padding_px"):
        if type(style[key]) is not int:
            raise ValueError(f"{key} must be an integer number of pixels")
    if not 8 <= style["text_height_px"] <= 128:
        raise ValueError("text_height_px must be 8..128")
    box = style["original_mask_xyxy"]
    if len(box) != 4 or any(type(x) not in (int, float) or not math.isfinite(x) for x in box):
        raise ValueError("original_mask_xyxy must contain four finite coordinates")
    if not (0 <= box[0] < box[2] and 0 <= box[1] < box[3] <= style["reference_height_px"]):
        raise ValueError("Invalid mask rectangle")
    clips = plan.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError("Plan needs a nonempty clips array")
    ids, sources, inputs = set(), set(), set()
    for c in clips:
        if not isinstance(c.get("id"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", c["id"]):
            raise ValueError("Clip id must be a safe filename token")
        if c["id"] in ids:
            raise ValueError(f"Duplicate clip id: {c['id']}")
        ids.add(c["id"])
        if type(c.get("day")) is not int or c["day"] < 1:
            raise ValueError(f"Invalid day for {c['id']}")
        if positive(c["start_seconds"], "start_seconds", zero=True) >= 86400:
            raise ValueError("start_seconds must be seconds within the given day")
        positive(c["expected_duration_s"], "expected_duration_s")
        inp = c.get("input")
        if inp is not None:
            if not isinstance(inp, str) or not inp.strip():
                raise ValueError(f"Invalid input for {c['id']}")
            if inp in inputs:
                raise ValueError(f"Repeated input video: {inp}")
            inputs.add(inp)
        source = c.get("source") or {}
        if source.get("video_id"):
            key = (source.get("dataset"), source["video_id"])
            if key in sources:
                raise ValueError(f"Repeated source video: {key}")
            sources.add(key)
    ordered = sorted(clips, key=lambda c: wallclock(plan, c))
    end = None
    for c in ordered:
        start = wallclock(plan, c)
        if end is not None and start < end - dt.timedelta(milliseconds=1):
            raise ValueError(f"Scheduled clips overlap at {c['id']}")
        end = start + dt.timedelta(seconds=c["expected_duration_s"])
    return plan


def import_manifest(manifest_path, base_date, timezone):
    """Adapter for carpenter caption-pipeline manifests, not @vb/core manifests."""
    path = Path(manifest_path).resolve()
    raw = path.read_bytes()
    manifest = json.loads(raw)
    clips = []
    for slot in manifest["slots"]:
        obs = slot.get("proposed_observation")
        if not obs:
            continue
        if slot.get("status") in ("excluded_by_user", "removed_from_video_plan"):
            raise ValueError(f"Excluded slot still has an observation: {slot['slot_id']}")
        source = slot.get("source") or {}
        duration = obs["duration_s"]
        if abs(obs["synthetic_end_s"] - obs["synthetic_start_s"] - duration) > .001:
            raise ValueError(f"Observation length mismatch: {slot['slot_id']}")
        if abs(source["end_s"] - source["start_s"] - duration) > .001:
            raise ValueError(f"Source length mismatch: {slot['slot_id']}")
        input_path = source.get("media_path")
        if input_path and not Path(input_path).is_absolute():
            input_path = str((path.parent / input_path).resolve())
        clips.append({"id": slot["slot_id"], "day": slot["day"], "label": slot["label"],
            "start_seconds": obs["synthetic_start_s"], "expected_duration_s": duration,
            "input": input_path, "source": {k: source.get(k) for k in
                ("dataset", "video_id", "start_s", "end_s")},
            "source_scope": slot.get("source_scope"), "review_status": slot.get("status")})
    plan = {"version": 1, "base_date": base_date, "timezone": timezone,
        "style": dict(DEFAULT_STYLE), "provenance": {"manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "schedule_sha256": manifest.get("schedule_sha256")},
        "clips": sorted(clips, key=lambda c: (c["day"], c["start_seconds"]))}
    return validate_plan(plan)


def resolve_input(value, plan_path, mappings):
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = Path(plan_path).resolve().parent / path
    path = Path(os.path.normpath(path))
    # Longest prefix first; match path components rather than a string prefix.
    for old, new in sorted(mappings, key=lambda item: len(str(item[0])), reverse=True):
        try:
            return (new / path.relative_to(old)).resolve()
        except ValueError:
            pass
    return path.resolve()


def find_font(explicit=None):
    if explicit:
        candidates = [Path(explicit)]
    else:
        candidates = [Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"),
                      Path("/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono-Bold.ttf")]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError("Install fonts-dejavu-core or pass --font /path/to/monospace-bold.ttf")


def fingerprint(path):
    st = path.stat()
    return {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def check_video(info, clip):
    from fractions import Fraction
    if Fraction(info["fps"]) != 1:
        raise ValueError("Input must already be 1fps; this command does not select frames")
    if abs(info["duration_s"] - clip["expected_duration_s"]) > 1.1:
        raise ValueError(f"File length {info['duration_s']}s differs from planned "
                         f"{clip['expected_duration_s']}s; will not trim or stretch")


def render_one(plan, clip, args, font, calibration, code_hash):
    inp = resolve_input(clip.get("input"), args.plan, args.path_map)
    row = {"id": clip["id"], "day": clip["day"],
           "synthetic_start": wallclock(plan, clip).isoformat(sep=" "), "input": str(inp) if inp else None}
    if inp is None or not inp.is_file():
        return {**row, "status": "pending_input" if inp else "missing_source_path"}
    try:
        before = fingerprint(inp)
        info = media.probe(inp)
        check_video(info, clip)
        recipe = {"version": 1, "source": before, "clip": clip,
                  "base_date": plan["base_date"], "timezone": plan["timezone"],
                  "style": {**DEFAULT_STYLE, **plan.get("style", {})},
                  "font_sha256": hashlib.sha256(font.read_bytes()).hexdigest(),
                  "calibration": calibration, "preset": args.preset, "crf": args.crf,
                  "threads": args.threads, "implementation_sha256": code_hash}
        key = digest(recipe)
        stem = f"day{clip['day']:02d}-{clip['id']}-{key[:16]}"
        directory = Path(args.output).resolve()
        output, sidecar = directory / f"{stem}.mp4", directory / f"{stem}.json"
        lockpath = directory / f".{stem}.lock"
        with lockpath.open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return {**row, "status": "owned_by_other_worker", "output": str(output)}
            if output.is_file() and sidecar.is_file():
                saved = json.loads(sidecar.read_text())
                if saved.get("recipe_sha256") == key and saved.get("output_fingerprint") == fingerprint(output):
                    current = media.probe(output)
                    check_video(current, clip)
                    if (current["width"], current["height"], current.get("frames")) != (
                            info["width"], info["height"], info.get("frames")):
                        raise ValueError("Existing output dimensions or frame count mismatch")
                    return {**row, "status": "reused", "output": str(output),
                            "duration_s": current["duration_s"]}
            temp = directory / f".{stem}.{os.getpid()}.part.mp4"
            try:
                result = media.render(inp, temp, display_epoch(plan, clip), recipe["style"],
                                      font, calibration, threads=args.threads,
                                      preset=args.preset, crf=args.crf)
                if fingerprint(inp) != before:
                    raise ValueError("Input changed during rendering; wait for an atomic completed input")
                os.replace(temp, output)
                write_json(sidecar, {"recipe_sha256": key, "recipe": recipe,
                    "output_fingerprint": fingerprint(output), "render": result,
                    "synthetic_start": row["synthetic_start"],
                    "clock_semantics": "synthetic local wall clock + output-local elapsed seconds; gaps jump"})
                return {**row, "status": "rendered", "output": str(output),
                        "duration_s": info["duration_s"], "elapsed_seconds": result["elapsed_seconds"]}
            finally:
                temp.unlink(missing_ok=True)
    except Exception as exc:
        return {**row, "status": "error", "error": str(exc)}


def run_render(args):
    font = find_font(args.font)
    code_hash = hashlib.sha256(Path(__file__).read_bytes() + Path(media.__file__).read_bytes()).hexdigest()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    report_path = Path(args.output) / f"run-{socket.gethostname()}-{os.getpid()}.json"
    while True:
        plan = validate_plan(json.loads(Path(args.plan).read_text()))
        clips = plan["clips"]
        if args.only:
            unknown = set(args.only) - {c["id"] for c in clips}
            if unknown:
                raise ValueError(f"Unknown --only ids: {sorted(unknown)}")
            clips = [c for c in clips if c["id"] in args.only]
        resolved = [resolve_input(c.get("input"), args.plan, args.path_map) for c in clips]
        resolved_present = [p for p in resolved if p is not None]
        if len(set(resolved_present)) != len(resolved_present):
            raise ValueError("Path mapping resolves multiple clips to the same video")
        style = {**DEFAULT_STYLE, **plan.get("style", {})}
        calibration = media.calibrate_font(font, target_height=style["text_height_px"])
        report = {"started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                  "plan_sha256": digest(plan), "selected_clips": len(clips), "clips": []}
        start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            jobs = [pool.submit(render_one, plan, c, args, font, calibration, code_hash) for c in clips]
            for future in concurrent.futures.as_completed(jobs):
                row = future.result()
                report["clips"].append(row)
                write_json(report_path, report)
                print(json.dumps(row, ensure_ascii=False), flush=True)
        report["elapsed_seconds"] = time.monotonic() - start
        report["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        counts = {status: sum(r["status"] == status for r in report["clips"]) for status in
                  ("rendered", "reused", "pending_input", "missing_source_path", "owned_by_other_worker", "error")}
        report["counts"] = counts
        write_json(report_path, report)
        print(json.dumps({"report": str(report_path), **counts}, ensure_ascii=False), flush=True)
        done = counts["rendered"] + counts["reused"] == len(clips)
        if done or not args.watch:
            return 0 if done else (1 if counts["error"] else 3)
        if counts["error"]:
            return 1  # Fail visibly rather than endlessly retry a bad file/spec.
        time.sleep(args.poll_seconds)


def bounded_int(low, high):
    def convert(value):
        number = int(value)
        if not low <= number <= high:
            raise argparse.ArgumentTypeError(f"Expected {low}..{high}")
        return number
    return convert


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("plan", help="Import current carpenter caption-pipeline placements")
    build.add_argument("--manifest", required=True)
    build.add_argument("--base-date", required=True, help="Day1 synthetic local date, e.g. 2025-07-07")
    build.add_argument("--timezone", default="Asia/Seoul")
    build.add_argument("--out", required=True)
    render = sub.add_parser("render", help="Stamp already-selected, complete 1fps input files")
    render.add_argument("--plan", required=True)
    render.add_argument("--output", required=True)
    render.add_argument("--font")
    render.add_argument("--workers", type=bounded_int(1, 16), default=2)
    render.add_argument("--threads", type=bounded_int(1, 32), default=2)
    render.add_argument("--preset", choices=("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"), default="veryfast")
    render.add_argument("--crf", type=bounded_int(0, 30), default=18)
    render.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW")
    render.add_argument("--only", action="append", default=[], metavar="CLIP_ID")
    render.add_argument("--watch", action="store_true")
    render.add_argument("--poll-seconds", type=bounded_int(1, 3600), default=30)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            if Path(args.manifest).resolve() == Path(args.out).resolve():
                raise ValueError("Plan output cannot overwrite the source manifest")
            plan = import_manifest(args.manifest, args.base_date, args.timezone)
            write_json(args.out, plan)
            print(json.dumps({"plan": str(Path(args.out).resolve()), "clips": len(plan["clips"]),
                              "base_date": plan["base_date"]}, ensure_ascii=False))
            return 0
        pairs = []
        for mapping in args.path_map:
            old, sep, new = mapping.partition("=")
            if not sep or not Path(old).is_absolute() or not Path(new).is_absolute():
                raise ValueError("--path-map must be /absolute/old=/absolute/new")
            pairs.append((Path(os.path.normpath(old)), Path(os.path.normpath(new))))
        args.path_map = pairs
        return run_render(args)
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f"timestamp: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
