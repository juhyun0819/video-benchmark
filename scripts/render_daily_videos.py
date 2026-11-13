#!/usr/bin/env python3
"""Assemble scheduled 1fps clips into daily videos and caption/source maps."""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

import daily_video_captions as captions
import daily_video_media as video
import timestamp_media as stamp
import timestamp_timeline as timeline


def clock_now():
    return dt.datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def assemble_mapping(plan, day, clip_results):
    selected = sorted((c for c in plan["clips"] if c["day"] == day), key=lambda c: c["start_seconds"])
    date = (dt.date.fromisoformat(plan["base_date"]) + dt.timedelta(days=day - 1)).isoformat()
    mapping = {"version": 1, "day": day, "date": date, "timezone": plan["timezone"],
               "segments": [], "clock_semantics": "Clock runs normally inside each clip and jumps across unrecorded gaps."}
    cursor, frame_cursor = 0.0, 0
    for clip in selected:
        result = clip_results[clip["id"]]
        info = result["metadata"]
        duration = info["duration_s"]
        source = clip.get("source") or {}
        synthetic = timeline.wallclock(plan, clip)
        mapping["segments"].append({"slot_id": clip["id"], "label": clip.get("label"),
            "day": day, "input": clip["input"], "normalized_file": result["output"],
            "output_start_s": cursor, "output_end_s": cursor + duration,
            "output_start_frame": frame_cursor, "output_end_frame_exclusive": frame_cursor + info["frames"],
            "input_duration_s": duration, "planned_duration_s": clip["expected_duration_s"],
            "synthetic_start": synthetic.isoformat(), "synthetic_start_s": clip["start_seconds"],
            "synthetic_end": (synthetic + dt.timedelta(seconds=duration)).isoformat(),
            "source_dataset": source.get("dataset"), "source_video_id": source.get("video_id"),
            "source_start_s": source.get("start_s", 0),
            "source_end_s": source.get("end_s", duration),
            "source_scope": clip.get("source_scope"), "review_status": clip.get("review_status")})
        cursor += duration
        frame_cursor += info["frames"]
    mapping.update(duration_s=cursor, frames=frame_cursor)
    return mapping


def render_segment(clip, plan, args, root, font, calibration, implementation):
    path = Path(clip["input"]) if clip.get("input") else None
    row = {"id": clip["id"], "day": clip["day"]}
    if path is None or not path.is_file():
        return {**row, "status": "pending_input", "input": str(path) if path else None}
    try:
        source_fp = timeline.fingerprint(path)
        info = stamp.probe(path)
        timeline.check_video(info, clip)
        if info["frames"] is None:
            raise ValueError("Input needs a frame count for exact daily composition")
        recipe = {"clip": clip, "input": source_fp, "base_date": plan["base_date"],
                  "style": plan["style"], "canvas": args.size, "calibration": calibration,
                  "font_sha256": hashlib.sha256(font.read_bytes()).hexdigest(),
                  "preset": args.preset, "crf": args.crf, "implementation": implementation}
        key = timeline.digest(recipe)
        output = root / "segments" / f"{clip['id']}-{key[:16]}.mp4"
        sidecar = output.with_suffix(".json")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.with_suffix(".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if output.is_file() and sidecar.is_file():
                saved = json.loads(sidecar.read_text())
                if saved.get("recipe_sha256") == key and saved.get("output_fingerprint") == timeline.fingerprint(output):
                    metadata = stamp.probe(output)
                    if (metadata["frames"] != info["frames"] or metadata["fps"] != "1/1" or
                            metadata["width"] != args.size or metadata["height"] != args.size or
                            abs(metadata["duration_s"] - info["duration_s"]) > .05):
                        raise ValueError("Cached normalized clip does not match its source")
                    return {**row, "status": "reused", "output": str(output), "metadata": metadata}
            temporary = output.with_name(f".{output.stem}.{os.getpid()}.part.mp4")
            temporary.unlink(missing_ok=True)
            try:
                rendered = video.render_clip(path, temporary, timeline.display_epoch(plan, clip),
                    plan["style"], font, calibration, threads=args.threads,
                    preset=args.preset, crf=args.crf, size=args.size)
                if source_fp != timeline.fingerprint(path):
                    raise ValueError("Input changed during processing")
                os.replace(temporary, output)
                metadata = stamp.probe(output)
                timeline.write_json(sidecar, {"recipe_sha256": key, "recipe": recipe,
                    "output_fingerprint": timeline.fingerprint(output), "render": rendered})
                return {**row, "status": "rendered", "output": str(output), "metadata": metadata,
                        "elapsed_seconds": rendered["elapsed_seconds"]}
            finally:
                temporary.unlink(missing_ok=True)
    except Exception as exc:
        return {**row, "status": "error", "error": str(exc)}


def finish_day(plan, day, clip_results, manifest, args, root):
    directory = root / f"day{day:02d}"
    directory.mkdir(exist_ok=True)
    mapping = assemble_mapping(plan, day, clip_results)
    mapping["canvas"] = [args.size, args.size]
    mapping["caption_transform_sha256"] = hashlib.sha256(Path(captions.__file__).read_bytes()).hexdigest()
    key = timeline.digest(mapping)
    destination = directory / "video.mp4"
    validation_path = directory / "validation.json"
    try:
        if (destination.is_file() and validation_path.is_file() and
                (directory / "mapping.json").is_file() and (directory / "captions.json").is_file()):
            saved = json.loads(validation_path.read_text())
            if saved.get("mapping_sha256") == key and saved.get("output_fingerprint") == timeline.fingerprint(destination):
                return {"day": day, "status": "complete", "output": str(destination),
                        "duration_s": mapping["duration_s"], "clips": len(mapping["segments"]), "reused": True}
        temporary = directory / f".video.{os.getpid()}.part.mp4"
        temporary.unlink(missing_ok=True)
        try:
            result = video.concat_clips([s["normalized_file"] for s in mapping["segments"]], temporary, size=args.size)
            observed = stamp.probe(temporary)
            if observed["frames"] != mapping["frames"] or abs(observed["duration_s"] - mapping["duration_s"]) > .05:
                raise ValueError("Concatenated video does not contain the exact planned frame count/duration")
            caption_data = captions.build_day_captions(mapping, manifest, manifest_path=args.manifest)
            timeline.write_json(directory / "mapping.json", mapping)
            timeline.write_json(directory / "captions.json", caption_data)
            os.replace(temporary, destination)
            timeline.write_json(validation_path, {"checked_at": clock_now(), "mapping_sha256": key,
                "output_fingerprint": timeline.fingerprint(destination), "metadata": observed,
                "concat_validation": result, "caption_coverage": caption_data.get("coverage"),
                "limits": "Frame/timing checks; not a certificate of visual activity, home, person, or day/night consistency."})
            return {"day": day, "status": "complete", "output": str(destination),
                    "duration_s": mapping["duration_s"], "clips": len(mapping["segments"])}
        finally:
            temporary.unlink(missing_ok=True)
    except Exception as exc:
        return {"day": day, "status": "error", "error": str(exc)}


def run(args):
    raw = Path(args.manifest).read_bytes()
    manifest = json.loads(raw)
    plan = timeline.import_manifest(args.manifest, args.base_date, args.timezone)
    if args.source_overrides:
        overrides = json.loads(Path(args.source_overrides).read_text())
        for clip in plan["clips"]:
            if clip["id"] in overrides:
                clip["input"] = str(Path(overrides[clip["id"]]).resolve())
        for slot in manifest["slots"]:
            if slot["slot_id"] in overrides:
                slot["source"]["media_path"] = overrides[slot["slot_id"]]
    timeline.validate_plan(plan)
    selected_days = sorted(set(args.day) if args.day else {c["day"] for c in plan["clips"]})
    unknown = set(selected_days) - {c["day"] for c in plan["clips"]}
    if unknown:
        raise ValueError(f"Days have no selected video: {unknown}")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # A single coordinator owns this output tree; FFmpeg work runs in parallel inside it.
    with (root / "coordinator.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another coordinator is already using this output directory")
        plan_path = root / "plan.json"
        if plan_path.exists() and timeline.digest(json.loads(plan_path.read_text())) != timeline.digest(plan):
            raise ValueError("Output directory belongs to a different plan; use a new output directory")
        timeline.write_json(plan_path, plan)
        timeline.write_json(root / "input-manifest.json", manifest)
        font = timeline.find_font(args.font)
        calibration = stamp.calibrate_font(font, plan["style"]["text_height_px"])
        implementation = hashlib.sha256(Path(__file__).read_bytes() + Path(video.__file__).read_bytes() + Path(stamp.__file__).read_bytes()).hexdigest()
        clip_results, daily_results = {}, {}
        report = {"started_at": clock_now(), "state": "running", "base_date": args.base_date,
                  "canvas": [args.size, args.size], "fps": 1, "audio": False,
                  "input_manifest_sha256": hashlib.sha256(raw).hexdigest(), "days_requested": selected_days,
                  "workers": args.workers, "threads_per_worker": args.threads}

        def save():
            report.update(updated_at=clock_now(), clips=list(clip_results.values()),
                          days=[daily_results.get(d, {"day": d, "status": "pending"}) for d in selected_days])
            report["completed_days"] = sum(d.get("status") == "complete" for d in daily_results.values())
            timeline.write_json(root / "status.json", report)

        save()
        while True:
            queue = [c for c in plan["clips"] if c["day"] in selected_days and
                     clip_results.get(c["id"], {}).get("status") not in ("rendered", "reused", "error")]
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(render_segment, c, plan, args, root, font, calibration, implementation): c for c in queue}
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    clip_results[result["id"]] = result
                    print(json.dumps({k: v for k, v in result.items() if k != "metadata"}, ensure_ascii=False), flush=True)
                    day = result["day"]
                    ids = [c["id"] for c in plan["clips"] if c["day"] == day]
                    if all(clip_results.get(i, {}).get("status") in ("rendered", "reused") for i in ids):
                        daily_results[day] = finish_day(plan, day, clip_results, manifest, args, root)
                        print(json.dumps(daily_results[day], ensure_ascii=False), flush=True)
                    save()
            failed = any(r["status"] == "error" for r in clip_results.values()) or any(r["status"] == "error" for r in daily_results.values())
            done = len(daily_results) == len(selected_days) and all(r["status"] == "complete" for r in daily_results.values())
            if done or failed or not args.wait_inputs:
                report["state"] = "complete" if done else ("error" if failed else "pending_inputs")
                report["finished_at"] = clock_now()
                save()
                return 0 if done else (1 if failed else 3)
            save()
            time.sleep(args.poll_seconds)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-date", required=True)
    parser.add_argument("--timezone", default="Asia/Seoul")
    parser.add_argument("--size", type=timeline.bounded_int(320, 4096), default=1024)
    parser.add_argument("--workers", type=timeline.bounded_int(1, 16), default=4)
    parser.add_argument("--threads", type=timeline.bounded_int(1, 32), default=4)
    parser.add_argument("--preset", choices=("veryfast", "fast", "medium"), default="veryfast")
    parser.add_argument("--crf", type=timeline.bounded_int(0, 30), default=18)
    parser.add_argument("--font")
    parser.add_argument("--source-overrides")
    parser.add_argument("--day", type=timeline.bounded_int(1, 366), action="append")
    parser.add_argument("--wait-inputs", action="store_true")
    parser.add_argument("--poll-seconds", type=timeline.bounded_int(1, 3600), default=30)
    args = parser.parse_args(argv)
    try:
        if args.size % 2:
            raise ValueError("--size must be even")
        return run(args)
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f"daily-render: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
