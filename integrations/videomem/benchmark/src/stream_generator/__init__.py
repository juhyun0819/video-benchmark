import math
from pathlib import Path
from src.common import run, probe, save

def manifest(plan, clips, fps=24):
    if fps < 1:
        raise ValueError("FPS must be positive")
    lookup = {c["clip_id"]: c for c in clips}
    if plan["days"] < 1 or not plan["events"]:
        raise ValueError("Routine must contain at least one event and one day")
    if any(not 1 <= e["synthetic_day"] <= plan["days"] for e in plan["events"]):
        raise ValueError("Event day outside configured range")
    if len(lookup) != len(clips):
        raise ValueError("Duplicate clip IDs in catalog")
    events, days, cursor = [], [], 0
    for day in range(1, plan["days"]+1):
        begin = cursor
        for event in [e for e in plan["events"] if e["synthetic_day"] == day]:
            clip = lookup[event["clip_id"]]
            if event["activity"] != clip["activity"]:
                raise ValueError("Routine activity differs from source clip label")
            frames = max(1, int(math.floor((clip["end_time"]-clip["start_time"])*fps + 0.5)))
            end = cursor + frames
            events.append({**clip, **event, "persona_id": plan["persona_id"],
                           "stream_start": cursor/fps, "stream_end": end/fps,
                           "day_start": (cursor-begin)/fps, "day_end": (end-begin)/fps,
                           "frames": frames})
            cursor = end
        days.append({"synthetic_day": day, "stream_start": begin/fps, "stream_end": cursor/fps})
    return {**plan, "events": events, "day_intervals": days, "duration": cursor/fps,
            "fps": fps, "timing_basis": "frame-quantized planned timeline", "rendered": False}

def concat(paths, output):
    listing = output.with_suffix(".concat.txt")
    listing.write_text("".join("file '" + str(p.resolve()).replace("'", "'\\''") + "'\n" for p in paths))
    try:
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
             "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(output)])
    finally:
        listing.unlink(missing_ok=True)

def generate(plan, clips, output, render=False, width=640, height=360, fps=24, remove_audio=True):
    from src.validation import validate
    if width < 2 or height < 2 or width % 2 or height % 2:
        raise ValueError("Output dimensions must be positive even numbers")
    data = manifest(plan, clips, fps)
    report = validate(data)
    if report["errors"]:
        raise ValueError(str(report["errors"]))
    output = Path(output).resolve()
    sources = {Path(e["source_video"]).resolve() for e in data["events"]}
    if any(output == s or output in s.parents for s in sources):
        raise ValueError("Output directory must not contain source videos")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new or empty output directory to preserve prior artifacts")
    output.mkdir(parents=True, exist_ok=True)
    save(output/"validation.json", report)
    if render:
        parts = output/"parts"
        parts.mkdir()
        daily = []
        for day in range(1, plan["days"]+1):
            paths = []
            for i, event in enumerate(data["events"]):
                if event["synthetic_day"] != day:
                    continue
                target = parts/f"{i:05d}.mp4"
                info = probe(event["source_video"])
                args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(event["start_time"]), "-i", event["source_video"]]
                if not remove_audio and not info["has_audio"]:
                    args += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
                args += ["-map", "0:v:0", "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,fps={fps},tpad=stop_mode=clone:stop_duration=1,setsar=1",
                         "-frames:v", str(event["frames"]), "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p", "-threads", "2"]
                if remove_audio:
                    args += ["-an"]
                else:
                    args += ["-map", "0:a:0" if info["has_audio"] else "1:a:0", "-af", "apad", "-t", str(event["frames"]/fps), "-c:a", "aac", "-ar", "48000", "-ac", "2"]
                run(args + [str(target)])
                paths.append(target)
            if paths:
                daypath = output/f"day_{day:02d}.mp4"
                concat(paths, daypath)
                daily.append(daypath)
        if not daily:
            raise ValueError("Cannot render an empty routine")
        concat(daily, output/"full_stream.mp4")
        actual = probe(output/"full_stream.mp4")["duration"]
        if abs(actual-data["duration"]) > max(0.15, 2/fps):
            raise RuntimeError(f"Rendered duration mismatch: {actual} vs {data['duration']}")
        data.update(rendered=True, actual_duration=actual)
    save(output/"manifest.json", data)
    return data
