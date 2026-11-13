#!/usr/bin/env python3
"""Render a block-mode timeline manifest to proxy videos (one per synthetic day).

The day's clips play back to back: each second shows the day and its synthetic
clock, the block (slot role), the real source (dataset, date and real time) and
the clip's caption. Spans without footage are cut out, so the clock jumps there.
No source media is read: this is a timing-exact stand-in until the real frames
(NextMe tars, Ego-Exo4D takes, YouTube, Ego4D) are downloaded.

usage: python3 -I scripts/render_proxy.py --manifest <manifest.json> --timeline <timeline.json>
       --out <dir> [--speed 1] [--fps 1] [--size 960x540] [--days 1,2,...] [--jobs 8]

--speed 1 renders real time at --fps (default 1 fps, as the benchmark videos).
--speed 60 renders a quick-look where one second of video is one minute of the day.
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor

ROLE_LABEL = {
    "home_morning": "HOME (morning)", "home_afternoon": "HOME (afternoon)", "home_evening": "HOME (evening)",
    "chores": "HOME (chores)", "work_am": "WORK (morning)", "work_pm": "WORK (afternoon)", "breakfast": "BREAKFAST",
    "lunch": "LUNCH", "dinner": "DINNER", "exercise": "EXERCISE (gym)", "hobby": "HOBBY 1 (LEGO)",
    "hobby_2": "HOBBY 2 (guitar)", "outing": "TRIP - outing", "outing_pm": "TRIP - outing (pm)",
    "lodging_morning": "TRIP - lodging (morning)", "lodging_evening": "TRIP - lodging (evening)", "commute": "COMMUTE",
}
# ASS colours are &HBBGGRR&
ROLE_COLOUR = {
    "home": "&H00AC7C4F&", "work": "&H004F79C0&", "meal": "&H003DA7D6&", "exercise": "&H003A58D0&",
    "hobby": "&H006E9E5E&", "hobby_2": "&H00D08F3A&", "trip": "&H00A0A03F&", "commute": "&H008A7F7A&",
}


def colour_of(role):
    if role.startswith("home") or role == "chores":
        return ROLE_COLOUR["home"]
    if role.startswith("work"):
        return ROLE_COLOUR["work"]
    if role in ("breakfast", "lunch", "dinner"):
        return ROLE_COLOUR["meal"]
    if role in ("outing", "outing_pm", "lodging_morning", "lodging_evening"):
        return ROLE_COLOUR["trip"]
    return ROLE_COLOUR.get(role, "&H00FFFFFF&")


def clock(sec):
    sec = int(round(sec))
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}"


def ass_time(sec):
    cs = int(round(sec * 100))
    return f"{cs // 360000}:{cs % 360000 // 6000:02d}:{cs % 6000 // 100:02d}.{cs % 100:02d}"


def esc(text):
    return text.replace("\\", "/").replace("{", "(").replace("}", ")").replace("\n", " ")


def real_source(c):
    media = c["source"].get("media", {})
    ds = c["source"]["dataset"]
    if "start_wallclock" in media:
        t = dt.datetime.strptime(media["start_wallclock"], "%Y-%m-%d %H:%M:%S") + dt.timedelta(seconds=c["source_start_s"])
        return f"{ds}  real {t:%Y-%m-%d %H:%M}"
    return f"{ds}  {c['source']['video_id'][:24]}  @{clock(c['source_start_s'])}"


def parse_hhmm(s):
    h, m = s.split(":")
    return int(h) * 3600 + int(m) * 60


def build_ass(day, slots, width, height, speed):
    base = (day["day_index"] - 1) * 86400
    t0, t1 = parse_hhmm(day["wake"]), parse_hhmm(day["sleep"])
    roles = {s["slot_id"]: s["role"] for s in slots}
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Head,DejaVu Sans,{height // 22},&H00FFFFFF&,&H00FFFFFF&,&H00000000&,&H00000000&,1,0,0,0,100,100,0,0,1,1,0,7,24,24,20,1
Style: Clock,DejaVu Sans Mono,{height // 14},&H00FFFFFF&,&H00FFFFFF&,&H00000000&,&H00000000&,1,0,0,0,100,100,0,0,1,1,0,9,24,24,16,1
Style: Role,DejaVu Sans,{height // 13},&H00FFFFFF&,&H00FFFFFF&,&H00000000&,&H00000000&,1,0,0,0,100,100,0,0,1,1,0,4,40,40,0,1
Style: Body,DejaVu Sans,{height // 26},&H00E6E6E6&,&H00FFFFFF&,&H00000000&,&H00000000&,0,0,0,0,100,100,0,0,1,1,0,7,40,40,0,1
Style: Foot,DejaVu Sans,{height // 34},&H00909090&,&H00FFFFFF&,&H00000000&,&H00000000&,0,0,0,0,100,100,0,0,1,0,0,1,24,24,14,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev = []

    def add(a, b, style, text):
        # a, b are seconds of output video (before the speed-up)
        if b <= a:
            return
        ev.append(f"Dialogue: 0,{ass_time(a / speed)},{ass_time(b / speed)},{style},,0,0,0,,{text}")

    # Spans without footage are cut out: clips play back to back and only the clock jumps.
    spans, prev_end = [], t0
    for c in sorted(day["clips"], key=lambda c: c["synthetic_start_s"]):
        a = max(c["synthetic_start_s"] - base, prev_end, t0)
        b = min(c["synthetic_end_s"] - base, t1)
        if b > a:
            spans.append((a, b, c))
            prev_end = b
    out = 0.0
    step = 60 if speed < 60 else 600
    for a, b, c in spans:
        role = roles.get(c["slot_id"], "commute")
        label = ROLE_LABEL.get(role, role.upper())
        add(out, out + b - a, "Role", f"{{\\pos(40,{int(height * 0.36)})\\c{colour_of(role)}}}{esc(label)}")
        caption = "\\N".join(textwrap.wrap(esc(c["provenance"]["spec"]["activity"].removeprefix("#Summary ")), 62)[:4])
        body = f"{{\\pos(40,{int(height * 0.46)})}}{esc(real_source(c))}\\N{{\\fs{height // 22}}}{caption}"
        add(out, out + b - a, "Body", body)
        for m in range(int(a - a % step), int(b), step):
            s, e = max(m, a), min(m + step, b)
            add(out + s - a, out + e - a, "Clock", clock(m))
        out += b - a
    title = f"D{day['day_index']}  {day['weekday']}  {day['day_type']}"
    add(0, out, "Head", esc(title))
    pace = "1 s = 1 s of footage" if speed == 1 else f"{speed:g}x speed: 1 s = {speed:g} s of footage"
    add(0, out, "Foot", f"PROXY render - spans without footage cut ({pace}); source frames not rendered yet")
    return head + "\n".join(ev) + "\n", out


def render(day, slots, out_dir, width, height, fps, speed):
    ass, dur = build_ass(day, slots, width, height, speed)
    name = f"D{day['day_index']:02d}"
    ass_path = os.path.join(out_dir, f"{name}.ass")
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(ass)
    out = os.path.join(out_dir, f"{name}.mp4")
    length = max(1.0, dur / speed)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
           f"color=c=0x141414:s={width}x{height}:r={fps}:d={length:.3f}", "-vf", f"ass={ass_path}",
           "-c:v", "libx264", "-preset", "veryfast", "-tune", "stillimage", "-crf", "30", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", out]
    subprocess.run(cmd, check=True)
    return name, length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--timeline", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--speed", type=float, default=1)
    ap.add_argument("--fps", type=float, default=1)
    ap.add_argument("--size", default="960x540")
    ap.add_argument("--days", default="")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    width, height = (int(x) for x in a.size.split("x"))
    m = json.load(open(a.manifest))
    t = json.load(open(a.timeline))
    slots = {d["day_index"]: d["slots"] for d in t["schedule"]["days"]}
    want = {int(x) for x in a.days.split(",") if x}
    days = [d for d in m["days"] if not want or d["day_index"] in want]
    os.makedirs(a.out, exist_ok=True)
    with ThreadPoolExecutor(a.jobs) as ex:
        done = list(ex.map(lambda d: render(d, slots[d["day_index"]], a.out, width, height, a.fps, a.speed), days))
    # The playlist lists every day rendered into the folder, so a --days rerun keeps the others.
    with open(os.path.join(a.out, "playlist.txt"), "w") as f:
        for name in sorted(p for p in os.listdir(a.out) if p.startswith("D") and p.endswith(".mp4")):
            f.write(f"file '{name}'\n")
    total = sum(length for _, length in done)
    print(json.dumps({"days": len(done), "video_seconds": round(total), "video_hours": round(total / 3600, 2), "out": a.out}))


if __name__ == "__main__":
    sys.exit(main())
