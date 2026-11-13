"""Render a block-mode timeline manifest to benchmark videos from real frames, one video per day.

Each synthetic second shows one real source frame: NextMe frames from the fetcher's 1024 cache,
Ego4D and YouTube frames cut from their videos (centre square, 1024, NextMe-like circle). Every
frame carries one timestamp line at the bottom left. Spans without footage are cut out, so the
clock jumps there. Spec: docs/superpowers/specs/2026-11-13-nextme-frame-render-design.md

    python3 scripts/render_frames.py <manifest> [--days 1|1-7,10] [--jobs N] [--warp 0] [--out ...]
"""
from __future__ import annotations

import bisect
import argparse
import datetime
import json
import math
import multiprocessing
import os
import shutil
import subprocess
import time
import sys
import tempfile
from collections import Counter
from collections.abc import Collection, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_nextme_frames import SAVED_RE, load_index, parse_days  # noqa: E402

START_DATE = datetime.date(2025, 4, 7)  # day 1 (a Monday)


class DayError(Exception):
    """Stops one day only; the other days go on."""


@dataclass(frozen=True)
class Pick:
    t: int  # absolute synthetic second
    clip: int  # index into day["clips"]
    src: float  # source time of this frame


def day_picks(day: dict) -> list[Pick]:
    """One frame per integer synthetic second inside each clip; gaps between clips get none."""
    picks: list[Pick] = []
    order = sorted(range(len(day["clips"])), key=lambda i: day["clips"][i]["synthetic_start_s"])
    for i in order:
        c = day["clips"][i]
        a, b = c["synthetic_start_s"], c["synthetic_end_s"]
        t = math.ceil(a)
        while t < b:
            picks.append(Pick(t, i, c["source_start_s"] + (t - a)))
            t += 1
    picks.sort(key=lambda p: p.t)
    return picks


def stamp_text(day_index: int, t: int) -> str:
    """'YYYY-MM-DD HH:MM:SS' for absolute second t of day `day_index` (no time zone)."""
    rel = t - (day_index - 1) * 86400
    date = START_DATE + datetime.timedelta(days=day_index - 1 + rel // 86400)
    rel %= 86400
    return f"{date.isoformat()} {rel // 3600:02d}:{rel % 3600 // 60:02d}:{rel % 60:02d}"


# ---------------------------------------------------------------- NextMe

NEXTME_SUBSTITUTE_S = 2  # a missing frame may be replaced by one this many seconds away
NEXTME_END_CLAMP_S = 5  # past the recording's end, hold the last frame this long


def nextme_frames(frames_dir: Path) -> dict[int, Path]:
    """Frame number -> cached 1024 JPEG (hidden temp files skipped)."""
    out: dict[int, Path] = {}
    if not Path(frames_dir).is_dir():
        return out
    for p in Path(frames_dir).iterdir():
        m = SAVED_RE.search(p.name)
        if m and not p.name.startswith("."):
            out[int(m.group(1))] = p
    return out


def nextme_last_frame(index_path: Path) -> int | None:
    """Last frame number of the recording, known only when the fetcher walked the whole tar."""
    if not Path(index_path).exists():
        return None
    known, _, ended = load_index(Path(index_path))
    return max(known) if ended and known else None


def pick_nextme(src: float, frames: Collection[int], last: int | None) -> tuple[int, str]:
    """(frame number, 'exact' | 'substituted' | 'clamped'); DayError when no frame may stand in."""
    n = math.ceil(src - 0.5)  # round half down
    if n in frames:
        return n, "exact"
    if last is not None and n > last:
        if n - last <= NEXTME_END_CLAMP_S and last in frames:
            return last, "clamped"
        raise DayError(f"NextMe frame {n} is {n - last} s past the recording end ({last})")
    for d in range(1, NEXTME_SUBSTITUTE_S + 1):
        for cand in (n - d, n + d):
            if cand in frames:
                return cand, "substituted"
    raise DayError(f"NextMe frame {n} missing with no frame within {NEXTME_SUBSTITUTE_S} s")


# ---------------------------------------------------------------- video timing


@dataclass
class VideoInfo:
    times: list[float]  # source time of each frame (packet pts - pts0), ascending
    pts0: float  # stream start_time
    width: int
    height: int
    rotation: int  # display-matrix rotation as ffprobe reports it, mod 360


def _run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tail = "\n".join(r.stderr.strip().splitlines()[-20:])
        raise DayError(f"{cmd[0]} failed ({r.returncode}): {tail}")
    return r.stdout


def probe_video(path: Path, cache_dir: Path | None = None) -> VideoInfo:
    """Packet times and stream facts from ffprobe (no decoding). Cached per file size and mtime."""
    path = Path(path)
    cache = None
    if cache_dir is not None:
        st = path.stat()
        cache = Path(cache_dir) / f"{path.name}.{st.st_size}-{st.st_mtime_ns}.json"
        if cache.exists():
            try:
                return VideoInfo(**json.loads(cache.read_text()))
            except (OSError, ValueError, TypeError):
                pass  # unreadable or cut short: probe again
    meta = json.loads(_run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=start_time,width,height:stream_side_data=rotation", "-of", "json", str(path)]))
    stream = meta["streams"][0]
    pts = sorted(float(x) for x in _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                         "packet=pts_time", "-of", "csv=p=0", str(path)]).split()
                 if x not in ("", "N/A"))
    if not pts:
        raise DayError(f"no video packets in {path}")
    try:
        pts0 = float(stream.get("start_time"))
    except (TypeError, ValueError):
        pts0 = pts[0]
    rotation = 0
    for sd in stream.get("side_data_list", []):
        if "rotation" in sd:
            rotation = int(round(float(sd["rotation"]))) % 360
    info = VideoInfo([p - pts0 for p in pts], pts0, int(stream["width"]), int(stream["height"]), rotation)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(f".{cache.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(asdict(info)))
        tmp.replace(cache)
    return info


def pick_video(src: float, times: list[float], fps: float) -> tuple[int, bool]:
    """(frame index, clamped): nearest frame, earlier on a tie; up to one frame past the end clamps."""
    if src > times[-1]:
        if src - times[-1] <= 1 / fps + 1e-9:
            return len(times) - 1, True
        raise DayError(f"source time {src:.3f} s is past the video end ({times[-1]:.3f} s)")
    i = bisect.bisect_left(times, src)
    if i == 0:
        return 0, False
    before, after = times[i - 1], times[i]
    return (i - 1 if src - before <= after - src else i), False


# ---------------------------------------------------------------- video frames

SELECT_BATCH = 120  # frames chosen per ffmpeg run (one seek each)
_ROTATE = {0: None, 90: "transpose=1", 180: "hflip,vflip", 270: "transpose=2"}


def extract_video_frames(path: Path, info: VideoInfo, indices: list[int], extra_rotation: int = 0,
                         size: int = 1024) -> Iterator[Image.Image]:
    """Yield RGB size x size frames for `indices` (non-decreasing; repeats allowed), in order.

    ffmpeg applies the display-matrix rotation (autorotate), then `extra_rotation` clockwise, a
    centre square crop and a Lanczos scale. Frames are chosen by exact pts with `select`."""
    if list(indices) != sorted(indices):
        raise ValueError("indices must be non-decreasing")
    if extra_rotation not in _ROTATE:
        raise ValueError(f"rotation must be one of {sorted(_ROTATE)}")
    counts = Counter(indices)
    uniq = sorted(counts)
    gaps = [b - a for a, b in zip(info.times, info.times[1:]) if b > a]
    tol = 0.25 * (min(gaps) if gaps else 1.0)
    frame_bytes = size * size * 3
    for b in range(0, len(uniq), SELECT_BATCH):
        batch = uniq[b:b + SELECT_BATCH]
        absolute = [info.times[i] + info.pts0 for i in batch]
        expr = "+".join(f"lt(abs(t-{a:.6f}),{tol:.6f})" for a in absolute)
        filters = [f"select='{expr}'"]
        if _ROTATE[extra_rotation]:
            filters.append(_ROTATE[extra_rotation])
        filters += ["crop='min(iw,ih)':'min(iw,ih)'", f"scale={size}:{size}:flags=lanczos"]
        # With -copyts, -ss counts from the file's start time while select's t is absolute.
        seek = max(0.0, info.times[batch[0]] - 1)
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{seek:.6f}", "-copyts",
               "-i", str(path), "-vf", ",".join(filters), "-fps_mode", "passthrough",
               "-frames:v", str(len(batch)), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        with tempfile.TemporaryFile() as err:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
            got = 0
            try:
                for idx in batch:
                    buf = proc.stdout.read(frame_bytes)
                    if len(buf) < frame_bytes:
                        break
                    got += 1
                    im = Image.frombytes("RGB", (size, size), buf)
                    for _ in range(counts[idx]):
                        yield im
                rc = proc.wait()
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                proc.stdout.close()
            if got != len(batch) or rc != 0:
                err.seek(0)
                tail = "\n".join(err.read().decode("utf-8", "replace").strip().splitlines()[-20:])
                raise DayError(f"ffmpeg gave {got}/{len(batch)} frames from {Path(path).name} (exit {rc}): {tail}")


# ---------------------------------------------------------------- picture

CIRCLE = {"center": (494, 507), "r_in": 500, "r_out": 560}  # measured on 300 NextMe frames
FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf")
OLD_TEXT_BOX = (0, 995, 200, 1024)  # NextMe's own HKT text lives inside this box
STAMP_SAMPLE = "0000-00-00 00:00:00"


def circle_mask(size: int = 1024, center=CIRCLE["center"], r_in: float = CIRCLE["r_in"],
                r_out: float = CIRCLE["r_out"]) -> Image.Image:
    """'L' mask: 255 inside r_in, linear fall to 0 at r_out, 0 outside."""
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.hypot(xx - center[0], yy - center[1])
    v = np.clip((r_out - r) / (r_out - r_in), 0.0, 1.0) * 255
    return Image.fromarray(np.rint(v).astype(np.uint8), "L")


def apply_circle(img: Image.Image, mask: Image.Image) -> Image.Image:
    return Image.composite(img, Image.new("RGB", img.size, (0, 0, 0)), mask)


def fisheye_warp(img: Image.Image, strength: float, center=CIRCLE["center"],
                 radius: float = CIRCLE["r_in"]) -> Image.Image:
    """Barrel warp: output radius u reads source radius u*(1+k*u^2)/(1+k) (k = strength), so the
    centre grows and the rim (u = 1) stays put. Bilinear; outside the picture is black."""
    if strength == 0:
        return img
    k = float(strength)
    src = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = src.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - center[0], yy - center[1]
    u2 = (dx * dx + dy * dy) / (radius * radius)
    scale = (1 + k * u2) / (1 + k)
    sx, sy = center[0] + dx * scale, center[1] + dy * scale
    x0, y0 = np.floor(sx).astype(np.int64), np.floor(sy).astype(np.int64)
    fx, fy = (sx - x0)[..., None], (sy - y0)[..., None]
    padded = np.zeros((h + 2, w + 2, 3), np.float32)
    padded[1:-1, 1:-1] = src

    def at(x, y):
        return padded[np.clip(y + 1, 0, h + 1), np.clip(x + 1, 0, w + 1)]

    out = (at(x0, y0) * (1 - fx) * (1 - fy) + at(x0 + 1, y0) * fx * (1 - fy)
           + at(x0, y0 + 1) * (1 - fx) * fy + at(x0 + 1, y0 + 1) * fx * fy)
    outside = (sx < -1) | (sx > w) | (sy < -1) | (sy > h)
    out[outside] = 0
    return Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8), "RGB")


class Stamper:
    """Black box over NextMe's own text, then our timestamp in white at the bottom left."""

    def __init__(self, font_path: Path = FONT_PATH, digit_px: int = 20, size: int = 1024):
        font_path = Path(font_path)
        if not font_path.is_file():
            raise FileNotFoundError(f"timestamp font not found: {font_path}")
        self.digit_px = digit_px
        self.font = min((ImageFont.truetype(str(font_path), n) for n in range(6, 4 * digit_px)),
                        key=lambda f: abs((f.getbbox("0")[3] - f.getbbox("0")[1]) - digit_px))
        self.pad = math.ceil(max(3, digit_px / 3))
        left, _, right, _ = self.font.getbbox(STAMP_SAMPLE)
        d_top, d_bottom = self.font.getbbox("0")[1], self.font.getbbox("0")[3]
        bottom = size - 10
        self.origin = (10 - left, bottom - d_bottom)
        ink = (10, bottom - (d_bottom - d_top), 10 + (right - left), bottom)
        box = (min(OLD_TEXT_BOX[0], ink[0] - self.pad), min(OLD_TEXT_BOX[1], ink[1] - self.pad),
               max(OLD_TEXT_BOX[2], ink[2] + self.pad), max(OLD_TEXT_BOX[3], ink[3] + self.pad))
        self.box = (max(0, box[0]), max(0, box[1]), min(size, box[2]), min(size, box[3]))

    def draw(self, img: Image.Image, text: str) -> Image.Image:
        out = img.copy()
        d = ImageDraw.Draw(out)
        d.rectangle((self.box[0], self.box[1], self.box[2] - 1, self.box[3] - 1), fill=(0, 0, 0))
        d.text(self.origin, text, font=self.font, fill=(255, 255, 255))
        return out


# ---------------------------------------------------------------- one day

RENDER_VERSION = 1
NEXTME = "nextme800"
VIDEO_LABEL = {"ego4d": "Ego4D", "youtube": "YouTube"}


@dataclass(frozen=True)
class RenderConfig:
    media_root: Path = Path("data/media")
    warp: float = 0.0
    rotation: dict = field(default_factory=lambda: {"ego4d": 0, "youtube": 0})
    crf: int = 18
    digit_px: int = 20
    font_path: Path = FONT_PATH
    size: int = 1024


def ENCODE_ARGS(crf: int) -> list[str]:
    return ["-an", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", "-g", "60",
            "-movflags", "+faststart"]


def settings_key(manifest_info: dict, cfg: RenderConfig) -> dict:
    """Everything that changes the rendered output; a day is redone when any of it differs."""
    manifest = {k: manifest_info.get(k) for k in ("id", "revision", "created_at")}  # not the path's spelling
    return {"render_version": RENDER_VERSION, "manifest": manifest, "start_date": START_DATE.isoformat(),
            "digit_px": cfg.digit_px, "font": Path(cfg.font_path).name, "circle": json.loads(json.dumps(CIRCLE)),
            "warp": cfg.warp, "rotation": dict(sorted(cfg.rotation.items())), "size": cfg.size,
            "encode": ENCODE_ARGS(cfg.crf)}


def video_path(media_root: Path, dataset: str, video_id: str) -> Path:
    if dataset == "ego4d":
        return Path(media_root) / "ego4d" / "v2" / "full_scale" / f"{video_id}.mp4"
    if dataset == "youtube":
        return Path(media_root) / "youtube" / f"{video_id}.mp4"
    raise DayError(f"unknown video dataset {dataset!r}")


class _NextmeCache:
    def __init__(self, media_root: Path):
        self.root = Path(media_root) / "nextme"
        self._frames: dict[str, dict[int, Path]] = {}
        self._last: dict[str, int | None] = {}

    def frames(self, vid: str) -> dict[int, Path]:
        if vid not in self._frames:
            self._frames[vid] = nextme_frames(self.root / "frames" / vid)
        return self._frames[vid]

    def last(self, vid: str) -> int | None:
        if vid not in self._last:
            self._last[vid] = nextme_last_frame(self.root / "index" / f"{vid}.jsonl")
        return self._last[vid]


def _runs(picks: list[Pick]) -> list[tuple[int, list[Pick]]]:
    """Consecutive picks of the same clip."""
    runs: list[tuple[int, list[Pick]]] = []
    for p in picks:
        if runs and runs[-1][0] == p.clip:
            runs[-1][1].append(p)
        else:
            runs.append((p.clip, [p]))
    return runs


def check_day(day: dict, cfg: RenderConfig) -> list[str]:
    """Missing sources for this day as readable lines; empty when the day can be rendered."""
    nm = _NextmeCache(cfg.media_root)
    missing_frames: dict[str, int] = {}
    problems: list[str] = []
    for ci, run in _runs(day_picks(day)):
        src = day["clips"][ci]["source"]
        vid = src["video_id"]
        if src["dataset"] == NEXTME:
            frames, last = nm.frames(vid), nm.last(vid)
            for p in run:
                try:
                    pick_nextme(p.src, frames, last)
                except DayError:
                    missing_frames[vid] = missing_frames.get(vid, 0) + 1
        else:
            line = f"{VIDEO_LABEL.get(src['dataset'], src['dataset'])} {vid}: file missing"
            if not video_path(cfg.media_root, src["dataset"], vid).is_file() and line not in problems:
                problems.append(line)
    return [f"NextMe {vid}: {n} frames missing" for vid, n in missing_frames.items()] + problems


def day_name(day_index: int) -> str:
    return f"D{day_index:02d}"


def day_is_done(out_dir: Path, day_index: int, key: dict) -> bool:
    name = day_name(day_index)
    rep, mp4 = Path(out_dir) / f"{name}.render.json", Path(out_dir) / f"{name}.mp4"
    if not (rep.exists() and mp4.exists()):
        return False
    try:
        return json.loads(rep.read_text()).get("settings") == key
    except (OSError, json.JSONDecodeError):
        return False


def render_day(manifest_info: dict, day: dict, cfg: RenderConfig, out_dir: Path) -> dict:
    """Render one day to Dnn.mp4 + Dnn.frames.jsonl + Dnn.render.json (written last, atomically)."""
    started = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = day_name(day["day_index"])
    for old in out_dir.glob(f".{name}.tmp-*"):
        old.unlink()
    tmp = {k: out_dir / f".{name}.tmp-{os.getpid()}.{k}" for k in ("mp4", "frames.jsonl", "render.json")}
    size = cfg.size
    stamper = Stamper(cfg.font_path, cfg.digit_px, size)
    mask = circle_mask(size)
    nm = _NextmeCache(cfg.media_root)
    picks = day_picks(day)
    report = {"day_index": day["day_index"], "frames": 0, "clip_frames": [], "substitutions": [], "clamped": [],
              "videos": {}}
    err = tempfile.TemporaryFile()
    enc = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{size}x{size}", "-r", "1", "-i", "-", *ENCODE_ARGS(cfg.crf), "-f", "mp4",
                            str(tmp["mp4"])], stdin=subprocess.PIPE, stderr=err)

    def encoder_tail() -> str:
        err.seek(0)
        return "\n".join(err.read().decode("utf-8", "replace").strip().splitlines()[-20:])

    try:
        with open(tmp["frames.jsonl"], "w") as fl:
            for ci, run in _runs(picks):
                c = day["clips"][ci]
                ds, vid = c["source"]["dataset"], c["source"]["video_id"]
                report["clip_frames"].append({"clip": ci, "video_id": vid, "frames": len(run)})
                for p, img, extra in _clip_frames(c, run, cfg, nm, mask, out_dir, report):
                    stamp = stamp_text(day["day_index"], p.t)
                    frame = stamper.draw(img, stamp)
                    try:
                        enc.stdin.write(frame.tobytes())
                    except BrokenPipeError:
                        raise DayError(f"encoder stopped: {encoder_tail()}") from None
                    fl.write(json.dumps({"t": p.t, "stamp": stamp, "dataset": ds, "video_id": vid,
                                         "slot_id": c.get("slot_id"), **extra}) + "\n")
                    report["frames"] += 1
        enc.stdin.close()
        if enc.wait() != 0:
            raise DayError(f"encoder failed: {encoder_tail()}")
        if report["frames"] != len(picks):
            raise DayError(f"rendered {report['frames']} frames, planned {len(picks)}")
        report.update({"circle": CIRCLE, "warp": cfg.warp,
                       "stamp": {"font": str(cfg.font_path), "digit_px": cfg.digit_px, "box": list(stamper.box)},
                       "encode": ENCODE_ARGS(cfg.crf), "manifest": manifest_info,
                       "settings": settings_key(manifest_info, cfg), "seconds": round(time.time() - started, 1)})
        tmp["render.json"].write_text(json.dumps(report, indent=1))
        os.replace(tmp["mp4"], out_dir / f"{name}.mp4")
        os.replace(tmp["frames.jsonl"], out_dir / f"{name}.frames.jsonl")
        os.replace(tmp["render.json"], out_dir / f"{name}.render.json")
        return report
    except BaseException:
        if enc.poll() is None:
            enc.kill()
            enc.wait()
        for f in tmp.values():
            f.unlink(missing_ok=True)
        raise
    finally:
        if enc.stdin and not enc.stdin.closed:
            try:
                enc.stdin.close()
            except BrokenPipeError:
                pass
        err.close()


def _clip_frames(c: dict, run: list[Pick], cfg: RenderConfig, nm: _NextmeCache, mask: Image.Image,
                 out_dir: Path, report: dict):
    """Yield (pick, size x size RGB image, frames.jsonl extras) for one clip's picks."""
    src = c["source"]
    ds, vid = src["dataset"], src["video_id"]
    if ds == NEXTME:
        frames, last = nm.frames(vid), nm.last(vid)
        for p in run:
            n, kind = pick_nextme(p.src, frames, last)
            if kind == "substituted":
                report["substitutions"].append({"video_id": vid, "src": p.src, "frame": n})
            elif kind == "clamped":
                report["clamped"].append({"dataset": ds, "video_id": vid, "src": p.src, "used": n})
            img = Image.open(frames[n])
            img = img.convert("RGB")
            if img.size != (cfg.size, cfg.size):
                raise DayError(f"NextMe {vid} frame {n} is {img.size}, expected {cfg.size}x{cfg.size}")
            yield p, img, {"frame": n, "kind": kind}
        return
    path = video_path(cfg.media_root, ds, vid)
    if not path.is_file():
        raise DayError(f"{VIDEO_LABEL.get(ds, ds)} {vid}: file missing")
    info = probe_video(path, Path(cfg.media_root) / ".probe")
    fps = float(src.get("media", {}).get("fps") or 30)
    chosen, kinds = [], []
    for p in run:
        k, clamped = pick_video(p.src, info.times, fps)
        if clamped:
            report["clamped"].append({"dataset": ds, "video_id": vid, "src": p.src, "used": k})
        chosen.append(k)
        kinds.append("clamped" if clamped else "nearest")
    report["videos"][vid] = {"pts0": info.pts0, "rotation_metadata": info.rotation,
                             "rotation_extra": cfg.rotation.get(ds, 0)}
    imgs = extract_video_frames(path, info, chosen, extra_rotation=cfg.rotation.get(ds, 0), size=cfg.size)
    for p, k, kind, img in zip(run, chosen, kinds, imgs):
        img = apply_circle(fisheye_warp(img, cfg.warp), mask)
        yield p, img, {"pts": round(info.times[k] + info.pts0, 6), "kind": kind}


# ---------------------------------------------------------------- command line

DEFAULT_OUT = Path("data/renders/nextme-p00/benchmark")
PREVIEW_CELL = 256


def check_tools(font_path: Path) -> list[str]:
    missing = [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]
    if "ffmpeg" not in missing:
        enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
        if "libx264" not in enc:
            missing.append("ffmpeg encoder libx264")
    if not Path(font_path).is_file():
        missing.append(f"font {font_path}")
    return missing


def warp_preview(manifest: dict, cfg: RenderConfig, strengths: list[float], out: Path) -> Path:
    """Grid image: one row per warp strength, three frames (start, middle, end) of the first Ego4D clip."""
    clip = next((c for d in manifest["days"] for c in d["clips"] if c["source"]["dataset"] == "ego4d"), None)
    if clip is None:
        raise DayError("no Ego4D clip in the chosen days")
    vid = clip["source"]["video_id"]
    path = video_path(cfg.media_root, "ego4d", vid)
    info = probe_video(path, Path(cfg.media_root) / ".probe")
    fps = float(clip["source"].get("media", {}).get("fps") or 30)
    s0, s1 = clip["source_start_s"], clip["source_end_s"]
    idx = sorted(pick_video(min(t, info.times[-1]), info.times, fps)[0] for t in (s0, (s0 + s1) / 2, s1 - 1))
    frames = list(extract_video_frames(path, info, idx, extra_rotation=cfg.rotation.get("ego4d", 0), size=cfg.size))
    mask = circle_mask(cfg.size)
    font = ImageFont.truetype(str(cfg.font_path), 20)
    grid = Image.new("RGB", (PREVIEW_CELL * len(frames), PREVIEW_CELL * len(strengths)))
    for r, k in enumerate(strengths):
        for c, img in enumerate(frames):
            cell = apply_circle(fisheye_warp(img, k), mask).resize((PREVIEW_CELL, PREVIEW_CELL), Image.LANCZOS)
            ImageDraw.Draw(cell).text((6, 4), f"warp {k:g}", font=font, fill=(255, 255, 0))
            grid.paste(cell, (c * PREVIEW_CELL, r * PREVIEW_CELL))
    Path(out).mkdir(parents=True, exist_ok=True)
    dest = Path(out) / "warp-preview.jpg"
    grid.save(dest, quality=92)
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--days", help="day indexes, e.g. 1 or 1-7,10 (default: all)")
    ap.add_argument("--jobs", type=int, default=4, help="days rendered at the same time")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--media-root", type=Path, default=Path("data/media"))
    ap.add_argument("--warp", type=float, default=0.0, help="fisheye warp strength for Ego4D/YouTube (0 = none)")
    ap.add_argument("--rotate-ego4d", type=int, default=0, choices=(0, 90, 180, 270))
    ap.add_argument("--rotate-youtube", type=int, default=0, choices=(0, 90, 180, 270))
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--warp-preview", help="comma-separated strengths: write warp-preview.jpg and exit")
    ap.add_argument("--force", action="store_true", help="re-render days that are up to date")
    args = ap.parse_args(argv)

    cfg = RenderConfig(media_root=args.media_root, warp=args.warp,
                       rotation={"ego4d": args.rotate_ego4d, "youtube": args.rotate_youtube}, crf=args.crf)
    missing = check_tools(cfg.font_path)
    if missing:
        print("cannot start, missing: " + ", ".join(missing), flush=True)
        return 2
    manifest = json.loads(args.manifest.read_text())
    wanted = parse_days(args.days)
    days = [d for d in manifest["days"] if wanted is None or d["day_index"] in wanted]
    info = {"path": str(args.manifest), "id": manifest.get("id"), "revision": manifest.get("revision"),
            "created_at": manifest.get("created_at")}

    if args.warp_preview:
        dest = warp_preview({**manifest, "days": days}, cfg, [float(x) for x in args.warp_preview.split(",")], args.out)
        print(dest, flush=True)
        return 0

    key = settings_key(info, cfg)
    todo, skipped = [], 0
    for d in days:
        if not args.force and day_is_done(args.out, d["day_index"], key):
            print(f"{day_name(d['day_index'])} up to date", flush=True)
            continue
        problems = check_day(d, cfg)
        if problems:
            print(f"day {d['day_index']}: " + "; ".join(problems), flush=True)
            skipped += 1
            continue
        todo.append(d)

    failed = 0
    if todo:
        ctx = multiprocessing.get_context("spawn")  # fork from a threaded process can deadlock
        pool = ProcessPoolExecutor(max_workers=max(1, args.jobs), mp_context=ctx)
        try:
            futures = {pool.submit(render_day, info, d, cfg, args.out): d["day_index"] for d in todo}
            for f in as_completed(futures):
                name = day_name(futures[f])
                try:
                    rep = f.result()
                    print(f"{name} done: {rep['frames']} frames in {rep['seconds']:.0f} s", flush=True)
                except Exception as e:  # one day's failure does not stop the others
                    failed += 1
                    print(f"{name} FAILED: {type(e).__name__}: {e}", flush=True)
        except KeyboardInterrupt:
            print("interrupted: queued days cancelled, running days stopped (rerun to continue)", flush=True)
            pool.shutdown(wait=False, cancel_futures=True)
            for proc in list((getattr(pool, "_processes", None) or {}).values()):
                proc.terminate()  # their hidden temp files are removed by the next run
            return 130
        pool.shutdown(wait=True)
    print(f"finished: {len(todo) - failed} rendered, {failed} failed, {skipped} skipped (missing sources)", flush=True)
    return 0 if not (failed or skipped) else 1


if __name__ == "__main__":
    sys.exit(main())
