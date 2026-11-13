"""Fetch only the NextMe-800 frames a manifest uses, by HTTP Range, resized to 1024x1024.

The tars are too large to download whole (204 tars, 4.5TB). This walks each tar's
headers with small Range requests and downloads only the picture/masked frames the
manifest's clips need. Design: docs/superpowers/specs/2026-11-13-nextme-frame-fetch-design.md

Run in your own shell with HF_TOKEN exported (the token is read from the environment
only and is never printed or written):

    python3 scripts/fetch_nextme_frames.py data/exports/<manifest>.json --days 1

Rerunning resumes: frames on disk are kept and each tar's walk continues from its index.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

BLOCK = 512
FRAME_RE = re.compile(r"/picture/masked/([^/]+__frame_(\d+)\.jpg)$")
SAVED_RE = re.compile(r"__frame_(\d+)\.jpg$")
DEFAULT_REPO = "mmm8383/NextMe-800"
DATASET = "nextme800"


class FetchError(Exception):
    pass


class TarFormatError(Exception):
    pass


class Stopped(Exception):
    pass


def pad(n: int) -> int:
    return (n + BLOCK - 1) // BLOCK * BLOCK


# ---------------------------------------------------------------- needed frames


def parse_days(text: str | None) -> set[int] | None:
    if not text:
        return None
    days: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            days.update(range(int(a), int(b) + 1))
        elif part:
            days.add(int(part))
    return days


def needed_frames(manifest: dict, days: set[int] | None = None) -> list[dict]:
    """One job per tar: frame numbers covering every NextMe clip, one extra frame each side."""
    jobs: dict[str, dict] = {}
    for day in manifest["days"]:
        if days is not None and day["day_index"] not in days:
            continue
        for clip in day["clips"]:
            src = clip["source"]
            if src["dataset"] != DATASET:
                continue
            tar = src["media"]["tar"]
            job = jobs.setdefault(
                tar, {"tar": tar, "video_id": src["video_id"], "frames": set(), "first_day": day["day_index"]}
            )
            lo = max(0, math.floor(clip["source_start_s"]) - 1)
            hi = math.ceil(clip["source_end_s"]) + 1
            job["frames"].update(range(lo, hi + 1))
    return sorted(jobs.values(), key=lambda j: (j["first_day"], j["video_id"]))


# ---------------------------------------------------------------- byte sources


class FileSource:
    """Local file (a whole tar, or a truncated partial download)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.size = self.path.stat().st_size
        self._f = open(self.path, "rb")
        self.requests = 0
        self.bytes = 0

    def read(self, off: int, n: int) -> bytes:
        n = min(n, self.size - off)
        if n <= 0:
            return b""
        self._f.seek(off)
        data = self._f.read(n)
        self.requests += 1
        self.bytes += len(data)
        return data

    def close(self):
        self._f.close()


class HttpSource:
    """A Hub file read by Range. The token goes only to the resolve URL, never to the CDN."""

    def __init__(self, url: str, token: str, session=None, retries: int = 8, backoff: float = 1.0,
                 max_backoff: float = 60.0, timeout: float = 120.0, sleep=time.sleep):
        import requests

        self._requests = requests
        self.url = url
        self._token = token
        self.s = session or requests.Session()
        self.retries = retries
        self.backoff = backoff
        self.max_backoff = max_backoff
        self.timeout = timeout
        self._sleep = sleep
        self._cdn: str | None = None
        self._cdn_auth = False
        self.size: int | None = None
        self.resolves = 0
        self.requests = 0
        self.bytes = 0
        self._resolve()

    def _wait(self, attempt: int):
        self._sleep(min(self.max_backoff, self.backoff * (2 ** attempt)))

    def _resolve(self):
        for attempt in range(self.retries + 1):
            try:
                r = self.s.get(self.url, headers={"Authorization": f"Bearer {self._token}", "Range": "bytes=0-0"},
                               allow_redirects=False, stream=True, timeout=self.timeout)
            except self._requests.RequestException:
                self._wait(attempt)
                continue
            with r:
                self.resolves += 1
                if r.status_code in (301, 302, 303, 307, 308):
                    self._cdn = urljoin(self.url, r.headers["Location"])
                    self._cdn_auth = False
                    linked = r.headers.get("X-Linked-Size")
                    if linked and self.size is None:
                        self.size = int(linked)
                elif r.status_code == 206:
                    self._cdn = self.url
                    self._cdn_auth = True
                    self.size = int(r.headers["Content-Range"].rsplit("/", 1)[1])
                elif r.status_code in (401, 403):
                    raise FetchError(f"Hub refused the token for {self.url} (HTTP {r.status_code}); "
                                     "check HF_TOKEN and that the dataset access is granted")
                elif r.status_code == 404:
                    raise FetchError(f"not found: {self.url}")
                elif r.status_code == 429 or r.status_code >= 500:
                    self._wait(attempt)
                    continue
                else:
                    raise FetchError(f"unexpected HTTP {r.status_code} from {self.url}")
            if self.size is None:
                self.size = self._probe_size()
            return
        raise FetchError(f"could not resolve {self.url} after {self.retries + 1} attempts")

    def _probe_size(self) -> int:
        r = self.s.get(self._cdn, headers=self._headers({"Range": "bytes=0-0"}), stream=True, timeout=self.timeout)
        with r:
            if r.status_code != 206:
                raise FetchError(f"expected 206 when probing size, got {r.status_code}")
            return int(r.headers["Content-Range"].rsplit("/", 1)[1])

    def _headers(self, extra: dict) -> dict:
        h = dict(extra)
        if self._cdn_auth:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    def read(self, off: int, n: int) -> bytes:
        n = min(n, self.size - off)
        if n <= 0:
            return b""
        end = off + n - 1
        last_error = "no attempt"
        for attempt in range(self.retries + 1):
            try:
                r = self.s.get(self._cdn, headers=self._headers({"Range": f"bytes={off}-{end}"}),
                               stream=True, timeout=self.timeout)
            except self._requests.RequestException as e:
                last_error = type(e).__name__
                self._wait(attempt)
                continue
            with r:
                if r.status_code in (401, 403):
                    last_error = f"HTTP {r.status_code} (expired URL?)"
                    self._resolve()
                    continue
                if r.status_code == 429 or r.status_code >= 500:
                    last_error = f"HTTP {r.status_code}"
                    self._wait(attempt)
                    continue
                if r.status_code != 206:
                    raise FetchError(f"expected 206 for bytes {off}-{end}, got {r.status_code}")
                cr = r.headers.get("Content-Range", "")
                if not cr.startswith(f"bytes {off}-{end}/"):
                    raise FetchError(f"Content-Range {cr!r} does not match bytes {off}-{end}")
                try:
                    data = b"".join(r.iter_content(1 << 20))
                except self._requests.RequestException as e:
                    last_error = type(e).__name__
                    self._wait(attempt)
                    continue
            self.requests += 1
            if len(data) != n:
                last_error = f"short body {len(data)}/{n}"
                self._wait(attempt)
                continue
            self.bytes += n
            return data
        raise FetchError(f"bytes {off}-{end} failed after {self.retries + 1} attempts: {last_error}")

    def close(self):
        self.s.close()


# ---------------------------------------------------------------- tar walking


@dataclass
class Member:
    name: str
    start: int  # offset of the first header of this member (long-name header included)
    data: int
    size: int
    type: bytes

    @property
    def next(self) -> int:
        return self.data + pad(self.size)


def _octal(field: bytes) -> int:
    if field and field[0] & 0x80:  # GNU base-256
        return int.from_bytes(field[1:], "big")
    text = field.split(b"\0", 1)[0].strip()
    return int(text, 8) if text else 0


def parse_header(block: bytes, off: int):
    """Return (type, name, size) or None for an end-of-archive zero block."""
    if block == b"\0" * BLOCK:
        return None
    if block[257:262] != b"ustar":
        raise TarFormatError(f"no ustar magic at offset {off}")
    stored = _octal(block[148:156])
    calc = sum(block[:148]) + 8 * 32 + sum(block[156:])
    if stored != calc:
        raise TarFormatError(f"bad header checksum at offset {off}")
    name = block[:100].split(b"\0", 1)[0].decode("utf-8", "replace")
    if block[257:263] == b"ustar\0":  # POSIX: prefix field
        prefix = block[345:500].split(b"\0", 1)[0].decode("utf-8", "replace")
        if prefix:
            name = f"{prefix}/{name}"
    return block[156:157], name, _octal(block[124:136])


def _pax_path(data: bytes) -> str | None:
    i = 0
    while i < len(data):
        sp = data.index(b" ", i)
        length = int(data[i:sp])
        key, _, value = data[sp + 1:i + length - 1].partition(b"=")
        if key == b"path":
            return value.decode("utf-8", "replace")
        i += length
    return None


class TarWalker:
    """Follows tar headers through a byte source. Reads 2KB windows after large members and
    4MB windows after small ones, so runs of small files cost few requests."""

    def __init__(self, src, header_window: int = 2048, small_window: int = 4 << 20, small_member: int = 256 << 10):
        self.src = src
        self.header_window = header_window
        self.small_window = small_window
        self.small_member = small_member
        self._buf_off = 0
        self._buf = b""
        self.end_reason: str | None = None  # "end" (end-of-archive block) or "truncated"

    def seed(self, off: int, data: bytes):
        self._buf_off, self._buf = off, data

    def _get(self, off: int, n: int, window: int) -> bytes:
        b0 = off - self._buf_off
        if 0 <= b0 and off + n <= self._buf_off + len(self._buf):
            return self._buf[b0:b0 + n]
        data = self.src.read(off, max(n, window))
        self.seed(off, data)
        return data[:n]

    def members(self, start: int = 0):
        off = start
        window = self.header_window
        long_name = None
        chain = None
        self.end_reason = None
        while True:
            if off + BLOCK > self.src.size:
                self.end_reason = "end" if off == self.src.size else "truncated"
                return
            block = self._get(off, BLOCK, window)
            if len(block) < BLOCK:
                self.end_reason = "truncated"
                return
            h = parse_header(block, off)
            if h is None:
                self.end_reason = "end"
                return
            typ, name, size = h
            if chain is None:
                chain = off
            if typ in (b"L", b"x", b"g"):
                if off + BLOCK + size > self.src.size:
                    self.end_reason = "truncated"
                    return
                data = self._get(off + BLOCK, size, window)
                if typ == b"L":
                    long_name = data.split(b"\0", 1)[0].decode("utf-8", "replace")
                elif typ == b"x":
                    long_name = _pax_path(data) or long_name
                off += BLOCK + pad(size)
                continue
            m = Member(long_name or name, chain, off + BLOCK, size, typ)
            long_name = None
            chain = None
            window = self.small_window if size < self.small_member else self.header_window
            yield m
            off = m.next


# ---------------------------------------------------------------- frames


def save_frame(jpeg: bytes, path: Path, size: int = 1024, quality: int = 95):
    from PIL import Image

    im = Image.open(io.BytesIO(jpeg))
    im.draft("RGB", (size, size))
    im = im.convert("RGB")
    w, h = im.size
    if w != h:
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    im = im.resize((size, size), Image.LANCZOS)
    tmp = path.with_name(f".{path.name}.tmp")
    im.save(tmp, "JPEG", quality=quality)
    os.replace(tmp, path)


def frames_on_disk(frames_dir: Path) -> set[int]:
    if not frames_dir.is_dir():
        return set()
    out = set()
    for p in frames_dir.iterdir():
        m = SAVED_RE.search(p.name)
        if m and not p.name.startswith("."):
            out.add(int(m.group(1)))
    return out


def load_index(path: Path):
    known: dict[int, dict] = {}
    resume = 0
    ended = False
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line cut by an interrupt
            if "n" in rec:
                known[rec["n"]] = rec
                resume = max(resume, rec["next"])
            elif "at" in rec:
                resume = max(resume, rec["at"])
            elif rec.get("end"):
                ended = True
    return known, resume, ended


class Progress:
    def __init__(self, path: Path | None, jobs: list[dict]):
        self.path = path
        self.lock = threading.Lock()
        self.started = time.time()
        self.tars = {j["video_id"]: {"tar": j["tar"], "state": "queued", "needed": len(j["frames"]), "present": 0,
                                     "missing": 0, "walked": 0, "bytes": 0, "requests": 0} for j in jobs}

    def update(self, vid: str, **kw):
        with self.lock:
            self.tars[vid].update(kw)

    def snapshot(self) -> dict:
        with self.lock:
            tars = {k: dict(v) for k, v in self.tars.items()}
        total = lambda key: sum(t[key] for t in tars.values())
        elapsed = time.time() - self.started
        states = {}
        for t in tars.values():
            states[t["state"]] = states.get(t["state"], 0) + 1
        return {"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_s": round(elapsed),
                "needed": total("needed"), "present": total("present"), "missing": total("missing"),
                "bytes": total("bytes"), "requests": total("requests"),
                "mb_per_s": round(total("bytes") / 1e6 / elapsed, 2) if elapsed > 0 else 0,
                "states": states, "tars": tars}

    def write(self):
        if not self.path:
            return
        tmp = self.path.with_name(f".{self.path.name}.tmp")
        tmp.write_text(json.dumps(self.snapshot(), indent=1))
        os.replace(tmp, self.path)


def fetch_tar(job: dict, src, out_root: Path, progress: Progress | None = None, stop: threading.Event | None = None,
              frame_size: int = 1024, quality: int = 95, walker_kw: dict | None = None) -> dict:
    vid = job["video_id"]
    need = set(job["frames"])
    frames_dir = out_root / "frames" / vid
    index_dir = out_root / "index"
    frames_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / f"{vid}.jsonl"
    known, resume, ended = load_index(index_path)
    remaining = need - frames_on_disk(frames_dir)
    walked = 0
    base_bytes = getattr(src, "bytes", 0)

    def report(state):
        if progress:
            progress.update(vid, state=state, present=len(need) - len(remaining), walked=walked,
                            bytes=src.bytes - base_bytes, requests=src.requests)

    def check_stop():
        if stop is not None and stop.is_set():
            raise Stopped()

    def store(name: str, jpeg: bytes):
        save_frame(jpeg, frames_dir / name, frame_size, quality)

    report("fetching")
    # Frames whose place is already indexed: read them directly.
    for n in sorted(remaining & known.keys()):
        check_stop()
        rec = known[n]
        store(rec["name"], src.read(rec["data"], rec["size"]))
        remaining.discard(n)
        report("fetching")

    walker = TarWalker(src, **(walker_kw or {}))
    if remaining and not ended:
        with open(index_path, "a") as idx:
            others = 0
            last_next = resume
            for m in walker.members(resume):
                check_stop()
                walked += 1
                last_next = m.next
                fm = FRAME_RE.search("/" + m.name) if m.type in (b"0", b"\0") else None
                if fm:
                    n = int(fm.group(2))
                    idx.write(json.dumps({"n": n, "name": fm.group(1), "data": m.data, "size": m.size,
                                          "next": m.next}) + "\n")
                    idx.flush()
                    if n in remaining:
                        blob = src.read(m.data, pad(m.size) + walker.header_window)
                        if len(blob) < m.size:
                            break  # truncated source
                        store(fm.group(1), blob[:m.size])
                        walker.seed(m.next, blob[pad(m.size):])
                        remaining.discard(n)
                        if not remaining:
                            break
                else:
                    others += 1
                    if others % 500 == 0:
                        idx.write(json.dumps({"at": m.next}) + "\n")
                        idx.flush()
                if walked % 50 == 0:
                    report("walking")
            else:
                if walker.end_reason == "end":
                    ended = True
                    idx.write(json.dumps({"end": True}) + "\n")
            if remaining or not ended:
                idx.write(json.dumps({"at": last_next}) + "\n")

    complete = not remaining
    result = {"tar": job["tar"], "video_id": vid, "tar_size": src.size, "needed": len(need),
              "present": len(need) - len(remaining), "missing": sorted(remaining) if ended else [],
              "walk_ended": ended, "complete": complete}
    if complete or ended:
        tmp = frames_dir / ".fetched.json.tmp"
        tmp.write_text(json.dumps(result, indent=1))
        os.replace(tmp, frames_dir / ".fetched.json")
    if progress:
        progress.update(vid, missing=len(result["missing"]))
    report("done" if complete else ("missing" if ended else "incomplete"))
    return result


# ---------------------------------------------------------------- main


def hub_url(endpoint: str, repo: str, revision: str, path: str) -> str:
    return f"{endpoint.rstrip('/')}/datasets/{repo}/resolve/{revision}/{path}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--days", help="day indexes to fetch, e.g. 1 or 1-7,10 (default: all)")
    ap.add_argument("--out", type=Path, default=Path("data/media/nextme"))
    ap.add_argument("--jobs", type=int, default=6, help="tars fetched at the same time")
    ap.add_argument("--limit", type=int, help="fetch only the first N tars (in day order)")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    ap.add_argument("--local-root", type=Path, help="read tars from <local-root>/<tar path> instead of the Hub")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = ap.parse_args(argv)

    manifest = json.loads(args.manifest.read_text())
    jobs = needed_frames(manifest, parse_days(args.days))
    if args.limit:
        jobs = jobs[:args.limit]
    total = sum(len(j["frames"]) for j in jobs)
    print(f"{len(jobs)} tars, {total} frames ({total / 3600:.1f} h at 1 fps)", flush=True)
    if args.dry_run:
        for j in jobs:
            print(f"  day {j['first_day']:2d}  {j['tar']}  {len(j['frames'])} frames")
        return 0

    token = None
    if not args.local_root:
        token = os.environ.get("HF_TOKEN")
        if not token:
            print("HF_TOKEN is not set. Export it in this shell first (do not use `hf auth login`).", file=sys.stderr)
            return 2

    args.out.mkdir(parents=True, exist_ok=True)
    progress = Progress(args.out / "fetch_status.json", jobs)
    stop = threading.Event()
    results: list[dict] = []
    errors: list[str] = []

    def run(job):
        vid = job["video_id"]
        src = None
        try:
            if args.local_root:
                src = FileSource(args.local_root / job["tar"])
            else:
                src = HttpSource(hub_url(args.endpoint, args.repo, args.revision, job["tar"]), token)
            r = fetch_tar(job, src, args.out, progress, stop)
            results.append(r)
            print(f"[{r['present']}/{r['needed']}] {vid} {'done' if r['complete'] else 'NOT complete'}"
                  + (f", {len(r['missing'])} missing" if r["missing"] else ""), flush=True)
        except Stopped:
            progress.update(vid, state="stopped")
        except Exception as e:  # keep the other tars going
            progress.update(vid, state="error", error=f"{type(e).__name__}: {e}")
            errors.append(f"{vid}: {type(e).__name__}: {e}")
            print(f"ERROR {vid}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        finally:
            if src is not None:
                src.close()

    last_line = 0.0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run, j) for j in jobs]
        try:
            while True:
                done, pending = wait(futures, timeout=5)
                progress.write()
                if time.time() - last_line > 60:
                    s = progress.snapshot()
                    print(f"{s['updated']}  frames {s['present']}/{s['needed']}  {s['bytes'] / 1e9:.1f} GB  "
                          f"{s['mb_per_s']} MB/s  {s['states']}", flush=True)
                    last_line = time.time()
                if not pending:
                    break
        except KeyboardInterrupt:
            print("stopping (frames written so far are kept; rerun to resume)...", flush=True)
            stop.set()
            for f in futures:
                f.cancel()
    progress.write()
    s = progress.snapshot()
    print(f"finished: frames {s['present']}/{s['needed']}, missing {s['missing']}, "
          f"{s['bytes'] / 1e9:.1f} GB, errors {len(errors)}", flush=True)
    return 1 if errors or stop.is_set() else 0


if __name__ == "__main__":
    sys.exit(main())
