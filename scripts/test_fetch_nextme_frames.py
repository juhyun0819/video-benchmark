"""Tests for fetch_nextme_frames.py. Run with: python3 -m pytest scripts/test_fetch_nextme_frames.py

All tars here are synthetic: tiny generated JPEGs in the NextMe layout."""
import io
import json
import random
import sys
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from fetch_nextme_frames import (FetchError, FileSource, HttpSource, Stopped, TarFormatError, TarWalker, fetch_tar,
                                 needed_frames, parse_days)

ROOT = "4-20_hkt1023-1224_120m_01b3b7f6-aafa-4e21-8145-e6d5a0c4f82a"


def frame_name(n):
    s = 4 + n
    return f"2026-04-20_10-{23 + s // 60:02d}-{s % 60:02d}_HKT__frame_{n:05d}.jpg"


def jpeg(n, size=48):
    buf = io.BytesIO()
    Image.new("RGB", (size, size), ((n * 37) % 256, (n * 11) % 256, 90)).save(buf, "JPEG", quality=90)
    return buf.getvalue()


def add(t, name, data=None, dir=False):
    info = tarfile.TarInfo(name)
    if dir:
        info.type = tarfile.DIRTYPE
        t.addfile(info)
    else:
        info.size = len(data)
        t.addfile(info, io.BytesIO(data))


def make_tar(path, frames, order="reverse", seed=0, eye=40, gnu=True):
    nums = sorted(frames)
    if order == "reverse":
        nums = nums[::-1]
    elif order == "random":
        random.Random(seed).shuffle(nums)
    with tarfile.open(path, "w", format=tarfile.GNU_FORMAT if gnu else tarfile.PAX_FORMAT) as t:
        add(t, ROOT, dir=True)
        add(t, f"{ROOT}/eye_tracking", dir=True)
        for i in range(eye):
            add(t, f"{ROOT}/eye_tracking/20260420-1023{i:02d}-167.jpg", bytes([i % 256]) * 3000)
        add(t, f"{ROOT}/audio", dir=True)
        add(t, f"{ROOT}/audio/audio.wav", b"\1" * 400_000)
        add(t, f"{ROOT}/picture", dir=True)
        add(t, f"{ROOT}/picture/masked", dir=True)
        for n in nums:
            add(t, f"{ROOT}/picture/masked/{frame_name(n)}", jpeg(n))
    return path


def job(frames, vid="2026-04-20_1023-1224"):
    return {"tar": f"raw/2026-04-20/{vid}.tar", "video_id": vid, "frames": set(frames), "first_day": 1}


def saved(out, vid="2026-04-20_1023-1224"):
    d = out / "frames" / vid
    return sorted(int(p.name.split("frame_")[1][:5]) for p in d.glob("*.jpg"))


def manifest(clips):
    return {"days": [{"day_index": d, "clips": [
        {"source": {"dataset": ds, "video_id": vid, "media": {"tar": f"raw/x/{vid}.tar"}},
         "source_start_s": a, "source_end_s": b} for (ds, vid, a, b) in cs]} for d, cs in clips.items()]}


# ---------------------------------------------------------------- needed frames


def test_needed_frames_pads_one_frame_each_side_and_merges_clips():
    m = manifest({1: [("nextme800", "v1", 10, 13), ("ego4d", "e", 0, 99), ("nextme800", "v1", 12, 14)],
                  2: [("nextme800", "v2", 0, 2)]})
    jobs = needed_frames(m)
    assert [j["video_id"] for j in jobs] == ["v1", "v2"]
    assert jobs[0]["frames"] == set(range(9, 16))
    assert jobs[1]["frames"] == {0, 1, 2, 3}  # no frame -1


def test_needed_frames_day_filter():
    m = manifest({1: [("nextme800", "v1", 0, 1)], 2: [("nextme800", "v2", 0, 1)], 3: [("nextme800", "v3", 0, 1)]})
    assert [j["video_id"] for j in needed_frames(m, parse_days("1,3"))] == ["v1", "v3"]
    assert parse_days("2-4,7") == {2, 3, 4, 7}


# ---------------------------------------------------------------- walking


@pytest.mark.parametrize("order", ["reverse", "random", "forward"])
def test_fetches_exactly_the_needed_frames_in_any_order(tmp_path, order):
    tar = make_tar(tmp_path / "a.tar", range(60), order=order, seed=3)
    out = tmp_path / "out"
    r = fetch_tar(job({5, 6, 7, 30, 59}), FileSource(tar), out, frame_size=32)
    assert r["complete"] and r["missing"] == []
    assert saved(out) == [5, 6, 7, 30, 59]
    img = Image.open(next((out / "frames" / "2026-04-20_1023-1224").glob("*frame_00030.jpg")))
    assert img.size == (32, 32)
    meta = json.loads((out / "frames" / "2026-04-20_1023-1224" / ".fetched.json").read_text())
    assert meta["present"] == 5 and meta["needed"] == 5


def test_pax_format_names_also_work(tmp_path):
    tar = make_tar(tmp_path / "a.tar", range(10), gnu=False)
    r = fetch_tar(job({2, 3}), FileSource(tar), tmp_path / "out", frame_size=32)
    assert r["complete"] and saved(tmp_path / "out") == [2, 3]


def test_stops_walking_once_all_needed_frames_are_found(tmp_path):
    tar = make_tar(tmp_path / "a.tar", range(200), order="reverse")
    src = FileSource(tar)
    fetch_tar(job({199, 198}), src, tmp_path / "out", frame_size=32, walker_kw={"small_window": 16384})
    with tarfile.open(tar) as t:
        third = [m for m in t.getmembers() if m.name.endswith(frame_name(196))][0]
    assert src.bytes < third.offset_data


def test_small_members_are_read_in_big_windows(tmp_path):
    tar = make_tar(tmp_path / "a.tar", range(3), eye=300)
    src = FileSource(tar)
    fetch_tar(job({0}), src, tmp_path / "out", frame_size=32)
    assert src.requests < 30  # 300 eye-tracking files + audio + 3 frames


def test_missing_frames_are_reported_after_a_full_walk(tmp_path):
    tar = make_tar(tmp_path / "a.tar", range(20))
    r = fetch_tar(job({3, 25, 26}), FileSource(tar), tmp_path / "out", frame_size=32)
    assert not r["complete"] and r["walk_ended"] and r["missing"] == [25, 26]
    assert saved(tmp_path / "out") == [3]


def test_truncated_source_is_not_reported_as_missing(tmp_path):
    tar = make_tar(tmp_path / "a.tar", range(50), order="reverse")
    cut = tmp_path / "cut.tar"
    with tarfile.open(tar) as t:
        second = [m for m in t.getmembers() if m.name.endswith(frame_name(48))][0]
    cut.write_bytes(tar.read_bytes()[:second.offset_data + 100])
    r = fetch_tar(job({49, 0}), FileSource(cut), tmp_path / "out", frame_size=32)
    assert not r["complete"] and not r["walk_ended"] and r["missing"] == []
    assert saved(tmp_path / "out") == [49]
    assert not (tmp_path / "out" / "frames" / "2026-04-20_1023-1224" / ".fetched.json").exists()


def test_bad_checksum_is_an_error(tmp_path):
    tar = make_tar(tmp_path / "a.tar", range(5))
    data = bytearray(tar.read_bytes())
    data[1] ^= 0xFF  # first header's name
    tar.write_bytes(bytes(data))
    with pytest.raises(TarFormatError):
        fetch_tar(job({1}), FileSource(tar), tmp_path / "out", frame_size=32)


class FailAfter(FileSource):
    def __init__(self, path, reads):
        super().__init__(path)
        self.left = reads

    def read(self, off, n):
        if self.left == 0:
            raise FetchError("simulated network loss")
        self.left -= 1
        return super().read(off, n)


def test_resume_continues_from_the_index_without_refetching(tmp_path):
    tar = make_tar(tmp_path / "a.tar", range(80), order="random", seed=7)
    out = tmp_path / "out"
    want = set(range(0, 80, 4))
    with pytest.raises(FetchError):
        fetch_tar(job(want), FailAfter(tar, 25), out, frame_size=32)
    first = saved(out)
    assert 0 < len(first) < len(want)
    src = FileSource(tar)
    r = fetch_tar(job(want), src, out, frame_size=32)
    assert r["complete"] and saved(out) == sorted(want)
    # Second run starts where the first stopped: it never rereads the start of the tar.
    full = FileSource(tar)
    fetch_tar(job(want), full, tmp_path / "fresh", frame_size=32)
    assert src.bytes < full.bytes


def test_more_days_later_reuses_the_index(tmp_path):
    tar = make_tar(tmp_path / "a.tar", range(40), order="reverse")
    out = tmp_path / "out"
    fetch_tar(job({39}), FileSource(tar), out, frame_size=32)
    fetch_tar(job({39, 10}), FileSource(tar), out, frame_size=32)
    src = FileSource(tar)
    r = fetch_tar(job({39, 10, 38}), src, out, frame_size=32)  # 38 is indexed already
    assert r["complete"] and saved(out) == [10, 38, 39]
    assert src.requests == 1


def test_stop_event_interrupts(tmp_path):
    tar = make_tar(tmp_path / "a.tar", range(10))
    ev = threading.Event()
    ev.set()
    with pytest.raises(Stopped):
        fetch_tar(job({1, 2}), FileSource(tar), tmp_path / "out", frame_size=32, stop=ev)


def test_walker_lists_every_member(tmp_path):
    tar = make_tar(tmp_path / "a.tar", range(7), eye=3)
    names = [m.name for m in TarWalker(FileSource(tar)).members()]
    with tarfile.open(tar) as t:
        assert [n.rstrip("/") for n in names] == [m.name for m in t.getmembers()]


# ---------------------------------------------------------------- HTTP


class QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        pass  # the client hangs up on purpose in test_http_refuses_a_full_body


class Hub:
    """Fake Hub: /resolve redirects to a signed /cdn URL; /cdn serves byte ranges."""

    def __init__(self, data):
        self.data = data
        self.resolves = 0
        self.expire_next = 0
        self.full_body = False
        self.auth_seen_on_cdn = False
        self.token_ok = True
        hub = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path.startswith("/resolve"):
                    hub.resolves += 1
                    if self.headers.get("Authorization") != "Bearer secret" or not hub.token_ok:
                        self.send_response(401)
                        self.end_headers()
                        return
                    self.send_response(302)
                    self.send_header("Location", f"/cdn?sig={hub.resolves}")
                    self.send_header("X-Linked-Size", str(len(hub.data)))
                    self.end_headers()
                    return
                if self.headers.get("Authorization"):
                    hub.auth_seen_on_cdn = True
                if hub.expire_next > 0:
                    hub.expire_next -= 1
                    self.send_response(403)
                    self.end_headers()
                    return
                if hub.full_body:
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(hub.data)))
                    self.end_headers()
                    self.wfile.write(hub.data)
                    return
                a, b = self.headers["Range"].split("=")[1].split("-")
                a, b = int(a), min(int(b), len(hub.data) - 1)
                body = hub.data[a:b + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {a}-{b}/{len(hub.data)}")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = QuietServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/resolve/raw/a.tar"

    def close(self):
        self.server.shutdown()


@pytest.fixture
def hub(tmp_path):
    h = Hub(make_tar(tmp_path / "a.tar", range(30), order="random", seed=1).read_bytes())
    yield h
    h.close()


def test_http_fetch_through_redirect_without_sending_the_token_to_the_cdn(tmp_path, hub):
    src = HttpSource(hub.url, "secret", backoff=0)
    assert src.size == len(hub.data)
    r = fetch_tar(job({4, 17}), src, tmp_path / "out", frame_size=32)
    assert r["complete"] and saved(tmp_path / "out") == [4, 17]
    assert not hub.auth_seen_on_cdn and hub.resolves == 1


def test_http_expired_url_is_resolved_again(tmp_path, hub):
    src = HttpSource(hub.url, "secret", backoff=0)
    hub.expire_next = 1
    assert src.read(0, 512) == hub.data[:512]
    assert hub.resolves == 2


def test_http_refuses_a_full_body(hub):
    src = HttpSource(hub.url, "secret", backoff=0)
    hub.full_body = True
    with pytest.raises(FetchError, match="expected 206"):
        src.read(0, 512)


def test_http_bad_token_is_a_clear_error(hub):
    with pytest.raises(FetchError, match="refused the token") as e:
        HttpSource(hub.url, "wrong", backoff=0)
    assert "wrong" not in str(e.value)
