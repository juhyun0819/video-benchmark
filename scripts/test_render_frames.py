"""Tests for render_frames.py. Run with: python3 -m pytest scripts/test_render_frames.py

Every fixture is synthetic: solid-colour JPEGs and ffmpeg-generated test videos."""
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import render_frames  # noqa: E402
from render_frames import (OLD_TEXT_BOX, DayError, RenderConfig, Stamper, check_day, day_is_done, main, render_day,
                           settings_key, warp_preview, apply_circle, circle_mask, day_picks, extract_video_frames, fisheye_warp, nextme_frames, nextme_last_frame, pick_nextme, pick_video, probe_video,
                           stamp_text)


def clip(a, b, s0, ds="nextme800", vid="v", slot="d1-s1"):
    return {"source": {"dataset": ds, "video_id": vid, "media": {"kind": "frames"}}, "video_id": vid,
            "source_start_s": s0, "source_end_s": s0 + (b - a), "synthetic_start_s": a, "synthetic_end_s": b,
            "slot_id": slot}


def make_video(path, n, fps=30, size=(64, 48), rotate=None, start=0.0):
    """Synthetic video whose frame k shows k in its colour (see frame_number)."""
    w, h = size
    raw = path.with_name(path.stem + ".raw.mp4") if rotate is not None else path
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
           "-i", f"nullsrc=s={w}x{h}:r={fps},geq=r='mod(N,50)*4':g='floor(N/50)*20':b='0'",
           "-frames:v", str(n), "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p"]
    if start and rotate is None:
        cmd += ["-output_ts_offset", str(start)]
    subprocess.run(cmd + [str(raw)], check=True)
    if rotate is not None:
        cmd = ["ffmpeg", "-y", "-v", "error", "-display_rotation:v:0", str(rotate), "-i", str(raw), "-c", "copy"]
        if start:
            cmd += ["-output_ts_offset", str(start)]
        subprocess.run(cmd + [str(path)], check=True)
        raw.unlink()
    return path


def make_marker_video(path):
    """48x48 black video with a white square in the top-left corner."""
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=black:s=48x48:r=30",
                    "-vf", "drawbox=x=0:y=0:w=10:h=10:color=white:t=fill", "-frames:v", "10",
                    "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", str(path)], check=True)
    return path


def frame_number(px):
    r, g = px[0], px[1]
    return round(g / 20) * 50 + round(r / 4)


# ---------------------------------------------------------------- frame list


def test_day_picks_one_per_integer_second_and_skips_gaps():
    day = {"day_index": 1, "clips": [clip(20, 22, 100), clip(10.5, 13.2, 7)], "no_video_spans": []}
    p = day_picks(day)
    assert [x.t for x in p] == [11, 12, 13, 20, 21]
    assert [x.clip for x in p] == [1, 1, 1, 0, 0]
    assert p[0].src == pytest.approx(7.5) and p[3].src == 100


def test_stamp_text_dates_and_midnight():
    assert stamp_text(1, 7 * 3600 + 48 * 60 + 12) == "2025-04-07 07:48:12"
    assert stamp_text(28, 27 * 86400 + 59) == "2025-05-04 00:00:59"
    assert stamp_text(1, 86400 + 5) == "2025-04-08 00:00:05"


# ---------------------------------------------------------------- NextMe


def test_pick_nextme_rounds_half_down():
    assert pick_nextme(4.5, {4, 5}, None) == (4, "exact")
    assert pick_nextme(4.51, {4, 5}, None) == (5, "exact")


def test_pick_nextme_substitutes_within_two_seconds_smaller_on_tie():
    assert pick_nextme(10, {8, 12}, None) == (8, "substituted")
    with pytest.raises(DayError):
        pick_nextme(10, {7, 13}, None)


def test_pick_nextme_holds_last_frame_up_to_five_seconds_at_recording_end():
    assert pick_nextme(9840, set(range(9800, 9837)), 9836) == (9836, "clamped")
    assert pick_nextme(9841, set(range(9800, 9837)), 9836) == (9836, "clamped")
    with pytest.raises(DayError):
        pick_nextme(9842, set(range(9800, 9837)), 9836)


def test_end_clamp_needs_a_finished_walk(tmp_path):
    idx = tmp_path / "v.jsonl"
    idx.write_text('{"n": 0, "name": "a__frame_00000.jpg", "data": 0, "size": 1, "next": 512}\n'
                   '{"n": 1, "name": "a__frame_00001.jpg", "data": 1024, "size": 1, "next": 2048}\n')
    assert nextme_last_frame(idx) is None
    idx.write_text(idx.read_text() + '{"end": true}\n')
    assert nextme_last_frame(idx) == 1
    assert nextme_last_frame(tmp_path / "absent.jsonl") is None
    with pytest.raises(DayError):
        pick_nextme(5, {0, 1}, None)  # unknown end: only the 2 s rule


def test_nextme_frames_reads_numbers_and_ignores_temp_files(tmp_path):
    for name in ["x__frame_00003.jpg", ".x__frame_00004.jpg.tmp", "x__frame_00012.jpg"]:
        (tmp_path / name).write_bytes(b"j")
    assert sorted(nextme_frames(tmp_path)) == [3, 12]
    assert nextme_frames(tmp_path / "absent") == {}


# ---------------------------------------------------------------- video timing


def test_pick_video_nearest_and_tie_takes_earlier():
    times = [i / 30 for i in range(90)]
    assert pick_video(1.0, times, 30) == (30, False)
    assert pick_video(1.0 + 1 / 60, times, 30) == (30, False)  # exact tie -> earlier


def test_pick_video_clamps_within_one_frame_and_errors_beyond():
    times = [i / 30 for i in range(90)]
    assert pick_video(times[-1] + 1 / 30, times, 30) == (89, True)
    with pytest.raises(DayError):
        pick_video(times[-1] + 2 / 30, times, 30)


def test_probe_video_reads_times_start_and_rotation(tmp_path):
    v = make_video(tmp_path / "a.mp4", 45, start=2.0, rotate=90)
    info = probe_video(v, cache_dir=tmp_path / "cache")
    assert len(info.times) == 45 and info.times[0] == 0 and info.times[1] == pytest.approx(1 / 30, abs=1e-3)
    assert info.pts0 == pytest.approx(2.0, abs=0.05)
    assert info.rotation in (90, 270)  # sign convention of the display matrix; recorded as read
    assert probe_video(v, cache_dir=tmp_path / "cache") == info and any((tmp_path / "cache").iterdir())


# ---------------------------------------------------------------- video frames


def numbers(imgs):
    return [frame_number(im.getpixel((im.width // 2, im.height // 2))) for im in imgs]


def test_extract_exact_frames_in_order(tmp_path):
    v = make_video(tmp_path / "a.mp4", 300)
    imgs = list(extract_video_frames(v, probe_video(v), [0, 30, 31, 299], size=32))
    assert numbers(imgs) == [0, 30, 31, 299]
    assert all(im.size == (32, 32) for im in imgs)


def test_extract_with_nonzero_start_time(tmp_path):
    v = make_video(tmp_path / "a.mp4", 90, start=1.5)
    assert numbers(extract_video_frames(v, probe_video(v), [45], size=32)) == [45]


def test_extract_repeats_duplicate_indices(tmp_path):
    v = make_video(tmp_path / "a.mp4", 60)
    assert numbers(extract_video_frames(v, probe_video(v), [59, 59, 59], size=32)) == [59, 59, 59]


def test_extract_center_crops_and_rotates(tmp_path):
    v = make_video(tmp_path / "a.mp4", 10, size=(64, 48))  # landscape -> 48x48 centre crop
    assert next(extract_video_frames(v, probe_video(v), [0], size=48)).size == (48, 48)
    p = make_video(tmp_path / "p.mp4", 10, size=(64, 48), rotate=90)  # portrait via display matrix
    assert next(extract_video_frames(p, probe_video(p), [0], size=48)).size == (48, 48)


def test_extra_rotation_turns_the_picture(tmp_path):
    v = make_marker_video(tmp_path / "m.mp4")
    up = next(extract_video_frames(v, probe_video(v), [0], size=48))
    turned = next(extract_video_frames(v, probe_video(v), [0], extra_rotation=90, size=48))
    assert up.getpixel((4, 4))[0] > 200 and turned.getpixel((43, 4))[0] > 200 and turned.getpixel((4, 4))[0] < 50


# ---------------------------------------------------------------- picture


def test_circle_mask_values():
    m = circle_mask()
    assert m.getpixel((494, 507)) == 255 and m.getpixel((494 + 499, 507)) == 255
    assert 100 < m.getpixel((494 + 375, 507 + 375)) < 155  # r = 530.3, halfway through the 500-560 band
    assert m.getpixel((0, 0)) == 0 and m.getpixel((1023, 1023)) == 0


def test_apply_circle_blackens_outside():
    red = Image.new("RGB", (1024, 1024), (255, 0, 0))
    out = apply_circle(red, circle_mask())
    assert out.getpixel((494, 507)) == (255, 0, 0) and out.getpixel((5, 5)) == (0, 0, 0)


def test_fisheye_zero_is_identity_and_positive_magnifies_centre():
    img = Image.new("RGB", (1024, 1024), (0, 0, 0))
    ImageDraw.Draw(img).ellipse((494 - 50, 507 - 50, 494 + 50, 507 + 50), fill=(255, 255, 255))
    ImageDraw.Draw(img).rectangle((494 + 490, 500, 494 + 499, 514), fill=(0, 255, 0))
    assert fisheye_warp(img, 0) is img
    w = fisheye_warp(img, 0.3)
    assert w.getpixel((494 + 60, 507))[0] > 200  # the disc grew past r = 50
    assert w.getpixel((494 + 495, 507))[1] > 200  # the rim stays where it was


def test_stamp_covers_old_text_and_sits_bottom_left():
    s = Stamper()
    red = Image.new("RGB", (1024, 1024), (255, 0, 0))
    out = s.draw(red, "2025-04-07 07:48:12")
    assert all(px != (255, 0, 0) for px in out.crop(OLD_TEXT_BOX).getdata())  # no original pixel survives
    white = [(x, y) for y in range(960, 1024) for x in range(0, 400) if out.getpixel((x, y)) == (255, 255, 255)]
    assert white and min(x for x, _ in white) >= 9 and max(y for _, y in white) <= 1014
    assert s.pad == 7 and s.box[0] == 0 and s.box[3] == 1024
    assert red.getpixel((5, 1010)) == (255, 0, 0)  # draws on a copy


def test_stamper_needs_the_font(tmp_path):
    with pytest.raises(FileNotFoundError):
        Stamper(font_path=tmp_path / "missing.ttf")


# ---------------------------------------------------------------- one day


def make_nextme(media, vid, frames, ended):
    d = media / "nextme" / "frames" / vid
    d.mkdir(parents=True, exist_ok=True)
    idx = media / "nextme" / "index"
    idx.mkdir(parents=True, exist_ok=True)
    lines = []
    for n in frames:
        Image.new("RGB", (1024, 1024), (n * 2 % 256, 64, 0)).save(d / f"x__frame_{n:05d}.jpg", quality=95)
        lines.append(json.dumps({"n": n, "name": f"x__frame_{n:05d}.jpg", "data": n * 1024, "size": 1, "next": n * 1024 + 512}))
    if ended:
        lines.append(json.dumps({"end": True}))
    (idx / f"{vid}.jsonl").write_text("\n".join(lines) + "\n")


def mini_day(tmp_path, nextme_present=True, nextme_frames=range(0, 3), ended=False, day_index=1, ego="e"):
    media = tmp_path / "media"
    if nextme_present:
        make_nextme(media, "v", nextme_frames, ended)
    vdir = media / "ego4d" / "v2" / "full_scale"
    vdir.mkdir(parents=True, exist_ok=True)
    if not (vdir / "e.mp4").exists():
        make_video(vdir / "e.mp4", 120)
    base = (day_index - 1) * 86400
    nm = clip(base + 0, base + 3, 0, vid="v", slot=f"d{day_index}-s1")
    eg = clip(base + 10, base + 13, 1, ds="ego4d", vid=ego, slot=f"d{day_index}-s2")
    eg["source"]["media"] = {"kind": "video", "fps": 30}
    day = {"day_index": day_index, "clips": [nm, eg], "no_video_spans": []}
    info = {"path": "m.json", "id": "test", "revision": 0, "created_at": "2026-01-01T00:00:00Z"}
    return day, RenderConfig(media_root=media), info


def ffprobe_json(path):
    return json.loads(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format",
                                      "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout)


def boom(*a, **k):
    raise DayError("simulated")


def test_render_day_outputs(tmp_path):
    day, cfg, info = mini_day(tmp_path)
    rep = render_day(info, day, cfg, tmp_path / "out")
    meta = ffprobe_json(tmp_path / "out" / "D01.mp4")
    v = [s for s in meta["streams"] if s["codec_type"] == "video"]
    assert len(meta["streams"]) == 1 and v[0]["width"] == v[0]["height"] == 1024
    assert v[0]["avg_frame_rate"] == "1/1" and int(v[0]["nb_read_frames"]) == rep["frames"] == 6
    lines = [json.loads(l) for l in (tmp_path / "out" / "D01.frames.jsonl").read_text().splitlines()]
    assert [l["dataset"] for l in lines] == ["nextme800"] * 3 + ["ego4d"] * 3
    assert lines[0]["stamp"] == "2025-04-07 00:00:00" and lines[3]["stamp"] == "2025-04-07 00:00:10"
    assert [l["frame"] for l in lines[:3]] == [0, 1, 2] and all("pts" in l for l in lines[3:])
    assert json.loads((tmp_path / "out" / "D01.render.json").read_text())["settings"] == settings_key(info, cfg)


def test_day_with_missing_nextme_folder_is_skipped(tmp_path):
    day, cfg, info = mini_day(tmp_path, nextme_present=False)
    assert check_day(day, cfg) == ["NextMe v: 3 frames missing"]
    day2, cfg2, _ = mini_day(tmp_path / "b", ego="nope")
    assert check_day(day2, cfg2) == ["Ego4D nope: file missing"]


def test_end_hold_is_reported(tmp_path):
    day, cfg, info = mini_day(tmp_path, nextme_frames=range(0, 2), ended=True)  # clip wants 0..2
    assert check_day(day, cfg) == []
    rep = render_day(info, day, cfg, tmp_path / "out")
    assert rep["clamped"] == [{"dataset": "nextme800", "video_id": "v", "src": 2, "used": 1}]


def test_failed_day_leaves_no_outputs(tmp_path, monkeypatch):
    day, cfg, info = mini_day(tmp_path)
    monkeypatch.setattr(render_frames, "extract_video_frames", boom)
    with pytest.raises(DayError):
        render_day(info, day, cfg, tmp_path / "out")
    assert sorted(p.name for p in (tmp_path / "out").iterdir() if p.name != ".probe") == []


def test_rerun_skips_same_settings_and_redoes_changed(tmp_path):
    day, cfg, info = mini_day(tmp_path)
    render_day(info, day, cfg, tmp_path / "out")
    assert day_is_done(tmp_path / "out", 1, settings_key(info, cfg))
    assert not day_is_done(tmp_path / "out", 1, settings_key(info, replace(cfg, rotation={"ego4d": 90, "youtube": 0})))
    assert not day_is_done(tmp_path / "out", 1, settings_key(info, replace(cfg, warp=0.2)))
    assert not day_is_done(tmp_path / "out", 2, settings_key(info, cfg))


def test_video_clamp_at_end_is_reported_and_marked(tmp_path):
    day, cfg, info = mini_day(tmp_path)
    day["clips"][1]["source_start_s"] = 1.99  # last pick at 3.99 s; the 120-frame video ends at 3.967 s
    rep = render_day(info, day, cfg, tmp_path / "out")
    assert [c["video_id"] for c in rep["clamped"]] == ["e"]
    lines = [json.loads(l) for l in (tmp_path / "out" / "D01.frames.jsonl").read_text().splitlines()]
    assert [l["kind"] for l in lines[3:]] == ["nearest", "nearest", "clamped"]


# ---------------------------------------------------------------- command line


def mini_manifest(tmp_path, days, day2_missing_video=False):
    out = []
    for i in range(1, days + 1):
        day, _, _ = mini_day(tmp_path, day_index=i, ego="nope" if (i == 2 and day2_missing_video) else "e")
        out.append(day)
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"id": "test-m", "revision": 0, "created_at": "2026-01-01T00:00:00Z", "days": out}))
    return p


def test_main_renders_ready_days_and_reports_missing(tmp_path, capsys):
    m = mini_manifest(tmp_path, days=2, day2_missing_video=True)
    args = [str(m), "--out", str(tmp_path / "out"), "--media-root", str(tmp_path / "media")]
    rc = main(args + ["--jobs", "2"])
    out = capsys.readouterr().out
    assert rc == 1 and (tmp_path / "out" / "D01.mp4").exists() and not (tmp_path / "out" / "D02.mp4").exists()
    assert "day 2: Ego4D nope: file missing" in out and "D01 done" in out
    assert main(args + ["--days", "1"]) == 0 and "D01 up to date" in capsys.readouterr().out


def test_main_refuses_without_tools(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(render_frames.shutil, "which", lambda name: None)
    assert main([str(mini_manifest(tmp_path, days=1)), "--out", str(tmp_path / "o")]) == 2
    assert "ffmpeg" in capsys.readouterr().out


def test_warp_preview_writes_one_image(tmp_path):
    m = json.loads(mini_manifest(tmp_path, days=1).read_text())
    p = warp_preview(m, RenderConfig(media_root=tmp_path / "media"), [0, 0.2], tmp_path / "out")
    im = Image.open(p)
    assert p.name == "warp-preview.jpg" and im.width >= 3 * 256 and im.height >= 2 * 256


# ---------------------------------------------------------------- final review fixes


def test_extract_stops_decoding_after_each_batch(tmp_path, monkeypatch):
    v = make_video(tmp_path / "long.mp4", 900)
    info = probe_video(v)
    real_popen = subprocess.Popen
    reports = []

    def popen_with_report(cmd, *a, **k):  # ffmpeg writes a verbose report per run
        rep = tmp_path / f"report{len(reports)}.log"
        reports.append(rep)
        env = dict(k.pop("env", None) or __import__("os").environ, FFREPORT=f"file={rep}:level=40")
        return real_popen(cmd, *a, env=env, **k)

    monkeypatch.setattr(render_frames, "SELECT_BATCH", 2)
    monkeypatch.setattr(render_frames.subprocess, "Popen", popen_with_report)
    assert numbers(extract_video_frames(v, info, [0, 1, 2], size=32)) == [0, 1, 2]
    decoded = int(__import__("re").search(r"(\d+) frames decoded", reports[0].read_text()).group(1))
    assert decoded < 100  # the first batch must not decode the remaining ~900 frames


def test_ctrl_c_cancels_remaining_days(tmp_path, monkeypatch, capsys):
    m = mini_manifest(tmp_path, days=3)

    def interrupted(futures):
        raise KeyboardInterrupt

    monkeypatch.setattr(render_frames, "as_completed", interrupted)
    rc = main([str(m), "--out", str(tmp_path / "out"), "--media-root", str(tmp_path / "media"), "--jobs", "1"])
    assert rc == 130 and "interrupted" in capsys.readouterr().out
    assert not (tmp_path / "out" / "D03.mp4").exists()


def test_corrupt_probe_cache_is_a_miss(tmp_path):
    v = make_video(tmp_path / "a.mp4", 30)
    first = probe_video(v, cache_dir=tmp_path / "cache")
    for f in (tmp_path / "cache").iterdir():
        f.write_text("{")  # cut mid-write
    assert probe_video(v, cache_dir=tmp_path / "cache") == first


def test_probe_cache_lives_under_media_not_renders(tmp_path):
    day, cfg, info = mini_day(tmp_path)
    render_day(info, day, cfg, tmp_path / "out")
    assert not (tmp_path / "out" / ".probe").exists()
    assert any((tmp_path / "media" / ".probe").iterdir())


def test_settings_key_ignores_how_the_manifest_path_was_written():
    cfg = RenderConfig()
    a = {"path": "data/exports/m.json", "id": "m", "revision": 0, "created_at": "x"}
    assert settings_key(a, cfg) == settings_key({**a, "path": "./data/exports/m.json"}, cfg)
    assert settings_key(a, cfg) != settings_key({**a, "revision": 1}, cfg)
