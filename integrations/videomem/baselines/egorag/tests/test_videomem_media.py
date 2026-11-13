"""Media boundary tests use tiny generated videos, never benchmark footage."""

import base64
import hashlib
import io
import json
import shutil
import subprocess

import pytest
from PIL import Image

from videomem_adapter.media import extract_frames, prepare_manifest


@pytest.fixture
def video(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe are required")
    path = tmp_path / "video.mp4"
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
        "testsrc2=size=160x96:rate=10", "-t", "2.4", "-c:v", "libx264",
        "-threads", "1", "-pix_fmt", "yuv420p", str(path),
    ], check=True, capture_output=True)
    return path


def source(video, **overrides):
    return {"day_id": "Day1", "day_order": 1, "path": str(video),
            "timeline_status": "direct_mp4", **overrides}


def test_manifest_splits_real_duration_hashes_and_preserves_unknown_clocks(video, tmp_path):
    rows = [source(video)]
    first = prepare_manifest(rows, clip_seconds=1, fingerprint_cache_dir=tmp_path / "hashes")
    second = prepare_manifest({"videos": rows}, clip_seconds=1,
                              fingerprint_cache_dir=tmp_path / "hashes")
    assert first == second
    assert [(clip["file_start_seconds"], clip["file_end_seconds"])
            for clip in first["clips"]] == [(0, 1), (1, 2), (2, 2.4)]
    assert first["videos"][0]["source_fingerprint"] == hashlib.sha256(video.read_bytes()).hexdigest()
    assert first["videos"][0]["virtual_start"] is None
    assert first["videos"][0]["recording_start"] is None
    assert len({clip["clip_id"] for clip in first["clips"]}) == 3
    manifest_path = tmp_path / "sources.json"
    manifest_path.write_text(json.dumps(rows))
    assert prepare_manifest(manifest_path, 1) == first


def test_components_are_sorted_by_explicit_day_order_and_keep_offsets(video):
    manifest = prepare_manifest([
        source(video, day_id="Day2", day_order=2),
        source(video, day_start_seconds=2.4, timeline_status="provisional_concatenation"),
        source(video, timeline_status="provisional_concatenation"),
    ], clip_seconds=2)
    assert [v["day_id"] for v in manifest["videos"]] == ["Day1", "Day1", "Day2"]
    assert manifest["clips"][2]["day_start_seconds"] == 2.4
    assert manifest["clips"][2]["file_start_seconds"] == 0
    assert manifest["clips"][2]["provenance"]["timeline_status"] == "provisional_concatenation"
    assert len({clip["clip_id"] for clip in manifest["clips"]}) == 6


@pytest.mark.parametrize("updates", [
    {"day_order": None}, {"day_order": True}, {"day_order": -1},
    {"day_start_seconds": -1}, {"day_start_seconds": float("nan")},
    {"timeline_status": "verified_by_guess"}, {"path": "relative.mp4"},
])
def test_invalid_source_metadata_is_rejected(video, updates):
    with pytest.raises(ValueError):
        prepare_manifest([source(video, **updates)])


def test_overlaps_and_conflicting_orders_are_rejected(video):
    with pytest.raises(ValueError, match="Overlapping"):
        prepare_manifest([source(video), source(video, day_start_seconds=2.3)])
    with pytest.raises(ValueError, match="Conflicting"):
        prepare_manifest([source(video), source(video, day_order=2)])
    with pytest.raises(ValueError, match="multiple days"):
        prepare_manifest([source(video), source(video, day_id="Day7")])


@pytest.mark.parametrize("clip_seconds", [0, -1, float("inf"), True])
def test_invalid_clip_sizes_are_rejected(video, clip_seconds):
    with pytest.raises(ValueError):
        prepare_manifest([source(video)], clip_seconds=clip_seconds)


def test_sampling_is_bounded_spans_clip_and_returns_real_jpegs(video):
    clip = prepare_manifest([source(video, day_start_seconds=10)], 30)["clips"][0]
    frames = extract_frames(clip, fps=10, max_frames=3, max_side=64)
    assert len(frames) == 3
    assert frames[0]["file_seconds"] == 0
    assert frames[-1]["file_seconds"] == pytest.approx(2.3)
    for frame in frames:
        assert 0 <= frame["file_seconds"] < 2.4
        assert frame["day_seconds"] == pytest.approx(10 + frame["file_seconds"])
        data = base64.b64decode(frame["jpeg_base64"])
        assert hashlib.sha256(data).hexdigest() == frame["sha256"]
        with Image.open(io.BytesIO(data)) as image:
            assert max(image.size) <= 64


def test_decoded_frame_cannot_cross_exclusive_cutoff(video):
    clip = prepare_manifest([source(video)], 30)["clips"][0]
    clip.update(file_start_seconds=0.05, file_end_seconds=0.1,
                day_start_seconds=10, day_end_seconds=10.05)
    assert extract_frames(clip) == []  # next frame is at 0.1, outside [0.05, 0.1)
    clip.update(file_end_seconds=0.15, day_end_seconds=10.1)
    frame, = extract_frames(clip)
    assert frame["file_seconds"] == pytest.approx(0.1)
    assert frame["day_seconds"] == pytest.approx(10.05)


def test_changed_source_invalidates_manifest_and_cached_hash(video, tmp_path):
    cache_dir = tmp_path / "hashes"
    manifest = prepare_manifest([source(video)], fingerprint_cache_dir=cache_dir)
    with video.open("ab") as handle:
        handle.write(b"metadata-change")
    with pytest.raises(ValueError, match="Source changed"):
        extract_frames(manifest["clips"][0])
    updated = prepare_manifest([source(video)], fingerprint_cache_dir=cache_dir)
    assert updated["videos"][0]["source_fingerprint"] != manifest["videos"][0]["source_fingerprint"]
    assert updated["clips"][0]["clip_id"] != manifest["clips"][0]["clip_id"]


def test_high_fps_deduplicates_actual_decoded_timestamps(video):
    clip = prepare_manifest([source(video)], 0.2)["clips"][0]
    frames = extract_frames(clip, fps=100, max_frames=16)
    timestamps = [frame["file_seconds"] for frame in frames]
    assert timestamps == sorted(set(timestamps))
    assert all(0 <= stamp < 0.2 for stamp in timestamps)


def test_nonvideo_file_is_rejected(tmp_path):
    path = tmp_path / "broken.mp4"
    path.write_text("not video data")
    with pytest.raises(ValueError, match="Cannot probe"):
        prepare_manifest([source(path)])
