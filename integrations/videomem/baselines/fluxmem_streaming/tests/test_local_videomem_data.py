"""Causal local-video contracts using tiny, CPU-generated MP4s and real pixels."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

from local_videomem.data import (
    FrameBudgetExceeded,
    decode_prefix,
    prepare_manifest,
    public_question,
    select_prefix,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def generated_media(tmp_path_factory):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None or shutil.which("ffprobe") is None:
        pytest.skip("Real video contract tests require ffmpeg and ffprobe")
    directory = tmp_path_factory.mktemp("causal_video_media")
    green = directory / "green.mp4"
    red_blue = directory / "red_blue.mp4"
    common = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    commands = [
        common + [
            "-f", "lavfi", "-i", "color=c=green:s=96x64:r=4:d=2",
            "-an", "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", str(green),
        ],
        common + [
            "-f", "lavfi", "-i", "color=c=red:s=96x64:r=4:d=2",
            "-f", "lavfi", "-i", "color=c=blue:s=96x64:r=4:d=2",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[out]", "-map", "[out]",
            "-an", "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", str(red_blue),
        ],
    ]
    for command in commands:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    return green, red_blue


@pytest.fixture
def sources(tmp_path, generated_media):
    green, red_blue = generated_media
    specification = [
        ("prior_day1.mp4", green, "DAY1", 1, 0.0),
        ("prior_day2.mp4", green, "DAY2", 2, 0.0),
        ("current_red_blue.mp4", red_blue, "DAY3", 3, 7.0),
        ("later_same_day.mp4", green, "DAY3", 3, 11.0),
        ("future_day4.mp4", green, "DAY4", 4, 0.0),
    ]
    rows = []
    for name, original, day_id, day_order, start in specification:
        path = tmp_path / name
        shutil.copyfile(original, path)
        rows.append({
            "path": str(path), "day_id": day_id, "day_order": day_order,
            "day_start_seconds": start, "timeline_status": "provisional_concatenation",
        })
    return [rows[4], rows[3], rows[1], rows[2], rows[0]]


def question(seconds=8.5):
    return {
        "question_id": "causal-color", "question": "What color is visible now?",
        "choices": ["red", "blue"],
        "query_time": {"day_id": "DAY3", "day_order": 3, "seconds": seconds},
    }


def only_current_source(sources):
    return next(row for row in sources if Path(row["path"]).name == "current_red_blue.mp4")


def require_av():
    return pytest.importorskip("av", reason="Actual sequential decoding requires PyAV")


def test_data_module_import_is_lazy_and_does_not_import_gpu_libraries():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import local_videomem.data; "
         "assert 'av' not in sys.modules; assert 'torch' not in sys.modules; "
         "assert 'transformers' not in sys.modules"],
        cwd=ROOT, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_prepare_accepts_list_object_and_json_and_sorts_source_timeline(sources, tmp_path):
    source_file = tmp_path / "sources.json"
    source_file.write_text(json.dumps({"videos": sources}), encoding="utf-8")
    manifests = [
        prepare_manifest(sources),
        prepare_manifest({"videos": sources}),
        prepare_manifest(str(source_file), fingerprint_cache_dir=tmp_path / "fingerprints"),
    ]
    expected_names = ["prior_day1.mp4", "prior_day2.mp4", "current_red_blue.mp4", "later_same_day.mp4", "future_day4.mp4"]
    for manifest in manifests:
        videos = manifest["videos"]
        assert [Path(row["path"]).name for row in videos] == expected_names
        assert [row["duration_seconds"] for row in videos] == pytest.approx([2, 2, 4, 2, 2], abs=0.05)
        assert [row["day_start_seconds"] for row in videos] == [0, 0, 7, 11, 0]
        for row in videos:
            assert Path(row["path"]).is_absolute()
            assert re.fullmatch(r"[0-9a-f]{64}", row["source_fingerprint"])
            assert row["source_fingerprint"] == hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()
            assert isinstance(row["provenance"]["source_state"], dict)
            assert row["provenance"]["source_state"]
    assert [row["source_fingerprint"] for row in manifests[0]["videos"]] == [row["source_fingerprint"] for row in manifests[2]["videos"]]


def test_prepare_defaults_day_offset_to_zero(sources):
    source = {key: value for key, value in sources[-1].items() if key != "day_start_seconds"}
    manifest = prepare_manifest([source])
    assert manifest["videos"][0]["day_start_seconds"] == 0


def test_public_question_drops_unknown_nested_fields_and_all_ground_truth(tmp_path):
    original = question()
    original["query_time"]["answer"] = "GT-LEAK-SENTINEL"
    original["query_time"]["evidence"] = {"path": str(tmp_path / "gt_only.mp4")}
    original.update({
        "answer": "blue", "gt_answer": "GT-LEAK-SENTINEL",
        "ground_truth": {"choice": "blue", "path": str(tmp_path / "gt_only.mp4")},
        "mapped_inputs": [{"video": str(tmp_path / "gt_only.mp4"), "time": [100, 110]}],
        "evidence": {"future": "GT-LEAK-SENTINEL"},
    })
    before = deepcopy(original)
    sanitized = public_question(original)
    assert sanitized == question()
    assert original == before
    assert "GT-LEAK-SENTINEL" not in json.dumps(sanitized)
    assert "gt_only.mp4" not in json.dumps(sanitized)
    minimal = {key: value for key, value in question().items() if key != "choices"}
    assert public_question(minimal)["choices"] == []


def test_select_prefix_includes_all_prior_days_and_exact_partial_current_file(sources):
    manifest = prepare_manifest(sources)
    segments = select_prefix(manifest, question()["query_time"])
    assert [Path(item["path"]).name for item in segments] == ["prior_day1.mp4", "prior_day2.mp4", "current_red_blue.mp4"]
    assert [item["day_order"] for item in segments] == [1, 2, 3]
    assert [item["file_start_seconds"] for item in segments] == [0, 0, 0]
    assert [item["file_end_seconds"] for item in segments] == pytest.approx([2, 2, 1.5])
    assert [item["day_start_seconds"] for item in segments] == [0, 0, 7]
    assert [item["day_end_seconds"] for item in segments] == pytest.approx([2, 2, 8.5])
    for item in segments:
        source = next(row for row in manifest["videos"] if row["path"] == item["path"])
        assert item["source_fingerprint"] == source["source_fingerprint"]
        assert item["provenance"] == source["provenance"]
    assert manifest["videos"][2]["duration_seconds"] == pytest.approx(4)


def test_exact_file_start_cutoff_excludes_that_file_but_keeps_prior_days(sources):
    manifest = prepare_manifest(sources)
    segments = select_prefix(manifest, question(7.0)["query_time"])
    assert [item["day_id"] for item in segments] == ["DAY1", "DAY2"]


def test_real_frames_are_causal_rgb_pixels_with_exact_source_pointers(sources, tmp_path, monkeypatch):
    av = require_av()
    manifest = prepare_manifest(sources)
    private = question()
    fake_gt = tmp_path / "must_never_open_gt_video.mp4"
    private.update({"gt_answer": "blue", "mapped_inputs": [{"path": str(fake_gt)}]})
    private["query_time"]["ground_truth"] = {"path": str(fake_gt)}
    opened = []
    actual_open = av.open

    def tracked_open(file, *args, **kwargs):
        opened.append(str(file))
        return actual_open(file, *args, **kwargs)

    actual_read_bytes = Path.read_bytes

    def no_whole_video_read(path):
        if path.suffix.lower() == ".mp4":
            pytest.fail("Sequential decoding must not load an entire MP4 with read_bytes")
        return actual_read_bytes(path)

    monkeypatch.setattr(av, "open", tracked_open)
    monkeypatch.setattr(Path, "read_bytes", no_whole_video_read)
    decoded = decode_prefix(manifest, private, fps=1.0, max_frames=6, max_side=32)
    frames, metadata = decoded["frames"], decoded["frame_metadata"]
    assert len(frames) == len(metadata) == 6
    assert decoded["fps"] == 1.0 and decoded["sampling"]
    assert decoded["segments"] == select_prefix(manifest, question()["query_time"])
    assert decoded["question"] == question()
    assert [Path(path).name for path in opened] == ["prior_day1.mp4", "prior_day2.mp4", "current_red_blue.mp4"]
    assert str(fake_gt) not in json.dumps(metadata)
    assert [row["file_seconds"] for row in metadata] == pytest.approx([0, 1, 0, 1, 0, 1])
    assert [row["day_seconds"] for row in metadata] == pytest.approx([0, 1, 0, 1, 7, 8])
    assert [row["day_order"] for row in metadata] == [1, 1, 2, 2, 3, 3]
    for image, row in zip(frames, metadata):
        assert image.mode == "RGB" and max(image.size) <= 32
        assert row["pixel_sha256"] == hashlib.sha256(image.tobytes()).hexdigest()
        assert row["path"] in opened
        red, green, blue = image.getpixel((image.width // 2, image.height // 2))
        if row["day_id"] == "DAY3":
            assert row["day_seconds"] < 8.5
            assert red > max(green, blue) + 100
        else:
            assert row["day_id"] in {"DAY1", "DAY2"}
            assert green > max(red, blue) + 60


def test_sampling_uses_actual_pts_at_or_after_grid_and_excludes_cutoff_frame(sources):
    require_av()
    source = only_current_source(sources)
    manifest = prepare_manifest([source])
    decoded = decode_prefix(manifest, question(8.0), fps=3.0, max_frames=3)
    # Actual 4 Hz frames first at/after the 3 Hz grid: 0, .5, .75.
    # The real frame at 1.0 s is exactly at the cutoff and must be excluded.
    assert [row["file_seconds"] for row in decoded["frame_metadata"]] == pytest.approx([0, 0.5, 0.75])
    assert all(row["day_seconds"] < 8.0 for row in decoded["frame_metadata"])


def test_target_before_cutoff_cannot_admit_a_decoded_frame_after_cutoff(sources):
    require_av()
    manifest = prepare_manifest([only_current_source(sources)])
    decoded = decode_prefix(manifest, question(7.4), fps=3.0, max_frames=2)
    # The target at 1/3 s would select the .5 s frame, which is already future.
    assert [row["file_seconds"] for row in decoded["frame_metadata"]] == pytest.approx([0])


def test_advancing_cutoff_reveals_real_blue_pixels(sources):
    require_av()
    manifest = prepare_manifest([only_current_source(sources)])
    decoded = decode_prefix(manifest, question(9.5), fps=1.0, max_frames=3)
    assert [row["file_seconds"] for row in decoded["frame_metadata"]] == pytest.approx([0, 1, 2])
    first = decoded["frames"][0].getpixel((0, 0))
    last = decoded["frames"][-1].getpixel((0, 0))
    assert first[0] > first[2] + 100
    assert last[2] > last[0] + 100


def test_frame_budget_rejects_all_past_days_before_opening_any_decoder(sources, monkeypatch):
    av = require_av()
    manifest = prepare_manifest(sources)

    def forbidden_open(*args, **kwargs):
        pytest.fail("Frame-budget preflight must run before opening any decoder")

    monkeypatch.setattr(av, "open", forbidden_open)
    with pytest.raises(FrameBudgetExceeded):
        decode_prefix(manifest, question(), fps=1.0, max_frames=5)


def test_decode_requires_an_explicit_frame_budget():
    with pytest.raises(TypeError, match="max_frames"):
        decode_prefix({"videos": []}, question())


def test_changed_source_is_rejected_before_decoding(sources, monkeypatch):
    av = require_av()
    manifest = prepare_manifest(sources)
    path = Path(only_current_source(sources)["path"])
    with path.open("ab") as stream:
        stream.write(b"changed-after-manifest")

    def forbidden_open(*args, **kwargs):
        pytest.fail("Changed source bytes must be rejected before decoder access")

    monkeypatch.setattr(av, "open", forbidden_open)
    with pytest.raises(ValueError, match="(?i)(chang|fingerprint|source|sha256)"):
        decode_prefix(manifest, question(), max_frames=6)


def test_replaced_source_with_restored_size_and_mtime_is_rejected(sources, monkeypatch, tmp_path):
    av = require_av()
    manifest = prepare_manifest(sources)
    path = Path(only_current_source(sources)["path"])
    before = path.stat()
    replacement = tmp_path / "replacement.mp4"
    replacement.write_bytes(b"X" * before.st_size)
    shutil.copystat(path, replacement)
    replacement.replace(path)

    def forbidden_open(*args, **kwargs):
        pytest.fail("Replaced source inode must be rejected before decoder access")

    monkeypatch.setattr(av, "open", forbidden_open)
    with pytest.raises(ValueError, match="(?i)(chang|fingerprint|source|sha256)"):
        decode_prefix(manifest, question(), max_frames=6)
