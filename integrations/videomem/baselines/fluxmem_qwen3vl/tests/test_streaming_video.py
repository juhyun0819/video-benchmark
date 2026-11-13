"""CPU integration tests for bounded, causal streaming video samples."""

from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from local_videomem.data import decode_prefix, prepare_manifest
from streaming.video import frame_key, iter_pairs, time_key


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def stream_media(tmp_path_factory):
    pytest.importorskip("av")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None or shutil.which("ffprobe") is None:
        pytest.skip("Real video tests require ffmpeg and ffprobe")
    directory = tmp_path_factory.mktemp("streaming_pair_media")
    green = directory / "green.mp4"
    red_blue = directory / "red_blue.mp4"
    common = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    commands = [
        common + ["-f", "lavfi", "-i", "color=c=green:s=96x64:r=4:d=3",
                  "-an", "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", str(green)],
        common + ["-f", "lavfi", "-i", "color=c=red:s=96x64:r=4:d=2",
                  "-f", "lavfi", "-i", "color=c=blue:s=96x64:r=4:d=2",
                  "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[out]", "-map", "[out]",
                  "-an", "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", str(red_blue)],
    ]
    for command in commands:
        subprocess.run(command, check=True, capture_output=True, timeout=30)
    return green, red_blue


@pytest.fixture
def stream_sources(tmp_path, stream_media):
    green, red_blue = stream_media
    specification = [
        ("prior1.mp4", green, "DAY1", 1, 0),
        ("prior2.mp4", green, "DAY2", 2, 0),
        ("current.mp4", red_blue, "DAY3", 3, 7),
        ("later.mp4", green, "DAY3", 3, 11),
        ("future.mp4", green, "DAY4", 4, 0),
    ]
    sources = []
    for name, original, day_id, day_order, offset in specification:
        path = tmp_path / name
        shutil.copyfile(original, path)
        sources.append({"path": str(path), "day_id": day_id,
                        "day_order": day_order, "day_start_seconds": offset})
    return sources


def cutoff(seconds=8.5):
    return {"day_id": "DAY3", "day_order": 3, "seconds": seconds}


def current_manifest(sources):
    return prepare_manifest([sources[2]])


def assert_closed(image):
    with pytest.raises(ValueError, match="closed"):
        image.getpixel((0, 0))


def test_streaming_video_imports_no_decoder_or_model_libraries():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import streaming.video; "
         "assert not {'av', 'torch', 'transformers'}.intersection(sys.modules)"],
        cwd=ROOT, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_prior_days_real_pixels_and_strict_future_exclusion(stream_sources, monkeypatch):
    import av

    manifest = prepare_manifest(list(reversed(stream_sources)))
    supplied = cutoff()
    supplied["ground_truth"] = {"path": "/must/not/open.mp4", "answer": "SENTINEL"}
    before = deepcopy(supplied)
    opened = []
    original_open = av.open

    def tracked_open(path, *args, **kwargs):
        opened.append(Path(path).name)
        return original_open(path, *args, **kwargs)

    original_read_bytes = Path.read_bytes

    def no_whole_video_read(path):
        if path.suffix == ".mp4":
            pytest.fail("Streaming decoding must not read a whole MP4 into RAM")
        return original_read_bytes(path)

    monkeypatch.setattr(av, "open", tracked_open)
    monkeypatch.setattr(Path, "read_bytes", no_whole_video_read)
    stats, pairs = {}, []
    for pair in iter_pairs(manifest, supplied, fps=1, max_side=32, stats=stats):
        pairs.append(pair)
        for image, row in zip(pair["frames"], pair["frame_metadata"]):
            assert image.mode == "RGB" and max(image.size) <= 32
            assert row["pixel_sha256"] == hashlib.sha256(image.tobytes()).hexdigest()
            assert row["decode_device"] == "cpu"
            assert frame_key(row) < time_key(cutoff())
            red, green, blue = image.getpixel((0, 0))
            if row["day_id"] == "DAY3":
                assert red > max(green, blue) + 100
            else:
                assert green > max(red, blue) + 60
    assert supplied == before
    assert opened == ["prior1.mp4", "prior2.mp4", "current.mp4"]
    assert [pair["pair_index"] for pair in pairs] == [0, 1, 2]
    rows = [row for pair in pairs for row in pair["frame_metadata"]]
    assert [row["day_order"] for row in rows] == [1, 1, 2, 2, 3, 3]
    assert [row["file_seconds"] for row in rows] == [0, 1, 0, 1, 0, 1]
    assert [row["position"] for row in rows] == [0, 1, 3, 4, 6, 7]
    assert "SENTINEL" not in json.dumps(rows)
    assert stats["sampled_frames"] == 8
    assert stats["pairs"] == 3 and stats["orphan_frames"] == 2
    assert stats["decoded_frames"] >= stats["sampled_frames"]
    assert stats["max_live_sampled_frames"] == 2
    for pair in pairs:
        assert len(set(row["source_id"] for row in pair["frame_metadata"])) == 1
        assert len(pair["metadata"]["frame_ids"]) == 2
        for image in pair["frames"]:
            assert_closed(image)


def test_metadata_and_1fps_pixels_match_existing_sampler(stream_sources):
    manifest = current_manifest(stream_sources)
    question = {"question_id": "comparison", "question": "What is visible?",
                "choices": [], "query_time": cutoff(11)}
    old = decode_prefix(manifest, question, fps=1, max_frames=4, max_side=32)
    try:
        seen = []
        for pair in iter_pairs(manifest, cutoff(11), max_side=32):
            for image, row in zip(pair["frames"], pair["frame_metadata"]):
                position = len(seen)
                assert row == old["frame_metadata"][position]
                assert image.tobytes() == old["frames"][position].tobytes()
                seen.append(row)
        assert len(seen) == 4
        assert seen[0]["pixel_sha256"] != seen[2]["pixel_sha256"]
    finally:
        for image in old["frames"]:
            image.close()


@pytest.mark.parametrize("seconds,expected_pairs,expected_orphans", [(8, 0, 1), (9, 1, 0), (10, 1, 1), (11, 2, 0)])
def test_partial_temporal_group_never_uses_future_or_padding(
        stream_sources, seconds, expected_pairs, expected_orphans):
    stats = {}
    pairs = list(iter_pairs(current_manifest(stream_sources), cutoff(seconds), stats=stats))
    assert len(pairs) == expected_pairs
    assert stats["orphan_frames"] == expected_orphans
    assert stats["sampled_frames"] == 2 * expected_pairs + expected_orphans
    assert all(frame_key(row) < time_key(cutoff(seconds))
               for pair in pairs for row in pair["frame_metadata"])


def test_exact_pts_excludes_frame_after_target_before_cutoff(stream_sources):
    stats = {}
    manifest = current_manifest(stream_sources)
    assert list(iter_pairs(manifest, cutoff(7.4), fps=3, stats=stats)) == []
    assert stats["sampled_frames"] == 1 and stats["orphan_frames"] == 1
    pairs = list(iter_pairs(manifest, cutoff(8), fps=3, stats=stats))
    assert [row["file_seconds"] for row in pairs[0]["frame_metadata"]] == [0, 0.5]
    assert stats["sampled_frames"] == 3 and stats["orphan_frames"] == 1
    assert pairs[0]["frontier_time"]["seconds"] == pytest.approx(7 + 2 / 3)


def test_available_time_frontier_and_peek_before_query(stream_sources):
    iterator = iter_pairs(current_manifest(stream_sources), cutoff(10.5))
    try:
        first = next(iterator)
        assert first["available_time"] == cutoff(8)
        assert first["frontier_time"] == cutoff(9)
        # Decode-only lookahead can be held while an earlier query is answered.
        second = next(iterator)
        assert frame_key(second) >= time_key(cutoff(9.5))
        assert second["available_time"] == cutoff(10)
        assert second["frontier_time"] == cutoff(10.5)
        second["frames"][0].getpixel((0, 0))
        for image in first["frames"]:
            assert_closed(image)
    finally:
        iterator.close()
    for image in second["frames"]:
        assert_closed(image)


def test_many_pairs_hold_only_two_sampled_images_and_close_on_early_exit(stream_sources):
    stats = {}
    iterator = iter_pairs(current_manifest(stream_sources), cutoff(11), fps=4, stats=stats)
    previous = None
    try:
        for _ in range(5):
            pair = next(iterator)
            if previous is not None:
                for image in previous["frames"]:
                    assert_closed(image)
            for image in pair["frames"]:
                image.getpixel((0, 0))
            previous = pair
        assert stats["sampled_frames"] == 10
        assert stats["pairs"] == 5
        assert stats["max_live_sampled_frames"] == 2
    finally:
        iterator.close()
    for image in previous["frames"]:
        assert_closed(image)


def test_source_change_before_decoder_open_rejected(stream_sources, monkeypatch):
    import av

    manifest = current_manifest(stream_sources)
    with Path(stream_sources[2]["path"]).open("ab") as handle:
        handle.write(b"changed")

    def forbidden_open(*args, **kwargs):
        pytest.fail("Source validation must precede decoder access")

    monkeypatch.setattr(av, "open", forbidden_open)
    with pytest.raises(ValueError, match="Source changed"):
        next(iter_pairs(manifest, cutoff(11)))


@pytest.mark.parametrize("finish", ["next", "close"])
def test_source_change_during_iteration_rejected_and_images_closed(stream_sources, finish):
    iterator = iter_pairs(current_manifest(stream_sources), cutoff(11))
    pair = next(iterator)
    with Path(stream_sources[2]["path"]).open("ab") as handle:
        handle.write(b"changed-during-decoding")
    with pytest.raises(ValueError, match="Source changed"):
        next(iterator) if finish == "next" else iterator.close()
    for image in pair["frames"]:
        assert_closed(image)


def test_later_source_changed_after_initial_preflight_rejected(stream_sources):
    iterator = iter_pairs(prepare_manifest(stream_sources), cutoff())
    first = next(iterator)
    with Path(stream_sources[1]["path"]).open("ab") as handle:
        handle.write(b"changed-before-its-decoder-opens")
    with pytest.raises(ValueError, match="Source changed"):
        next(iterator)
    for image in first["frames"]:
        assert_closed(image)


def test_time_helpers_use_chronology_and_exact_frame_fraction():
    assert time_key({"query_time": cutoff(8.5)}) == (3, Fraction(17, 2))
    assert frame_key({"available_time": cutoff(8.5)}) == time_key(cutoff())
    frame = {"day_id": "DAY1", "day_order": 1, "day_seconds": 1 / 3,
             "day_seconds_fraction": "1/3"}
    assert frame_key(frame) == (1, Fraction(1, 3))
    assert frame_key(frame) < time_key(cutoff(0))


@pytest.mark.parametrize("kwargs", [{"fps": 0}, {"fps": float("nan")}, {"max_side": 0}, {"max_side": True}])
def test_invalid_sampling_options_rejected(kwargs):
    with pytest.raises(ValueError):
        next(iter_pairs({"videos": []}, cutoff(), **kwargs))
