"""Additional temporal regressions with generated footage and explicit doubles."""

import copy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from videomem_adapter.common import public_question, read
from videomem_adapter.media import prepare_manifest
from videomem_adapter.pipeline import MemoryNotReady, Pipeline


@pytest.fixture(scope="module")
def small_media(tmp_path_factory):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe are required")
    root = tmp_path_factory.mktemp("temporal-extra-videos")
    paths = []
    for color in ("red", "blue"):
        path = root / f"synthetic_{color}.mp4"
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
            f"color=c={color}:s=64x64:r=10:d=4", "-c:v", "libx264", "-threads", "1",
            "-pix_fmt", "yuv420p", str(path),
        ], check=True, capture_output=True)
        paths.append(path)
    config = read(Path(__file__).resolve().parents[1] / "configs/videomem.json")
    config.update(test_only=True, index_engine="sqlite", clip_seconds=2, top_k=3)
    config["frames"] = {"fps": 1, "max_frames": 1, "max_side": 64}
    config["embedding"] = {"backend": "fixture", "test_only": True, "dimensions": 32}
    config["models"] = {
        role: {"backend": "fixture", "model": "temporal-extra-test-double-v1"}
        for role in config["models"]
    }
    return paths, config


def source(path, day=1):
    return {"day_id": f"Day{day}", "day_order": day, "path": str(path),
            "timeline_status": "direct_mp4"}


def question(seconds=2, text="What color was the scene?"):
    return {"question_id": "temporal-extra", "question": text, "choices": ["red", "blue"],
            "query_time": {"day_id": "Day1", "day_order": 1, "seconds": seconds}}


def pipeline(small_media, work, *, two_days=False):
    paths, config = small_media
    rows = [source(paths[0])]
    if two_days:
        rows.append(source(paths[1], day=2))
    manifest = prepare_manifest(rows, clip_seconds=2)
    return Pipeline(copy.deepcopy(config), manifest, work)


def test_future_day_selection_stops_before_future_summary_is_exposed(small_media, tmp_path, monkeypatch):
    pipe = pipeline(small_media, tmp_path, two_days=True)
    assert not pipe.create()["errors"]
    future = pipe.summaries({"day_id": "Day2", "day_order": 2, "seconds": 4})
    assert any("blue" in row["text"].lower() for row in future["day"])

    original = pipe.models.generate
    observed = []

    def malicious_day(role, system, prompt, frames=None):
        observed.append({"role": role, "prompt": prompt})
        if role == "range_selector":
            return "[2]"
        return original(role, system, prompt, frames=frames)

    monkeypatch.setattr(pipe.models, "generate", malicious_day)
    with pytest.raises(ValueError, match="unavailable or future day"):
        pipe.query(question())
    range_calls = [call for call in observed if call["role"] == "range_selector"]
    assert len(range_calls) == 1  # no follow-up hour selector with Day2 descriptions
    assert "blue" not in json.dumps(observed).lower()
    assert "'date': 2" not in range_calls[0]["prompt"]


@pytest.mark.parametrize("cutoff,expected_status", [
    (1.999, "no_observed_evidence"), (2.001, "complete"), (3.999, "complete"),
])
def test_fractional_cutoff_excludes_crossing_clip_without_failing_query(
        small_media, tmp_path, cutoff, expected_status):
    pipe = pipeline(small_media, tmp_path)
    assert not pipe.create()["errors"]
    result = pipe.query(question(cutoff))
    assert result["status"] == expected_status
    expected = [clip["clip_id"] for clip in pipe.manifest["clips"]
                if clip["day_end_seconds"] <= cutoff]
    assert result["eligible_clip_ids"] == expected
    assert all(hit["clip"]["day_end_seconds"] <= cutoff for hit in result["retrieved"])
    assert pipe.manifest["clips"][1]["clip_id"] not in result["eligible_clip_ids"]


def test_missing_preprocessing_is_distinct_from_empty_observation_prefix(small_media, tmp_path):
    pipe = pipeline(small_media, tmp_path)
    empty = pipe.query(question(1.999))
    assert empty["status"] == "no_observed_evidence"
    assert pipe.models.calls == 0
    with pytest.raises(MemoryNotReady, match="Run create"):
        pipe.query(question(2))
    assert pipe.models.calls == 0
    created = pipe.create(limit=1)
    assert len(created["new"]) == 1 and not created["errors"]
    assert pipe.query(question(2))["status"] == "complete"
    with pytest.raises(MemoryNotReady, match="Run create"):
        pipe.query(question(4))


def test_changed_source_rejects_stale_manifest_and_cannot_reuse_caption_cache(small_media, tmp_path):
    paths, config = small_media
    local = tmp_path / "mutable_test_copy.mp4"
    shutil.copyfile(paths[0], local)
    work = tmp_path / "run"
    first_manifest = prepare_manifest([source(local)], 2)
    first = Pipeline(config, first_manifest, work)
    assert len(first.create()["new"]) == 2

    # Mutate only the private fixture copy; adding bytes changes source identity
    # while preserving a decodable MP4 and its duration.
    with local.open("ab") as handle:
        handle.write(b"changed-synthetic-fixture")
    with pytest.raises(ValueError, match="Source video changed"):
        Pipeline(config, first_manifest, work)
    changed_manifest = prepare_manifest([source(local)], 2)
    assert (changed_manifest["videos"][0]["source_fingerprint"]
            != first_manifest["videos"][0]["source_fingerprint"])
    second = Pipeline(config, changed_manifest, work)
    assert second.index_namespace != first.index_namespace
    with pytest.raises(MemoryNotReady):
        second.query(question())
    rebuilt = second.create()
    assert len(rebuilt["new"]) == 2 and rebuilt["reused"] == [] and not rebuilt["errors"]


@pytest.mark.parametrize("top_k", [0, -1, True])
def test_nonpositive_or_boolean_top_k_is_rejected(small_media, tmp_path, top_k):
    paths, config = small_media
    config = copy.deepcopy(config)
    config["top_k"] = top_k
    manifest = prepare_manifest([source(paths[0])], 2)
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        Pipeline(config, manifest, tmp_path)


def test_day_zero_is_rejected_in_sources_and_question_cutoffs(small_media):
    paths, _ = small_media
    with pytest.raises(ValueError, match="day_order.*positive"):
        prepare_manifest([source(paths[0], day=0)])
    row = question()
    row["query_time"].update(day_id="Day0", day_order=0)
    with pytest.raises(ValueError, match="day_order.*positive"):
        public_question(row)


def test_last_time_selector_keeps_entire_multidigit_sentence_suffix(small_media, tmp_path, monkeypatch):
    pipe = pipeline(small_media, tmp_path)
    original = pipe.models.generate
    selected_id = "DAY1-00000000-00000200_11"
    observed_selector_prompts = []

    def long_caption_and_selection(role, system, prompt, frames=None):
        if role == "captioner":
            assert frames  # preprocessing still performs real fixture MP4 decoding
            return ". ".join(f"Synthetic red color sentence {index}" for index in range(12)) + "."
        if role == "recency_selector":
            observed_selector_prompts.append(prompt)
            return "ID: " + selected_id
        return original(role, system, prompt, frames=frames)

    monkeypatch.setattr(pipe.models, "generate", long_caption_and_selection)
    assert not pipe.create()["errors"]
    result = pipe.query(question(text="What color was the last scene?"))
    assert result["status"] == "complete"
    assert observed_selector_prompts and selected_id in observed_selector_prompts[0]
    hit, = result["retrieved"]
    assert hit["sentence_id"] == selected_id
    assert hit["document"] == "Synthetic red color sentence 11."
    nearby = pipe.index.get_caption(selected_id, 1)["documents"]
    assert "Synthetic red color sentence 10." in nearby
    assert "Synthetic red color sentence 11." in nearby
    assert "Synthetic red color sentence 1." not in nearby
