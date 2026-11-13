"""Structural tests: real native retrieval/clustering, explicitly fake model calls."""
import copy
import pickle
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


def stamps(day, start, count=2):
    return [{"stream_id": "s", "day": day, "global_frame_id": start + i,
             "global_timestamp": float(start + i), "source_timestamp": float(i),
             "source_id": f"day{day}", "source_frame_index": i} for i in range(count)]


def pictures(color="red", count=2):
    return [Image.new("RGB", (16, 16), color) for _ in range(count)]


def ego_config():
    return {"test_only": True, "index_engine": "sqlite", "top_k": 3,
            "embedding": {"backend": "fixture", "dimensions": 16},
            "models": {role: {"backend": "fixture", "model": "pixel-contract-test-double-v1"}
                       for role in ["captioner", "summary_10min", "summary_hour", "summary_day",
                                    "range_selector", "query_rewriter", "recency_selector",
                                    "evidence", "answerer"]}}


def test_egorag_resume_retains_index_captions_images_and_summaries():
    from baselines.streambench.adapters.egorag import EgoRAGMethod
    continuous = EgoRAGMethod(4_000_000, ego_config())
    continuous.initialize()
    continuous.observe(pictures(), stamps(1, 0))
    continuous.finalize_chunk()
    day1 = pickle.loads(pickle.dumps(continuous.state_dict()))
    resumed = EgoRAGMethod(4_000_000, ego_config())
    resumed.load_state_dict(day1)
    for method in [continuous, resumed]:
        method.observe(pictures("blue"), stamps(2, 2))
        method.finalize_chunk()
    left, right = continuous.state_dict(), resumed.state_dict()
    # SQLite's file-change counters differ after deserialize; records, index
    # rows and all model-relevant metadata must still match exactly.
    databases = []
    for state in (left, right):
        db = sqlite3.connect(":memory:")
        db.deserialize(state.pop("sqlite"))
        databases.append({table: list(db.execute("SELECT * FROM " + table + " ORDER BY 1,2"))
                          for table in ("cache", "sentences")})
        db.close()
    assert left == right and databases[0] == databases[1]
    answer = resumed.answer("What color was the scene?", ["red", "blue"])
    assert answer["answer"] == "A"
    assert answer["query_read_bytes"] > 0
    assert answer["num_text_items_used"] > 0
    assert resumed.memory_bytes()["visual_memory_bytes"] > 0
    assert len(resumed.state_dict()["manifest"]["clips"]) == 2


def test_egorag_queries_cannot_add_calls_or_summaries_to_observation_state():
    from baselines.streambench.adapters.egorag import EgoRAGMethod
    method = EgoRAGMethod(4_000_000, ego_config())
    method.initialize()
    method.observe(pictures(), stamps(1, 0))
    method.finalize_chunk()
    before = method.state_dict()
    first = method.answer("What color was the scene?", ["red", "blue"])
    method.answer("What color was the last scene?", ["red", "blue"])
    assert method.state_dict() == before
    assert method.answer("What color was the scene?", ["red", "blue"])["answer"] == first["answer"]


class ColorEncoder:
    def __call__(self, frames):
        return torch.tensor(np.array([np.array(image).mean(axis=(0, 1)) for image in frames]), dtype=torch.float32)


class ColorCaptioner:
    def __call__(self, frames, timestamps):
        return ["red scene" if np.array(image)[0, 0, 0] > 0 else "blue scene" for image in frames]


class TextReasoner:
    def score(self, question, choices, captions):
        return [1] * len(captions)

    def answer(self, question, choices, captions):
        return "A" if "red scene" in captions else "B"


def tree(budget=4_000_000):
    from baselines.streambench.adapters.videotree import VideoTreeMethod
    return VideoTreeMethod(budget, encoder=ColorEncoder(), captioner=ColorCaptioner(),
                           reasoner=TextReasoner(), test_only=True,
                           initial_clusters=2, maximum_clusters=2)


def test_videotree_real_kernels_resume_and_query_independence():
    method = tree()
    method.initialize()
    method.observe(pictures() + pictures("blue"), stamps(1, 0, 4))
    saved = pickle.loads(pickle.dumps(method.state_dict()))
    resumed = tree()
    resumed.load_state_dict(saved)
    for item in [method, resumed]:
        item.observe(pictures(), stamps(2, 4))
        item.finalize_chunk()
    assert torch.equal(method.state_dict()["features"], resumed.state_dict()["features"])
    before = resumed.state_dict()
    answer = resumed.answer("What color?", ["red", "blue"])
    assert answer["answer"] == "A"
    assert answer["num_retrieved_items"] == 2
    assert answer["query_read_bytes"] >= 6 * 3 * 4
    assert answer["selected_frame_ids"] == [0, 2]
    assert torch.equal(resumed.state_dict()["features"], before["features"])
    assert resumed.state_dict()["captions"] == before["captions"]


@pytest.mark.parametrize("kind", ["ego", "tree"])
def test_retain_all_methods_reject_budget_instead_of_evicting(kind):
    from baselines.streambench.adapters.egorag import EgoRAGMethod
    from baselines.streambench.method import UnsupportedBudget
    method = EgoRAGMethod(4_000_000, ego_config()) if kind == "ego" else tree()
    method.initialize()
    method.observe(pictures(), stamps(1, 0))
    method.finalize_chunk()
    method.memory_budget_bytes = method.memory_bytes()["persistent_total_bytes"] - 1
    with pytest.raises(UnsupportedBudget):
        method.finalize_chunk()


def test_videotree_rejects_missing_native_backend_instead_of_substitution():
    from baselines.streambench.adapters.videotree import create_method
    with pytest.raises((ValueError, FileNotFoundError), match="encoder|checkpoint|config"):
        create_method({"method": "videotree", "memory_budget_bytes": 4_000_000})


@pytest.mark.parametrize("kind", ["ego", "tree"])
def test_text_tree_reject_reordered_or_foreign_stream_frames(kind):
    from baselines.streambench.adapters.egorag import EgoRAGMethod
    method = EgoRAGMethod(4_000_000, ego_config()) if kind == "ego" else tree()
    method.initialize()
    method.observe(pictures(), stamps(1, 0))
    with pytest.raises(ValueError, match="chronological|stream"):
        method.observe(pictures(), stamps(1, 0))


def test_native_depth_expansion_keeps_zero_based_first_frame():
    from baselines.streambench.adapters.videotree import build_tree
    features = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    selected, info = build_tree(features, ["first", "second", "third", "fourth"],
                                lambda captions: [1] * len(captions), initial_clusters=2,
                                maximum_clusters=2, seed=0)
    assert selected == [0, 2]
    assert info["cluster_count"] == 2


@pytest.mark.parametrize("size", [1, 2, 3])
def test_tree_handles_tiny_prefix_without_dropping_high_relevance_roots(size):
    from baselines.streambench.adapters.videotree import build_tree
    features = torch.eye(3)[:size]
    selected, _ = build_tree(features, [str(i) for i in range(size)],
                              lambda captions: [3] * len(captions), initial_clusters=size,
                              maximum_clusters=size)
    assert selected == list(range(size))


def test_single_root_cluster_handles_multiple_frames():
    from baselines.streambench.adapters.videotree import build_tree
    selected, _ = build_tree(torch.tensor([[1., 0.], [1., 0.]]), ["one", "two"],
                              lambda captions: [1], initial_clusters=1, maximum_clusters=1)
    assert selected == [0]


def test_relevance_maps_temporal_caption_order_back_to_native_cluster_labels():
    from baselines.streambench.adapters.videotree import build_tree
    features = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    selected, info = build_tree(features, ["red", "red", "blue", "blue"],
                                 lambda captions: [3 if c == "red" else 1 for c in captions],
                                 initial_clusters=2, maximum_clusters=2)
    assert info["relevance_by_cluster"][info["cluster_labels"][0]] == 3
    assert info["relevance_by_cluster"][info["cluster_labels"][2]] == 1


def test_high_relevance_singleton_children_are_not_lost():
    from baselines.streambench.adapters.videotree import build_tree
    selected, _ = build_tree(torch.eye(3), ["one", "two", "three"],
                              lambda captions: [3], initial_clusters=1, maximum_clusters=1)
    assert selected == [0, 1, 2]


@pytest.mark.parametrize("kind", ["ego", "tree"])
def test_fresh_process_checkpoint_resume_matches_continuous(kind, tmp_path):
    from baselines.streambench.adapters.egorag import EgoRAGMethod
    method = EgoRAGMethod(4_000_000, ego_config()) if kind == "ego" else tree()
    method.initialize()
    method.observe(pictures(), stamps(1, 0))
    method.finalize_chunk()
    checkpoint = tmp_path / "day1.pt"
    method.save_checkpoint(checkpoint)
    program = '''
import json, runpy, sys
from pathlib import Path
from baselines.streambench.adapters.egorag import EgoRAGMethod
m = runpy.run_path("baselines/streambench/tests/test_streambench_text_tree.py")
kind, checkpoint, target = sys.argv[1:]
method = EgoRAGMethod(4_000_000, m["ego_config"]()) if kind == "ego" else m["tree"]()
method.load_checkpoint(checkpoint)
method.observe(m["pictures"]("blue"), m["stamps"](2, 2))
method.finalize_chunk()
answer = method.answer("What color was the scene?", ["red", "blue"])
Path(target).write_text(json.dumps({"answer": answer["answer"], "memory": method.memory_bytes(),
                                  "frames": len(method.state_dict().get("timestamps", []))}))
'''
    target = tmp_path / "continued.json"
    subprocess.run([sys.executable, "-c", program, kind, str(checkpoint), str(target)],
                   check=True, cwd=Path(__file__).resolve().parents[3], capture_output=True, text=True)
    method.observe(pictures("blue"), stamps(2, 2))
    method.finalize_chunk()
    import json
    resumed = json.loads(target.read_text())
    assert resumed["answer"] == method.answer("What color was the scene?", ["red", "blue"])["answer"]
    assert resumed["memory"] == method.memory_bytes()
    if kind == "tree":
        assert resumed["frames"] == 4


def test_egorag_failed_qa_restores_sqlite_without_cached_partial_answers():
    from baselines.streambench.adapters.egorag import EgoRAGMethod
    method = EgoRAGMethod(4_000_000, ego_config())
    method.initialize()
    method.observe(pictures(), stamps(1, 0))
    method.finalize_chunk()
    before = method.state_dict()
    generate = method._pipe.models.generate
    def failing(role, *args, **kwargs):
        if role == "answerer":
            raise RuntimeError("intentional model failure")
        return generate(role, *args, **kwargs)
    method._pipe.models.generate = failing
    with pytest.raises(RuntimeError, match="intentional"):
        method.answer("What color was the scene?", ["red", "blue"])
    assert method.state_dict() == before


def api_language_config(monkeypatch):
    monkeypatch.setenv("EGORAG_API_KEY", "constructor-test-no-request")
    monkeypatch.setenv("EGORAG_BASE_URL", "https://example.invalid/v1")
    return {"models": {role: {"backend": "openai_compatible", "model": "constructor-test-model"}
                       for role in ("captioner", "relevance", "answerer")}}


def artifact_config(tmp_path, monkeypatch):
    import json
    encoder, processor = tmp_path / "encoder", tmp_path / "processor"
    encoder.mkdir(exist_ok=True)
    processor.mkdir(exist_ok=True)
    (encoder / "config.json").write_text(json.dumps({"model_type": "clip"}))
    for name in ("modeling_evaclip.py", "configuration_evaclip.py"):
        (encoder / name).write_text("# Constructor integrity test; never imported\n")
    (encoder / "model.safetensors").write_bytes(b"structural-only-weight-placeholder")
    (processor / "preprocessor_config.json").write_text('{"size": 224}')
    return {"memory_budget_bytes": 4_000_000,
            "encoder": {"checkpoint": str(encoder), "processor_checkpoint": str(processor),
                        "revision": "a" * 40, "processor_revision": "b" * 40, "device": "cpu"},
            "language_config": api_language_config(monkeypatch)}


def test_real_language_constructor_maps_relevance_to_supported_native_role(monkeypatch):
    from baselines.streambench.adapters.videotree import VideoTreeLanguageBackend
    language = VideoTreeLanguageBackend(api_language_config(monkeypatch))
    assert set(language.models.roles) == {"captioner", "evidence", "answerer"}
    dispatched = []
    def generate(role, *args, **kwargs):
        assert role in language.models.roles
        dispatched.append(role)
        return "[3]"
    monkeypatch.setattr(language.models, "generate", generate)
    assert language.score("question", [], ["caption"]) == [3]
    assert dispatched == ["evidence"]


@pytest.mark.parametrize("changed", ["weights", "processor", "endpoint", "inventory"])
def test_videotree_checkpoint_rejects_changed_real_backend_artifacts(changed, tmp_path, monkeypatch):
    from baselines.streambench.adapters.videotree import create_method
    config = artifact_config(tmp_path, monkeypatch)
    method = create_method(config)
    method.initialize()
    state = method.snapshot()
    if changed == "weights":
        (tmp_path / "encoder/model.safetensors").write_bytes(b"different weights")
    elif changed == "processor":
        (tmp_path / "processor/preprocessor_config.json").write_text('{"size": 336}')
    elif changed == "inventory":
        (tmp_path / "encoder/added.safetensors").write_bytes(b"new loader input")
    else:
        monkeypatch.setenv("EGORAG_BASE_URL", "https://different.invalid/v1")
    resumed = create_method(config)
    with pytest.raises(ValueError, match="checkpoint.*mismatch"):
        resumed.restore(state)
    if changed != "endpoint":
        with pytest.raises(ValueError, match="changed"):
            method.restore(state)


def test_videotree_identity_covers_local_language_checkpoint(tmp_path, monkeypatch):
    from baselines.streambench.adapters.videotree import create_method
    config = artifact_config(tmp_path, monkeypatch)
    local = tmp_path / "language"
    local.mkdir()
    (local / "config.json").write_text('{"model_type":"qwen3_vl"}')
    (local / "model.safetensors").write_bytes(b"constructor-only-never-loaded")
    (local / "tokenizer.json").write_text('{}')
    config["language_config"] = {"models": {role: {
        "backend": "hf_local", "model": str(local), "revision": "c" * 40, "device": "cuda:0"}
        for role in ("captioner", "relevance", "answerer")}}
    method = create_method(config)
    method.initialize()
    state = method.snapshot()
    (local / "model.safetensors").write_bytes(b"changed local language weights")
    with pytest.raises(ValueError, match="checkpoint.*mismatch"):
        create_method(config).restore(state)


def test_videotree_empty_archive_is_explicit_abstention():
    method = tree()
    method.initialize()
    result = method.answer("What happened?", ["A", "B"])
    assert result["status"] == "no_observed_evidence"
    assert result["answer"] == ""


def test_egorag_native_range_miss_is_explicit_abstention(monkeypatch):
    from baselines.streambench.adapters.egorag import EgoRAGMethod
    method = EgoRAGMethod(4_000_000, ego_config())
    method.initialize()
    method.observe(pictures(), stamps(1, 0))
    method.finalize_chunk()
    before = method.state_dict()
    original = method._pipe.models.generate
    def generate(role, system, prompt, **kwargs):
        if role == "range_selector" and "HHMMSSFF" in system:
            return "[00000100]\n[00000200]"
        return original(role, system, prompt, **kwargs)
    monkeypatch.setattr(method._pipe.models, "generate", generate)
    result = method.answer("What color?", ["red", "blue"])
    assert result["status"] == "no_observed_evidence"
    assert result["num_retrieved_items"] == 0
    assert result["answer"] == ""
    assert method.state_dict() == before
