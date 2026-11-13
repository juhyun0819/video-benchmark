"""AKS structural tests; model doubles do not represent accuracy evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch


def test_aks_selector_module_exists():
    assert importlib.util.find_spec("baselines.streambench.adapters.aks") is not None


@pytest.mark.parametrize("scores,n,depth,t1,want", [
    ([0., 1., 0., 0., 0., 0., 0., 0.], 2, 1, .2, [0, 1]),
    ([0., 1., 0., 0., 0., 0., 0., 0.], 2, 1, .8, [1, 4]),
    ([1.] * 8, 4, 1, .8, [0, 1, 4, 5]),
    ([1.] * 8, 3, 1, .8, [0, 4]),
    ([.2, .8], 4, 1, .8, [0, 1]),
])
def test_native_aks_relevance_coverage_and_depth_quotas(scores, n, depth, t1, want):
    from baselines.streambench.adapters.aks import adaptive_select
    assert adaptive_select(scores, max_num_frames=n, all_depth=depth, t1=t1) == want


def test_aks_rejects_nonfinite_scores_and_zero_depth_quotas():
    from baselines.streambench.adapters.aks import adaptive_select
    with pytest.raises(ValueError, match="finite"):
        adaptive_select([0., float("nan")])
    with pytest.raises(ValueError, match="depth"):
        adaptive_select([0.] * 16, max_num_frames=2, all_depth=5)


class Scorer:
    def encode_images(self, frames):
        return torch.tensor(np.array([np.array(frame).mean((0, 1)) for frame in frames]), dtype=torch.float32)

    def score(self, features, question):
        text = torch.tensor([1., 0., 0.]) if "red" in question else torch.tensor([0., 0., 1.])
        return torch.nn.functional.cosine_similarity(features, text[None], dim=-1).tolist()


class Answerer:
    def answer(self, frames, question, choices, timestamps):
        return "A" if np.array(frames[0])[0, 0, 0] else "B"


def method(budget=4_000_000):
    from baselines.streambench.adapters.aks import AKSMethod
    return AKSMethod(budget, scorer=Scorer(), answerer=Answerer(), test_only=True,
                      max_num_frames=2, all_depth=1)


def stamps(day, start, count=2):
    return [{"stream_id": "stream", "day": day, "global_frame_id": start + i,
             "global_timestamp": float(start + i), "source_timestamp": float(i),
             "source_id": f"day{day}", "source_frame_index": i} for i in range(count)]


def pictures():
    return [Image.new("RGB", (16, 16), color) for color in ("red", "blue")]


def test_aks_observes_without_questions_preserves_pixels_and_isolates_qa():
    item = method()
    item.initialize()
    item.observe(pictures(), stamps(1, 0))
    item.finalize_chunk()
    before = copy.deepcopy(item.state_dict())
    result = item.answer("Find red", ["red", "blue"])
    assert result["answer"] == "A"
    assert result["query_read_bytes"] > 2 * 3 * 4
    assert result["num_retrieved_items"] == 2
    assert result["retrieved_frame_ids"] == [0, 1]
    item.answer("Find blue", ["red", "blue"])
    assert torch.equal(item.state_dict()["features"], before["features"])
    assert item.state_dict()["frames_png"] == before["frames_png"]
    assert item.memory_bytes()["visual_memory_bytes"] >= 2 * 3 * 4 + sum(map(len, before["frames_png"]))


def test_aks_rejects_over_budget_without_eviction_or_hidden_original_video():
    from baselines.streambench.method import UnsupportedBudget
    item = method()
    item.initialize()
    item.observe(pictures(), stamps(1, 0))
    item.memory_budget_bytes = item.memory_bytes()["persistent_total_bytes"] - 1
    with pytest.raises(UnsupportedBudget):
        item.finalize_chunk()
    assert len(item.state_dict()["frames_png"]) == 2


def test_aks_fresh_process_resumption_matches_continuous(tmp_path):
    item = method()
    item.initialize()
    item.observe(pictures(), stamps(1, 0))
    item.finalize_chunk()
    checkpoint = tmp_path / "day1"
    item.save_checkpoint(checkpoint)
    code = '''
import runpy, sys, json
from pathlib import Path
m=runpy.run_path("baselines/streambench/tests/test_streambench_aks.py")
item=m["method"]()
item.load_checkpoint(sys.argv[1])
item.observe(m["pictures"](),m["stamps"](2,2))
item.finalize_chunk()
Path(sys.argv[2]).write_text(json.dumps({"result":item.answer("Find red",["red","blue"]),"memory":item.memory_bytes()}))
'''
    target = tmp_path / "result.json"
    subprocess.run([sys.executable, "-c", code, str(checkpoint), str(target)], check=True,
                   cwd=Path(__file__).resolve().parents[3], capture_output=True, text=True)
    item.observe(pictures(), stamps(2, 2))
    item.finalize_chunk()
    reloaded = json.loads(target.read_text())
    assert reloaded["result"] == item.answer("Find red", ["red", "blue"])
    assert reloaded["memory"] == item.memory_bytes()


def test_aks_factory_requires_real_explicit_local_checkpoints():
    from baselines.streambench.adapters.aks import create_method
    with pytest.raises(ValueError, match="scorer|checkpoint"):
        create_method({"method": "aks", "memory_budget_bytes": 4_000_000})


def test_aks_checkpoint_rejects_changed_selector_config():
    item = method()
    item.initialize()
    item.observe(pictures(), stamps(1, 0))
    state = item.state_dict()
    state["parameters"]["t1"] = 0.7
    with pytest.raises(ValueError, match="config"):
        item.load_state_dict(state)
