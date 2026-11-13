"""CPU checks for runner observations and the public model-input boundary."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_videomem import data, run


class ObservedMemory:
    vision_start_token_id = 3
    vision_end_token_id = 4

    def __init__(self, kept=(0, 1, 3, 4, 5)):
        self.kept = kept
        self.temporal_repetitions = 1
        self.spatial_repetitions = 2

    def _apply_adjacent_pruning(self):
        pass

    def _long_term_memory_merge_per_frames(self):
        pass

    def process_memory_streaming(self, *, hidden_states, input_ids):
        for _ in range(self.temporal_repetitions):
            self._apply_adjacent_pruning()
        for _ in range(self.spatial_repetitions):
            self._long_term_memory_merge_per_frames()
        indices = torch.tensor(self.kept)
        self.last_result = (hidden_states[:, indices], None, None, None, [indices])
        return self.last_result


def memory_inputs():
    return {"input_ids": torch.tensor([[1, 3, 6, 6, 4, 9]]),
            "hidden_states": torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)}


def test_instrumentation_resets_each_call_and_returns_original_result():
    records = []
    memory = ObservedMemory()
    run.instrument(memory, records)
    inputs = memory_inputs()
    first = memory.process_memory_streaming(**inputs)
    assert first is memory.last_result
    memory.temporal_repetitions, memory.spatial_repetitions = 2, 1
    second = memory.process_memory_streaming(**inputs)
    assert second is memory.last_result
    assert [(row["temporal_calls"], row["spatial_calls"]) for row in records] == [(1, 2), (2, 1)]
    for row in records:
        assert (row["input_tokens"], row["retained_tokens"], row["dropped_tokens"]) == (6, 5, 1)
        assert (row["visual_tokens_before"], row["visual_tokens_after"]) == (2, 1)
    assert torch.equal(second[0], inputs["hidden_states"][:, [0, 1, 3, 4, 5]])
    assert not torch.cuda.is_initialized()


def test_instrumentation_rejects_dropped_nonvisual_prompt_token():
    records = []
    memory = ObservedMemory(kept=(1, 3, 4, 5))
    run.instrument(memory, records)
    with pytest.raises(RuntimeError, match="removed nonvisual prompt tokens"):
        memory.process_memory_streaming(**memory_inputs())
    assert records == []
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("prediction", ["C", "   "], ids=["public_wire", "empty_output_rejected"])
def test_runner_public_wire_and_empty_output_on_cpu(tmp_path, monkeypatch, prediction):
    """Exercise main with an explicit CPU double; no GPU or model files are read."""
    import qwen2_5_vl_fluxmem
    import qwen_vl_utils_fluxmem

    question = {
        "question_id": "public-question", "question": "What is visible?",
        "choices": ["First option", "Second option", "Third option"],
        "query_time": {"day_id": "Day3", "day_order": 3, "seconds": 30,
                       "answer": "NESTED_ANSWER_SECRET"},
        "answer": "ANSWER_SECRET", "gt_answer": "GT_SECRET",
        "mapped_inputs": [{"path": "/private/MAPPED_INPUT_SECRET.json"}],
    }
    manifest_path, questions_path = tmp_path / "manifest.json", tmp_path / "questions.json"
    manifest_path.write_text(json.dumps({"private_annotation": "MANIFEST_SECRET"}))
    questions_path.write_text(json.dumps([question]))
    output = tmp_path / "result"
    frames = [Image.new("RGB", (28, 28), color="blue"), Image.new("RGB", (28, 28), color="green")]
    captured = {}

    def decode_prefix(manifest, public, **kwargs):
        assert set(public) == {"question_id", "question", "choices", "query_time"}
        assert set(public["query_time"]) == {"day_id", "day_order", "seconds"}
        return {
            "frames": frames,
            "segments": [{"source_id": "public-source", "day_id": "Day3", "day_order": 3,
                          "file_start_seconds": 0, "file_end_seconds": 30,
                          "day_start_seconds": 0, "day_end_seconds": 30,
                          "path": "/private/ACTIVITY_LABEL_SECRET.mp4", "annotation": "ANNOTATION_SECRET"}],
            "frame_metadata": [{"day_id": "Day3", "file_seconds": index, "day_seconds": index,
                                "path": "/private/FRAME_LABEL_SECRET.jpg"} for index in (0, 1)],
        }

    class CpuBatch(dict):
        def __getattr__(self, name):
            return self[name]

        def to(self, destination):
            captured["requested_device"] = destination
            assert all(value.device.type == "cpu" for value in self.values())
            return self

    class Processor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            assert kwargs["local_files_only"]
            return cls()

        def apply_chat_template(self, messages, **kwargs):
            captured["messages"] = messages
            assert messages[1]["content"][1]["type"] == "video"
            assert messages[1]["content"][1]["video"] is frames
            return "rendered public prompt"

        def __call__(self, **kwargs):
            assert kwargs["videos"] == [frames]
            assert kwargs["fps"] == [1.0]
            return CpuBatch(input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.ones(1, 3, dtype=torch.long))

        def decode(self, *args, **kwargs):
            return prediction

    class Model:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            assert kwargs["local_files_only"] and kwargs["attn_implementation"] == "sdpa"
            result = cls()
            result.model = type("Decoder", (), {"fluxmem": ObservedMemory()})()
            result.generation_config = type("Generation", (), {"eos_token_id": 2})()
            return result

        def eval(self):
            return self

        def requires_grad_(self, value):
            assert value is False

        def generate(self, **kwargs):
            assert kwargs["use_fluxmem"] is False
            assert kwargs["input_ids"].device.type == "cpu"
            return torch.tensor([[1, 2, 3, 7, 2]])

    def vision_info(messages, **kwargs):
        assert messages is captured["messages"]
        return None, [frames], {"fps": [1.0]}

    monkeypatch.setattr(run, "bind_gpu", lambda index: {"physical_index": index, "uuid": "CPU_TEST_DOUBLE", "name": "CPU test"})
    monkeypatch.setattr(run, "verify_weights", lambda *args: {"revision": "CPU_TEST_NO_CHECKPOINT"})
    monkeypatch.setattr(data, "decode_prefix", decode_prefix)
    monkeypatch.setattr(qwen2_5_vl_fluxmem, "Qwen2_5_VLForConditionalGeneration", Model)
    monkeypatch.setattr(qwen2_5_vl_fluxmem, "Qwen2_5_VLProcessor", Processor)
    monkeypatch.setattr(qwen_vl_utils_fluxmem, "process_vision_info", vision_info)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(sys, "argv", ["run.py", "--manifest", str(manifest_path), "--questions", str(questions_path),
                                     "--output", str(output), "--fluxmem", "off"])
    if prediction.strip():
        run.main()
    else:
        with pytest.raises(SystemExit) as error:
            run.main()
        assert error.value.code == 1
    prompt = captured["messages"][1]["content"][0]["text"]
    assert "What is visible?" in prompt and "C. Third option" in prompt
    assert "SECRET" not in prompt
    rows = json.loads((output / "predictions.json").read_text())
    assert "SECRET" not in json.dumps(rows[0]["question"])
    assert rows[0]["status"] == ("complete" if prediction.strip() else "execution_error")
    if not prediction.strip():
        assert "empty answer" in rows[0]["error"]
    assert not torch.cuda.is_initialized()
