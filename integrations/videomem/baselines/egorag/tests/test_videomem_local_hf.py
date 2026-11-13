"""Local-HF contracts using fake modules: no CUDA, network, or model weights."""

import base64
from contextlib import nullcontext
import hashlib
import io
import json
import subprocess
import sys
from types import SimpleNamespace

from PIL import Image
import pytest

from videomem_adapter.backends import BackendConfigurationError, ModelExecutionError, Models
from videomem_adapter.local_hf import LocalHF


REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
UUID4 = "GPU-00000000-0000-0000-0000-000000000004"
UUID5 = "GPU-00000000-0000-0000-0000-000000000005"
MASK = UUID4 + "," + UUID5


def local_config(tmp_path, **extra):
    return {"models": {"captioner": {
        "backend": "hf_local", "model": str(tmp_path), "revision": REVISION,
        "device": "cuda:0", "dtype": "bfloat16", "attn_implementation": "sdpa",
        "max_tokens": 4, "temperature": 0, "min_pixels": 65536, "max_pixels": 262144,
        **extra,
    }}}


def frame():
    stream = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(stream, format="JPEG")
    return {"jpeg_base64": base64.b64encode(stream.getvalue()).decode(),
            "day_id": "Day1", "clip_id": "clip1", "file_seconds": 2.5, "day_seconds": 32.5}


@pytest.fixture
def fake_runtime(monkeypatch):
    state = SimpleNamespace(processor_loads=[], model_loads=[], templates=[], moves=[], generations=[],
                            continuation=[10, 2], decoded="local caption", decoded_ids=[], error=None)

    class Inputs(dict):
        def to(self, device):
            state.moves.append(device)
            return self

    class Processor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            state.processor_loads.append((path, kwargs))
            return cls()

        def apply_chat_template(self, messages, **kwargs):
            captured = []
            for message in messages:
                parts = []
                for part in message["content"]:
                    if part["type"] == "image":
                        assert isinstance(part["image"], Image.Image)
                        parts.append({"type": "image", "mode": part["image"].mode,
                                      "size": part["image"].size, "pixel": part["image"].getpixel((0, 0))})
                    else:
                        parts.append(dict(part))
                captured.append({"role": message["role"], "content": parts})
            state.templates.append((captured, kwargs))
            return Inputs(input_ids=[[101, 102, 103]], attention_mask=[[1, 1, 1]])

        def batch_decode(self, ids, **kwargs):
            state.decoded_ids.append(ids)
            assert kwargs == {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
            return [state.decoded]

    class Model:
        generation_config = SimpleNamespace(eos_token_id=[2, 3])

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == MASK
            state.model_loads.append((path, kwargs))
            return cls()

        def eval(self):
            return self

        def generate(self, **kwargs):
            state.generations.append(kwargs)
            if state.error:
                raise state.error
            return [[101, 102, 103] + state.continuation]

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        __version__="2.8.0-mock", bfloat16="mock-bfloat16", float16="mock-float16",
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 2,
                             get_device_properties=lambda index: SimpleNamespace(uuid=(UUID4, UUID5)[index])),
        inference_mode=nullcontext,
    ))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        __version__="4.57.3-mock", AutoProcessor=Processor, Qwen3VLForConditionalGeneration=Model,
    ))
    monkeypatch.setattr("videomem_adapter.local_hf.subprocess.run", lambda *args, **kwargs:
                        SimpleNamespace(stdout="4, " + UUID4 + "\n5, " + UUID5 + "\n"))
    monkeypatch.setattr(Models, "_http", lambda *args, **kwargs: pytest.fail("local backend called HTTP"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", MASK)
    return state


def test_configuration_is_lazy_and_needs_no_api_credentials(fake_runtime, monkeypatch, tmp_path):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    monkeypatch.delenv("EGORAG_API_KEY", raising=False)
    monkeypatch.delenv("EGORAG_BASE_URL", raising=False)
    monkeypatch.setattr("videomem_adapter.local_hf.subprocess.run", lambda *a, **k: pytest.fail("configuration queried GPUs"))
    Models(local_config(tmp_path), tmp_path / "calls.jsonl")
    assert not fake_runtime.model_loads and not fake_runtime.processor_loads
    assert not (tmp_path / "calls.jsonl").exists()


@pytest.mark.parametrize("extra", [
    {"model": "Qwen/Qwen3-VL-8B-Instruct"}, {"revision": "main"}, {"device": "cuda:4"},
    {"device": "cpu"}, {"dtype": "auto"}, {"attn_implementation": "flash_attention_2"},
    {"max_tokens": True}, {"max_tokens": 0}, {"temperature": -1}, {"temperature": float("nan")},
    {"min_pixels": 0}, {"max_pixels": 1},
])
def test_invalid_configuration_is_rejected_without_load(tmp_path, extra):
    with pytest.raises(BackendConfigurationError):
        Models(local_config(tmp_path, **extra), tmp_path / "calls.jsonl")


def test_local_images_timestamps_cpu_processing_and_trace(fake_runtime, tmp_path):
    trace = tmp_path / "calls.jsonl"
    models = Models(local_config(tmp_path), trace)
    models.protocol_context = {"namespace": "local-test"}
    image = frame()
    assert models.generate("captioner", "exact system", "exact prompt", [image]) == "local caption"
    messages, kwargs = fake_runtime.templates[0]
    assert messages[0] == {"role": "system", "content": [{"type": "text", "text": "exact system"}]}
    assert messages[1]["content"][0] == {"type": "text", "text": "exact prompt"}
    assert messages[1]["content"][1] == {"type": "text", "text":
        'Frame metadata: {"clip_id": "clip1", "day_id": "Day1", "day_seconds": 32.5, "file_seconds": 2.5, "position": 0}'}
    assert messages[1]["content"][2] == {"type": "image", "mode": "RGB", "size": (8, 8), "pixel": (254, 0, 0)}
    assert kwargs == {"tokenize": True, "add_generation_prompt": True, "return_dict": True,
                      "return_tensors": "pt", "images_kwargs": {"device": "cpu"}}
    assert fake_runtime.moves == ["cuda:0"]
    assert fake_runtime.decoded_ids == [[[10, 2]]]
    load = fake_runtime.model_loads[0][1]
    assert load == {"dtype": "mock-bfloat16", "device_map": {"": "cuda:0"}, "attn_implementation": "sdpa",
                    "revision": REVISION, "local_files_only": True, "trust_remote_code": False}
    assert fake_runtime.generations[0]["max_new_tokens"] == 4
    assert fake_runtime.generations[0]["do_sample"] is False
    record = json.loads(trace.read_text())
    assert record["protocol_context"] == {"namespace": "local-test"}
    assert record["backend"] == "hf_local" and record["revision"] == REVISION
    assert record["decoded_frame_count"] == 1 and record["image_decode_device"] == "cpu"
    assert record["local_runtime"]["physical_gpu_index"] == 4
    assert record["local_runtime"]["physical_gpu_uuid"] == UUID4
    assert record["local_runtime"]["image_preprocess_device"] == "cpu"
    assert record["status"] == "ok" and record["test_only"] is False
    assert models.token_usage == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    assert image["jpeg_base64"] not in trace.read_text() and "endpoint" not in record


def test_two_lazy_models_are_shared_across_roles(fake_runtime, tmp_path):
    config = local_config(tmp_path)
    settings = config["models"]["captioner"]
    config["models"].update({"evidence": dict(settings),
        "summary_day": {**settings, "device": "cuda:1", "max_tokens": 8},
        "answerer": {**settings, "device": "cuda:1", "temperature": 0.1}})
    models = Models(config, tmp_path / "calls.jsonl")
    for role in ("captioner", "evidence", "summary_day", "answerer"):
        models.generate(role, "system", "prompt")
    assert len(fake_runtime.model_loads) == len(fake_runtime.processor_loads) == 2
    assert [call[1]["device_map"] for call in fake_runtime.model_loads] == [{"": "cuda:0"}, {"": "cuda:1"}]
    assert fake_runtime.generations[-1]["do_sample"] is True
    assert fake_runtime.generations[-1]["temperature"] == 0.1
    assert fake_runtime.templates[2][0][1]["content"] == [{"type": "text", "text": "prompt"}]


@pytest.mark.parametrize("continuation, fails", [([10, 11, 12, 13], True), ([10, 11, 12, 2], False)])
def test_token_limit_rejects_truncation_but_allows_eos(fake_runtime, tmp_path, continuation, fails):
    fake_runtime.continuation = continuation
    models = Models(local_config(tmp_path), tmp_path / "calls.jsonl")
    if fails:
        with pytest.raises(ModelExecutionError, match="truncated"):
            models.generate("captioner", "system", "prompt")
    else:
        assert models.generate("captioner", "system", "prompt") == "local caption"
    record = json.loads((tmp_path / "calls.jsonl").read_text())
    assert record["finish_reason"] == ("length" if fails else "stop")
    assert record["usage"]["completion_tokens"] == 4
    assert ("response" in record) is not fails


def test_empty_output_and_decode_failure_never_fall_back(fake_runtime, tmp_path):
    models = Models(local_config(tmp_path), tmp_path / "calls.jsonl")
    invalid = {"jpeg_base64": base64.b64encode(b"not an image").decode()}
    with pytest.raises(ModelExecutionError, match="decode actual image pixels"):
        models.generate("captioner", "system", "prompt", [invalid])
    assert fake_runtime.model_loads == []
    fake_runtime.decoded = ""
    with pytest.raises(ModelExecutionError, match="no text"):
        models.generate("captioner", "system", "prompt")


def test_unexpected_model_error_does_not_log_exception_payload(fake_runtime, tmp_path):
    fake_runtime.error = RuntimeError("potentially private image or credential payload")
    models = Models(local_config(tmp_path), tmp_path / "calls.jsonl")
    with pytest.raises(ModelExecutionError, match="Model execution failed"):
        models.generate("captioner", "system", "prompt")
    assert "potentially private" not in (tmp_path / "calls.jsonl").read_text()


@pytest.mark.parametrize("mask", ["", "0", "0,1", "5,4", "4", "4,5,6", UUID5 + "," + UUID4])
def test_disallowed_gpu_masks_never_load(fake_runtime, monkeypatch, tmp_path, mask):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    models = Models(local_config(tmp_path), tmp_path / "calls.jsonl")
    with pytest.raises(ModelExecutionError, match="CUDA"):
        models.generate("captioner", "system", "prompt")
    assert fake_runtime.model_loads == []


def test_numeric_mask_resolves_before_torch_import(fake_runtime, monkeypatch):
    monkeypatch.delitem(sys.modules, "torch")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    local = LocalHF()
    assert local._guard_devices() == (MASK, {"4": UUID4, "5": UUID5})
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == MASK
    assert __import__("os").environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert local._requested_mask == "4,5"
    assert "torch" not in sys.modules


def test_numeric_mask_after_torch_import_fails(fake_runtime, monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    with pytest.raises(ModelExecutionError, match="torch was imported"):
        Models(local_config(tmp_path), tmp_path / "calls.jsonl").generate("captioner", "system", "prompt")
    assert not fake_runtime.model_loads


def test_cuda_runtime_with_stale_visibility_mask_never_loads(fake_runtime, monkeypatch, tmp_path):
    monkeypatch.setattr(sys.modules["torch"].cuda, "get_device_properties",
                        lambda index: SimpleNamespace(uuid="GPU-00000000-0000-0000-0000-000000000000"))
    with pytest.raises(ModelExecutionError, match="runtime devices do not match"):
        Models(local_config(tmp_path), tmp_path / "calls.jsonl").generate("captioner", "system", "prompt")
    assert not fake_runtime.model_loads


@pytest.mark.parametrize("output", ["", "4, " + UUID4, "4, " + UUID4 + "\n4, " + UUID5,
                                     "4, " + UUID4 + "\n5, " + UUID4, "4, invalid\n5, " + UUID5])
def test_unverified_gpu_identity_never_loads(fake_runtime, monkeypatch, tmp_path, output):
    monkeypatch.setattr("videomem_adapter.local_hf.subprocess.run", lambda *a, **k: SimpleNamespace(stdout=output))
    with pytest.raises(ModelExecutionError, match="Cannot verify"):
        Models(local_config(tmp_path), tmp_path / "calls.jsonl").generate("captioner", "system", "prompt")
    assert not fake_runtime.model_loads


def test_smi_failure_never_loads(fake_runtime, monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "nvidia-smi")
    monkeypatch.setattr("videomem_adapter.local_hf.subprocess.run", fail)
    with pytest.raises(ModelExecutionError, match="Cannot verify"):
        Models(local_config(tmp_path), tmp_path / "calls.jsonl").generate("captioner", "system", "prompt")
    assert not fake_runtime.model_loads


def verified_local_config(tmp_path, weight=b"original weight bytes"):
    directory = tmp_path / "checkpoint"
    directory.mkdir(exist_ok=True)
    (directory / "config.json").write_text('{"model_type":"qwen3_vl"}')
    (directory / "model.safetensors.index.json").write_text(
        '{"weight_map":{"test.weight":"model-00001-of-00001.safetensors"}}')
    (directory / "model-00001-of-00001.safetensors").write_bytes(weight)
    files = [{"path": path.name, "bytes": path.stat().st_size,
              "hash": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(directory.iterdir())]
    manifest = tmp_path / "verification.json"
    manifest.write_text(json.dumps({"repo_id": "Qwen/Qwen3-VL-8B-Instruct", "revision": REVISION,
        "verified_files": files, "file_count": len(files), "total_bytes": sum(f["bytes"] for f in files)}))
    return local_config(directory, verified_manifest=str(manifest))


def test_actual_checkpoint_identity_is_traced(fake_runtime, tmp_path):
    config = verified_local_config(tmp_path)
    config["models"]["evidence"] = dict(config["models"]["captioner"])
    models = Models(config, tmp_path / "calls.jsonl")
    assert models.cache_identity["captioner"] == models.cache_identity["evidence"]
    models.generate("captioner", "system", "prompt")
    record = json.loads((tmp_path / "calls.jsonl").read_text())
    identity = record["checkpoint_identity"]
    assert identity["verification"] == "verified"
    assert len(identity["content_sha256"]) == 64
    assert identity["content_sha256"] == models.cache_identity["captioner"]["content_sha256"]


def test_pipeline_checks_changed_weights_before_cache_and_uses_actual_identity(fake_runtime, tmp_path):
    from videomem_adapter.pipeline import Pipeline

    config = verified_local_config(tmp_path)
    config.update(test_only=True, index_engine="sqlite", embedding={"backend": "fixture"})
    manifest = {"clips": [], "videos": []}
    first = Pipeline(config, manifest, tmp_path / "run")
    assert first.call("captioner", "system", "same prompt") == "local caption"
    assert first.call("captioner", "system", "same prompt") == "local caption"
    assert first.models.calls == 1 and first.hits == 1
    assert first.provenance["local_checkpoint_identity"] == first.models.cache_identity

    verified_local_config(tmp_path, b"different weight bytes")
    with pytest.raises(BackendConfigurationError):
        first.call("captioner", "system", "same prompt")
    with pytest.raises(BackendConfigurationError):
        first.create(limit=0)
    second = Pipeline(config, manifest, tmp_path / "run")
    assert second.namespace != first.namespace
    second.call("captioner", "system", "same prompt")
    assert second.models.calls == 1 and second.hits == 0


def test_pinned_embedding_is_verified_before_pipeline_caches(tmp_path):
    checkpoint = tmp_path / "clip.safetensors"
    checkpoint.write_bytes(b"tiny fake checkpoint")
    config = {"test_only": True, "models": {"captioner": {"backend": "fixture"}},
        "embedding": {"backend": "openclip", "checkpoint": str(checkpoint), "revision": REVISION,
                      "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}}
    models = Models(config, tmp_path / "calls.jsonl")
    assert models.cache_identity["embedding"]["verification"] == "verified"
    checkpoint.write_bytes(b"replacement fake checkpoint")
    with pytest.raises(BackendConfigurationError):
        models.assert_checkpoints_unchanged()
