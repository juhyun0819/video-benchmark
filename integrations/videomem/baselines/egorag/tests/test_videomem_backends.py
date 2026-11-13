"""Backend contracts with an in-process HTTP transport stub; no network or downloads."""

import base64
from contextlib import contextmanager
import hashlib
import io
import json
import math
import sys
from types import SimpleNamespace
from urllib import error, parse

import pytest
from PIL import Image

from videomem_adapter.backends import (
    BackendConfigurationError, Embeddings, ModelExecutionError, Models, ROLES,
)


def image_frame(color):
    output = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(output, format="JPEG")
    raw = output.getvalue()
    return {
        "jpeg_base64": base64.b64encode(raw).decode("ascii"),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "file_seconds": 2.5, "day_seconds": 3602.5,
    }


def fixture_config():
    return {"test_only": True, "models": {role: {"backend": "fixture"} for role in ROLES}}


def production_config(**extra):
    return {"models": {"captioner": {
        "backend": "openai_compatible", "model": "explicit-user-model-revision",
        "api_key_env": "TEST_EGORAG_KEY", "base_url_env": "TEST_EGORAG_URL",
        "temperature": 0.25, "max_tokens": 13, "timeout": 2, **extra,
    }}}


@contextmanager
def completion_stub(monkeypatch, content="caption from configured server", finish_reason="stop", status=200):
    captured = []

    def urlopen(req, timeout):
        assert timeout == 2
        captured.append({
            "path": parse.urlsplit(req.full_url).path,
            "authorization": req.get_header("Authorization"),
            "payload": json.loads(req.data),
        })
        body = json.dumps({
            "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 17, "completion_tokens": 4, "total_tokens": 21},
            "error": "provider may echo sk-private-test-secret",
        }).encode("utf-8")
        if status != 200:
            raise error.HTTPError(req.full_url, status, "provider error", {}, io.BytesIO(body))
        return io.BytesIO(body)

    with monkeypatch.context() as context:
        context.setattr("videomem_adapter.backends.request.urlopen", urlopen)
        yield "http://127.0.0.1:12345/v1", captured


@pytest.mark.parametrize("factory", [
    lambda: Models({"models": {"captioner": {"backend": "fixture"}}}, "unused.jsonl"),
    lambda: Embeddings({"backend": "fixture"}),
])
def test_fixture_requires_explicit_test_only(factory):
    with pytest.raises(BackendConfigurationError, match="test_only"):
        factory()


def test_production_requires_model_and_credentials(monkeypatch, tmp_path):
    monkeypatch.delenv("TEST_EGORAG_KEY", raising=False)
    with pytest.raises(BackendConfigurationError, match="explicit model"):
        Models(production_config(model=None), tmp_path / "calls.jsonl")
    with pytest.raises(BackendConfigurationError, match="TEST_EGORAG_KEY"):
        Models(production_config(), tmp_path / "calls.jsonl")
    monkeypatch.setenv("TEST_EGORAG_KEY", "private-secret")
    monkeypatch.delenv("TEST_EGORAG_URL", raising=False)
    with pytest.raises(BackendConfigurationError, match="TEST_EGORAG_URL"):
        Models(production_config(), tmp_path / "calls.jsonl")
    assert not (tmp_path / "calls.jsonl").exists()


def test_http_wire_payload_and_redacted_exact_trace(monkeypatch, tmp_path):
    frame = image_frame("red")
    trace = tmp_path / "calls.jsonl"
    with completion_stub(monkeypatch) as (url, captured):
        monkeypatch.setenv("TEST_EGORAG_URL", url)
        monkeypatch.setenv("TEST_EGORAG_KEY", "sk-private-test-secret")
        models = Models(production_config(), trace)
        answer = models.generate("captioner", "Exact system\nline", "Exact prompt\nline", [frame])
    assert answer == "caption from configured server"
    assert captured[0]["path"] == "/v1/chat/completions"
    assert captured[0]["authorization"] == "Bearer sk-private-test-secret"
    payload = captured[0]["payload"]
    assert payload["model"] == "explicit-user-model-revision"
    assert payload["temperature"] == 0.25 and payload["max_tokens"] == 13
    assert payload["messages"][0] == {"role": "system", "content": "Exact system\nline"}
    assert payload["messages"][1]["content"][0] == {"type": "text", "text": "Exact prompt\nline"}
    assert payload["messages"][1]["content"][1] == {
        "type": "text", "text": 'Frame metadata: {"day_seconds": 3602.5, "file_seconds": 2.5, "position": 0}',
    }
    assert payload["messages"][1]["content"][2]["image_url"]["url"].endswith(frame["jpeg_base64"])
    log = trace.read_text()
    assert frame["jpeg_base64"] not in log and "sk-private-test-secret" not in log
    record = json.loads(log)
    assert record["system"] == "Exact system\nline" and record["prompt"] == "Exact prompt\nline"
    assert record["images"][0]["sha256"] == frame["sha256"]
    assert record["images"][0]["day_seconds"] == 3602.5
    assert record["elapsed_seconds"] >= 0 and record["status"] == "ok"
    assert models.calls == 1
    assert models.token_usage == {"prompt_tokens": 17, "completion_tokens": 4, "total_tokens": 21}


def test_answer_wire_interleaves_exact_day_clip_and_time_labels(monkeypatch, tmp_path):
    first = {**image_frame("red"), "day_id": "DAY1", "clip_id": "first-clip", "untrusted_extra": "must not reach model or log"}
    second = {**image_frame("blue"), "day_id": "DAY2", "clip_id": "second-clip", "day_seconds": 7201.0, "file_seconds": 1.0}
    config = production_config()
    config["models"]["answerer"] = config["models"].pop("captioner")
    with completion_stub(monkeypatch) as (url, captured):
        monkeypatch.setenv("TEST_EGORAG_URL", url)
        monkeypatch.setenv("TEST_EGORAG_KEY", "sk-private-test-secret")
        models = Models(config, tmp_path / "calls.jsonl")
        models.protocol_context = {"namespace": "test-protocol", "manifest_hash": "test-manifest-hash"}
        models.generate("answerer", "Original system", "Original prompt", [first, second])
    contents = captured[0]["payload"]["messages"][1]["content"]
    labels = [
        'Frame metadata: {"clip_id": "first-clip", "day_id": "DAY1", "day_seconds": 3602.5, "file_seconds": 2.5, "position": 0}',
        'Frame metadata: {"clip_id": "second-clip", "day_id": "DAY2", "day_seconds": 7201.0, "file_seconds": 1.0, "position": 1}',
    ]
    assert contents == [
        {"type": "text", "text": "Original prompt"},
        {"type": "text", "text": labels[0]},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + first["jpeg_base64"]}},
        {"type": "text", "text": labels[1]},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + second["jpeg_base64"]}},
    ]
    trace = (tmp_path / "calls.jsonl").read_text()
    record = json.loads(trace)
    assert [image["wire_text"] for image in record["images"]] == labels
    assert [image["day_id"] for image in record["images"]] == ["DAY1", "DAY2"]
    assert [image["clip_id"] for image in record["images"]] == ["first-clip", "second-clip"]
    assert record["system"] == "Original system" and record["prompt"] == "Original prompt"
    assert record["protocol_context"] == {"namespace": "test-protocol", "manifest_hash": "test-manifest-hash"}
    assert "protocol_context" not in captured[0]["payload"]
    assert "test-protocol" not in json.dumps(captured[0]["payload"])
    assert "must not reach model or log" not in trace + json.dumps(captured)
    assert first["jpeg_base64"] not in trace and second["jpeg_base64"] not in trace


@pytest.mark.parametrize("finish", ["length", "max_tokens", "max_output_tokens"])
def test_output_limit_is_execution_failure_not_partial_answer(monkeypatch, tmp_path, finish):
    with completion_stub(monkeypatch, content="A", finish_reason=finish) as (url, _):
        monkeypatch.setenv("TEST_EGORAG_URL", url)
        monkeypatch.setenv("TEST_EGORAG_KEY", "sk-private-test-secret")
        models = Models(production_config(), tmp_path / "calls.jsonl")
        with pytest.raises(ModelExecutionError, match="truncated"):
            models.generate("captioner", "system", "prompt")
    record = json.loads((tmp_path / "calls.jsonl").read_text())
    assert record["status"] == "error" and "response" not in record
    assert record["usage"]["total_tokens"] == 21


def test_http_failure_never_falls_back_or_logs_provider_error_body(monkeypatch, tmp_path):
    with completion_stub(monkeypatch, status=401) as (url, _):
        monkeypatch.setenv("TEST_EGORAG_URL", url)
        monkeypatch.setenv("TEST_EGORAG_KEY", "sk-private-test-secret")
        models = Models(production_config(), tmp_path / "calls.jsonl")
        with pytest.raises(ModelExecutionError, match="401"):
            models.generate("captioner", "system", "prompt")
    log = (tmp_path / "calls.jsonl").read_text()
    assert "sk-private-test-secret" not in log
    assert models.calls == 1 and json.loads(log)["status"] == "error"


def test_trace_redacts_secret_even_if_in_model_text(monkeypatch, tmp_path):
    with completion_stub(monkeypatch, content="echo sk-private-test-secret") as (url, _):
        monkeypatch.setenv("TEST_EGORAG_URL", url)
        monkeypatch.setenv("TEST_EGORAG_KEY", "sk-private-test-secret")
        models = Models(production_config(), tmp_path / "calls.jsonl")
        models.generate("captioner", "system sk-private-test-secret", "prompt sk-private-test-secret")
    log = (tmp_path / "calls.jsonl").read_text()
    assert "sk-private-test-secret" not in log
    assert log.count("[REDACTED_API_KEY]") == 3


def test_fixture_changes_caption_evidence_and_answer_from_decoded_pixels(tmp_path):
    models = Models(fixture_config(), tmp_path / "calls.jsonl")
    red, blue = image_frame("red"), image_frame("blue")
    for role in ("captioner", "evidence", "answerer"):
        first = models.generate(role, "same", "same", [red])
        second = models.generate(role, "same", "same", [blue])
        assert first != second
        if role == "answerer":
            assert (first, second) == ("A", "B")
        else:
            assert "red" in first and "blue" in second
    assert models.calls == 6
    records = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert all(record["test_only"] for record in records)


def test_fixture_vision_does_not_guess_without_pixels(tmp_path):
    models = Models(fixture_config(), tmp_path / "calls.jsonl")
    for role in ("captioner", "evidence", "answerer"):
        with pytest.raises(ModelExecutionError, match="requires decoded"):
            models.generate(role, "system", "The answer is red; choose A")
    invalid = image_frame("blue")
    invalid["jpeg_base64"] = base64.b64encode(b"not decoded pixels").decode("ascii")
    invalid.pop("sha256")
    with pytest.raises(ModelExecutionError, match="decode actual image pixels"):
        models.generate("answerer", "system", "red", [invalid])


def test_frame_hash_mismatch_is_rejected(tmp_path):
    models = Models(fixture_config(), tmp_path / "calls.jsonl")
    frame = image_frame("red")
    frame["sha256"] = "wrong"
    with pytest.raises(ModelExecutionError, match="SHA256"):
        models.generate("captioner", "system", "prompt", [frame])


def test_fixture_selection_uses_upstream_text_contract(tmp_path):
    models = Models(fixture_config(), tmp_path / "calls.jsonl")
    assert models.generate("range_selector", "choose one day", "summaries") == "[1]"
    assert models.generate("range_selector", "HHMMSSFF", "summaries") == "[00000000]\n[23595999]"
    result = models.generate("recency_selector", "choose last", "Search Results: DAY1-01000000-01003000_0 DAY1-01003000-01010000_1")
    assert result == "ID: DAY1-01003000-01010000_1"


def test_signed_hash_embeddings_are_deterministic_test_only():
    embeddings = Embeddings({"test_only": True, "embedding": {"backend": "fixture", "dimensions": 128}})
    vectors = embeddings.embed(["red scene", "red scene", "blue scene"])
    assert vectors[0] == vectors[1] and vectors[0] != vectors[2]
    assert all(math.isclose(sum(value * value for value in vector), 1) for vector in vectors)
    assert any(value < 0 for vector in vectors for value in vector)
    assert embeddings.embed([]) == []


def test_openclip_is_lazy_and_matches_official_defaults(monkeypatch):
    torch = pytest.importorskip("torch")
    loaded = []

    class FakeModel:
        def to(self, device):
            assert device == "cpu"

        def eval(self):
            pass

        def encode_text(self, tokens):
            return tokens

    def create(**kwargs):
        loaded.append(kwargs)
        return FakeModel(), None, None

    def get_tokenizer(**kwargs):
        assert kwargs == {"model_name": "ViT-B-32"}
        return lambda texts: torch.tensor([[3.0, 4.0] for _ in texts])

    monkeypatch.setitem(sys.modules, "open_clip", SimpleNamespace(create_model_and_transforms=create, get_tokenizer=get_tokenizer))
    embeddings = Embeddings({"backend": "openclip"})
    assert loaded == []
    assert embeddings.embed(["text"])[0] == pytest.approx([0.6, 0.8])
    assert loaded == [{"model_name": "ViT-B-32", "pretrained": "laion2b_s34b_b79k"}]
    embeddings.embed(["second"])
    assert len(loaded) == 1
