"""Explicit model backends and auditable, image-redacted request traces.

The fixture backends are deliberately non-semantic test doubles. They must be
enabled with ``test_only: true`` and are never a production fallback.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Mapping, Optional
from urllib import error, parse, request


ROLES = (
    "captioner", "summary_10min", "summary_hour", "summary_day",
    "range_selector", "query_rewriter", "recency_selector", "evidence", "answerer",
)


class BackendConfigurationError(ValueError):
    """A required, explicitly configured backend setting is missing or invalid."""


class ModelExecutionError(RuntimeError):
    """An attempted inference failed, including truncated or empty outputs."""


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _positive_number(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise BackendConfigurationError("%s must be a positive finite number" % name)


class Models:
    """Dispatch each pipeline role to its explicitly configured backend.

    ``config`` is the adapter configuration containing ``models`` and optional
    ``test_only``. Frames contain ``jpeg_base64``, ``sha256``, ``file_seconds``,
    ``day_seconds`` and optional ``day_id``/``clip_id``. Each wire image is
    preceded by a text label with its supplied identity and timestamps. Only
    image metadata and these exact labels are persisted to the JSONL trace.
    ``calls`` counts attempted inference calls, including failed HTTP calls.
    """

    def __init__(self, config: Mapping[str, Any], log_path: Any):
        self.config = dict(config)
        self.roles = {name: dict(value) for name, value in config.get("models", {}).items()}
        self.test_only = config.get("test_only") is True
        self.log_path = Path(log_path)
        self.calls = 0
        self.total_elapsed_seconds = 0.0
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._secrets = []
        self._connections = {}
        self._local_hf = None
        self._checkpoint_integrity = None
        self.cache_identity = {}
        if not self.roles:
            raise BackendConfigurationError("models must explicitly configure at least one role")
        for role, settings in self.roles.items():
            if role not in ROLES:
                raise BackendConfigurationError("Unknown model role: %s" % role)
            backend = settings.get("backend")
            if backend == "fixture":
                if not self.test_only:
                    raise BackendConfigurationError("fixture models require explicit test_only: true")
            elif backend == "openai_compatible":
                self._connections[role] = self._validate_connection(role, settings)
            elif backend == "hf_local":
                from .local_hf import LocalHF

                if self._local_hf is None:
                    self._local_hf = LocalHF()
                self._local_hf.validate_settings(role, settings)
            else:
                raise BackendConfigurationError("Unsupported backend for role %s" % role)

        embedding = config.get("embedding", {})
        pinned_embedding = (embedding.get("backend") == "openclip"
                            and embedding.get("sha256") is not None)
        if self._local_hf is not None or pinned_embedding:
            from .local_integrity import CheckpointIntegrity

            self._checkpoint_integrity = CheckpointIntegrity()
            for role, settings in self.roles.items():
                if settings["backend"] == "hf_local":
                    self.cache_identity[role] = self._checkpoint_integrity.register(settings)
            if pinned_embedding:
                self.cache_identity["embedding"] = self._checkpoint_integrity.register_file(
                    embedding.get("checkpoint"), embedding["sha256"], embedding.get("revision"))

    def assert_checkpoints_unchanged(self):
        """Fail before inference or cache reuse if verified local files changed."""
        if self._checkpoint_integrity is not None:
            self._checkpoint_integrity.assert_unchanged()

    def _validate_connection(self, role: str, settings: Mapping[str, Any]):
        model = settings.get("model")
        if not _nonempty(model) or model.strip().lower() in {"none", "null", "unset", "changeme", "your-model"}:
            raise BackendConfigurationError("Role %s requires an explicit model identifier" % role)
        key_env = settings.get("api_key_env", "EGORAG_API_KEY")
        url_env = settings.get("base_url_env", "EGORAG_BASE_URL")
        if not _nonempty(key_env) or not _nonempty(url_env):
            raise BackendConfigurationError("Role %s requires API key and base URL environment variable names" % role)
        key = os.environ.get(key_env)
        if not _nonempty(key):
            raise BackendConfigurationError("Role %s requires environment variable %s" % (role, key_env))
        self._secrets.append(key)
        base_url = os.environ.get(url_env)
        if not _nonempty(base_url):
            raise BackendConfigurationError("Role %s requires environment variable %s" % (role, url_env))
        parsed = parse.urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise BackendConfigurationError("Role %s base URL must be an HTTP(S) URL without credentials, query or fragment" % role)
        timeout = settings.get("timeout", 120)
        max_tokens = settings.get("max_tokens", 1024)
        _positive_number(timeout, "timeout")
        _positive_number(max_tokens, "max_tokens")
        if not isinstance(max_tokens, int):
            raise BackendConfigurationError("max_tokens must be an integer")
        temperature = settings.get("temperature", 0)
        if isinstance(temperature, bool) or not isinstance(temperature, (float, int)) or not math.isfinite(temperature):
            raise BackendConfigurationError("temperature must be a finite number")
        endpoint = base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        return endpoint, key

    def _sanitize(self, value: Any, image_payloads=()) -> Any:
        if isinstance(value, str):
            for secret in sorted(set(self._secrets), key=len, reverse=True):
                value = value.replace(secret, "[REDACTED_API_KEY]")
            value = re.sub(r"data:image/[^;\s]+;base64,[A-Za-z0-9+/=]+", "[REDACTED_IMAGE_DATA]", value)
            for payload in image_payloads:
                value = value.replace(payload, "[REDACTED_IMAGE_DATA]")
            return value
        if isinstance(value, dict):
            return {key: self._sanitize(item, image_payloads) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._sanitize(item, image_payloads) for item in value]
        return value

    @staticmethod
    def _frames(frames: Optional[List[Dict[str, Any]]]):
        wire_images, metadata, decoded = [], [], []
        for position, frame in enumerate(frames or []):
            if not isinstance(frame, dict) or not _nonempty(frame.get("jpeg_base64")):
                raise ModelExecutionError("Each frame requires jpeg_base64 image bytes")
            try:
                raw = base64.b64decode(frame["jpeg_base64"], validate=True)
            except (ValueError, TypeError):
                raise ModelExecutionError("A frame contains invalid image base64") from None
            digest = hashlib.sha256(raw).hexdigest()
            if frame.get("sha256") and frame["sha256"] != digest:
                raise ModelExecutionError("A frame SHA256 does not match the supplied image")
            item = {"position": position, "sha256": digest, "byte_count": len(raw), "mime_type": "image/jpeg"}
            for key in ("day_id", "clip_id"):
                value = frame.get(key)
                if value is not None:
                    if isinstance(value, bool) or not (isinstance(value, int) or _nonempty(value)):
                        raise ModelExecutionError("Frame identifiers must be nonempty strings or integers")
                    item[key] = value
            for key in ("file_seconds", "day_seconds"):
                value = frame.get(key)
                if value is not None:
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ModelExecutionError("Frame timestamps must be finite numbers")
                    item[key] = value
            label = {key: item[key] for key in ("position", "day_id", "clip_id", "day_seconds", "file_seconds") if key in item}
            item["wire_text"] = "Frame metadata: " + json.dumps(label, sort_keys=True, ensure_ascii=False, allow_nan=False)
            metadata.append(item)
            decoded.append(raw)
            wire_images.append({"type": "text", "text": item["wire_text"]})
            wire_images.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + frame["jpeg_base64"]}})
        return wire_images, metadata, decoded

    def generate(self, role: str, system: str, prompt: str, frames=None) -> str:
        self.assert_checkpoints_unchanged()
        if role not in self.roles:
            raise BackendConfigurationError("No backend configured for role %s" % role)
        if not isinstance(system, str) or not isinstance(prompt, str):
            raise ModelExecutionError("system and prompt must be strings")
        settings = self.roles[role]
        wire_images, image_metadata, decoded = self._frames(frames)
        content = [{"type": "text", "text": prompt}] + wire_images if wire_images else prompt
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        self.calls += 1
        started = time.perf_counter()
        record = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "protocol_context": getattr(self, "protocol_context", {}),
            "call_index": self.calls, "role": role, "backend": settings["backend"],
            "model": settings.get("model"), "revision": settings.get("revision"),
            "system": system, "prompt": prompt, "images": image_metadata,
            "test_only": settings["backend"] == "fixture", "status": "error",
            "usage": None,
        }
        try:
            if settings["backend"] == "fixture":
                result = self._fixture(role, system, prompt, decoded)
                record["finish_reason"] = "fixture"
            else:
                payload = {
                    "model": settings["model"], "messages": messages,
                    "temperature": settings.get("temperature", 0),
                    "max_tokens": settings.get("max_tokens", 1024),
                }
                record["parameters"] = {key: value for key, value in payload.items() if key != "messages"}
                if settings["backend"] == "hf_local":
                    record["checkpoint_identity"] = {key: value for key, value in self.cache_identity[role].items()
                                                     if key != "files"}
                    response = self._local_hf.generate(settings, system, prompt, decoded, image_metadata, record)
                else:
                    endpoint, key = self._connections[role]
                    record["endpoint"] = endpoint
                    response = self._http(endpoint, key, payload, settings.get("timeout", 120))
                usage = response.get("usage")
                if isinstance(usage, dict):
                    record["usage"] = {
                        key: value for key, value in usage.items()
                        if key in self.token_usage and isinstance(value, int) and not isinstance(value, bool) and value >= 0
                    }
                    for key, value in record["usage"].items():
                        self.token_usage[key] += value
                choices = response.get("choices")
                if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                    raise ModelExecutionError("Model response has no completion choice")
                choice = choices[0]
                finish = choice.get("finish_reason")
                record["finish_reason"] = finish
                if finish in {"length", "max_tokens", "max_output_tokens"}:
                    raise ModelExecutionError("Model output was truncated at its output token limit")
                if finish not in {"stop", None}:
                    raise ModelExecutionError("Model response did not finish normally")
                message = choice.get("message")
                result = message.get("content") if isinstance(message, dict) else None
            if not isinstance(result, str) or not result.strip():
                raise ModelExecutionError("Model response contains no text")
            record["status"] = "ok"
            record["response"] = result
            return result
        except ModelExecutionError as exc:
            record["error"] = str(exc)
            raise
        except Exception:
            # Provider exceptions and response bodies can contain credentials.
            record["error"] = "Model execution failed"
            raise ModelExecutionError("Model execution failed for role %s" % role) from None
        finally:
            elapsed = time.perf_counter() - started
            self.total_elapsed_seconds += elapsed
            record["elapsed_seconds"] = elapsed
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as trace:
                trace.write(json.dumps(self._sanitize(record, [frame["jpeg_base64"] for frame in frames or []]), ensure_ascii=False, allow_nan=False) + "\n")

    @staticmethod
    def _http(endpoint, key, payload, timeout):
        req = request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # urllib must never forward the credential to a redirect destination.
        req.add_unredirected_header("Authorization", "Bearer " + key)
        try:
            with request.urlopen(req, timeout=timeout) as stream:
                response = json.load(stream)
        except error.HTTPError as exc:
            # Do not persist potentially secret-bearing server error bodies.
            raise ModelExecutionError("Model HTTP request failed with status %s" % exc.code) from None
        except (error.URLError, TimeoutError, OSError):
            raise ModelExecutionError("Model HTTP connection failed or timed out") from None
        except (ValueError, UnicodeError):
            raise ModelExecutionError("Model HTTP response is not valid JSON") from None
        if not isinstance(response, dict):
            raise ModelExecutionError("Model HTTP response is not a JSON object")
        return response

    @staticmethod
    def _fixture(role, system, prompt, images):
        """A color-reading smoke double; never a benchmark inference backend."""
        if role in {"captioner", "evidence", "answerer"} and images:
            from PIL import Image, ImageStat

            colors = []
            for raw in images:
                try:
                    with Image.open(io.BytesIO(raw)) as image:
                        means = ImageStat.Stat(image.convert("RGB")).mean
                except Exception:
                    raise ModelExecutionError("Fixture could not decode actual image pixels") from None
                color = ("red", "green", "blue")[max(range(3), key=lambda index: means[index])]
                colors.append(color)
            if role == "evidence":
                return "I can provide evidence. Evidence: " + ", ".join(colors) + " scene (decoded-pixel test fixture)."
            if role == "answerer":
                if colors[-1] not in {"red", "blue"}:
                    raise ModelExecutionError("Fixture answerer only supports red/A and blue/B test scenes")
                return {"red": "A", "blue": "B"}[colors[-1]]
            return "Test fixture decoded frame colors: " + ", ".join(colors) + "."
        if role in {"captioner", "evidence", "answerer"}:
            raise ModelExecutionError("Fixture %s requires decoded image frames" % role)
        if role.startswith("summary_"):
            return "Test fixture summary: " + prompt
        if role == "range_selector":
            return "[00000000]\n[23595999]" if "HHMMSSFF" in system else "[1]"
        if role == "query_rewriter":
            return "Keywords: [color]\nSearch Time Range: [DAY1 23595999]"
        if role == "recency_selector":
            identifiers = re.findall(r"DAY\d+-\d{8}-\d{8}_\d+", prompt)
            if identifiers:
                return "ID: " + identifiers[-1]
            raise ModelExecutionError("Fixture recency selector received no candidate identifiers")
        raise ModelExecutionError("Fixture role %s requires an explicitly injected test implementation" % role)


class Embeddings:
    """Normalized OpenCLIP text embeddings using the upstream Chroma defaults.

    Model imports and checkpoint loading happen only on first ``embed``. The
    fixture alternative is signed token hashing, explicitly NOT semantic CLIP.
    """

    def __init__(self, config: Mapping[str, Any]):
        settings = config.get("embedding", config)
        self.settings = dict(settings)
        self.backend = settings.get("backend", "openclip")
        self.test_only = config.get("test_only") is True or settings.get("test_only") is True
        self.model_name = settings.get("model_name", "ViT-B-32")
        self.checkpoint = settings.get("checkpoint", "laion2b_s34b_b79k")
        self.device = settings.get("device", "cpu")
        self._model = self._tokenizer = self._torch = None
        self.calls = 0
        if self.backend == "fixture":
            if not self.test_only:
                raise BackendConfigurationError("fixture embeddings require explicit test_only: true")
            self.dimensions = settings.get("dimensions", 128)
            if isinstance(self.dimensions, bool) or not isinstance(self.dimensions, int) or self.dimensions < 2:
                raise BackendConfigurationError("fixture dimensions must be an integer >= 2")
        elif self.backend != "openclip":
            raise BackendConfigurationError("Unsupported embedding backend")
        elif not all(_nonempty(value) for value in (self.model_name, self.checkpoint, self.device)):
            raise BackendConfigurationError("OpenCLIP model_name, checkpoint and device must be explicit strings")

    def _load(self):
        if self._model is not None:
            return
        try:
            import open_clip
            import torch
        except ImportError:
            raise BackendConfigurationError("OpenCLIP embeddings require open_clip_torch and torch; no fallback is permitted") from None
        model, _, _ = open_clip.create_model_and_transforms(model_name=self.model_name, pretrained=self.checkpoint)
        model.to(self.device)
        model.eval()
        tokenizer = open_clip.get_tokenizer(model_name=self.model_name)
        self._model = model
        self._tokenizer = tokenizer
        self._torch = torch

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
            raise ValueError("embed expects a list of strings")
        if not texts:
            return []
        self.calls += 1
        if self.backend == "fixture":
            result = []
            for text in texts:
                vector = [0.0] * self.dimensions
                for token in re.findall(r"\w+", text.lower()) or [""]:
                    digest = hashlib.sha256(token.encode("utf-8")).digest()
                    bucket = int.from_bytes(digest[:8], "big") % self.dimensions
                    vector[bucket] += 1.0 if digest[8] & 1 else -1.0
                norm = math.sqrt(sum(value * value for value in vector))
                result.append([value / norm if norm else value for value in vector])
            return result
        self._load()
        with self._torch.no_grad():
            tokens = self._tokenizer(texts).to(self.device)
            features = self._model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            return features.detach().cpu().tolist()

    def __call__(self, input):
        """The ``input`` parameter name also satisfies Chroma's callable API."""
        return self.embed(list(input))
