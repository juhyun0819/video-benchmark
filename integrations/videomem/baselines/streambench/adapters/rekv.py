"""Pinned official ReKV: layer KV archive, native retrieval, exact continuation.

The archive is deliberately not replaced with a FIFO budget policy. Native
fixed allocations and growing CPU archives are all charged to the common B.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

from baselines.streambench.accounting import measure_state
from baselines.streambench.method import StreamingMemoryMethod, UnsupportedBudget
from .native import _package, _temporary_modules, NativeVisualMemory, local_model_fingerprint


UPSTREAM_REVISION = "5a17285d5e1a7488228eb0098b5e586d72a73e3a"
MODEL_ID = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
TRANSFORMERS_REVISION = "66bc4def9505fa7c7fe4aa7a248c34a026bb552b"
VENDOR = Path(__file__).resolve().parents[1] / "vendor/rekv"
PACKAGE = "_streambench_rekv_model"
_POLICY = None


def verify_source():
    manifest = json.loads((VENDOR / "provenance.json").read_text())
    if manifest["revision"] != UPSTREAM_REVISION:
        raise ValueError("Unexpected ReKV source revision")
    for record in manifest["files"]:
        raw = (VENDOR / record["path"]).read_bytes()
        if record.get("trailing_newline_added"):
            raw = raw[:-1] if raw.endswith(b"\n") else raw
        actual = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if actual != record["git_blob"]:
            raise ValueError("Pinned ReKV source changed: " + record["path"])
    return manifest


def load_policy():
    """Import the unchanged attention/cache classes without model dependencies."""
    global _POLICY
    if _POLICY is None:
        verify_source()
        _package(PACKAGE, VENDOR / "model")
        cache = importlib.import_module(PACKAGE + ".attention.kv_cache_manager")
        rope = importlib.import_module(PACKAGE + ".attention.rope")
        _POLICY = SimpleNamespace(**{name: getattr(cache, name) for name in (
            "ContextManager", "MemoryUnit", "VectorTensor", "CudaCache")},
            RotaryEmbeddingESM=rope.RotaryEmbeddingESM)
    return _POLICY


def _sync():
    import torch
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


def encode_cache_graph(root):
    """Expose every native field as a plain graph, preserving shared objects.

    CUDA events are completed before export; they carry scheduling rather than
    semantic memory. Restore recreates completed events. Unknown state fails
    closed instead of silently excluding a newly introduced cache field.
    """
    import torch
    policy = load_policy()
    classes = {getattr(policy, name): name for name in vars(policy)}
    _sync()
    objects, memo = [], {}

    def encode(value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, torch.dtype):
            return {"dtype": str(value).split(".")[-1]}
        if isinstance(value, torch.device):
            return {"device": str(value)}
        if id(value) in memo:
            return {"ref": memo[id(value)]}
        index = len(objects)
        memo[id(value)] = index
        record = {}
        objects.append(record)
        if isinstance(value, torch.Tensor):
            record.update(kind="tensor", value=value, device=str(value.device), pinned=value.is_pinned())
        elif isinstance(value, torch.cuda.Event):
            record.update(kind="completed_event")
        elif isinstance(value, dict):
            record.update(kind="dict", items=[(encode(k), encode(v)) for k, v in value.items()])
        elif isinstance(value, (list, tuple, set)):
            record.update(kind=type(value).__name__, items=[encode(item) for item in value])
        elif type(value) in classes:
            kind = classes[type(value)]
            if kind == "RotaryEmbeddingESM":
                attrs = {name: getattr(value, name) for name in (
                    "base", "distance_scale", "inv_freq", "_seq_len_cached", "_cos_cached", "_sin_cached")}
            else:
                attrs = {name: item for name, item in vars(value).items() if name != "Attn"}
            record.update(kind=kind, attrs=encode(attrs))
        else:
            raise TypeError(f"Unaccounted ReKV persistent object: {type(value).__name__}")
        return {"ref": index}

    root_ref = encode(root)
    return {"version": 1, "root": root_ref, "objects": objects}


def decode_cache_graph(state, *, device="cuda:0"):
    """Own the saved graph; preserve CPU archives and reconstruct GPU aliases."""
    import torch
    if state.get("version") != 1:
        raise ValueError("Unsupported ReKV cache graph version")
    policy = load_policy()
    state = deepcopy(state)
    memo, allocations = {}, {}

    def decode(value):
        if not isinstance(value, dict):
            return value
        if "dtype" in value:
            return getattr(torch, value["dtype"])
        if "device" in value:
            return torch.device(device if value["device"].startswith("cuda") else value["device"])
        index = value["ref"]
        if index in memo:
            return memo[index]
        record = state["objects"][index]
        kind = record["kind"]
        if kind == "tensor":
            tensor = record["value"]
            target = torch.device(device if record["device"].startswith("cuda") else record["device"])
            old = tensor.untyped_storage()
            key = (str(tensor.device), old.data_ptr(), old.nbytes(), str(target))
            if key not in allocations:
                raw = torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(old, 0, (old.nbytes(),), (1,))
                raw = raw.to(target)
                if record["pinned"] and target.type == "cpu" and torch.cuda.is_available():
                    raw = raw.pin_memory()
                allocations[key] = raw.untyped_storage()
            result = torch.empty(0, dtype=tensor.dtype, device=target).set_(
                allocations[key], tensor.storage_offset(), tensor.size(), tensor.stride())
        elif kind == "completed_event":
            if not str(device).startswith("cuda"):
                raise ValueError("A native CUDA-event checkpoint requires a CUDA device")
            result = torch.cuda.Event()
            result.record(torch.cuda.current_stream(device))
        elif kind == "dict":
            result = {}
            memo[index] = result
            result.update((decode(k), decode(v)) for k, v in record["items"])
        elif kind in {"list", "tuple", "set"}:
            result = []
            memo[index] = result
            result.extend(decode(item) for item in record["items"])
            if kind == "tuple":
                result = tuple(result)
            elif kind == "set":
                result = set(result)
        elif hasattr(policy, kind):
            cls = getattr(policy, kind)
            result = cls.__new__(cls)
            memo[index] = result
            attrs = decode(record["attrs"])
            if kind == "RotaryEmbeddingESM":
                torch.nn.Module.__init__(result)
                result.register_buffer("inv_freq", attrs.pop("inv_freq"), persistent=False)
            vars(result).update(attrs)
            if kind == "ContextManager":
                attention = importlib.import_module(PACKAGE + ".attention.dot_production_attention")
                result.Attn, _ = attention.get_multi_stage_dot_production_attention(result.fattn)
                if result.async_global_stream and str(device).startswith("cuda"):
                    cache_module = importlib.import_module(PACKAGE + ".attention.kv_cache_manager")
                    if cache_module.GLOBAL_STREAM is None:
                        cache_module.GLOBAL_STREAM = torch.cuda.Stream(device=device)
        else:
            raise ValueError("Unknown ReKV graph kind: " + str(kind))
        memo[index] = result
        return result
    return decode(state["root"])


def _graph_components(graph):
    tensors = [record["value"] for record in graph["objects"] if record["kind"] == "tensor"]
    metadata = {**graph, "objects": [
        {key: value for key, value in record.items() if key != "value"} if record["kind"] == "tensor" else record
        for record in graph["objects"]]}
    return tensors, metadata


class ReKVMemory(StreamingMemoryMethod):
    """Official training-free ReKV with all native CPU/GPU state budgeted."""

    def __init__(self, config, *, model=None):
        super().__init__(config["memory_budget_bytes"])
        self.config = deepcopy(config)
        self.config.setdefault("method", "rekv")
        if self.config["method"] != "rekv":
            raise ValueError("ReKVMemory requires method='rekv'")
        if model is not None and not config.get("synthetic_test_mode"):
            raise ValueError("Injected ReKV models require explicit synthetic_test_mode")
        if config.get("synthetic_test_mode") and (model is None or not config.get("encoder_id")):
            raise ValueError("Synthetic structural runs require model and encoder_id")
        self._model = model
        self._model_fingerprint = None
        self._initialized = False
        self._last_record = None
        self._frames_seen = 0
        self.compression_count = self.eviction_count = 0

    def initialize(self):
        verify_source()
        if self._model is None:
            self._load_model()
        self._model.kv_cache = None
        self._model.encode_init_prompt()
        self._last_record, self._frames_seen = None, 0
        self._initialized = True
        self.minimum_memory_bytes = 0
        self.minimum_memory_bytes = self.memory_bytes()["persistent_total_bytes"]
        self.enforce_budget()
        return self

    def _position_embedding(self):
        if self._model.kv_cache:
            return self._model.kv_cache[0].position_embedding
        return None

    def state_dict(self):
        if not self._initialized:
            raise RuntimeError("initialize() must precede checkpoint/observation")
        graph = encode_cache_graph({"kv_cache": self._model.kv_cache,
            "position_embedding": self._position_embedding(), "init_prompt_ids": self._model.init_prompt_ids})
        return {"version": 1, "identity": self.config, "upstream_revision": UPSTREAM_REVISION,
                "model_fingerprint": self._model_fingerprint,
                "cache": graph, "last_record": self._last_record, "frames_seen": self._frames_seen,
                "minimum_memory_bytes": self.minimum_memory_bytes}

    def persistent_state(self):
        state = self.state_dict()
        tensors, graph_meta = _graph_components(state["cache"])
        return {"kv_memory_bytes": tensors, "metadata_bytes": {**state, "cache": graph_meta}}

    def load_state_dict(self, state):
        if (state.get("version") != 1 or state.get("identity") != self.config
                or state.get("upstream_revision") != UPSTREAM_REVISION):
            raise ValueError("ReKV checkpoint identity mismatch")
        if self._model is None:
            self._load_model()
        if state.get("model_fingerprint") != self._model_fingerprint:
            raise ValueError("ReKV checkpoint local model identity mismatch")
        device = "cpu" if self.config.get("synthetic_test_mode") else self.config.get("device", "cuda:0")
        restored = decode_cache_graph(state["cache"], device=device)
        self._model.kv_cache = restored["kv_cache"]
        self._model.init_prompt_ids = restored["init_prompt_ids"]
        if hasattr(self._model, "language_model"):
            self._model.language_model.model.position_bias = restored["position_embedding"]
        self._last_record, self._frames_seen = deepcopy(state["last_record"]), state["frames_seen"]
        self.minimum_memory_bytes = state["minimum_memory_bytes"]
        self._initialized = True

    def observe(self, frames, timestamps):
        import numpy as np
        import torch
        from PIL import Image
        if len(frames) != len(timestamps):
            raise ValueError("frames and timestamps must align")
        before = deepcopy(self.state_dict())
        try:
            for frame, timestamp in zip(frames, timestamps):
                record = NativeVisualMemory._checked_time(self, timestamp)
                if not isinstance(frame, Image.Image) and not self.config.get("synthetic_test_mode"):
                    raise TypeError("Native ReKV requires RGB PIL images")
                array = np.asarray(frame)
                if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
                    raise ValueError("ReKV frames must be uint8 RGB")
                with torch.inference_mode():
                    self._model.encode_video(np.stack([array]), encode_chunk_size=1)
                self._last_record = record
                self._frames_seen += 1
                self.enforce_budget()
        except Exception:
            self.load_state_dict(before)
            raise
        return self.finalize_chunk()

    def finalize_chunk(self):
        return {**self.enforce_budget(), "frames_seen_total": self._frames_seen,
                "archive_blocks_per_layer": [layer.num_global_block for layer in self._model.kv_cache]}

    def answer(self, question, choices=None):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("A nonempty public question is required")
        if not self._frames_seen:
            return {"answer": "", "status": "insufficient_observation", "query_read_bytes": 0,
                    "num_kv_positions_used": 0, "num_retrieved_items": 0}
        before = deepcopy(self.state_dict())
        retained_tensors, _ = _graph_components(before["cache"])
        read_upper_bound = measure_state({"kv_memory_bytes": retained_tensors})["kv_memory_bytes"]
        reads, indices = [], []
        instrumented = []
        prompt = question
        if choices:
            prompt += "\n" + "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices))
        # The official question method resets retrieval indices before returning.
        # Observe its native retrieval boundary to record actual per-layer reads.
        for layer in self._model.kv_cache:
            original = layer.get_retrieved_kv
            def measured(query=None, *, _original=original, _layer=layer):
                key, value = _original(query)
                reads.append((key.numel() * key.element_size() + value.numel() * value.element_size(), key.shape[-2]))
                indices.append(deepcopy(_layer.retrieved_block_indices))
                return key, value
            layer.get_retrieved_kv = measured
            instrumented.append(layer)
        try:
            answer = self._model.question_answering(
                {"question": question, "prompt": self._model.get_prompt(prompt, mc=bool(choices))},
                max_new_tokens=self.config.get("max_new_tokens", 128))
            layers = self._model.kv_cache
            representative_bytes = 0
            for layer in layers:
                if layer.num_global_block > layer.topk:
                    representative_bytes += sum(vector.get_data().numel() * vector.data.element_size()
                                                for vector in layer.block_k)
                elif not layer.init_exc:
                    keys = layer.global_remainder[0][:, :, layer.n_init:]
                    if keys.shape[-2] // layer.block_size > layer.topk:
                        representative_bytes += keys.numel() * keys.element_size()
            if not reads:
                # Structural injected encoders can exercise native retrieval
                # directly without executing the CUDA transfer path.
                indices = [deepcopy(layer.retrieved_block_indices) for layer in layers]
                for layer, selected in zip(layers, indices):
                    for unit, block_ids in enumerate(selected or []):
                        for block_id in block_ids:
                            if layer.init_exc:
                                block = layer.global_blocks[unit][block_id]
                                selected_bytes = sum(t.numel() * t.element_size() for t in block.cpu_data)
                            else:
                                start = layer.n_init + block_id * layer.block_size
                                selected_bytes = sum(t[unit, :, start:start + layer.block_size].numel() * t.element_size()
                                                     for t in layer.global_remainder)
                            reads.append((selected_bytes, layer.block_size))
            return {"answer": answer, "query_read_bytes": read_upper_bound,
                    "query_read_bytes_is_upper_bound": True,
                    "read_accounting": "Full live native KV/cache/RoPE tensor storage, deduplicated; selected KV payload is reported separately",
                    "query_selected_kv_bytes": sum(row[0] for row in reads),
                    "query_representative_scan_bytes": representative_bytes,
                    "num_kv_positions_used": sum(row[1] for row in reads),
                    "num_retrieved_items": sum(len(unit) for layer in indices for unit in (layer or [])),
                    "retrieved_items": indices, "num_visual_tokens_used": 0, "num_text_items_used": 0}
        finally:
            for layer in instrumented:
                del layer.get_retrieved_kv
            self.load_state_dict(before)

    def reset(self):
        self.protocol_state = {}
        return self.initialize()

    def _load_model(self):
        import torch
        model_path = Path(self.config.get("model_path", ""))
        if not self.config.get("model_path") or not (model_path / "config.json").is_file():
            raise ValueError("ReKV requires a local model_path containing config.json")
        if self.config.get("model_id", MODEL_ID) != MODEL_ID:
            raise ValueError("This pinned adapter supports the official LLaVA-OneVision Qwen2 7B checkpoint")
        config = json.loads((model_path / "config.json").read_text())
        language = config.get("text_config", {})
        if config.get("model_type") != "llava_onevision" or language.get("model_type") != "qwen2":
            raise ValueError("ReKV model_path must contain the native OneVision/Qwen2 architecture")
        n_local, topk = self.config.get("n_local", 15000), self.config.get("topk", 64)
        max_cached = self.config.get("max_cached_block", 128)
        chunk_size = self.config.get("retrieval_chunk_size", 1)
        if (any(type(value) is not int or value < 1 for value in (n_local, topk, max_cached, chunk_size))
                or n_local < 196 or topk * 196 > n_local or topk > max_cached or topk % chunk_size):
            raise ValueError("Invalid native ReKV local/retrieval/cache capacities")
        dtype_name = self.config.get("dtype", "float16")
        if dtype_name not in {"float16", "bfloat16", "float32"}:
            raise ValueError("Unsupported ReKV dtype")
        dtype = getattr(torch, dtype_name)
        width = language["hidden_size"] // language["num_attention_heads"]
        per_token = 2 * language["num_key_value_heads"] * width * language["num_hidden_layers"] * torch.empty((), dtype=dtype).element_size()
        minimum = (max_cached + topk) * 196 * per_token
        if minimum > self.memory_budget_bytes:
            raise UnsupportedBudget("Native ReKV GPU pool and retrieval buffer alone exceed B; CPU archive is additional",
                memory_budget_bytes=self.memory_budget_bytes, minimum_memory_bytes=minimum)
        fingerprint = local_model_fingerprint(model_path)
        device = self.config.get("device", "cuda:0")
        if not str(device).startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("The pinned ReKV backend requires CUDA; no CPU surrogate is used")
        torch.cuda.set_device(device)
        load_policy()
        import sys
        package = sys.modules[PACKAGE]
        attention = importlib.import_module(PACKAGE + ".attention")
        with _temporary_modules({"model": package, "model.attention": attention}):
            patch = importlib.import_module(PACKAGE + ".patch")
            abstract = importlib.import_module(PACKAGE + ".abstract_rekv")
            with _temporary_modules({"model.patch": patch, "model.abstract_rekv": abstract}):
                implementation = importlib.import_module(PACKAGE + ".llava_onevision_rekv")
        processor = implementation.LlavaOnevisionProcessor.from_pretrained(str(model_path), local_files_only=True)
        initial = "<|im_start|>system \nYou are a helpful assistant.<|im_end|><|im_start|>user "
        initial_ids = processor.tokenizer(initial, return_tensors="pt").input_ids.to(device)
        model = implementation.LlavaOneVision_ReKV.from_pretrained(
            str(model_path), local_files_only=True, device_map={"": device}, low_cpu_mem_usage=True,
            torch_dtype=dtype, processor=processor, n_frame_tokens=196, init_prompt_ids=initial_ids,
            n_local=n_local, topk=topk, chunk_size=chunk_size)
        native = {"n_init": initial_ids.shape[1], "n_local": n_local, "fattn": self.config.get("fattn", True),
                  "block_size": 196, "topk": topk, "chunk_size": chunk_size,
                  "max_cached_block": max_cached, "exc_block_size": 196,
                  "pin_memory": self.config.get("pin_memory", True),
                  "async_global_stream": self.config.get("async_global_stream", True)}
        model.language_model = patch.patch_hf(model.language_model, **native)
        self._model = model.eval().requires_grad_(False)
        self._model_fingerprint = fingerprint


def create_method(config):
    return ReKVMemory(config)
