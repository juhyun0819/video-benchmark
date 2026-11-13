"""Continual adapters around the repository's unchanged native visual banks.

Model dependencies are lazy. CPU structural tests inject an encoder explicitly;
production never substitutes synthetic features for unavailable pretrained models.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from copy import deepcopy
import importlib
import importlib.util
import hashlib
import math
from pathlib import Path
import sys
import types

from baselines.streambench.accounting import measure_state
from baselines.streambench.method import StreamingMemoryMethod, UnsupportedBudget


ROOT = Path(__file__).resolve().parents[2]
ALIASES = {"fifo_qwen3vl": "fifo", "recent_window": "fifo",
           "reservoir_qwen3vl": "reservoir", "fluxmem": "fluxmem_streaming"}
METHODS = {"fifo", "reservoir", "streamforest", "fluxmem_streaming", "fluxmem_qwen3vl"}
SF_DEFAULT = {"st_memory_windows": [1, 18], "st_memory_tokens": [729, 128],
              "event_split_window": 8, "long_memory_tokens_per_frame": 64,
              "long_memory_tokens_quota": 5120,
              "sim_weight_g": 0.4, "time_weight_a": 0.2, "merge_weight_b": 0.4}
PUBLIC_TIME = {"stream_id", "day", "global_frame_id", "source_timestamp",
               "global_timestamp", "source_id", "source_frame_index"}
QUEUES = {"fifo": ("_grids",), "reservoir": ("_slots",),
          "fluxmem_streaming": ("_short", "_medium", "_long"),
          "fluxmem_qwen3vl": ("_short", "_medium", "_long"), "streamforest": ()}


def local_model_fingerprint(path):
    """Hash local inference artifacts once when loading a frozen model.

    Checkpoint identity must describe actual local bytes, not only a mutable
    directory name. No weight tensor is materialized by this streaming hash.
    """
    root = Path(path).expanduser().resolve()
    suffixes = {".json", ".safetensors", ".bin", ".model", ".txt", ".jinja"}
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in suffixes)
    if not (root / "config.json").is_file() or not any(p.suffix in {".safetensors", ".bin"} for p in paths):
        raise ValueError("Local model identity requires config.json and materialized model weights")
    digest = hashlib.sha256()
    for artifact in paths:
        before = artifact.stat()
        digest.update(artifact.relative_to(root).as_posix().encode() + b"\0")
        digest.update(str(before.st_size).encode() + b"\0")
        with artifact.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        after = artifact.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise RuntimeError("Local model artifact changed during fingerprinting")
    return digest.hexdigest()


def _package(name, directory):
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(directory)]
        package.__package__ = name
        package.__spec__ = importlib.util.spec_from_loader(name, loader=None, is_package=True)
        sys.modules[name] = package
    return sys.modules[name]


@contextmanager
def _temporary_modules(mapping):
    """Only used during synchronous import; existing baseline namespaces survive."""
    previous = {key: sys.modules.get(key) for key in mapping}
    sys.modules.update(mapping)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


def _native_modules(method):
    if method in {"fifo", "reservoir", "streamforest"}:
        directory, package = {
            "fifo": ("fifo_qwen3vl", "fifo_videomem"),
            "reservoir": ("reservoir_qwen3vl", "reservoir_videomem"),
            "streamforest": ("streamforest", "streamforest_videomem"),
        }[method]
        name = "_streambench_" + package
        _package(name, ROOT / directory / package)
        module = importlib.import_module(name + ".memory")
        cls = getattr(module, {"fifo": "Qwen3FIFOMemory", "reservoir": "Qwen3ReservoirMemory",
                               "streamforest": "StreamForestMemory"}[method])
        return cls, module
    directory = ROOT / method
    name = "_streambench_" + method
    package = _package(name, directory / "streaming")
    kernel_name = name + "_kernel"
    kernel_package = _package(kernel_name, directory / "models/qwen2-5-vl/src/qwen2_5_vl_fluxmem")
    kernel = importlib.import_module(kernel_name + ".fluxmem")
    helper = types.ModuleType(name + ".upstream_kernel")
    helper.FluxMem = kernel.FluxMem
    aliases = {"streaming": package, "streaming.upstream_kernel": helper,
               "qwen2_5_vl_fluxmem": kernel_package, "qwen2_5_vl_fluxmem.fluxmem": kernel}
    with _temporary_modules(aliases):
        base = importlib.import_module(name + ".memory")
        if method == "fluxmem_qwen3vl":
            module = importlib.import_module(name + ".qwen3_memory")
            return module.Qwen3PersistentMemory, module
    return base.PersistentMemory, base


def _split_tensors(value, tensors):
    import torch
    if isinstance(value, torch.Tensor):
        tensors.append(value)
        return None
    if isinstance(value, dict):
        return {key: _split_tensors(item, tensors) for key, item in value.items()}
    if isinstance(value, (list, tuple, deque)):
        return [_split_tensors(item, tensors) for item in value]
    return value


def _owned_to_device(value, device):
    """Copy a complete graph while preserving all tensor storage/view aliases."""
    import torch
    owned = deepcopy(value)
    storages = {}

    def move(item):
        if isinstance(item, torch.Tensor):
            if item.device == torch.device(device):
                return item
            old = item.untyped_storage()
            key = (str(item.device), old.data_ptr(), old.nbytes())
            if key not in storages:
                raw = torch.empty(0, dtype=torch.uint8, device=item.device).set_(
                    old, 0, (old.nbytes(),), (1,))
                storages[key] = raw.to(device).untyped_storage()
            return torch.empty(0, dtype=item.dtype, device=device).set_(
                storages[key], item.storage_offset(), item.size(), item.stride())
        if isinstance(item, dict):
            return {key: move(entry) for key, entry in item.items()}
        if isinstance(item, list):
            return [move(entry) for entry in item]
        if isinstance(item, tuple):
            return tuple(move(entry) for entry in item)
        return item
    return move(owned)


class NativeVisualMemory(StreamingMemoryMethod):
    """One uninterrupted stream, preserving each native memory representation."""

    def __init__(self, config, *, encoder=None, answerer=None):
        self.config = deepcopy(config)
        self.method = ALIASES.get(config.get("method"), config.get("method"))
        if self.method == "fluxmem_reference":
            raise NotImplementedError("fluxmem_reference rebuilds memory per prefill; use fluxmem_streaming for continual memory")
        if self.method not in METHODS:
            raise ValueError(f"Unknown native memory method: {self.method}")
        super().__init__(config["memory_budget_bytes"])
        self.config["method"] = self.method
        if (encoder is not None or answerer is not None) and not config.get("synthetic_test_mode"):
            raise ValueError("Injected encoders/answerers require explicit synthetic_test_mode")
        if config.get("synthetic_test_mode") and (encoder is None or not config.get("encoder_id")):
            raise ValueError("Synthetic structural runs require encoder and encoder_id")
        self._encoder, self._answerer = encoder, answerer
        self._model = self._processor = self._tokenizer = None
        self._model_fingerprint = None
        self._bank_class, self._bank_module = _native_modules(self.method)
        self.bank = None
        self._initialized = False

    def _new_bank(self):
        if self.method in {"fifo", "reservoir"}:
            capacity = self.config.get("capacity_grids")
            if capacity is not None and (type(capacity) is not int or capacity < 1):
                raise ValueError("capacity_grids must be a positive integer")
            if capacity is not None and any(self.config.get(key) is not None for key in ("memory_frames", "memory_tokens")):
                raise ValueError("capacity_grids cannot be combined with another native capacity")
            kwargs = {key: self.config[key] for key in ("memory_frames", "memory_tokens") if self.config.get(key) is not None}
            if not kwargs:
                kwargs["memory_frames"] = 2 * (capacity or 1)
            if self.method == "reservoir":
                kwargs["seed"] = self.config.get("seed", 0)
            return self._bank_class(**kwargs)
        if self.method == "streamforest":
            return self._bank_class(self.config.get("hidden_size", 1152),
                                    self.config.get("num_attention_heads", 16),
                                    config=deepcopy(self.config.get("memory_config", SF_DEFAULT)))
        return self._bank_class(short_frames=self.config.get("short_frames", 8),
            medium_frames=self.config.get("medium_frames", 64), long_frames=self.config.get("long_frames"),
            gamma=self.config.get("gamma", 0.5), tas_policy=self.config.get("tas_policy", "upstream"))

    def initialize(self):
        self.bank = self._new_bank()
        self._pending_frame = None
        self._last_record = None
        self._frames_seen = self._groups_seen = 0
        self.compression_count = self.eviction_count = 0
        self._capacity_resolved = False
        self._initialized = True
        self.minimum_memory_bytes = 0
        self.minimum_memory_bytes = self.memory_bytes()["persistent_total_bytes"]
        self.enforce_budget()
        return self

    def _ensure_initialized(self):
        if not self._initialized:
            raise RuntimeError("initialize() must precede observation or checkpoint operations")

    def _bank_state(self):
        excluded = {"_manager", "_kernel", "_rng", *QUEUES[self.method]}
        result = {"attributes": {key: value for key, value in vars(self.bank).items() if key not in excluded}}
        result["queues"] = {name: [vars(grid) for grid in getattr(self.bank, name)] for name in QUEUES[self.method]}
        if self.method == "streamforest":
            result["manager"] = self.bank._manager_state()
        if self.method == "reservoir":
            result["rng_state"] = self.bank._rng.getstate()
        return result

    def state_dict(self):
        self._ensure_initialized()
        # Live tensors intentionally retain aliases. Base snapshot/torch.save
        # own the complete object graph, instead of cloning tensors separately.
        return {"version": 1, "identity": self.config, "bank": self._bank_state(),
                "model_fingerprint": self._model_fingerprint,
                "pending_frame": self._pending_frame, "last_record": self._last_record,
                "frames_seen": self._frames_seen, "groups_seen": self._groups_seen,
                "compression_count": self.compression_count, "eviction_count": self.eviction_count,
                "capacity_resolved": self._capacity_resolved,
                "minimum_memory_bytes": self.minimum_memory_bytes}

    def persistent_state(self):
        tensors = []
        metadata = _split_tensors(self.state_dict(), tensors)
        return {"visual_memory_bytes": tensors, "metadata_bytes": metadata}

    def load_state_dict(self, state):
        if state.get("version") != 1 or state.get("identity") != self.config:
            raise ValueError("Native checkpoint identity does not match method/budget/model/config")
        fingerprint = state.get("model_fingerprint")
        if fingerprint is not None:
            current = self._model_fingerprint
            if current is None:
                current = local_model_fingerprint(self.config.get("model_path", self.config.get("model", "")))
            if fingerprint != current:
                raise ValueError("Native checkpoint local model identity mismatch")
        import torch
        device = self.config.get("device", "cpu" if self.config.get("synthetic_test_mode") else "cuda:0")
        saved = _owned_to_device(state["bank"], device)
        bank = self._new_bank()
        expected = set(vars(bank)) - {"_manager", "_kernel", "_rng", *QUEUES[self.method]}
        if set(saved["attributes"]) != expected or set(saved["queues"]) != set(QUEUES[self.method]):
            raise ValueError("Native checkpoint is missing complete bank state")
        if self.method == "streamforest":
            attrs = saved["attributes"]
            bank = self._bank_class(attrs["hidden_size"], attrs["num_attention_heads"], config=attrs["config"])
        vars(bank).update(saved["attributes"])
        if bank._dtype is not None:
            bank._device = torch.device(device)
        grid_type = getattr(self._bank_module, "_DeepStackGrid", getattr(self._bank_module, "_Grid", None))
        for name, records in saved["queues"].items():
            values = [grid_type(**record) for record in records]
            setattr(bank, name, values if self.method == "reservoir" else deque(values))
        if self.method == "streamforest":
            if set(saved.get("manager", {})) != set(bank._manager_state()):
                raise ValueError("Native checkpoint is missing complete StreamForest manager state")
            for name, value in saved["manager"].items():
                setattr(bank._manager, name, value)
            bank._validate_restored_manager()
        if self.method == "reservoir":
            bank._rng.setstate(saved["rng_state"])
        self.bank = bank
        self._pending_frame = deepcopy(state["pending_frame"])
        self._last_record = deepcopy(state["last_record"])
        self._frames_seen, self._groups_seen = state["frames_seen"], state["groups_seen"]
        self.compression_count, self.eviction_count = state["compression_count"], state["eviction_count"]
        self._capacity_resolved = state["capacity_resolved"]
        self.minimum_memory_bytes = state["minimum_memory_bytes"]
        self._initialized = True
        self._model_fingerprint = fingerprint

    @staticmethod
    def _frame_payload(frame):
        import torch
        from PIL import Image
        if isinstance(frame, torch.Tensor):
            return {"kind": "tensor", "pixels": frame.detach().cpu().clone()}
        if not isinstance(frame, Image.Image) or frame.mode != "RGB":
            raise ValueError("Native visual observation requires RGB PIL frames")
        import numpy as np
        return {"kind": "rgb", "pixels": torch.from_numpy(np.array(frame, copy=True))}

    @staticmethod
    def _unpack_frame(payload):
        if payload["kind"] == "tensor":
            return payload["pixels"]
        from PIL import Image
        return Image.fromarray(payload["pixels"].numpy(), mode="RGB")

    def _checked_time(self, record):
        if not isinstance(record, dict) or set(record) != PUBLIC_TIME:
            raise ValueError("timestamps must contain exactly the public frame fields; no questions/labels")
        for name in ("day", "global_frame_id", "source_frame_index"):
            if type(record[name]) is not int or record[name] < (1 if name == "day" else 0):
                raise ValueError(f"Invalid {name}")
        for name in ("source_timestamp", "global_timestamp"):
            if isinstance(record[name], bool) or not isinstance(record[name], (int, float)) or not math.isfinite(record[name]) or record[name] < 0:
                raise ValueError(f"Invalid {name}")
        if any(not isinstance(record[key], str) or not record[key] for key in ("source_id", "stream_id")):
            raise ValueError("Frame source_id and stream_id must be nonempty")
        if self._last_record is not None:
            last = self._last_record
            if (record["stream_id"] != last["stream_id"] or record["day"] < last["day"]
                    or record["global_frame_id"] <= last["global_frame_id"]
                    or record["global_timestamp"] <= last["global_timestamp"]):
                raise ValueError("Frames must belong to one strictly chronological stream")
        return deepcopy(record)

    def _resolve_capacity(self, raw_bytes):
        if self._capacity_resolved:
            return
        if self.method in {"fifo", "reservoir"} and not any(self.config.get(key) is not None for key in ("capacity_grids", "memory_frames", "memory_tokens")):
            queue = self.bank._grids if self.method == "fifo" else self.bank._slots
            record_bytes = measure_state({"visual_memory_bytes": vars(queue[0])})["persistent_total_bytes"]
            actual = self.memory_bytes()["persistent_total_bytes"]
            reserve = max(raw_bytes, self.config.get("pending_frame_reserve_bytes", 0)) + 16384
            capacity = (self.memory_budget_bytes - (actual - record_bytes) - reserve) // (record_bytes + 256)
            if capacity < 1:
                raise UnsupportedBudget("Budget cannot fit one native grid plus pending-frame/protocol reserve",
                    memory_budget_bytes=self.memory_budget_bytes,
                    minimum_memory_bytes=actual + reserve + 256)
            self.bank.memory_frames = int(capacity) * 2
            self.minimum_memory_bytes = actual + reserve + 256
        self._capacity_resolved = True

    def _resolve_streamforest_quota(self, features):
        if self._capacity_resolved or self.config.get("memory_config") is not None:
            self._capacity_resolved = True
            return
        cfg = deepcopy(SF_DEFAULT)
        h, size = features.shape[1], features.element_size()
        n, s = cfg["st_memory_windows"]
        nt, st = cfg["st_memory_tokens"]
        minimum_quota = cfg["event_split_window"] * cfg["long_memory_tokens_per_frame"]
        fixed = (features.shape[0] + n * nt + (s + cfg["event_split_window"] - 1) * st) * h * size
        fixed += self.memory_bytes()["persistent_total_bytes"] + 16384
        unit = cfg["long_memory_tokens_per_frame"]
        quota = ((self.memory_budget_bytes - fixed) // (h * size) // unit) * unit
        if quota < minimum_quota:
            raise UnsupportedBudget("Budget is below native StreamForest recent/event memory floor",
                memory_budget_bytes=self.memory_budget_bytes,
                minimum_memory_bytes=fixed + minimum_quota * h * size)
        cfg["long_memory_tokens_quota"] = int(quota)
        self.bank = self._bank_class(self.bank.hidden_size, self.bank.num_attention_heads, config=cfg)
        self.minimum_memory_bytes = fixed + minimum_quota * h * size
        self._capacity_resolved = True

    def _track_update(self, previous, input_tokens):
        current = self.bank.stats()
        discarded = sum(current.get(key, 0) - previous.get(key, 0)
                        for key in ("evicted_tokens", "rejected_tokens"))
        # Count arrivals which actually reduced visual rows, separately from
        # whole-record sampling/replacement/eviction. No method policy changes.
        if previous["total_tokens"] + input_tokens - discarded > current["total_tokens"]:
            self.compression_count += 1
        self.eviction_count = current.get("evicted_grids", 0)

    def _validate_temporal_sampling(self, timestamps):
        if self.method != "fluxmem_streaming":
            return
        rate = self.config.get("temporal_tokens_per_second", 2)
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
            raise ValueError("temporal_tokens_per_second must be positive and finite")
        pending_time = None if self._pending_frame is None else self._pending_frame["timestamp"]["global_timestamp"]
        previous_id = self.bank._last_time_id
        for record in timestamps:
            timestamp = record.get("global_timestamp") if isinstance(record, dict) else None
            if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or timestamp < 0:
                raise ValueError("Invalid global_timestamp")
            if pending_time is None:
                pending_time = timestamp
                continue
            time_id = int((pending_time + timestamp) / 2 * rate)
            if previous_id is not None and time_id <= previous_id:
                raise ValueError(
                    f"Unsupported Qwen2.5 sampling: temporal coordinate {time_id} does not advance "
                    f"after {previous_id} at temporal_tokens_per_second={rate:g}; "
                    f"regular FPS must be <= {2 * rate:g}")
            previous_id, pending_time = time_id, None

    def observe(self, frames, timestamps):
        self._ensure_initialized()
        if len(frames) != len(timestamps):
            raise ValueError("frames and timestamps must align")
        self._validate_temporal_sampling(timestamps)
        before = deepcopy(self.state_dict())
        try:
            for frame, timestamp in zip(frames, timestamps):
                record = self._checked_time(timestamp)
                if not self.config.get("synthetic_test_mode"):
                    from PIL import Image
                    if not isinstance(frame, Image.Image):
                        raise TypeError("Encoded tensors are only accepted in synthetic_test_mode")
                if self.method == "streamforest":
                    if self._encoder is None:
                        self._load_model()
                    features = self._encoder(frame)
                    self._resolve_streamforest_quota(features)
                    previous = self.bank.stats()
                    self.bank.ingest(features, time_id=record["global_frame_id"],
                        timestamp_seconds=record["global_timestamp"], metadata=record)
                    self._track_update(previous, features.shape[0])
                    self._groups_seen += 1
                elif self._pending_frame is None:
                    self._pending_frame = {**self._frame_payload(frame), "timestamp": record}
                else:
                    if self._encoder is None:
                        self._load_model()
                    pending = self._pending_frame
                    first = self._unpack_frame(pending)
                    try:
                        projection = self._encoder([first, frame])
                    finally:
                        if pending["kind"] == "rgb":
                            first.close()
                    stamp = (pending["timestamp"]["global_timestamp"] + record["global_timestamp"]) / 2
                    time_id = self._groups_seen
                    if self.method == "fluxmem_streaming":
                        rate = self.config.get("temporal_tokens_per_second", 2)
                        time_id = int(stamp * rate)
                    metadata = {"memory_id": f"group_{self._groups_seen}",
                                "native_timestamp_seconds": stamp,
                                "frames": [pending["timestamp"], record]}
                    extras = {"deepstack": projection["deepstack"]} if self.method != "fluxmem_streaming" else {}
                    previous = self.bank.stats()
                    self.bank.ingest(projection["embeddings"], projection["height"], projection["width"],
                                     time_id, stamp, metadata, **extras)
                    self.bank.advance()
                    self._track_update(previous, projection["embeddings"].shape[0])
                    self._pending_frame = None
                    self._groups_seen += 1
                    self._resolve_capacity(pending["pixels"].numel() * pending["pixels"].element_size())
                self._last_record = record
                self._frames_seen += 1
                self.enforce_budget()
        except Exception:
            self.load_state_dict(before)
            raise
        return self.finalize_chunk()

    def finalize_chunk(self):
        self._ensure_initialized()
        # Do not pad or discard an unmatched observed frame at chunk/day ends.
        report = self.enforce_budget()
        stats = self.bank.stats()
        return {**report, "native_stats": stats, "frames_seen_total": self._frames_seen,
                "encoded_groups": self._groups_seen, "pending_frames": int(self._pending_frame is not None),
                "eviction_count": self.eviction_count, "compression_count": self.compression_count}

    def answer(self, question, choices=None):
        self._ensure_initialized()
        if not isinstance(question, str) or not question.strip():
            raise ValueError("A nonempty public question is required")
        prompt = "Use only the observed video. Question: " + question
        if choices:
            prompt += "\n" + "\n".join(f"{chr(65 + i)}. {value}" for i, value in enumerate(choices))
            prompt += "\nAnswer with the option letter only."
        snapshot = self.bank.snapshot()
        tensors = []
        metadata = _split_tensors(snapshot, tensors)
        read = measure_state({"visual_memory_bytes": tensors, "metadata_bytes": metadata})["persistent_total_bytes"]
        count = snapshot["stats"]["total_tokens"]
        if count:
            if self._answerer is None:
                self._load_model()
            result = self._answerer(snapshot, prompt)
        else:
            result = {"prediction": None, "status": "insufficient_observation"}
        return {**result, "answer": result.get("answer", result.get("prediction")) or "",
                "query_read_bytes": read, "num_visual_tokens_used": count,
                "num_retrieved_items": len(snapshot.get("records", [])) if count else 0,
                "num_kv_positions_used": 0, "num_text_items_used": 0}

    def reset(self):
        self.protocol_state = {}
        return self.initialize()

    def _load_model(self):
        """Load only the requested local frozen checkpoint; never download."""
        import torch
        model_path = self.config.get("model_path", self.config.get("model"))
        if not model_path or not Path(model_path).is_dir():
            raise ValueError("A local model_path directory is required for native inference")
        fingerprint = local_model_fingerprint(model_path)
        if self._model_fingerprint is not None and self._model_fingerprint != fingerprint:
            raise ValueError("Local model identity changed since checkpoint restore")
        device = self.config.get("device", "cuda:0")
        dtype_name = self.config.get("dtype", "float32" if str(device).startswith("cpu") else "bfloat16")
        if dtype_name not in {"float32", "float16", "bfloat16"}:
            raise ValueError("Unsupported native inference dtype")
        dtype = getattr(torch, dtype_name)
        if self.method == "streamforest":
            package = self._bank_module.__package__
            verify = importlib.import_module(package + ".checkpoint")
            if not self.config.get("verification_file"):
                raise ValueError("StreamForest requires verification_file for its pinned local checkpoint")
            verify.verify_checkpoint(model_path, self.config["verification_file"])
            importlib.import_module(package + ".upstream").verify_source()
            bridge = importlib.import_module(package + ".bridge")
            self._model, self._tokenizer, self._processor = bridge.load_model(model_path, device=device, dtype=dtype)
            vision = self._model.get_vision_tower().config
            if (vision.hidden_size, vision.num_attention_heads) != (self.bank.hidden_size, self.bank.num_attention_heads):
                raise ValueError("Configured bank does not match the released vision architecture")
            self._encoder = lambda frame: bridge.encode_frame(self._model, self._processor, frame)
            self._answerer = lambda snapshot, prompt: bridge.generate_from_memory(
                self._model, self._tokenizer, snapshot, prompt, self.config.get("max_new_tokens", 192))
            self._model_fingerprint = fingerprint
            return
        if self.method == "fluxmem_streaming":
            path = ROOT / "fluxmem_streaming/models/qwen2-5-vl/src"
            existing = sys.modules.get("qwen2_5_vl_fluxmem")
            if existing is not None and not Path(existing.__file__).resolve().is_relative_to(path):
                raise RuntimeError("Another Qwen2.5 implementation is already loaded; use a fresh process")
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
            module = importlib.import_module("qwen2_5_vl_fluxmem")
            cls, processor_cls = module.Qwen2_5_VLForConditionalGeneration, module.Qwen2_5_VLProcessor
            name = "_streambench_fluxmem_streaming"
            _package(name, ROOT / "fluxmem_streaming/streaming")
            bridge = importlib.import_module(name + ".model_bridge")
            pixels = self.config.get("max_pixels", 256 * 28 * 28)
        else:
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
            cls, processor_cls = Qwen3VLForConditionalGeneration, AutoProcessor
            name = "_streambench_qwen3_bridge"
            _package(name, ROOT / "fluxmem_qwen3vl/streaming")
            bridge = importlib.import_module(name + ".qwen3_bridge")
            pixels = self.config.get("max_pixels", 256 * 32 * 32)
        self._model = cls.from_pretrained(model_path, local_files_only=True, torch_dtype=dtype,
            attn_implementation=self.config.get("attn_implementation", "sdpa"), device_map={"": str(device)}).eval()
        self._model.requires_grad_(False)
        self._processor = processor_cls.from_pretrained(model_path, local_files_only=True)
        self._encoder = lambda pair: bridge.encode_pair(self._model, self._processor, pair, pixels)
        self._answerer = lambda snapshot, prompt: bridge.generate_from_memory(
            self._model, self._processor, snapshot, prompt, self.config.get("max_new_tokens", 192))
        self._model_fingerprint = fingerprint


def create_method(config):
    return NativeVisualMemory(config)
