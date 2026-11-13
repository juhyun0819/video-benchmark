"""Recent-frame SimpleStream adaptation using the pinned Qwen2.5-VL-7B.

This is the explicit 7B adaptation, not a released SimpleStream 8B/4B model.
Observation retains only query-independent visual embeddings. Every answer
starts a fresh backend generation; decoder KV is never continuation state.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, is_dataclass
import math

from baselines.streambench.method import StreamingMemoryMethod


MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
PUBLIC_TIME = {"stream_id", "day", "global_frame_id", "source_timestamp",
               "global_timestamp", "source_id", "source_frame_index"}


def _check_projection(features, grid_thw):
    import torch
    if (not isinstance(features, torch.Tensor) or features.device.type != "cpu"
            or features.dtype != torch.bfloat16 or features.ndim != 2
            or features.shape[0] < 1 or features.shape[1] != 3584
            or not torch.isfinite(features).all()):
        raise ValueError("Encoder features must be finite CPU BF16 [rows,3584]")
    if (not isinstance(grid_thw, torch.Tensor) or grid_thw.device.type != "cpu"
            or grid_thw.dtype != torch.int64 or tuple(grid_thw.shape) not in ((3,), (1, 3))):
        raise ValueError("Encoder grid must be CPU int64 [3] or [1,3]")
    grid = grid_thw.reshape(3)
    if (grid[0].item() != 1 or torch.any(grid <= 0)
            or grid[1].item() % 2 or grid[2].item() % 2
            or grid.prod().item() != features.shape[0] * 4):
        raise ValueError("Encoder grid must describe one spatially merged image")


def _partition_features(value, features):
    """Expose live feature allocations and all remaining state for accounting."""
    import torch
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.bfloat16 and value.ndim == 2 and value.shape[-1] == 3584:
            features.append(value)
            return None
        return value
    if isinstance(value, dict):
        return {key: _partition_features(item, features) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_partition_features(item, features) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _partition_features(getattr(value, field.name), features)
                for field in fields(value)}
    return value


class SimpleStreamMethod(StreamingMemoryMethod):
    def __init__(self, config, *, backend=None):
        super().__init__(config["memory_budget_bytes"])
        self.config = deepcopy(config)
        self.config.setdefault("method", "simplestream")
        if self.config["method"] != "simplestream":
            raise ValueError("SimpleStreamMethod requires method='simplestream'")
        self.recent_frames = config.get("recent_frames", 4)
        if type(self.recent_frames) is not int or self.recent_frames < 1:
            raise ValueError("recent_frames must be a positive integer")
        self.synthetic = config.get("synthetic_test_mode", False)
        if type(self.synthetic) is not bool:
            raise ValueError("synthetic_test_mode must be boolean")
        if backend is not None and not self.synthetic:
            raise ValueError("Injected backends require explicit synthetic_test_mode")
        if self.synthetic and (backend is None or not isinstance(config.get("encoder_id"), str)
                               or not config["encoder_id"].strip()):
            raise ValueError("Synthetic tests require an explicit backend and encoder_id")
        if config.get("model_id", MODEL_ID) != MODEL_ID or config.get("model_revision", MODEL_REVISION) != MODEL_REVISION:
            raise ValueError("This adapter supports only the pinned Qwen2.5-VL-7B adaptation")
        if config.get("dtype", "bfloat16") != "bfloat16":
            raise ValueError("The pinned visual representation requires bfloat16")
        self._backend = backend
        self._memory = None
        self._identity = None
        self._frames_seen = 0
        self._last_record = None

    def initialize(self):
        from baselines.simplestream.memory import RecentWindow
        if self._backend is None:
            from baselines.simplestream.backend import Qwen25Backend
            if not self.config.get("model_path") or not self.config.get("ledger_path"):
                raise ValueError("Set model_path and ledger_path for the verified local checkpoint")
            self._backend = Qwen25Backend(
                self.config["model_path"], self.config["ledger_path"],
                device=self.config.get("device", "cuda:2"),
                max_new_tokens=self.config.get("max_new_tokens", 128),
                attn_implementation=self.config.get("attn_implementation", "sdpa"),
            )
        identity = {"config": self.config, "backend": deepcopy(self._backend.identity),
                    "variant": "simplestream_qwen25_vl_7b_adaptation",
                    "recent_frames": self.recent_frames,
                    "persistent_decoder_kv": False}
        if self._identity is not None and identity != self._identity:
            raise ValueError("SimpleStream backend identity changed after initialization")
        self._identity = identity
        if self._memory is None:
            self._memory = RecentWindow(self.recent_frames)
        self.enforce_budget()

    def _ensure_initialized(self):
        if self._memory is None:
            self.initialize()

    @property
    def eviction_count(self):
        return max(0, self._frames_seen - self.recent_frames)

    @property
    def compression_count(self):
        return 0

    def persistent_state(self):
        if self._memory is None:
            raise RuntimeError("Initialize the adapter before measuring its state")
        features = []
        metadata = _partition_features(self.state_dict(), features)
        return {"visual_memory_bytes": features, "metadata_bytes": metadata}

    def state_dict(self):
        self._ensure_initialized()
        return {"version": 1, "identity": self._identity, "memory": self._memory.state_dict(copy=False),
                "frames_seen": self._frames_seen, "last_record": self._last_record}

    def load_state_dict(self, state):
        from baselines.simplestream.memory import RecentWindow
        self._ensure_initialized()
        if state.get("version") != 1 or state.get("identity") != self._identity:
            raise ValueError("SimpleStream checkpoint model/processor/window/config identity differs")
        memory = RecentWindow.from_state_dict(state["memory"])
        seen = state["frames_seen"]
        last = state["last_record"]
        if type(seen) is not int or seen < 0 or ((seen == 0) != (last is None)):
            raise ValueError("Invalid SimpleStream checkpoint continuation cursor")
        if last is not None and (set(last) != PUBLIC_TIME or last["global_frame_id"] != seen - 1):
            raise ValueError("SimpleStream checkpoint frame cursor mismatch")
        if state["memory"].get("window_frames") != self.recent_frames or state["memory"].get("observed_count") != seen:
            raise ValueError("SimpleStream checkpoint memory window/cursor mismatch")
        retained = memory.state_dict(copy=False)["frames"]
        if [frame["frame_id"] for frame in retained] != list(range(max(0, seen - self.recent_frames), seen)):
            raise ValueError("SimpleStream checkpoint does not contain the complete recent-frame window")
        if retained and retained[-1]["metadata"] != last:
            raise ValueError("SimpleStream checkpoint frame provenance/cursor mismatch")
        for frame in retained:
            _check_projection(frame["features"], frame["grid_thw"])
        self._memory = memory
        self._frames_seen = seen
        self._last_record = deepcopy(last)

    def _validate_records(self, timestamps):
        previous = self._last_record
        for record in timestamps:
            if not isinstance(record, dict) or set(record) != PUBLIC_TIME:
                raise ValueError("Observe accepts only the seven public frame timestamp fields")
            for name in ("global_frame_id", "source_frame_index"):
                if type(record[name]) is not int or record[name] < 0:
                    raise ValueError(f"{name} must be a nonnegative integer")
            if type(record["day"]) is not int or record["day"] < 1:
                raise ValueError("day must be a positive integer")
            for name in ("stream_id", "source_id"):
                if not isinstance(record[name], str) or not record[name]:
                    raise ValueError(f"{name} must be nonempty text")
            for name in ("global_timestamp", "source_timestamp"):
                value = record[name]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"{name} must be finite and nonnegative")
            if previous is None:
                if record["global_frame_id"] != 0 or record["day"] != 1:
                    raise ValueError("An independent stream must begin at frame zero on day one")
            elif (record["stream_id"] != previous["stream_id"]
                  or record["global_frame_id"] != previous["global_frame_id"] + 1
                  or record["global_timestamp"] <= previous["global_timestamp"]
                  or record["day"] not in (previous["day"], previous["day"] + 1)):
                raise ValueError("Observations must be consecutive and chronological in one stream")
            previous = record

    def observe(self, frames, timestamps):
        from baselines.simplestream.memory import EncodedFrame
        self._ensure_initialized()
        if not frames or len(frames) != len(timestamps):
            raise ValueError("Observation requires matching nonempty frames and timestamps")
        self._validate_records(timestamps)
        if not self.synthetic:
            from PIL import Image
            if any(not isinstance(frame, Image.Image) for frame in frames):
                raise TypeError("Production observation requires raw PIL frames")
        before = deepcopy(self.state_dict())
        try:
            encoded = self._backend.encode_frames(frames)
            if len(encoded) != len(frames):
                raise ValueError("The encoder must return exactly one feature/grid pair per frame")
            for (features, grid_thw), record in zip(encoded, timestamps):
                _check_projection(features, grid_thw)
                self._memory.observe(EncodedFrame(
                    frame_id=record["global_frame_id"], timestamp=record["global_timestamp"],
                    features=features, grid_thw=grid_thw, metadata=deepcopy(record),
                ))
                self._frames_seen += 1
                self._last_record = deepcopy(record)
                self.enforce_budget()
        except BaseException:
            self.load_state_dict(before)
            raise
        return self.enforce_budget()

    def finalize_chunk(self):
        self._ensure_initialized()
        return self.enforce_budget()

    def answer(self, question, choices=None):
        self._ensure_initialized()
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be nonempty text")
        choices = [] if choices is None else choices
        if not isinstance(choices, list) or any(not isinstance(value, str) for value in choices):
            raise ValueError("choices must be a list of strings")
        if self._last_record is None:
            return {"answer": "", "status": "insufficient_observation", "query_read_bytes": 0,
                    "num_retrieved_items": 0, "num_visual_tokens_used": 0, "retrieved_frame_ids": []}
        # The query receives owned features, so neither successful generation nor
        # an exceptional backend can mutate the continuing observation state.
        selected = self._memory.selected(self._last_record["global_timestamp"])
        response = self._backend.answer(selected, question, choices)
        if not isinstance(response, dict) or not isinstance(response.get("answer"), str):
            raise ValueError("SimpleStream backend must return answer text")
        frame_ids = [frame.frame_id for frame in selected]
        if response.get("used_frame_ids", frame_ids) != frame_ids:
            raise ValueError("Backend did not use the selected recent-frame window")
        visual_tokens = sum(frame.features.shape[0] for frame in selected)
        if response.get("visual_tokens", visual_tokens) != visual_tokens:
            raise ValueError("Backend visual token count differs from the selected window")
        read_bytes = sum(frame.features.numel() * frame.features.element_size()
                         + frame.grid_thw.numel() * frame.grid_thw.element_size() for frame in selected)
        return {**response, "query_read_bytes": read_bytes, "query_read_bytes_is_upper_bound": False,
                "read_accounting": "selected visual embeddings and image grid payload; excludes transient QA KV",
                "num_retrieved_items": len(selected), "num_visual_tokens_used": visual_tokens,
                "retrieved_frame_ids": frame_ids,
                "retrieved_items": [{"frame_id": frame.frame_id, "timestamp": frame.timestamp} for frame in selected]}

    def reset(self):
        self.protocol_state = {}
        self._memory = None
        self._frames_seen = 0
        self._last_record = None
        self.initialize()


def create_method(config):
    return SimpleStreamMethod(config)
