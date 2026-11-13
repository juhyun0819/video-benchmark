"""Uniform Algorithm R over complete Qwen3-VL two-frame temporal grids.

The reservoir preserves main features and all three DeepStack streams without
compression. It uses its own seeded random generator and has no FIFO/FluxMem
dependency. Sampling decisions occur before an answer can observe the bank.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Integral, Real
import random
from typing import Any

import torch


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _timestamp(value: Any, name: str = "timestamp_seconds") -> float:
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


@dataclass
class _Grid:
    embeddings: torch.Tensor
    deepstack: list[torch.Tensor]
    positions: torch.Tensor
    height: int
    width: int
    time_id: int
    timestamp_seconds: float
    metadata: dict[str, Any]

    @property
    def tokens(self) -> int:
        return self.height * self.width


class Qwen3ReservoirMemory:
    """Uniformly sample at most K complete grids from all grids observed so far.

    Supply exactly one budget and an explicit seed. ``memory_frames`` counts
    sampled frames (two per grid); ``memory_tokens`` counts main feature rows.
    Fixed grid shape makes either budget a fixed reservoir size K. DeepStack
    streams consume additional bytes, included in the tensor-byte statistics.

    After filling K slots, the nth observed grid draws j uniformly from [0, n).
    If j < K it replaces slot j; otherwise the new grid is discarded. Internal
    slots stay in sampling order. Snapshots sort records chronologically only
    when exporting them to the model, without changing future sampling.
    """

    temporal_patch_size = 2
    deepstack_levels = 3
    _STATE_VERSION = 1

    def __init__(self, *, memory_frames: int | None = None,
                 memory_tokens: int | None = None, seed: int) -> None:
        if (memory_frames is None) == (memory_tokens is None):
            raise ValueError("supply exactly one of memory_frames or memory_tokens")
        if memory_frames is not None:
            memory_frames = _integer(memory_frames, "memory_frames", minimum=2)
            if memory_frames % self.temporal_patch_size:
                raise ValueError("memory_frames must be even (two frames per temporal grid)")
        if memory_tokens is not None:
            memory_tokens = _integer(memory_tokens, "memory_tokens", minimum=1)
        self.seed = _integer(seed, "seed")
        self.memory_frames = memory_frames
        self.memory_tokens = memory_tokens
        self._rng = random.Random(self.seed)
        self._slots: list[_Grid] = []
        self._shape: tuple[int, int, int] | None = None
        self._dtype: torch.dtype | None = None
        self._device: torch.device | None = None
        # These describe the stream, including arrivals rejected by sampling.
        self._last_time_id: int | None = None
        self._last_timestamp: float | None = None
        self._last_response_time_id: int | None = None
        self._last_observation: dict[str, Any] | None = None
        self._total_ingested = self._total_encoded_tokens = 0
        self._evicted_grids = self._evicted_tokens = 0
        self._rejected_grids = self._rejected_tokens = 0
        self._response_count = 0
        self._pending = False

    def _capacity(self, tokens_per_grid: int | None = None) -> int | None:
        if self.memory_frames is not None:
            return self.memory_frames // self.temporal_patch_size
        if tokens_per_grid is None and self._shape is not None:
            tokens_per_grid = self._shape[0] * self._shape[1]
        return None if tokens_per_grid is None else self.memory_tokens // tokens_per_grid

    def _validate_input(self, embeddings, height, width, time_id,
                        timestamp_seconds, metadata, deepstack) -> None:
        if self._pending:
            raise RuntimeError("advance() must follow the previous ingest()")
        height = _integer(height, "height", minimum=1)
        width = _integer(width, "width", minimum=1)
        if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
            raise ValueError("embeddings must be a [height * width, hidden_dim] tensor")
        if embeddings.shape[0] != height * width or embeddings.shape[1] < 1:
            raise ValueError("embeddings shape must match the grid and a positive hidden dimension")
        if not embeddings.is_floating_point() or not bool(torch.isfinite(embeddings).all()):
            raise ValueError("embeddings must contain finite floating point values")
        if not isinstance(deepstack, (list, tuple)) or len(deepstack) != self.deepstack_levels:
            raise ValueError("deepstack must contain exactly three aligned tensors")
        for level, tensor in enumerate(deepstack):
            if not isinstance(tensor, torch.Tensor) or tensor.shape != embeddings.shape:
                raise ValueError(f"deepstack[{level}] must match the embeddings shape")
            if tensor.dtype != embeddings.dtype or tensor.device != embeddings.device:
                raise ValueError(f"deepstack[{level}] must match the embeddings dtype and device")
            if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"deepstack[{level}] must contain finite floating point values")
        time_id = _integer(time_id, "time_id")
        timestamp_seconds = _timestamp(timestamp_seconds)
        if self._last_time_id is not None and time_id <= self._last_time_id:
            raise ValueError("time_id must strictly increase, including rejected arrivals")
        if self._last_timestamp is not None and timestamp_seconds <= self._last_timestamp:
            raise ValueError("timestamp_seconds must strictly increase, including rejected arrivals")
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a dictionary")
        if self._shape is not None:
            if (height, width, embeddings.shape[1]) != self._shape:
                raise ValueError("all grids must share height, width and hidden dimension")
            if embeddings.dtype != self._dtype or embeddings.device != self._device:
                raise ValueError("all grids must share dtype and device")
        if self._capacity(height * width) < 1:
            raise ValueError("memory_tokens must fit at least one complete incoming temporal grid")

    @staticmethod
    def _owned_grid(embeddings, height, width, time_id, timestamp_seconds,
                    metadata, deepstack) -> _Grid:
        height, width, time_id = int(height), int(width), int(time_id)
        rows = torch.arange(height, device=embeddings.device).repeat_interleave(width)
        columns = torch.arange(width, device=embeddings.device).repeat(height)
        positions = torch.stack((torch.full_like(rows, time_id), rows, columns))
        return _Grid(embeddings.detach().clone(),
                     [tensor.detach().clone() for tensor in deepstack], positions,
                     height, width, time_id, float(timestamp_seconds), deepcopy(metadata))

    @torch.no_grad()
    def ingest(self, embeddings: torch.Tensor, height: int, width: int,
               time_id: int, timestamp_seconds: float, metadata: dict[str, Any],
               deepstack: list[torch.Tensor]) -> dict[str, Any]:
        """Observe one grid, then fill, replace, or reject it using Algorithm R."""
        self._validate_input(embeddings, height, width, time_id,
                             timestamp_seconds, metadata, deepstack)
        # Finish validation and copying before drawing from the local RNG.
        grid = self._owned_grid(embeddings, height, width, time_id,
                                timestamp_seconds, metadata, deepstack)
        capacity = self._capacity(grid.tokens)
        seen = self._total_ingested + 1
        draw = None
        removed = None
        if len(self._slots) < capacity:
            slot = len(self._slots)
            self._slots.append(grid)
        else:
            draw = self._rng.randrange(seen)
            slot = draw if draw < capacity else None
            if slot is not None:
                removed = self._slots[slot]
                self._slots[slot] = grid
                self._evicted_grids += 1
                self._evicted_tokens += removed.tokens
            else:
                self._rejected_grids += 1
                self._rejected_tokens += grid.tokens
        self._shape = (grid.height, grid.width, int(embeddings.shape[1]))
        self._dtype, self._device = embeddings.dtype, embeddings.device
        self._last_time_id, self._last_timestamp = grid.time_id, grid.timestamp_seconds
        self._total_ingested = seen
        self._total_encoded_tokens += grid.tokens
        self._pending = True
        self._last_observation = {
            "time_id": grid.time_id, "timestamp_seconds": grid.timestamp_seconds,
            "policy": "reservoir", "triggered": False, "seen_grids": seen,
            "accepted": slot is not None, "reservoir_slot": slot, "draw_index": draw,
            "replaced_time_id": None if removed is None else removed.time_id,
        }
        return deepcopy(self._last_observation)

    def advance(self) -> dict[str, Any]:
        """Close this arrival; reservoir selection already enforced the budget."""
        if not self._pending:
            raise RuntimeError("advance() requires a preceding ingest()")
        self._pending = False
        return self.stats()

    def should_respond(self) -> bool:
        return False

    def mark_response(self) -> None:
        if self._last_time_id is None:
            raise RuntimeError("cannot mark a response before ingesting a grid")
        self._last_response_time_id = self._last_time_id
        self._response_count += 1

    @property
    def last_observation(self) -> dict[str, Any] | None:
        return deepcopy(self._last_observation)

    def stats(self) -> dict[str, Any]:
        total_tokens = sum(grid.tokens for grid in self._slots)
        tensor_bytes = cpu_bytes = deepstack_bytes = 0
        for grid in self._slots:
            for tensor in (grid.embeddings, grid.positions, *grid.deepstack):
                size = tensor.numel() * tensor.element_size()
                tensor_bytes += size
                if tensor.device.type == "cpu":
                    cpu_bytes += size
            deepstack_bytes += sum(t.numel() * t.element_size() for t in grid.deepstack)
        return {
            "tier_grids": {"reservoir": len(self._slots)},
            "tier_tokens": {"reservoir": total_tokens}, "total_tokens": total_tokens,
            "retained_frames": len(self._slots) * self.temporal_patch_size,
            "total_ingested": self._total_ingested,
            "total_encoded_tokens": self._total_encoded_tokens,
            "response_count": self._response_count,
            "evicted_grids": self._evicted_grids, "evicted_tokens": self._evicted_tokens,
            "rejected_grids": self._rejected_grids, "rejected_tokens": self._rejected_tokens,
            "discarded_grids": self._evicted_grids + self._rejected_grids,
            "discarded_tokens": self._evicted_tokens + self._rejected_tokens,
            "retained_tensor_bytes": tensor_bytes, "retained_cpu_bytes": cpu_bytes,
            "deepstack_tensor_bytes": deepstack_bytes,
            "pending_advance": self._pending,
            "budget": {"memory_frames": self.memory_frames,
                       "memory_tokens": self.memory_tokens,
                       "capacity_grids": self._capacity()},
            "seed": self.seed,
        }

    def snapshot(self) -> dict[str, Any]:
        """Export detached chronological records without reordering sample slots."""
        grids = sorted(self._slots, key=lambda grid: grid.time_id)
        if grids:
            embeddings = torch.cat([g.embeddings for g in grids]).detach().clone()
            positions = torch.cat([g.positions for g in grids], dim=1).detach().clone()
            deepstack = [torch.cat([g.deepstack[level] for g in grids]).detach().clone()
                         for level in range(self.deepstack_levels)]
        else:
            embeddings = torch.empty((0, 0))
            positions = torch.empty((3, 0), dtype=torch.long)
            deepstack = [torch.empty((0, 0)) for _ in range(self.deepstack_levels)]
        records = [{"time_id": g.time_id, "timestamp_seconds": g.timestamp_seconds,
                    "height": g.height, "width": g.width, "tier": "reservoir",
                    "input_tokens": g.tokens, "retained_tokens": g.tokens,
                    "metadata": deepcopy(g.metadata)} for g in grids]
        return {"embeddings": embeddings, "positions": positions,
                "deepstack": deepstack, "records": records, "stats": self.stats(),
                "memory_truncated": bool(self._evicted_grids or self._rejected_grids)}

    def state_dict(self) -> dict[str, Any]:
        """Save slots, RNG and stream state; load only trusted local checkpoints."""
        slots = [{"embeddings": g.embeddings.detach().cpu().clone(),
                  "deepstack": [t.detach().cpu().clone() for t in g.deepstack],
                  "height": g.height, "width": g.width, "time_id": g.time_id,
                  "timestamp_seconds": g.timestamp_seconds,
                  "metadata": deepcopy(g.metadata)} for g in self._slots]
        return {
            "kind": "qwen3_reservoir", "version": self._STATE_VERSION,
            "config": {"memory_frames": self.memory_frames,
                       "memory_tokens": self.memory_tokens, "seed": self.seed},
            "slots": slots, "rng_state": deepcopy(self._rng.getstate()),
            "total_ingested": self._total_ingested,
            "total_encoded_tokens": self._total_encoded_tokens,
            "evicted_grids": self._evicted_grids, "evicted_tokens": self._evicted_tokens,
            "rejected_grids": self._rejected_grids, "rejected_tokens": self._rejected_tokens,
            "response_count": self._response_count,
            "last_response_time_id": self._last_response_time_id,
            "last_time_id": self._last_time_id, "last_timestamp": self._last_timestamp,
            "last_observation": deepcopy(self._last_observation),
            "pending_advance": self._pending,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any], *,
                        device: str | torch.device = "cpu") -> "Qwen3ReservoirMemory":
        """Restore slots directly, without replaying ingestion or drawing samples."""
        if (not isinstance(state, dict) or state.get("kind") != "qwen3_reservoir"
                or type(state.get("version")) is not int
                or state["version"] != cls._STATE_VERSION):
            raise ValueError("unsupported Qwen3 reservoir memory checkpoint")
        config, records = state.get("config"), state.get("slots")
        if not isinstance(config, dict) or not isinstance(records, list):
            raise ValueError("checkpoint must contain config and slots")
        bank = cls(memory_frames=config.get("memory_frames"),
                   memory_tokens=config.get("memory_tokens"), seed=config.get("seed"))
        names = ("total_ingested", "total_encoded_tokens", "evicted_grids", "evicted_tokens",
                 "rejected_grids", "rejected_tokens", "response_count")
        counts = {name: _integer(state.get(name), name) for name in names}
        pending = state.get("pending_advance")
        if not isinstance(pending, bool):
            raise ValueError("checkpoint pending_advance must be a boolean")
        try:
            bank._rng.setstate(deepcopy(state.get("rng_state")))
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError("checkpoint has invalid local RNG state") from exc
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("checkpoint slots must be dictionaries")
            embeddings, levels = record.get("embeddings"), record.get("deepstack")
            if (not isinstance(embeddings, torch.Tensor) or not isinstance(levels, (list, tuple))
                    or len(levels) != cls.deepstack_levels
                    or any(not isinstance(t, torch.Tensor) for t in levels)):
                raise ValueError("checkpoint slot is missing its aligned visual tensors")
            inputs = {"embeddings": embeddings.to(device), "height": record.get("height"),
                      "width": record.get("width"), "time_id": record.get("time_id"),
                      "timestamp_seconds": record.get("timestamp_seconds"),
                      "metadata": record.get("metadata"),
                      "deepstack": [tensor.to(device) for tensor in levels]}
            # Slots are intentionally unordered. Only fixed shape/dtype/device
            # are established here; chronological stream state is set below.
            bank._validate_input(**inputs)
            grid = bank._owned_grid(**inputs)
            bank._slots.append(grid)
            bank._shape = (grid.height, grid.width, int(grid.embeddings.shape[1]))
            bank._dtype, bank._device = grid.embeddings.dtype, grid.embeddings.device

        discarded = counts["evicted_grids"] + counts["rejected_grids"]
        if counts["total_ingested"] != len(records) + discarded:
            raise ValueError("checkpoint grid counters are inconsistent")
        last_id, last_time = state.get("last_time_id"), state.get("last_timestamp")
        observation = state.get("last_observation")
        response_id = state.get("last_response_time_id")
        if records:
            capacity = bank._capacity()
            if len(records) != min(capacity, counts["total_ingested"]):
                raise ValueError("checkpoint retained slots do not match the budget and stream count")
            tokens = bank._slots[0].tokens
            if (counts["total_encoded_tokens"] != counts["total_ingested"] * tokens
                    or counts["evicted_tokens"] != counts["evicted_grids"] * tokens
                    or counts["rejected_tokens"] != counts["rejected_grids"] * tokens):
                raise ValueError("checkpoint token counters are inconsistent")
            ordered = sorted(bank._slots, key=lambda grid: grid.time_id)
            if any(a.time_id >= b.time_id or a.timestamp_seconds >= b.timestamp_seconds
                   for a, b in zip(ordered, ordered[1:])):
                raise ValueError("checkpoint retained grids have invalid chronological order")
            last_id = _integer(last_id, "last_time_id")
            last_time = _timestamp(last_time, "last_timestamp")
            if last_id < ordered[-1].time_id or last_time < ordered[-1].timestamp_seconds:
                raise ValueError("checkpoint last observed time precedes a retained grid")
            bank._validate_observation(observation, counts["total_ingested"], last_id, last_time)
        elif (any(counts.values()) or pending or response_id is not None
              or last_id is not None or last_time is not None or observation is not None):
            raise ValueError("empty checkpoint has nonempty stream state")
        if response_id is not None:
            response_id = _integer(response_id, "last_response_time_id")
        if ((counts["response_count"] == 0) != (response_id is None)
                or (response_id is not None and response_id > last_id)):
            raise ValueError("checkpoint response state is inconsistent")
        for name, value in counts.items():
            setattr(bank, "_" + name, value)
        bank._last_time_id, bank._last_timestamp = last_id, last_time
        bank._last_response_time_id = response_id
        bank._last_observation = deepcopy(observation)
        bank._pending = pending
        return bank

    def _validate_observation(self, observation, seen, last_id, last_time) -> None:
        if (not isinstance(observation, dict) or observation.get("policy") != "reservoir"
                or observation.get("triggered") is not False
                or observation.get("time_id") != last_id
                or observation.get("timestamp_seconds") != last_time
                or observation.get("seen_grids") != seen
                or not isinstance(observation.get("accepted"), bool)):
            raise ValueError("checkpoint last observation does not match stream state")
        capacity = self._capacity()
        slot, draw = observation.get("reservoir_slot"), observation.get("draw_index")
        removed = observation.get("replaced_time_id")
        if seen <= capacity:
            if draw is not None or removed is not None or observation["accepted"] is not True:
                raise ValueError("checkpoint fill observation is inconsistent")
        else:
            draw = _integer(draw, "last observation draw_index")
            if draw >= seen or observation["accepted"] != (draw < capacity):
                raise ValueError("checkpoint sampling draw is inconsistent")
        if observation["accepted"]:
            slot = _integer(slot, "last observation reservoir_slot")
            if (slot >= len(self._slots) or self._slots[slot].time_id != last_id
                    or self._slots[slot].timestamp_seconds != last_time
                    or (draw is not None and slot != draw)):
                raise ValueError("checkpoint accepted observation does not match its slot")
            if seen > capacity:
                removed = _integer(removed, "last observation replaced_time_id")
                if removed >= last_id:
                    raise ValueError("checkpoint replaced grid must precede its replacement")
        elif (slot is not None or removed is not None
              or any(g.time_id >= last_id or g.timestamp_seconds >= last_time for g in self._slots)):
            raise ValueError("checkpoint rejected observation is inconsistent")
