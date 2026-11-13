"""Uncompressed, strictly bounded FIFO memory for Qwen3-VL temporal grids.

The bank stores each two-frame temporal grid as an indivisible record. Main
features, all three DeepStack streams, positions, and provenance move together.
There is no FluxMem dependency, feature selection, merging, or novelty trigger.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any

import torch


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


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


class Qwen3FIFOMemory:
    """Keep the newest complete grids under exactly one configured budget.

    ``memory_frames`` counts sampled raw frames, two per temporal grid.
    ``memory_tokens`` counts rows of the main projected features; aligned
    DeepStack rows use additional bytes and are included in byte accounting.
    A token budget's unused remainder never permits a partial grid.

    ``ingest`` owns detached copies and evicts immediately, so snapshots taken
    before ``advance`` still satisfy the budget. ``advance`` only closes the
    arrival phase expected by the shared causal scheduler.
    """

    temporal_patch_size = 2
    deepstack_levels = 3
    _STATE_VERSION = 1

    def __init__(self, *, memory_frames: int | None = None,
                 memory_tokens: int | None = None) -> None:
        if (memory_frames is None) == (memory_tokens is None):
            raise ValueError("supply exactly one of memory_frames or memory_tokens")
        if memory_frames is not None:
            memory_frames = _integer(memory_frames, "memory_frames", minimum=2)
            if memory_frames % self.temporal_patch_size:
                raise ValueError("memory_frames must be even (two frames per temporal grid)")
        if memory_tokens is not None:
            memory_tokens = _integer(memory_tokens, "memory_tokens", minimum=1)
        self.memory_frames = memory_frames
        self.memory_tokens = memory_tokens
        self._grids: deque[_Grid] = deque()
        self._shape: tuple[int, int, int] | None = None
        self._dtype: torch.dtype | None = None
        self._device: torch.device | None = None
        self._last_time_id: int | None = None
        self._last_timestamp: float | None = None
        self._last_response_time_id: int | None = None
        self._total_ingested = 0
        self._total_encoded_tokens = 0
        self._evicted_grids = 0
        self._evicted_tokens = 0
        self._response_count = 0
        self._pending = False
        self._last_observation: dict[str, Any] | None = None

    def _capacity(self, tokens_per_grid: int | None = None) -> int | None:
        if self.memory_frames is not None:
            return self.memory_frames // self.temporal_patch_size
        if tokens_per_grid is None and self._shape is not None:
            tokens_per_grid = self._shape[0] * self._shape[1]
        if tokens_per_grid is None:
            return None
        return self.memory_tokens // tokens_per_grid

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
        if self._last_time_id is not None and time_id <= self._last_time_id:
            raise ValueError("time_id must strictly increase")
        if (isinstance(timestamp_seconds, bool) or not isinstance(timestamp_seconds, Real)
                or not math.isfinite(timestamp_seconds) or timestamp_seconds < 0):
            raise ValueError("timestamp_seconds must be finite and nonnegative")
        if self._last_timestamp is not None and timestamp_seconds <= self._last_timestamp:
            raise ValueError("timestamp_seconds must strictly increase")
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a dictionary")
        if self._shape is not None:
            if (height, width, embeddings.shape[1]) != self._shape:
                raise ValueError("all grids must share height, width and hidden dimension")
            if embeddings.dtype != self._dtype or embeddings.device != self._device:
                raise ValueError("all grids must share dtype and device")
        if self._capacity(height * width) < 1:
            raise ValueError("memory_tokens must fit at least one complete incoming temporal grid")

    @torch.no_grad()
    def ingest(self, embeddings: torch.Tensor, height: int, width: int,
               time_id: int, timestamp_seconds: float, metadata: dict[str, Any],
               deepstack: list[torch.Tensor]) -> dict[str, Any]:
        """Append one complete grid and enforce the budget before returning."""
        self._validate_input(embeddings, height, width, time_id,
                             timestamp_seconds, metadata, deepstack)
        height, width, time_id = int(height), int(width), int(time_id)
        # Prepare every owned object before mutating the bank.
        owned_embeddings = embeddings.detach().clone()
        owned_deepstack = [tensor.detach().clone() for tensor in deepstack]
        owned_metadata = deepcopy(metadata)
        rows = torch.arange(height, device=embeddings.device).repeat_interleave(width)
        columns = torch.arange(width, device=embeddings.device).repeat(height)
        positions = torch.stack((torch.full_like(rows, time_id), rows, columns))
        grid = _Grid(owned_embeddings, owned_deepstack, positions, height, width,
                     time_id, float(timestamp_seconds), owned_metadata)
        next_grids = deque(self._grids)
        next_grids.append(grid)
        removed: list[_Grid] = []
        while len(next_grids) > self._capacity(grid.tokens):
            removed.append(next_grids.popleft())

        self._grids = next_grids
        self._shape = (height, width, int(embeddings.shape[1]))
        self._dtype, self._device = embeddings.dtype, embeddings.device
        self._last_time_id, self._last_timestamp = time_id, float(timestamp_seconds)
        self._total_ingested += 1
        self._total_encoded_tokens += grid.tokens
        self._evicted_grids += len(removed)
        self._evicted_tokens += sum(old.tokens for old in removed)
        self._pending = True
        self._last_observation = {"time_id": time_id,
                                  "timestamp_seconds": float(timestamp_seconds),
                                  "policy": "fifo", "triggered": False}
        return deepcopy(self._last_observation)

    def advance(self) -> dict[str, Any]:
        """Close the arrival phase; the budget was already enforced by ingest."""
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
        total_tokens = sum(grid.tokens for grid in self._grids)
        tensor_bytes = cpu_bytes = deepstack_bytes = 0
        for grid in self._grids:
            for tensor in (grid.embeddings, grid.positions, *grid.deepstack):
                size = tensor.numel() * tensor.element_size()
                tensor_bytes += size
                if tensor.device.type == "cpu":
                    cpu_bytes += size
            deepstack_bytes += sum(t.numel() * t.element_size() for t in grid.deepstack)
        return {
            "tier_grids": {"fifo": len(self._grids)},
            "tier_tokens": {"fifo": total_tokens},
            "total_tokens": total_tokens,
            "retained_frames": len(self._grids) * self.temporal_patch_size,
            "total_ingested": self._total_ingested,
            "total_encoded_tokens": self._total_encoded_tokens,
            "response_count": self._response_count,
            "evicted_grids": self._evicted_grids,
            "evicted_tokens": self._evicted_tokens,
            "retained_tensor_bytes": tensor_bytes,
            "retained_cpu_bytes": cpu_bytes,
            "deepstack_tensor_bytes": deepstack_bytes,
            "pending_advance": self._pending,
            "budget": {"memory_frames": self.memory_frames,
                       "memory_tokens": self.memory_tokens,
                       "capacity_grids": self._capacity()},
        }

    def snapshot(self) -> dict[str, Any]:
        """Return isolated aligned features and chronological audit records."""
        if self._grids:
            embeddings = torch.cat([g.embeddings for g in self._grids]).detach().clone()
            positions = torch.cat([g.positions for g in self._grids], dim=1).detach().clone()
            deepstack = [torch.cat([g.deepstack[level] for g in self._grids]).detach().clone()
                         for level in range(self.deepstack_levels)]
        else:
            embeddings = torch.empty((0, 0))
            positions = torch.empty((3, 0), dtype=torch.long)
            deepstack = [torch.empty((0, 0)) for _ in range(self.deepstack_levels)]
        records = [{"time_id": g.time_id, "timestamp_seconds": g.timestamp_seconds,
                    "height": g.height, "width": g.width, "tier": "fifo",
                    "input_tokens": g.tokens, "retained_tokens": g.tokens,
                    "metadata": deepcopy(g.metadata)} for g in self._grids]
        return {"embeddings": embeddings, "positions": positions,
                "deepstack": deepstack, "records": records, "stats": self.stats()}

    def state_dict(self) -> dict[str, Any]:
        """Make a detached CPU checkpoint; load only trusted local checkpoints."""
        grids = [{"embeddings": g.embeddings.detach().cpu().clone(),
                  "deepstack": [t.detach().cpu().clone() for t in g.deepstack],
                  "height": g.height, "width": g.width, "time_id": g.time_id,
                  "timestamp_seconds": g.timestamp_seconds,
                  "metadata": deepcopy(g.metadata)} for g in self._grids]
        return {
            "kind": "qwen3_fifo", "version": self._STATE_VERSION,
            "config": {"memory_frames": self.memory_frames, "memory_tokens": self.memory_tokens},
            "grids": grids,
            "total_ingested": self._total_ingested,
            "total_encoded_tokens": self._total_encoded_tokens,
            "evicted_grids": self._evicted_grids,
            "evicted_tokens": self._evicted_tokens,
            "response_count": self._response_count,
            "last_response_time_id": self._last_response_time_id,
            "pending_advance": self._pending,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any], *,
                        device: str | torch.device = "cpu") -> "Qwen3FIFOMemory":
        """Restore this bank's checkpoint, validating budget and stream invariants."""
        if (not isinstance(state, dict) or state.get("kind") != "qwen3_fifo"
                or state.get("version") != cls._STATE_VERSION):
            raise ValueError("unsupported Qwen3 FIFO memory checkpoint")
        config, records = state.get("config"), state.get("grids")
        if not isinstance(config, dict) or not isinstance(records, list):
            raise ValueError("checkpoint must contain config and grids")
        bank = cls(memory_frames=config.get("memory_frames"),
                   memory_tokens=config.get("memory_tokens"))
        names = ("total_ingested", "total_encoded_tokens", "evicted_grids",
                 "evicted_tokens", "response_count")
        counts = {name: _integer(state.get(name), name) for name in names}
        pending = state.get("pending_advance")
        if not isinstance(pending, bool):
            raise ValueError("checkpoint pending_advance must be a boolean")
        response_id = state.get("last_response_time_id")
        if response_id is not None:
            response_id = _integer(response_id, "last_response_time_id")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("checkpoint grids must be dictionaries")
            embeddings, levels = record.get("embeddings"), record.get("deepstack")
            if (not isinstance(embeddings, torch.Tensor) or not isinstance(levels, (list, tuple))
                    or len(levels) != cls.deepstack_levels
                    or any(not isinstance(t, torch.Tensor) for t in levels)):
                raise ValueError("checkpoint grid is missing its aligned visual tensors")
            bank.ingest(embeddings.to(device), record.get("height"), record.get("width"),
                        record.get("time_id"), record.get("timestamp_seconds"),
                        record.get("metadata"), [tensor.to(device) for tensor in levels])
            bank.advance()
        if bank._evicted_grids:
            raise ValueError("checkpoint grids exceed its configured budget")
        if counts["total_ingested"] != len(records) + counts["evicted_grids"]:
            raise ValueError("checkpoint grid counters are inconsistent")
        if records:
            tokens = bank._grids[-1].tokens
            if (counts["total_encoded_tokens"] != counts["total_ingested"] * tokens
                    or counts["evicted_tokens"] != counts["evicted_grids"] * tokens):
                raise ValueError("checkpoint token counters are inconsistent")
        elif any(counts.values()) or pending or response_id is not None:
            raise ValueError("empty checkpoint has nonempty stream state")
        if ((counts["response_count"] == 0) != (response_id is None)
                or (response_id is not None and response_id > bank._last_time_id)):
            raise ValueError("checkpoint response state is inconsistent")
        bank._total_ingested = counts["total_ingested"]
        bank._total_encoded_tokens = counts["total_encoded_tokens"]
        bank._evicted_grids = counts["evicted_grids"]
        bank._evicted_tokens = counts["evicted_tokens"]
        bank._response_count = counts["response_count"]
        bank._last_response_time_id = response_id
        bank._pending = pending
        return bank
