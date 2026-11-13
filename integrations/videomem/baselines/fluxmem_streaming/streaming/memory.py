"""Persistent projected-token memory; the released model sources stay untouched.

Call ``ingest -> snapshot/query (optional) -> advance`` for every temporal grid.
The current grid enters short memory before a query; compression happens only in
``advance``. Capacities are sampled raw-frame counts, converted with ``// 2``.
Even an empty grid occupies a medium/long queue slot.

``upstream`` TAS delegates to the released kernel (inclusive Otsu comparisons
and its extra >0.8 similarity drop). ``paper`` uses separate backward/forward
Otsu thresholds, strict > comparisons and their OR, without that extra drop.
Both use the released 128-bin [0, 2] Otsu implementation: first maximizing bin,
bin-center threshold, float32 histograms, and a median fallback for a constant
or single-bin distribution. Thus a constant-distance input has no strictly
above-threshold tokens, even if that constant describes a large scene change.
The backward trigger uses this strict rule with either TAS policy, requires a
previous successful response, and tests r > gamma. Distances within 1e-6 of
zero are zeroed *only for the trigger* to suppress normalization roundoff.
SDC always delegates to the released kernel, including its no-merge behavior
when its Otsu distribution is degenerate.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, fields
import math
from numbers import Integral, Real
from typing import Any

import torch
import torch.nn.functional as F

from qwen2_5_vl_fluxmem.fluxmem import FluxMem


@dataclass
class _Grid:
    embeddings: torch.Tensor
    positions: torch.Tensor
    normalized: torch.Tensor | None
    backward: torch.Tensor | None
    time_id: int
    timestamp_seconds: float
    height: int
    width: int
    metadata: dict[str, Any]

    @property
    def tokens(self) -> int:
        return int(self.embeddings.shape[0])


class PersistentMemory:
    """Incremental short/medium/long memory shared by every question in a stream.

    Only full short grids, pruned medium grids and merged long grids are stored;
    a transition considers at most two grids. No full-prefix replay is used.
    ``long_frames=None`` preserves the upstream unlimited long-memory behavior;
    a finite long capacity evicts the oldest complete temporal-grid record.
    Inputs and snapshots are detached copies owned by their recipient.
    """

    temporal_patch_size = 2
    _STATE_VERSION = 1

    def __init__(
        self,
        short_frames: int = 8,
        medium_frames: int = 64,
        long_frames: int | None = None,
        gamma: float = 0.5,
        tas_policy: str = "upstream",
    ) -> None:
        for name, value in (("short_frames", short_frames), ("medium_frames", medium_frames),
                            ("long_frames", long_frames)):
            if name == "long_frames" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer raw-frame count")
        if short_frames // self.temporal_patch_size < 1:
            raise ValueError("short_frames must provide at least one temporal grid (2 raw frames)")
        if isinstance(gamma, bool) or not isinstance(gamma, Real) or not math.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("gamma must be finite and between 0 and 1")
        if tas_policy not in {"upstream", "paper"}:
            raise ValueError("tas_policy must be 'upstream' or 'paper'")
        self.short_frames = int(short_frames)
        self.medium_frames = int(medium_frames)
        self.long_frames = None if long_frames is None else int(long_frames)
        self.short_capacity = self.short_frames // 2
        self.medium_capacity = self.medium_frames // 2
        self.long_capacity = None if self.long_frames is None else self.long_frames // 2
        self.gamma = float(gamma)
        self.tas_policy = tas_policy
        self._kernel = FluxMem(vision_start_token_id=0, vision_end_token_id=1,
                               short_term_frames=self.short_frames, mid_term_frames=self.medium_frames)
        self._short: deque[_Grid] = deque()
        self._medium: deque[_Grid] = deque()
        self._long: deque[_Grid] = deque()
        self._pending = False
        self._shape: tuple[int, int, int] | None = None
        self._dtype: torch.dtype | None = None
        self._device: torch.device | None = None
        self._last_time_id: int | None = None
        self._last_timestamp: float | None = None
        self._last_response_time_id: int | None = None
        self._response_count = 0
        self._total_ingested = 0
        self._total_encoded_tokens = 0
        self._evicted_grids = 0
        self._evicted_tokens = 0
        self._last_observation: dict[str, Any] | None = None

    def _validate_input(self, embeddings, height, width, time_id, timestamp_seconds, metadata) -> None:
        if self._pending:
            raise RuntimeError("advance() must follow the previous ingest() before another ingest()")
        for name, value in (("height", height), ("width", width)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
            raise ValueError("embeddings must be a [height * width, hidden_dim] tensor")
        if embeddings.shape[0] != height * width or embeddings.shape[1] < 1:
            raise ValueError("embeddings shape must match height * width and a positive hidden dimension")
        if not embeddings.is_floating_point() or not bool(torch.isfinite(embeddings).all()):
            raise ValueError("embeddings must contain finite floating point values")
        if isinstance(time_id, bool) or not isinstance(time_id, Integral) or time_id < 0:
            raise ValueError("time_id must be a nonnegative integer")
        if self._last_time_id is not None and time_id <= self._last_time_id:
            raise ValueError("time_id must strictly increase; duplicate or out-of-order grids are rejected")
        if (isinstance(timestamp_seconds, bool) or not isinstance(timestamp_seconds, Real)
                or not math.isfinite(timestamp_seconds) or timestamp_seconds < 0):
            raise ValueError("timestamp_seconds must be finite and nonnegative")
        if self._last_timestamp is not None and timestamp_seconds <= self._last_timestamp:
            raise ValueError("timestamp_seconds must strictly increase")
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a dictionary")
        if self._shape is not None:
            if (height, width, embeddings.shape[1]) != self._shape:
                raise ValueError("all grids in one stream must share height, width and hidden dimension")
            if embeddings.dtype != self._dtype or embeddings.device != self._device:
                raise ValueError("all grids in one stream must share dtype and device")

    @torch.no_grad()
    def ingest(
        self,
        embeddings: torch.Tensor,
        height: int,
        width: int,
        time_id: int,
        timestamp_seconds: float,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Append one observed grid and compute its causal backward trigger.

        Metadata is copied for provenance, never read by matching or triggering.
        An observed grid is always available in full until ``advance``.
        """
        self._validate_input(embeddings, height, width, time_id, timestamp_seconds, metadata)
        embeddings = embeddings.detach().clone()
        height, width, time_id = int(height), int(width), int(time_id)
        h = torch.arange(height, device=embeddings.device).repeat_interleave(width)
        w = torch.arange(width, device=embeddings.device).repeat(height)
        positions = torch.stack((torch.full_like(h, time_id), h, w))
        grid = _Grid(embeddings, positions, F.normalize(embeddings, p=2, dim=1, eps=1e-8), None,
                     time_id, float(timestamp_seconds), height, width, deepcopy(metadata))
        threshold = None
        score = 0.0
        if self._short:
            data = self._pair_data(grid, self._short[-1])
            paired, _, distances, _ = self._kernel._max_sim_against_frames(
                query_indices=data["indices"][grid.time_id], neighbor_frames=[self._short[-1].time_id],
                height_ids=data["positions"][1], width_ids=data["positions"][2],
                frame_grids=data["grids"], hidden_norm=data["normalized"],
                offsets_3x3=self._kernel._neighbor_offsets(embeddings.device, include_center=True),
            )
            grid.backward = torch.full((grid.tokens,), -1.0, dtype=embeddings.dtype, device=embeddings.device)
            grid.backward[paired] = distances
            trigger_distances = distances.detach().float().clamp(0.0, 2.0)
            trigger_distances[trigger_distances <= 1e-6] = 0.0
            threshold = self._threshold(trigger_distances)
            score = float((trigger_distances > threshold).float().mean().item())
        self._short.append(grid)
        self._shape = (height, width, int(embeddings.shape[1]))
        self._dtype, self._device = embeddings.dtype, embeddings.device
        self._last_time_id, self._last_timestamp = time_id, float(timestamp_seconds)
        self._total_ingested += 1
        self._total_encoded_tokens += grid.tokens
        self._pending = True
        self._last_observation = {
            "time_id": time_id,
            "timestamp_seconds": float(timestamp_seconds),
            "backward_score": score,
            "backward_threshold": threshold,
            "has_prior_response": self._last_response_time_id is not None,
            "triggered": self._last_response_time_id is not None and score > self.gamma,
            "gamma": self.gamma,
        }
        return deepcopy(self._last_observation)

    @staticmethod
    def _threshold(distances: torch.Tensor) -> float:
        # Avoid torch.var's degrees-of-freedom warning for a one-token grid.
        if distances.numel() == 1:
            return float(distances.detach().float().clamp(0.0, 2.0).item())
        threshold = FluxMem._otsu_threshold(distances, fallback_to_median=True)
        if threshold is None:
            raise RuntimeError("cannot threshold an empty distance distribution")
        return threshold

    def _pair_data(self, query: _Grid, neighbor: _Grid) -> dict[str, Any]:
        positions = torch.cat((query.positions, neighbor.positions), dim=1)
        normalized = torch.cat((query.normalized, neighbor.normalized), dim=0)
        visual = torch.arange(positions.shape[1], device=positions.device)
        _, indices, grids = self._kernel._frame_grids(
            visual, positions[0], positions[1], positions[2], (query.height, query.width)
        )
        return {"positions": positions, "normalized": normalized, "indices": indices, "grids": grids}

    def _prune(self, grid: _Grid, neighbor: _Grid) -> _Grid:
        data = self._pair_data(grid, neighbor)
        keep = torch.zeros(data["positions"].shape[1], dtype=torch.bool, device=grid.embeddings.device)
        backward = torch.full_like(keep, -1.0, dtype=grid.embeddings.dtype)
        if grid.backward is not None:
            backward[:grid.tokens] = grid.backward
        if self.tas_policy == "upstream":
            self._kernel._apply_adjacent_pruning(
                frame_id=grid.time_id, buffer_frames=[neighbor.time_id],
                frame_to_indices=data["indices"], frame_grids=data["grids"],
                height_ids=data["positions"][1], width_ids=data["positions"][2],
                hidden_norm=data["normalized"], pair_distance_cache=backward,
                keep_mask=keep, pair_distance_threshold=None,
                offsets_3x3=self._kernel._neighbor_offsets(grid.embeddings.device, include_center=True),
            )
        else:
            paired, _, forward, _ = self._kernel._max_sim_against_frames(
                query_indices=data["indices"][grid.time_id], neighbor_frames=[neighbor.time_id],
                height_ids=data["positions"][1], width_ids=data["positions"][2],
                frame_grids=data["grids"], hidden_norm=data["normalized"],
                offsets_3x3=self._kernel._neighbor_offsets(grid.embeddings.device, include_center=True),
            )
            forward = forward.float().clamp(0.0, 2.0)
            keep[paired] = forward > self._threshold(forward)
            if grid.backward is not None:
                previous = grid.backward.float().clamp(0.0, 2.0)
                keep[:grid.tokens] |= previous > self._threshold(previous)
        return self._select(grid, keep[:grid.tokens], keep_normalized=True)

    @staticmethod
    def _select(grid: _Grid, keep: torch.Tensor, *, keep_normalized: bool) -> _Grid:
        return _Grid(grid.embeddings[keep].detach().clone(), grid.positions[:, keep].detach().clone(),
                     grid.normalized[keep].detach().clone() if keep_normalized else None,
                     None, grid.time_id, grid.timestamp_seconds, grid.height, grid.width, grid.metadata)

    def _merge(self, grid: _Grid) -> _Grid:
        keep = torch.ones(grid.tokens, dtype=torch.bool, device=grid.embeddings.device)
        self._kernel._long_term_memory_merge_per_frames(
            batch_index=0, long_frames={grid.time_id},
            visual_indices=torch.arange(grid.tokens, device=grid.embeddings.device),
            time_ids=grid.positions[0], height_ids=grid.positions[1], width_ids=grid.positions[2],
            hidden_states=grid.embeddings.unsqueeze(0), hidden_norm=grid.normalized,
            keep_mask=keep, grid_hw=(grid.height, grid.width),
        )
        return self._select(grid, keep, keep_normalized=False)

    @torch.no_grad()
    def advance(self) -> dict[str, Any]:
        """Complete one arrival by cascading overflow and enforcing long capacity."""
        if not self._pending:
            raise RuntimeError("advance() requires a preceding ingest()")
        while len(self._short) > self.short_capacity:
            oldest = self._short.popleft()
            self._medium.append(self._prune(oldest, self._short[0]))
        while len(self._medium) > self.medium_capacity:
            self._long.append(self._merge(self._medium.popleft()))
        if self.long_capacity is not None:
            while len(self._long) > self.long_capacity:
                oldest = self._long.popleft()
                self._evicted_grids += 1
                self._evicted_tokens += oldest.tokens
        self._pending = False
        return self.stats()

    def should_respond(self) -> bool:
        """Whether the newest observation crosses gamma after a prior response."""
        return bool(self._last_observation and self._last_observation["triggered"]
                    and self._last_response_time_id != self._last_time_id)

    def mark_response(self) -> None:
        """Record a successful answer without resetting any memory queue."""
        if self._last_time_id is None:
            raise RuntimeError("cannot mark a response before ingesting a grid")
        self._last_response_time_id = self._last_time_id
        self._response_count += 1

    @property
    def last_observation(self) -> dict[str, Any] | None:
        return deepcopy(self._last_observation)

    def _tiers(self):
        return (("long", self._long), ("medium", self._medium), ("short", self._short))

    def stats(self) -> dict[str, Any]:
        """Small JSON-safe counts, including all physically retained tensor bytes."""
        tier_grids = {name: len(queue) for name, queue in self._tiers()}
        tier_tokens = {name: sum(grid.tokens for grid in queue) for name, queue in self._tiers()}
        tensor_bytes = cpu_bytes = 0
        for _, queue in self._tiers():
            for grid in queue:
                for tensor in (grid.embeddings, grid.positions, grid.normalized, grid.backward):
                    if tensor is not None:
                        size = tensor.numel() * tensor.element_size()
                        tensor_bytes += size
                        if tensor.device.type == "cpu":
                            cpu_bytes += size
        return {"tier_grids": tier_grids, "tier_tokens": tier_tokens,
                "total_tokens": sum(tier_tokens.values()), "total_ingested": self._total_ingested,
                "total_encoded_tokens": self._total_encoded_tokens, "response_count": self._response_count,
                "evicted_grids": self._evicted_grids, "evicted_tokens": self._evicted_tokens,
                "retained_tensor_bytes": tensor_bytes, "retained_cpu_bytes": cpu_bytes,
                "pending_advance": self._pending}

    def snapshot(self) -> dict[str, Any]:
        """Return isolated detached tensors and provenance in chronological order."""
        grids = [(tier, grid) for tier, queue in self._tiers() for grid in queue]
        if grids:
            embeddings = torch.cat([grid.embeddings for _, grid in grids]).detach().clone()
            positions = torch.cat([grid.positions for _, grid in grids], dim=1).detach().clone()
        else:
            embeddings = torch.empty((0, 0))
            positions = torch.empty((3, 0), dtype=torch.long)
        records = [{"time_id": grid.time_id, "timestamp_seconds": grid.timestamp_seconds,
                    "height": grid.height, "width": grid.width, "tier": tier,
                    "input_tokens": grid.height * grid.width, "retained_tokens": grid.tokens,
                    "metadata": deepcopy(grid.metadata)} for tier, grid in grids]
        return {"embeddings": embeddings, "positions": positions, "records": records, "stats": self.stats()}

    def state_dict(self) -> dict[str, Any]:
        """Return an isolated CPU checkpoint, including pending phase/trigger state.

        Suitable for torch.save to a trusted local checkpoint. Question and label
        payloads need not be present: only caller-supplied metadata is preserved.
        """
        queues = {}
        for tier, queue in self._tiers():
            queues[tier] = []
            for grid in queue:
                record = {}
                for field in fields(_Grid):
                    value = getattr(grid, field.name)
                    record[field.name] = value.detach().cpu().clone() if isinstance(value, torch.Tensor) else deepcopy(value)
                queues[tier].append(record)
        counters = {name: deepcopy(getattr(self, name)) for name in (
            "_pending", "_shape", "_last_time_id", "_last_timestamp", "_last_response_time_id",
            "_response_count", "_total_ingested", "_total_encoded_tokens", "_evicted_grids",
            "_evicted_tokens", "_last_observation",
        )}
        return {"version": self._STATE_VERSION,
                "config": {"short_frames": self.short_frames, "medium_frames": self.medium_frames,
                           "long_frames": self.long_frames, "gamma": self.gamma, "tas_policy": self.tas_policy},
                "queues": queues, "counters": counters}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any], *, device: str | torch.device = "cpu") -> "PersistentMemory":
        """Restore a trusted checkpoint; continuation never re-encodes old grids."""
        if state.get("version") != cls._STATE_VERSION:
            raise ValueError("unsupported persistent memory checkpoint version")
        bank = cls(**state["config"])
        for tier, queue in bank._tiers():
            for saved in state["queues"][tier]:
                record = {key: value.detach().to(device).clone() if isinstance(value, torch.Tensor) else deepcopy(value)
                          for key, value in saved.items()}
                queue.append(_Grid(**record))
        allowed = {
            "_pending", "_shape", "_last_time_id", "_last_timestamp", "_last_response_time_id",
            "_response_count", "_total_ingested", "_total_encoded_tokens", "_evicted_grids",
            "_evicted_tokens", "_last_observation",
        }
        if set(state["counters"]) != allowed:
            raise ValueError("invalid persistent memory checkpoint counters")
        for name, value in state["counters"].items():
            setattr(bank, name, deepcopy(value))
        if bank._short:
            bank._dtype = bank._short[-1].embeddings.dtype
            bank._device = bank._short[-1].embeddings.device
        return bank
