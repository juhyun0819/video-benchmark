"""Persistent Qwen3-VL memory with three aligned DeepStack visual streams.

Only the final projected visual features determine TAS and SDC decisions. Every
DeepStack level follows the same TAS mask, and SDC averages its rows over the
same connected components at the anchors chosen by the released main kernel.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch
import torch.nn.functional as F

from .memory import PersistentMemory, _Grid


@dataclass
class _DeepStackGrid(_Grid):
    deepstack: list[torch.Tensor]


def _with_deepstack(grid: _Grid, deepstack: list[torch.Tensor]) -> _DeepStackGrid:
    return _DeepStackGrid(**{field.name: getattr(grid, field.name) for field in fields(_Grid)},
                          deepstack=deepstack)


def _sdc_components(grid: _Grid, kernel: Any) -> torch.Tensor:
    """Return the released SDC component id for each retained input row.

    Adapted from FluxMem's ``_long_term_memory_merge_per_frames`` in
    ``models/qwen2-5-vl/src/qwen2_5_vl_fluxmem/fluxmem.py``. Preserve its spatial
    edges, float32 normalization rule, Otsu degeneracy behavior, inclusive edge
    threshold, and label propagation exactly. The official kernel still owns
    main feature averaging and anchor selection; this helper only exposes its
    otherwise private component membership for the auxiliary streams.
    """
    device = grid.embeddings.device
    num_nodes = grid.tokens
    node_indices = torch.arange(num_nodes, dtype=torch.long, device=device)
    if num_nodes <= 1:
        return node_indices

    neighbor_offsets = kernel._neighbor_offsets(device, include_center=False)
    padded_grid = torch.full((grid.height + 2, grid.width + 2), -1,
                             dtype=torch.long, device=device)
    grid_heights = grid.positions[1].to(torch.long) + 1
    grid_widths = grid.positions[2].to(torch.long) + 1
    padded_grid[grid_heights, grid_widths] = node_indices
    neighbor_heights = grid_heights.view(-1, 1) + neighbor_offsets[:, 0].view(1, -1)
    neighbor_widths = grid_widths.view(-1, 1) + neighbor_offsets[:, 1].view(1, -1)
    neighbor_mat = padded_grid[neighbor_heights, neighbor_widths]
    source_mat = node_indices.unsqueeze(1).expand_as(neighbor_mat)
    upper_triangle = (neighbor_mat >= 0) & (neighbor_mat > source_mat)
    if not bool(upper_triangle.any().item()):
        return node_indices

    edge_i = source_mat[upper_triangle]
    edge_j = neighbor_mat[upper_triangle]
    if grid.normalized is not None and grid.normalized.dtype == torch.float32:
        features_norm = grid.normalized
    else:
        features_norm = F.normalize(grid.embeddings.float(), p=2, dim=1, eps=1e-8)
    edge_distances = 1.0 - (features_norm[edge_i] * features_norm[edge_j]).sum(dim=1)
    edge_distances = edge_distances.clamp(0.0, 2.0)
    threshold = kernel._otsu_threshold(edge_distances)
    if threshold is None:
        return node_indices
    keep_edges = edge_distances <= threshold
    if not bool(keep_edges.any().item()):
        return node_indices
    edge_i, edge_j = edge_i[keep_edges], edge_j[keep_edges]

    labels = node_indices.clone()
    for _ in range(max(1, num_nodes)):
        min_labels = torch.minimum(labels[edge_i], labels[edge_j])
        new_labels = labels.clone()
        new_labels.scatter_reduce_(0, edge_i, min_labels, reduce="amin", include_self=True)
        new_labels.scatter_reduce_(0, edge_j, min_labels, reduce="amin", include_self=True)
        new_labels = new_labels[new_labels]
        if torch.equal(new_labels, labels):
            break
        labels = new_labels
    _, inverse = torch.unique(labels, return_inverse=True)
    return inverse


class Qwen3PersistentMemory(PersistentMemory):
    """Three DeepStack levels retained and compressed alongside main features.

    ``ingest`` accepts the base memory arguments plus ``deepstack``: exactly
    three tensors with the same shape, dtype and device as ``embeddings``.
    Snapshots expose ``deepstack`` in the same chronological row order as
    ``embeddings`` and ``positions``. Queue ownership bounds auxiliary storage
    automatically when the base memory evicts an old temporal grid.
    """

    deepstack_levels = 3
    _DEEPSTACK_STATE_VERSION = 1

    @classmethod
    def _validate_deepstack(cls, embeddings: torch.Tensor,
                            deepstack: list[torch.Tensor]) -> None:
        if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
            raise ValueError("embeddings must be a [height * width, hidden_dim] tensor")
        if not isinstance(deepstack, (list, tuple)) or len(deepstack) != cls.deepstack_levels:
            raise ValueError("deepstack must contain exactly three aligned visual tensors")
        for level, tensor in enumerate(deepstack):
            if not isinstance(tensor, torch.Tensor) or tensor.shape != embeddings.shape:
                raise ValueError(f"deepstack[{level}] must have the same [tokens, hidden_dim] shape as embeddings")
            if tensor.dtype != embeddings.dtype or tensor.device != embeddings.device:
                raise ValueError(f"deepstack[{level}] must share the embeddings dtype and device")
            if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"deepstack[{level}] must contain finite floating point values")

    @torch.no_grad()
    def ingest(
        self,
        embeddings: torch.Tensor,
        height: int,
        width: int,
        time_id: int,
        timestamp_seconds: float,
        metadata: dict[str, Any],
        deepstack: list[torch.Tensor],
    ) -> dict[str, Any]:
        """Append one grid, validating all four streams before changing memory."""
        self._validate_deepstack(embeddings, deepstack)
        owned_deepstack = [tensor.detach().clone() for tensor in deepstack]
        observation = super().ingest(embeddings, height, width, time_id,
                                     timestamp_seconds, metadata)
        self._short[-1] = _with_deepstack(self._short[-1], owned_deepstack)
        return observation

    @staticmethod
    def _select(grid: _DeepStackGrid, keep: torch.Tensor, *, keep_normalized: bool) -> _DeepStackGrid:
        selected = PersistentMemory._select(grid, keep, keep_normalized=keep_normalized)
        return _with_deepstack(selected, [level[keep].detach().clone() for level in grid.deepstack])

    def _merge(self, grid: _DeepStackGrid) -> _DeepStackGrid:
        components = _sdc_components(grid, self._kernel)
        keep = torch.ones(grid.tokens, dtype=torch.bool, device=grid.embeddings.device)
        # The released kernel remains authoritative for main means and anchors.
        self._kernel._long_term_memory_merge_per_frames(
            batch_index=0, long_frames={grid.time_id},
            visual_indices=torch.arange(grid.tokens, device=grid.embeddings.device),
            time_ids=grid.positions[0], height_ids=grid.positions[1], width_ids=grid.positions[2],
            hidden_states=grid.embeddings.unsqueeze(0), hidden_norm=grid.normalized,
            keep_mask=keep, grid_hw=(grid.height, grid.width),
        )
        selected = self._select(grid, keep, keep_normalized=False)
        if not grid.tokens or bool(keep.all().item()):
            return selected

        num_components = int(components.max().item()) + 1
        counts = torch.zeros(num_components, dtype=torch.float32, device=components.device)
        counts.index_add_(0, components, torch.ones_like(components, dtype=torch.float32))
        selected_components = components[keep]
        merged_rows = counts[selected_components] > 1.0
        for original, retained in zip(grid.deepstack, selected.deepstack):
            component_sum = torch.zeros((num_components, original.shape[1]),
                                        dtype=torch.float32, device=original.device)
            component_sum.index_add_(0, components, original.float())
            component_mean = component_sum / counts.clamp_min(1.0).unsqueeze(1)
            retained[merged_rows] = component_mean[selected_components[merged_rows]].to(original.dtype)
        return selected

    def stats(self) -> dict[str, Any]:
        stats = super().stats()
        auxiliary_bytes = auxiliary_cpu_bytes = 0
        for _, queue in self._tiers():
            for grid in queue:
                for tensor in grid.deepstack:
                    size = tensor.numel() * tensor.element_size()
                    auxiliary_bytes += size
                    if tensor.device.type == "cpu":
                        auxiliary_cpu_bytes += size
        stats["deepstack_tensor_bytes"] = auxiliary_bytes
        stats["deepstack_cpu_bytes"] = auxiliary_cpu_bytes
        stats["retained_tensor_bytes"] += auxiliary_bytes
        stats["retained_cpu_bytes"] += auxiliary_cpu_bytes
        return stats

    def snapshot(self) -> dict[str, Any]:
        snapshot = super().snapshot()
        grids = [grid for _, queue in self._tiers() for grid in queue]
        snapshot["deepstack"] = [
            torch.cat([grid.deepstack[level] for grid in grids]).detach().clone()
            if grids else torch.empty((0, 0))
            for level in range(self.deepstack_levels)
        ]
        return snapshot

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["deepstack_version"] = self._DEEPSTACK_STATE_VERSION
        for tier, queue in self._tiers():
            for record, grid in zip(state["queues"][tier], queue):
                record["deepstack"] = [tensor.detach().cpu().clone() for tensor in grid.deepstack]
        return state

    @classmethod
    def from_state_dict(cls, state: dict[str, Any], *, device: str | torch.device = "cpu") -> "Qwen3PersistentMemory":
        """Restore a trusted checkpoint, including pending and DeepStack state."""
        if state.get("deepstack_version") != cls._DEEPSTACK_STATE_VERSION:
            raise ValueError("unsupported Qwen3 DeepStack memory checkpoint version")
        base_state = {**state, "queues": {
            tier: [{key: value for key, value in record.items() if key != "deepstack"}
                   for record in records]
            for tier, records in state["queues"].items()
        }}
        bank = super().from_state_dict(base_state, device=device)
        for tier, queue in bank._tiers():
            for index, (grid, record) in enumerate(zip(queue, state["queues"][tier])):
                saved_levels = record.get("deepstack")
                if not isinstance(saved_levels, (list, tuple)) or len(saved_levels) != cls.deepstack_levels:
                    raise ValueError("checkpoint grid is missing its three DeepStack levels")
                if any(not isinstance(tensor, torch.Tensor) for tensor in saved_levels):
                    raise ValueError("checkpoint DeepStack levels must be tensors")
                deepstack = [tensor.detach().to(device).clone() for tensor in saved_levels]
                cls._validate_deepstack(grid.embeddings, deepstack)
                queue[index] = _with_deepstack(grid, deepstack)
        return bank
