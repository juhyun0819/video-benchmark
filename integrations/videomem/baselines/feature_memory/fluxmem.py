"""Released FluxMem TAS/SDC kernels with shared-image input and a byte cap.

This adapter reuses the repository's tested persistent extension and unmodified
released compression kernels. It is explicitly not the native full-model path:
one independently encoded image supplies each grid, tier capacities may shrink
to fit a byte cap, and pressure applies extra native SDC passes to a fixed point.
Oldest finalized tokens are evicted only when that compression cannot fit the cap.
The default native capacities are 4 short grids and 128 medium grids (the
release calls these 8 and 256 raw frames, with temporal_patch_size=2).
"""
from __future__ import annotations

import __future__
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import types

import numpy as np
import torch

from baselines.streambench.accounting import measure_state


_ROOT = Path(__file__).resolve().parents[1]
_OFFICIAL = _ROOT / 'fluxmem_reference/models/qwen2-5-vl/src/qwen2_5_vl_fluxmem'
_PERSISTENT = _ROOT / 'fluxmem_streaming/streaming/memory.py'
_PACKAGE = '_feature_memory_official_fluxmem'
_EXTRA_FIELDS = {'merge_count': '<i8', 'start_frame': '<i8', 'end_frame': '<i8',
                 'start_timestamp': '<f8', 'end_timestamp': '<f8'}


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def official_classes():
    """Load only memory kernels, avoiding model/package initialization.

    Deferred annotations make the original Python >=3.10 type hints importable
    in this experiment's Python 3.9 environment. Only the persistent adapter's
    import target changes; no kernel expression, threshold or operation changes.
    """
    package = types.ModuleType(_PACKAGE)
    package.__path__ = [str(_OFFICIAL)]
    sys.modules[_PACKAGE] = package
    for name, path in [('utils', _OFFICIAL / 'utils.py'),
                       ('fluxmem', _OFFICIAL / 'fluxmem.py'),
                       ('persistent', _PERSISTENT)]:
        module_name = _PACKAGE + '.' + name
        module = types.ModuleType(module_name)
        module.__file__ = str(path)
        module.__package__ = _PACKAGE
        source = path.read_text()
        if name == 'persistent':
            source = source.replace('from qwen2_5_vl_fluxmem.fluxmem import FluxMem',
                                    f'from {_PACKAGE}.fluxmem import FluxMem')
        sys.modules[module_name] = module
        exec(compile(source, str(path), 'exec', flags=__future__.annotations.compiler_flag,
                     dont_inherit=True), module.__dict__)
    return (sys.modules[_PACKAGE + '.fluxmem'].FluxMem,
            sys.modules[_PACKAGE + '.persistent'].PersistentMemory)


def kernel_provenance():
    return {'upstream': 'https://github.com/ShareLab-SII/FluxMem',
            'revision': '6fdbf65b6f2af1a94ca41d542be5be86a565d338',
            'files': {str(p.relative_to(_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [_OFFICIAL / 'fluxmem.py', _OFFICIAL / 'utils.py', _PERSISTENT]},
            'kernel_math_modified': False,
            'type_annotations_deferred_for_python39': True,
            'merge_provenance': 'read-only return trace of official SDC counts and anchor indices'}


@lru_cache(maxsize=1)
def _provenance_class():
    _, PersistentMemory = official_classes()

    class ProvenanceMemory(PersistentMemory):
        """Attach exact SDC cluster sizes without changing its feature math."""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.num_removed_temporal_duplicates = 0
            self.num_merged_tokens = 0
            self.num_merged_anchors = 0

        def _prune(self, grid, neighbor):
            result = super()._prune(grid, neighbor)
            self.num_removed_temporal_duplicates += grid.tokens - result.tokens
            return result

        def _merge(self, grid):
            # Each call handles one frame only. Capture official final cluster
            # sizes and anchor positions; no cluster/mean computation is copied.
            captured = {}
            target = self._kernel._long_term_memory_merge_per_frames.__func__.__code__
            previous_trace = sys.gettrace()

            def trace(frame, event, arg):
                if frame.f_code is target:
                    if event == 'return':
                        values = frame.f_locals
                        if 'anchor_global' in values and 'merge_clusters' in values:
                            captured['anchors'] = values['anchor_global'].detach().cpu().clone()
                            captured['counts'] = values['counts'][values['merge_clusters']].to(
                                device='cpu', dtype=torch.int64).clone()
                    return trace
                return previous_trace(frame, event, arg) if previous_trace else None

            # super()._merge mutates its owned feature grid to official means.
            sys.settrace(trace)
            try:
                result = super()._merge(grid)
            finally:
                sys.settrace(previous_trace)
            counts = torch.ones(grid.tokens, dtype=torch.int64)
            if captured:
                counts[captured['anchors']] = captured['counts']
            original_indices = grid.positions[1] * grid.width + grid.positions[2]
            result_indices = result.positions[1] * grid.width + result.positions[2]
            # Position IDs remain unique anchors in a frame; searchsorted works
            # because released _select preserves original spatial ordering.
            selected = torch.searchsorted(original_indices.cpu(), result_indices.cpu())
            result.metadata = dict(result.metadata)
            result.metadata['merge_counts'] = counts[selected].clone()
            self.num_merged_tokens += grid.tokens - result.tokens
            self.num_merged_anchors += int((result.metadata['merge_counts'] > 1).sum())
            return result

    return ProvenanceMemory


class FluxMemByteMemory:
    """Bounded FluxMem tiers and disk records with native compression first.

    All retained tier tensors, indices and scalar state are charged against the
    same hard cap as the compressed disk records. Merge workspaces and the one
    shared incoming feature tensor are transient, reported separately by runner.
    """
    method = 'fluxmem'
    algorithm = 'official_fluxmem_tas_sdc_shared_image_native_fixedpoint_bytecap_fallback'

    def __init__(self, directory, hard_budget_bytes, hidden_dim=3584, dtype='bfloat16',
                 tokens_per_frame=676, height=26, width=26, retention_ratio=None,
                 seed=0, native_short_grids=4, native_medium_grids=128,
                 tier_budget_fraction=0.2, shard_bytes=64 * 1024 * 1024):
        from .byte_memory import DiskTokenStore
        dtype = {'bf16': 'bfloat16', 'fp16': 'float16', 'fp32': 'float32'}.get(dtype, dtype)
        if dtype not in ('float16', 'bfloat16', 'float32'):
            raise ValueError('Unsupported feature dtype')
        if (type(hard_budget_bytes) is not int or hard_budget_bytes <= 0
                or type(tokens_per_frame) is not int or tokens_per_frame <= 0
                or tokens_per_frame != height * width):
            raise ValueError('Positive byte cap and tokens_per_frame == height*width required')
        if not 0 < tier_budget_fraction < 1:
            raise ValueError('tier_budget_fraction must be between zero and one')
        if native_short_grids < 1 or native_medium_grids < 0:
            raise ValueError('Native tier grid capacities are invalid')
        self.directory = Path(directory)
        self.hard_budget_bytes = hard_budget_bytes
        self.hidden_dim, self.dtype = int(hidden_dim), dtype
        self.tokens_per_frame, self.height, self.width = tokens_per_frame, int(height), int(width)
        self.retention_ratio, self.seed = retention_ratio, seed
        element = torch.empty((), dtype=getattr(torch, dtype)).element_size()
        # Worst retained grid: embeddings + normalized copy + backward values
        # + 3 int64 position coordinates + generous fixed metadata allowance.
        self._grid_reserve = tokens_per_frame * (2 * hidden_dim * element + element + 24) + 4096
        # Snapshot hash inventories grow with shard count. Reserve their
        # bounded worst-case metadata as well, so an otherwise full bank does
        # not overshoot its cap when serialized with thousands of shard hashes.
        snapshot_manifest_reserve = 8192 + math.ceil(hard_budget_bytes / shard_bytes) * 384
        fixed_reserve = max(65536, snapshot_manifest_reserve)
        native_grids = native_short_grids + native_medium_grids
        available_grids = min(native_grids, int(hard_budget_bytes * tier_budget_fraction
                                               - fixed_reserve) // self._grid_reserve)
        if available_grids < 1:
            # A single causal comparison grid is mandatory. Keep it when the
            # hard cap still leaves room for at least one compressed record.
            available_grids = 1
        self.short_grids = min(native_short_grids, available_grids)
        self.medium_grids = min(native_medium_grids, available_grids - self.short_grids)
        self.tier_reserved_bytes = fixed_reserve + self._grid_reserve * (self.short_grids + self.medium_grids)
        record_bytes = hidden_dim * element + 32 + 40
        if hard_budget_bytes - self.tier_reserved_bytes < 8192 + record_bytes:
            raise ValueError('Byte cap cannot fit one FluxMem comparison grid plus one stored token')
        self._core = _provenance_class()(short_frames=2 * self.short_grids,
                                        medium_frames=2 * self.medium_grids,
                                        long_frames=None, tas_policy='upstream')
        self.store = DiskTokenStore(self.directory, hard_budget_bytes - self.tier_reserved_bytes,
                                   hidden_dim=hidden_dim, dtype=dtype, extra_fields=_EXTRA_FIELDS,
                                   control_reserve=8192, shard_bytes=shard_bytes)
        self._cursor = 0
        self._evicted_tokens = 0
        self._total_frames = 0
        self._total_tokens = 0
        self._total_feature_bytes = 0
        self._last_frame_id = None
        self._last_timestamp = None
        self._update_seconds = 0.0
        self._finalized_tokens = 0
        self._bank_sdc_fixedpoint = False
        self._native_recompression_events = 0
        self._native_recompression_bytes_saved = 0
        self._native_bank_compactions = 0
        self._pressure_merged_tokens = 0
        self._pressure_merged_anchors = 0
        self._pressure_merged_members = 0
        self._fallback_eviction_events = 0
        self._fallback_eviction_bytes = 0
        self._last_fallback_cause = None
        self.config = {'method': self.method, 'algorithm': self.algorithm,
            'native_short_grids': native_short_grids, 'native_medium_grids': native_medium_grids,
            'actual_short_grids': self.short_grids, 'actual_medium_grids': self.medium_grids,
            'tier_budget_fraction': tier_budget_fraction, 'tier_reserved_bytes': self.tier_reserved_bytes,
            'snapshot_manifest_reserve_bytes': snapshot_manifest_reserve,
            'single_image_grid_instead_of_native_two_frame_patch': True,
            'temporal_pruning_policy': 'released inclusive Otsu OR + similarity >0.8 direct drop',
            'spatial_merging_policy': 'released 3x3 Otsu connected components + mean + nearest anchor',
            'byte_overflow_policy': 'native SDC fixed-point recompression of incoming and existing frame groups first; '
                                    'oldest compressed token eviction only if fixed-point records still exceed cap',
            'native_pressure_adaptation': 'additional released SDC passes until token count no longer decreases; '
                                          'one ordered in-place old-bank pass, then every incoming grid reaches fixed point',
            'recompressed_centroid_mean': 'unchanged native unweighted mean of retained centroids; provenance merge counts sum original memberships',
            'provenance_scope': 'SDC merge counts and one-frame spans; TAS removal is not a merge',
            'full_membership_lists_stored': False, 'upstream': kernel_provenance()}

    def _control(self):
        return {'method': self.method, 'algorithm': self.algorithm,
                'hard_budget_bytes': self.hard_budget_bytes, 'retention_ratio': self.retention_ratio,
                'hidden_dim': self.hidden_dim, 'dtype': self.dtype,
                'tokens_per_frame': self.tokens_per_frame, 'height': self.height, 'width': self.width,
                'tier_reserved_bytes': self.tier_reserved_bytes,
                'short_grids': self.short_grids, 'medium_grids': self.medium_grids,
                'fifo_cursor': self._cursor, 'total_frames_seen': self._total_frames,
                'total_visual_tokens_seen': self._total_tokens,
                'full_feature_bytes_seen': self._total_feature_bytes,
                'last_frame_id': self._last_frame_id, 'last_timestamp': self._last_timestamp,
                'num_removed_temporal_duplicates': self._core.num_removed_temporal_duplicates,
                'num_merged_tokens': self._core.num_merged_tokens,
                'num_merged_anchors': self._core.num_merged_anchors,
                'evicted_compressed_tokens': self._evicted_tokens,
                'finalized_compressed_tokens': self._finalized_tokens,
                'bank_sdc_fixedpoint': self._bank_sdc_fixedpoint,
                'native_recompression_events': self._native_recompression_events,
                'native_recompression_bytes_saved': self._native_recompression_bytes_saved,
                'native_bank_compactions': self._native_bank_compactions,
                'pressure_merged_tokens': self._pressure_merged_tokens,
                'pressure_merged_anchors': self._pressure_merged_anchors,
                'pressure_merged_members': self._pressure_merged_members,
                'fallback_eviction_events': self._fallback_eviction_events,
                'fallback_eviction_bytes': self._fallback_eviction_bytes,
                'last_fallback_cause': self._last_fallback_cause,
                'update_seconds': self._update_seconds}

    def _tier_bytes(self):
        # Expose live objects directly; state_dict() would clone every tensor.
        result = measure_state({'visual_memory_bytes': [self._core._short, self._core._medium],
                                'metadata_bytes': {'control': self._control(),
                                  'last_observation': self._core._last_observation,
                                  'configuration': self.config,
                                  'kernel_parameters': {'short': self._core.short_frames,
                                                       'medium': self._core.medium_frames,
                                                       'direct_drop_sim_threshold': 0.8}}})
        return result['persistent_total_bytes']

    def _native_sdc_once(self, features, metadata):
        """Run the unchanged SDC kernel and accumulate bounded provenance."""
        if not len(features):
            return features, metadata, 0, 0
        if not bool((metadata['frame_id'] == metadata['frame_id'][0]).all()):
            raise ValueError('Native spatial recompression requires one source frame per group')
        hidden = features.detach().clone()
        n = len(hidden)
        indices = torch.arange(n, dtype=torch.int64)
        token_indices = metadata['token_index']
        time_ids = metadata['frame_id']
        heights, widths = token_indices // self.width, token_indices % self.width
        keep = torch.ones(n, dtype=torch.bool)
        captured = {}
        kernel = self._core._kernel
        target = kernel._long_term_memory_merge_per_frames.__func__.__code__
        previous_trace = sys.gettrace()

        def trace(frame, event, arg):
            if frame.f_code is target:
                if event == 'return':
                    values = frame.f_locals
                    if 'anchor_global' in values and 'inverse' in values:
                        captured.update({key: values[key].detach().cpu().clone()
                                         for key in ['anchor_global', 'inverse', 'merge_clusters', 'kept_frame_indices']})
                return trace
            return previous_trace(frame, event, arg) if previous_trace else None

        sys.settrace(trace)
        try:
            kernel._long_term_memory_merge_per_frames(
                batch_index=0, long_frames={int(time_ids[0])}, visual_indices=indices,
                time_ids=time_ids, height_ids=heights, width_ids=widths,
                hidden_states=hidden.unsqueeze(0), keep_mask=keep,
                hidden_norm=None, grid_hw=(self.height, self.width))
        finally:
            sys.settrace(previous_trace)
        result_meta = {key: value[keep].clone() for key, value in metadata.items()}
        merged_anchors = merged_members = 0
        if captured:
            prior_counts = metadata['merge_count'][captured['kept_frame_indices']]
            inverse = captured['inverse']
            combined = torch.zeros(int(inverse.max()) + 1, dtype=torch.int64)
            combined.index_add_(0, inverse, prior_counts)
            counts = metadata['merge_count'].clone()
            merged_counts = combined[captured['merge_clusters']]
            counts[captured['anchor_global']] = merged_counts
            result_meta['merge_count'] = counts[keep].clone()
            merged_anchors = len(merged_counts)
            merged_members = int(merged_counts.sum())
        return hidden[keep].contiguous(), result_meta, merged_anchors, merged_members

    def _native_sdc_fixed_point(self, features, metadata):
        while True:
            before = len(features)
            result, result_meta, anchors, members = self._native_sdc_once(features, metadata)
            self._native_recompression_events += 1
            removed = before - len(result)
            if removed < 0:
                raise RuntimeError('Released SDC unexpectedly increased token count')
            self._native_recompression_bytes_saved += removed * self.store.record_bytes
            self._pressure_merged_tokens += removed
            self._pressure_merged_anchors += anchors
            self._pressure_merged_members += members
            features, metadata = result, result_meta
            if removed == 0:
                return features, metadata
            # Every continuing pass strictly reduces a finite positive integer;
            # there is no arbitrary pass limit and no history/video replay.

    def _compact_bank_to_native_fixed_point(self):
        if self._cursor != 0 or self._fallback_eviction_events:
            raise RuntimeError('Old bank must reach native fixed point before any FIFO wrap')
        write_position = 0
        pending_features, pending_meta, pending_frame = [], [], None

        def emit_group():
            nonlocal write_position, pending_features, pending_meta
            if not pending_features:
                return
            values = torch.cat(pending_features)
            metadata = {key: torch.cat([row[key] for row in pending_meta]) for key in pending_meta[0]}
            if len(values) > self.tokens_per_frame:
                raise RuntimeError('Stored frame group exceeds its original token count')
            values, metadata = self._native_sdc_fixed_point(values, metadata)
            if len(values):
                self.store.write_slots(np.arange(write_position, write_position + len(values), dtype=np.int64),
                                       values, metadata)
                write_position += len(values)
            pending_features, pending_meta = [], []

        for batch in self.store.iter_batches(max_tokens=max(1, self.tokens_per_frame)):
            frame_ids = batch['metadata']['frame_id']
            start = 0
            while start < len(frame_ids):
                current = int(frame_ids[start])
                end = start + 1
                while end < len(frame_ids) and int(frame_ids[end]) == current:
                    end += 1
                if pending_frame is not None and current != pending_frame:
                    emit_group()
                pending_frame = current
                # Clone the bounded piece so a view never retains an unrelated
                # whole batch while waiting for the next frame fragment.
                pending_features.append(batch['features'][start:end].clone())
                pending_meta.append({key: value[start:end].clone()
                                     for key, value in batch['metadata'].items()})
                start = end
        emit_group()
        self.store.truncate(write_position)
        self._native_bank_compactions += 1
        self._bank_sdc_fixedpoint = True

    def _append_fifo(self, features, metadata):
        n = len(features)
        if not n:
            return
        original_n = n
        if self._bank_sdc_fixedpoint or self.store.count + n > self.store.capacity:
            features, metadata = self._native_sdc_fixed_point(features, metadata)
            n = len(features)
            if self.store.count + n > self.store.capacity and not self._bank_sdc_fixedpoint:
                self._compact_bank_to_native_fixed_point()
        fallback_removed = 0
        if n > self.store.capacity:
            skip = n - self.store.capacity
            features = features[skip:]
            metadata = {key: value[skip:] for key, value in metadata.items()}
            n = len(features)
            self._evicted_tokens += skip
            fallback_removed += skip
        room = self.store.capacity - self.store.count
        add = min(n, room)
        if add:
            self.store.append(features[:add], {k: v[:add] for k, v in metadata.items()})
        rest = n - add
        if rest:
            slots = (np.arange(rest, dtype=np.int64) + self._cursor) % self.store.capacity
            self.store.write_slots(slots, features[add:], {k: v[add:] for k, v in metadata.items()})
            self._cursor = (self._cursor + rest) % self.store.capacity
            self._evicted_tokens += rest
            fallback_removed += rest
        if fallback_removed:
            if not self._bank_sdc_fixedpoint:
                raise RuntimeError('FIFO fallback attempted before native fixed-point compression')
            self._fallback_eviction_events += 1
            self._fallback_eviction_bytes += fallback_removed * self.store.record_bytes
            self._last_fallback_cause = 'native_sdc_fixed_point_exceeds_cap_after_reserved_native_tiers'
        self._finalized_tokens += original_n

    @torch.no_grad()
    def update(self, features, frame_id, timestamp, metadata=None):
        started = time.perf_counter()
        if (features.device.type != 'cpu' or features.shape != (self.tokens_per_frame, self.hidden_dim)
                or features.dtype != getattr(torch, self.dtype)):
            raise ValueError('FluxMem requires the same CPU [tokens,hidden] feature dtype and shape')
        self._core.ingest(features, self.height, self.width, frame_id, timestamp,
                          {'frame_id': frame_id, 'timestamp': float(timestamp)})
        self._core.advance()
        while self._core._long:
            grid = self._core._long.popleft()
            token_indices = (grid.positions[1] * self.width + grid.positions[2]).cpu()
            count = grid.tokens
            frame_ids = torch.full((count,), grid.time_id, dtype=torch.int64)
            timestamps = torch.full((count,), grid.timestamp_seconds, dtype=torch.float64)
            values = {'global_token_id': grid.time_id * self.tokens_per_frame + token_indices,
                      'frame_id': frame_ids, 'timestamp': timestamps, 'token_index': token_indices,
                      'merge_count': grid.metadata['merge_counts'],
                      'start_frame': frame_ids, 'end_frame': frame_ids,
                      'start_timestamp': timestamps, 'end_timestamp': timestamps}
            self._append_fifo(grid.embeddings.contiguous(), values)
        self._total_frames += 1
        self._total_tokens += len(features)
        self._total_feature_bytes += features.numel() * features.element_size()
        self._last_frame_id, self._last_timestamp = int(frame_id), float(timestamp)
        self._update_seconds += time.perf_counter() - started
        self.store.set_control(self._control())
        self.assert_budget()

    def stats(self):
        disk = self.store.stats()
        tier_bytes = self._tier_bytes()
        tier_tokens = sum(grid.tokens for queue in (self._core._short, self._core._medium) for grid in queue)
        retained = self.store.count + tier_tokens
        actual = disk['actual_memory_bytes'] + tier_bytes
        feature_bytes = retained * self.hidden_dim * torch.empty((), dtype=getattr(torch, self.dtype)).element_size()
        merge_anchors = self._core.num_merged_anchors + self._pressure_merged_anchors
        merge_members = (self._core.num_merged_tokens + self._core.num_merged_anchors
                         + self._pressure_merged_members)
        return {**self._control(), 'actual_memory_bytes': actual,
                'compressed_store_bytes': disk['actual_memory_bytes'], 'tier_persistent_bytes': tier_bytes,
                'retained_visual_tokens': retained, 'retained_tokens': retained,
                'feature_bytes': feature_bytes, 'metadata_bytes': actual - feature_bytes,
                'approximate_frame_equivalent': retained / self.tokens_per_frame,
                'compression_ratio': self._total_feature_bytes / actual if actual else 0.0,
                'payload_compression_ratio': self._total_tokens / retained if retained else 0.0,
                'num_merged_tokens': self._core.num_merged_tokens + self._pressure_merged_tokens,
                'average_merge_size': merge_members / merge_anchors if merge_anchors else 1.0,
                'average_merge_size_basis': 'original memberships per merge operation, including extra native passes',
                'effective_retention_ratio': actual / self._total_feature_bytes if self._total_feature_bytes else 0.0,
                'budget_utilization': actual / self.hard_budget_bytes,
                'tensor_shape': [retained, self.hidden_dim], 'config': self.config}

    def assert_budget(self):
        stats = self.stats()
        if stats['tier_persistent_bytes'] > self.tier_reserved_bytes:
            raise RuntimeError('FluxMem tier state exceeded its conservative byte reservation')
        if stats['actual_memory_bytes'] > self.hard_budget_bytes:
            raise RuntimeError(json.dumps({'method': self.method, 'timestamp': self._last_timestamp,
                'overshoot_bytes': stats['actual_memory_bytes'] - self.hard_budget_bytes,
                'cause': 'persistent FluxMem state exceeds absolute hard byte cap'}))
        return stats

    def snapshot(self, directory, timestamp=None):
        """Stream disk records and save bounded tiers; no historical replay."""
        directory = Path(directory)
        self.assert_budget()
        self.store.snapshot(directory, self._control())
        core_file = directory / 'fluxmem_tiers.pt'
        torch.save(self._core.state_dict(), core_file)
        config_file = directory / 'fluxmem_config.json'
        config_file.write_text(json.dumps({**self.config, 'core_file_sha256': _sha256_file(core_file)},
                                         indent=2) + '\n')
        # Snapshot copies are archival outputs; independently bound the complete
        # logical state that a loader needs, including serialized tier overhead.
        serialized = sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
        if serialized > self.hard_budget_bytes:
            raise RuntimeError(f'FluxMem serialized snapshot exceeds cap by {serialized-self.hard_budget_bytes} bytes')
        result = {**self.stats(), 'timestamp': timestamp,
                  'snapshot_serialized_bytes': serialized,
                  'core_file_sha256': _sha256_file(core_file)}
        return result

    def iter_batches(self, max_tokens=1024):
        """All retained records; caller can sort by recorded source timestamps."""
        yield from self.store.iter_batches(max_tokens=max_tokens)
        for queue in (self._core._medium, self._core._short):
            for grid in queue:
                for start in range(0, grid.tokens, max_tokens):
                    end = min(grid.tokens, start + max_tokens)
                    ids = (grid.positions[1, start:end] * self.width + grid.positions[2, start:end]).cpu()
                    count = end - start
                    frame = torch.full((count,), grid.time_id, dtype=torch.int64)
                    stamp = torch.full((count,), grid.timestamp_seconds, dtype=torch.float64)
                    yield {'features': grid.embeddings[start:end].detach().cpu().clone(),
                           'metadata': {'global_token_id': grid.time_id * self.tokens_per_frame + ids,
                                        'frame_id': frame, 'timestamp': stamp, 'token_index': ids,
                                        'merge_count': torch.ones(count, dtype=torch.int64),
                                        'start_frame': frame, 'end_frame': frame,
                                        'start_timestamp': stamp, 'end_timestamp': stamp}}

    def flush(self):
        self.store.set_control(self._control())
        self.store.flush()

    def close(self):
        self.store.set_control(self._control())
        self.store.close()


def load_snapshot(directory, working_directory):
    """Restore trusted local tensor state and independently checked disk records."""
    from .byte_memory import DiskTokenStore
    directory = Path(directory)
    # The generic disk loader validates its record hashes; callers should only
    # load their own local tensor snapshots (torch.save carries Python metadata).
    store = DiskTokenStore.load(directory, working_directory)
    control = store.control
    obj = FluxMemByteMemory.__new__(FluxMemByteMemory)
    obj.directory = Path(working_directory)
    obj.store = store
    obj.config = json.loads((directory / 'fluxmem_config.json').read_text())
    for attr, key in [('hard_budget_bytes', 'hard_budget_bytes'), ('hidden_dim', 'hidden_dim'),
                      ('dtype', 'dtype'), ('tokens_per_frame', 'tokens_per_frame'),
                      ('height', 'height'), ('width', 'width'), ('retention_ratio', 'retention_ratio'),
                      ('tier_reserved_bytes', 'tier_reserved_bytes'), ('short_grids', 'short_grids'),
                      ('medium_grids', 'medium_grids'), ('_cursor', 'fifo_cursor'),
                      ('_total_frames', 'total_frames_seen'), ('_total_tokens', 'total_visual_tokens_seen'),
                      ('_total_feature_bytes', 'full_feature_bytes_seen'), ('_last_frame_id', 'last_frame_id'),
                      ('_last_timestamp', 'last_timestamp'), ('_update_seconds', 'update_seconds'),
                      ('_evicted_tokens', 'evicted_compressed_tokens'), ('_finalized_tokens', 'finalized_compressed_tokens'),
                      ('_bank_sdc_fixedpoint', 'bank_sdc_fixedpoint'),
                      ('_native_recompression_events', 'native_recompression_events'),
                      ('_native_recompression_bytes_saved', 'native_recompression_bytes_saved'),
                      ('_native_bank_compactions', 'native_bank_compactions'),
                      ('_pressure_merged_tokens', 'pressure_merged_tokens'),
                      ('_pressure_merged_anchors', 'pressure_merged_anchors'),
                      ('_pressure_merged_members', 'pressure_merged_members'),
                      ('_fallback_eviction_events', 'fallback_eviction_events'),
                      ('_fallback_eviction_bytes', 'fallback_eviction_bytes'),
                      ('_last_fallback_cause', 'last_fallback_cause')]:
        setattr(obj, attr, control[key])
    obj.seed = 0
    if _sha256_file(directory / 'fluxmem_tiers.pt') != obj.config.pop('core_file_sha256'):
        raise ValueError('FluxMem tier snapshot hash mismatch')
    state = torch.load(directory / 'fluxmem_tiers.pt', map_location='cpu', weights_only=True)
    obj._core = _provenance_class().from_state_dict(state)
    for key in ['num_removed_temporal_duplicates', 'num_merged_tokens', 'num_merged_anchors']:
        setattr(obj._core, key, control[key])
    obj.assert_budget()
    return obj


class FluxMemSnapshotReader:
    """Read compressed records and bounded native tiers without copying shards."""
    def __init__(self, directory, verify=True):
        from .byte_memory import SnapshotReader
        directory = Path(directory)
        self.directory = directory
        self.store = SnapshotReader(directory)
        self.control = self.store.control
        self.config = json.loads((directory / 'fluxmem_config.json').read_text())
        try:
            if verify:
                self.store.verify()
            if _sha256_file(directory / 'fluxmem_tiers.pt') != self.config['core_file_sha256']:
                raise ValueError('FluxMem tier snapshot hash mismatch')
            state = torch.load(directory / 'fluxmem_tiers.pt', map_location='cpu', weights_only=True)
            self._core = _provenance_class().from_state_dict(state)
            for key in ['num_removed_temporal_duplicates', 'num_merged_tokens', 'num_merged_anchors']:
                setattr(self._core, key, self.control[key])
            for key in ['tokens_per_frame', 'height', 'width']:
                setattr(self, key, self.control[key])
            if self._core._total_ingested != self.control['total_frames_seen']:
                raise ValueError('FluxMem core and store frame counters differ')
            if self._core._total_encoded_tokens != self.control['total_visual_tokens_seen']:
                raise ValueError('FluxMem core and store token counters differ')
            if self.count > self.control['total_visual_tokens_seen']:
                raise ValueError('FluxMem snapshot retains more tokens than observed')
        except BaseException:
            self.store.close()
            raise

    def iter_batches(self, max_tokens=1024):
        yield from FluxMemByteMemory.iter_batches(self, max_tokens=max_tokens)

    @property
    def count(self):
        return self.store.count + sum(grid.tokens for queue in (self._core._medium, self._core._short)
                                      for grid in queue)

    @property
    def max_timestamp(self):
        """Newest retained short-tier observation; disk rows are earlier inputs."""
        stamps = [grid.timestamp_seconds for queue in (self._core._medium, self._core._short)
                  for grid in queue if grid.tokens]
        return max(stamps) if stamps else self.control['last_timestamp']

    def verify(self):
        checked = self.store.verify()
        if _sha256_file(self.directory / 'fluxmem_tiers.pt') != self.config['core_file_sha256']:
            raise ValueError('FluxMem tier snapshot hash mismatch')
        if self.max_timestamp != self.control['last_timestamp']:
            raise ValueError('FluxMem latest short-tier timestamp differs from its control')
        return {'disk_verification': checked, 'retained_visual_tokens': self.count,
                'disk_tokens': self.store.count, 'tier_tokens': self.count - self.store.count,
                'max_timestamp': self.max_timestamp, 'core_sha256_verified': True}

    def close(self):
        self.store.close()
