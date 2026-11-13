"""One-pass, token-bounded CPU memories for detached visual features.

The budget counts feature rows, not complete frames. A frame can therefore be
partly retained. No frame pixels or hidden full-history feature store is kept.

``uniform`` is deterministic online dyadic thinning of the *token* timeline:
global token indices are equally spaced by a power-of-two stride. This is a
bounded approximation to fixed-count linspace sampling, whose desired old
points cannot in general be recovered after eviction in a one-pass stream.
The stride doubles when necessary and occupancy can be below B. Reservoir
sampling instead gives every observed token the same inclusion probability.
"""
from __future__ import annotations

from copy import deepcopy
import math
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import torch

from baselines.streambench.accounting import measure_state


ALGORITHMS = {
    'fifo': 'token_fifo_ring',
    'uniform': 'dyadic_uniform_token_stride',
    'reservoir': 'algorithm_r_token_reservoir_pcg64',
}


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


class TokenMemory:
    """Retain at most ``budget`` visual rows with owned, fixed-capacity storage.

    ``actual_memory_bytes`` charges complete tensor allocations, including
    unused capacity, and deterministic scalar/RNG state using the existing
    streambench accountant. Python allocator headers, model weights, decoder
    buffers, update workspace and temporary snapshot copies are not this
    persistent-bank measure; runtime RSS is a separate metric.
    """

    def __init__(self, method, budget, seed=0):
        if method not in ALGORITHMS:
            raise ValueError(f'method must be one of {tuple(ALGORITHMS)}')
        self.method = method
        self.budget = _integer(budget, 'budget', 1)
        self.seed = _integer(seed, 'seed')
        self._rng = np.random.Generator(np.random.PCG64(self.seed)) if method == 'reservoir' else None
        self._features = None
        self._global_token_ids = None
        self._frame_ids = None
        self._frame_timestamps = None
        self._token_indices = None
        self._hidden_dim = None
        self._dtype = None
        self._count = 0
        self._cursor = 0
        self._stride = 1
        self._total_frames = 0
        self._total_tokens = 0
        self._total_feature_bytes = 0
        self._last_frame_id = None
        self._last_timestamp = None

    def _buffers(self):
        return {
            'features': self._features,
            'global_token_ids': self._global_token_ids,
            'frame_ids': self._frame_ids,
            'frame_timestamps': self._frame_timestamps,
            'token_indices': self._token_indices,
        }

    def _control(self):
        return {
            'schema_version': 1, 'method': self.method, 'budget': self.budget,
            'seed': self.seed, 'hidden_dim': self._hidden_dim,
            'dtype': str(self._dtype) if self._dtype is not None else None,
            'retained_tokens': self._count, 'fifo_cursor': self._cursor,
            'uniform_stride': self._stride, 'total_frames_seen': self._total_frames,
            'total_tokens_seen': self._total_tokens,
            'total_feature_bytes_seen': self._total_feature_bytes,
            'last_frame_id': self._last_frame_id, 'last_timestamp': self._last_timestamp,
            'rng_state': deepcopy(self._rng.bit_generator.state) if self._rng is not None else None,
        }

    def persistent_state(self):
        """Expose all owned persistent feature, index, counter and RNG contents."""
        return {
            'visual_memory_bytes': self._features,
            'metadata_bytes': {
                **{k: v for k, v in self._buffers().items() if k != 'features'},
                'control': self._control(),
            },
        }

    def _allocate(self, features):
        self._hidden_dim = int(features.shape[1])
        self._dtype = features.dtype
        self._features = torch.empty((self.budget, self._hidden_dim), dtype=self._dtype)
        self._global_token_ids = torch.empty(self.budget, dtype=torch.int64)
        self._frame_ids = torch.empty(self.budget, dtype=torch.int64)
        self._frame_timestamps = torch.empty(self.budget, dtype=torch.float64)
        self._token_indices = torch.empty(self.budget, dtype=torch.int64)

    def _validate(self, features, frame_id, timestamp):
        frame_id = _integer(frame_id, 'frame_id')
        if (isinstance(timestamp, bool) or not isinstance(timestamp, Real)
                or not math.isfinite(float(timestamp)) or timestamp < 0):
            raise ValueError('timestamp must be finite and nonnegative')
        timestamp = float(timestamp)
        if self._last_frame_id is not None and frame_id <= self._last_frame_id:
            raise ValueError('frame_id must strictly increase')
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError('timestamp must strictly increase')
        if (not isinstance(features, torch.Tensor) or features.ndim != 2
                or min(features.shape) < 1 or features.device.type != 'cpu'
                or features.layout != torch.strided
                or features.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
            raise ValueError('features must be a nonempty CPU [tokens, hidden_dim] fp16/bf16/fp32 tensor')
        if self._hidden_dim is not None and (features.shape[1] != self._hidden_dim or features.dtype != self._dtype):
            raise ValueError('feature hidden dimension and dtype must stay constant')
        if frame_id > np.iinfo(np.int64).max or self._total_tokens + len(features) > np.iinfo(np.int64).max:
            raise ValueError('token/frame index exceeds int64 storage')
        return frame_id, timestamp

    def _write_slice(self, destination, features, start, stop, frame_id, timestamp):
        count = stop - start
        slots = slice(destination, destination + count)
        self._features[slots].copy_(features[start:stop])
        self._global_token_ids[slots] = torch.arange(self._total_tokens + start, self._total_tokens + stop)
        self._frame_ids[slots] = frame_id
        self._frame_timestamps[slots] = timestamp
        self._token_indices[slots] = torch.arange(start, stop)

    def _write_indices(self, slots, source_indices, features, frame_id, timestamp):
        if len(slots) == 0:
            return
        slots = torch.as_tensor(np.asarray(slots, dtype=np.int64))
        source_indices = torch.as_tensor(np.asarray(source_indices, dtype=np.int64))
        self._features.index_copy_(0, slots, features.index_select(0, source_indices))
        self._global_token_ids[slots] = source_indices + self._total_tokens
        self._frame_ids[slots] = frame_id
        self._frame_timestamps[slots] = timestamp
        self._token_indices[slots] = source_indices

    @torch.no_grad()
    def update(self, features, frame_id, timestamp):
        """Consume exactly one frame; no feature row from a later frame is read."""
        frame_id, timestamp = self._validate(features, frame_id, timestamp)
        features = features.detach()
        if self._features is None:
            self._allocate(features)
        n = len(features)
        if self.method == 'fifo':
            if n >= self.budget:
                self._write_slice(0, features, n - self.budget, n, frame_id, timestamp)
                self._cursor = 0
                self._count = self.budget
            else:
                first = min(n, self.budget - self._cursor)
                self._write_slice(self._cursor, features, 0, first, frame_id, timestamp)
                if first < n:
                    self._write_slice(0, features, first, n, frame_id, timestamp)
                self._cursor = (self._cursor + n) % self.budget
                self._count = min(self.budget, self._count + n)
        elif self.method == 'uniform':
            end = self._total_tokens + n
            stride = self._stride
            while (end - 1) // stride + 1 > self.budget:
                stride *= 2
            if stride != self._stride:
                keep = torch.nonzero(self._global_token_ids[:self._count] % stride == 0, as_tuple=False).flatten()
                # The bank itself always has exactly B allocated slots. These
                # compacting copies are bounded update workspace, not history.
                for buffer in self._buffers().values():
                    buffer[:len(keep)].copy_(buffer.index_select(0, keep))
                self._count = len(keep)
                self._stride = stride
            first_global = ((self._total_tokens + stride - 1) // stride) * stride
            indices = np.arange(first_global, end, stride, dtype=np.int64) - self._total_tokens
            slots = np.arange(self._count, self._count + len(indices), dtype=np.int64)
            self._write_indices(slots, indices, features, frame_id, timestamp)
            self._count += len(indices)
        else:
            fill = min(n, self.budget - self._count)
            if fill:
                self._write_slice(self._count, features, 0, fill, frame_id, timestamp)
                self._count += fill
            indices = np.arange(fill, n, dtype=np.int64)
            if len(indices):
                # Algorithm R: each token i draws j uniformly in [0, i].
                # Apply only the last replacement per slot in this frame;
                # this is exactly the final state of sequential replacements.
                draws = self._rng.integers(0, self._total_tokens + indices + 1, dtype=np.int64)
                accepted = draws < self.budget
                slots, chosen = draws[accepted], indices[accepted]
                if len(slots):
                    _, reversed_positions = np.unique(slots[::-1], return_index=True)
                    last = len(slots) - 1 - reversed_positions
                    self._write_indices(slots[last], chosen[last], features, frame_id, timestamp)
        self._total_frames += 1
        self._total_tokens += n
        self._total_feature_bytes += features.numel() * features.element_size()
        self._last_frame_id = frame_id
        self._last_timestamp = timestamp
        if self._count > self.budget:
            raise AssertionError('retained token budget exceeded')

    def stats(self):
        accounting = measure_state(self.persistent_state())
        if self._count:
            # Correspondence is one timestamp per frame ID; preserve chronological
            # frame order even when reservoir physical slots are unordered.
            ids, positions = np.unique(self._frame_ids[:self._count].numpy(), return_index=True)
            timestamps = self._frame_timestamps[torch.from_numpy(positions)].tolist()
            payload = self._count * self._hidden_dim * self._features.element_size()
        else:
            ids, timestamps, payload = [], [], 0
        actual = accounting['persistent_total_bytes']
        return {
            'method': self.method, 'algorithm': ALGORITHMS[self.method], 'budget_tokens': self.budget,
            'total_frames_seen': self._total_frames, 'total_tokens_seen': self._total_tokens,
            'total_feature_bytes_seen': self._total_feature_bytes,
            'retained_items': len(ids), 'retained_visual_tokens': self._count,
            'retained_frame_ids': [int(x) for x in ids], 'retained_frame_timestamps': timestamps,
            'feature_bytes': payload, 'allocated_feature_bytes': accounting['visual_memory_bytes'],
            'metadata_bytes': accounting['metadata_bytes'], 'actual_memory_bytes': actual,
            'compression_ratio': self._total_feature_bytes / payload if payload else None,
            'actual_memory_compression_ratio': self._total_feature_bytes / actual if actual else None,
            'compression_ratio_definition': 'all_seen_feature_payload_bytes / retained_feature_payload_bytes',
            'actual_memory_definition': 'persistent owned allocations + scalar/RNG payload; excludes Python headers and transient workspace',
            'hidden_dimension': self._hidden_dim, 'dtype': str(self._dtype),
            'uniform_stride': self._stride if self.method == 'uniform' else None,
            'last_timestamp': self._last_timestamp,
        }

    def snapshot(self):
        """Return detached chronological copies; no bank storage is shared."""
        if self._count:
            order = torch.argsort(self._global_token_ids[:self._count])
            tensors = {k: v.index_select(0, order).contiguous() for k, v in self._buffers().items()}
        else:
            tensors = {}
        return {'tensors': tensors, 'metadata': self.stats(), 'state': self._control()}

    def save_snapshot(self, path, timestamp):
        """Write a torch checkpoint and return its actual serialized byte size.

        This helper refuses existing destinations. The runner may instead write
        snapshot tensors with safetensors and the metadata/control as JSON.
        """
        if (not isinstance(timestamp, Real) or not math.isfinite(timestamp)
                or timestamp < 0 or (self._last_timestamp is not None and timestamp <= self._last_timestamp)):
            raise ValueError('snapshot timestamp must be later than every observed frame')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        value = self.snapshot()
        value['metadata']['timestamp'] = float(timestamp)
        with path.open('xb') as handle:
            torch.save(value, handle)
        return {**value['metadata'], 'serialized_file_bytes': path.stat().st_size,
                'serialized_path': str(path)}
