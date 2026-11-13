"""Model-independent, query-independent memory retaining the latest N frames.

Tensor ownership is CPU-only. Normal selection and state export return owned
copies; ``state_dict(copy=False)`` borrows live tensors solely for accounting.
Byte accounting includes tensor payloads and canonical JSON state descriptors,
not Python allocator overhead or process RSS.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
import re

import torch


SCHEMA = 'simplestream_recent_window_v1'
_FEATURE_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
_GRID_DTYPES = {torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64}
_HISTORY_FIELDS = ('first_observed_frame_id', 'first_observed_timestamp',
                   'last_observed_frame_id', 'last_observed_timestamp',
                   'evicted_through_frame_id', 'evicted_through_timestamp')
_STATE_KEYS = {'schema', 'window_frames', 'observed_count', 'frames', *_HISTORY_FIELDS}
_FRAME_KEYS = {'frame_id', 'timestamp', 'features', 'feature_shape', 'feature_dtype',
               'grid_thw', 'grid_shape', 'grid_dtype', 'metadata'}


def _positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _frame_id(value):
    if type(value) is not int or value < 0:
        raise ValueError('frame_id must be a nonnegative integer')
    return value


def _timestamp(value, name='timestamp'):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be finite numeric seconds')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite numeric seconds')
    return value


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def _metadata(value):
    if not isinstance(value, dict):
        raise ValueError('Frame metadata must be a JSON dictionary')

    def inspect(node):
        if isinstance(node, dict):
            for key, item in node.items():
                if not isinstance(key, str):
                    raise ValueError('Frame metadata keys must be strings')
                normalized = re.sub(r'[^a-z0-9]+', '_', key.lower()).strip('_')
                flags = {'query_independent', 'query_free', 'question_independent'}
                if normalized in flags:
                    if item is not True:
                        raise ValueError('Memory metadata must describe query-independent observations')
                elif (normalized in {'question', 'questions', 'query', 'queries', 'answer',
                        'answers', 'label', 'labels', 'qa', 'ground_truth', 'target_answer'} or
                        normalized.startswith(('question_', 'answer_', 'query_', 'label_', 'ground_truth_'))):
                    raise ValueError(f'Question/answer/label metadata is not allowed: {key}')
                inspect(item)
        elif isinstance(node, list):
            for item in node:
                inspect(item)
        elif node is not None and type(node) not in (str, int, float, bool):
            raise ValueError('Frame metadata must contain only JSON values')

    inspect(value)
    try:
        return json.loads(_json(value))
    except (ValueError, TypeError) as error:
        raise ValueError('Frame metadata contains an invalid/nonfinite JSON value') from error


@dataclass(frozen=True)
class EncodedFrame:
    frame_id: int
    timestamp: float
    features: torch.Tensor
    grid_thw: torch.Tensor
    metadata: dict

    def __post_init__(self):
        frame_id, timestamp = _frame_id(self.frame_id), _timestamp(self.timestamp)
        metadata = _metadata(self.metadata)
        features, grid = self.features, self.grid_thw
        if (not isinstance(features, torch.Tensor) or features.layout != torch.strided or
                features.ndim != 2 or min(features.shape) <= 0 or features.dtype not in _FEATURE_DTYPES):
            raise ValueError('Features must be a nonempty dense floating tensor [rows, hidden]')
        if (not isinstance(grid, torch.Tensor) or grid.layout != torch.strided or
                tuple(grid.shape) not in ((3,), (1, 3)) or grid.dtype not in _GRID_DTYPES):
            raise ValueError('grid_thw must be an integer tensor [3] or [1,3]')
        features = features.detach().to(device='cpu', copy=True).contiguous()
        grid = grid.detach().to(device='cpu', dtype=torch.int64, copy=True).reshape(3).contiguous()
        if not bool(torch.isfinite(features).all()):
            raise ValueError('Encoded features must be finite')
        if not bool((grid > 0).all()):
            raise ValueError('Every grid_thw dimension must be positive')
        object.__setattr__(self, 'frame_id', frame_id)
        object.__setattr__(self, 'timestamp', timestamp)
        object.__setattr__(self, 'features', features)
        object.__setattr__(self, 'grid_thw', grid)
        object.__setattr__(self, 'metadata', metadata)


def _owned_frame(frame):
    if not isinstance(frame, EncodedFrame):
        raise TypeError('observe expects one EncodedFrame')
    return EncodedFrame(frame.frame_id, frame.timestamp, frame.features, frame.grid_thw, frame.metadata)


def _frame_record(frame, clone_tensors):
    return dict(frame_id=frame.frame_id, timestamp=frame.timestamp,
        features=frame.features.clone() if clone_tensors else frame.features,
        feature_shape=list(frame.features.shape), feature_dtype=str(frame.features.dtype),
        grid_thw=frame.grid_thw.clone() if clone_tensors else frame.grid_thw,
        grid_shape=list(frame.grid_thw.shape), grid_dtype=str(frame.grid_thw.dtype),
        metadata=_metadata(frame.metadata))


class RecentWindow:
    def __init__(self, window_frames):
        self._window_frames = _positive_integer(window_frames, 'window_frames')
        self._observed_count = 0
        self._frames = deque()
        for name in _HISTORY_FIELDS:
            setattr(self, '_'+name, None)

    @property
    def window_frames(self):
        return self._window_frames

    @property
    def observed_count(self):
        return self._observed_count

    def observe(self, frame):
        # Validate and acquire ownership before changing any live frame/history.
        owned = _owned_frame(frame)
        if self._observed_count and (owned.frame_id <= self._last_observed_frame_id or
                                     owned.timestamp <= self._last_observed_timestamp):
            raise ValueError('Observed frame IDs and timestamps must both strictly increase')
        if not self._observed_count:
            self._first_observed_frame_id = owned.frame_id
            self._first_observed_timestamp = owned.timestamp
        if len(self._frames) == self.window_frames:
            evicted = self._frames.popleft()
            self._evicted_through_frame_id = evicted.frame_id
            self._evicted_through_timestamp = evicted.timestamp
        self._frames.append(owned)
        self._last_observed_frame_id = owned.frame_id
        self._last_observed_timestamp = owned.timestamp
        self._observed_count += 1

    def selected(self, cutoff=None):
        if cutoff is None:
            return [_owned_frame(frame) for frame in self._frames]
        cutoff = _timestamp(cutoff, 'cutoff')
        if not self._observed_count or cutoff < self._first_observed_timestamp:
            return []
        if self._evicted_through_timestamp is not None and cutoff < self._last_observed_timestamp:
            raise ValueError('Historical cutoff requires evicted frames; this recent window cannot reconstruct them')
        return [_owned_frame(frame) for frame in self._frames if frame.timestamp <= cutoff]

    def state_dict(self, *, copy=True):
        """Return a validated-format snapshot; copy=False borrows read-only tensors.

        The borrowed form is only for live-storage accounting. Mutating its
        tensors violates ownership; ordinary callers should use the default.
        Metadata always remains an owned copy, including in the borrowed form.
        """
        if type(copy) is not bool:
            raise ValueError('copy must be a boolean')
        state = dict(schema=SCHEMA, window_frames=self.window_frames, observed_count=self.observed_count,
                     frames=[_frame_record(frame, copy) for frame in self._frames])
        state.update({name:getattr(self, '_'+name) for name in _HISTORY_FIELDS})
        return state

    @classmethod
    def from_state_dict(cls, state):
        if not isinstance(state, dict) or set(state) != _STATE_KEYS or state.get('schema') != SCHEMA:
            raise ValueError('Invalid SimpleStream recent-window state schema/fields')
        memory = cls(state['window_frames'])
        count, records = state['observed_count'], state['frames']
        if type(count) is not int or count < 0 or not isinstance(records, list) or len(records) != min(count, memory.window_frames):
            raise ValueError('State frame coverage does not match its observation count/window')
        frames = []
        for record in records:
            if not isinstance(record, dict) or set(record) != _FRAME_KEYS:
                raise ValueError('Invalid encoded-frame state fields')
            features, grid = record['features'], record['grid_thw']
            if (not isinstance(features, torch.Tensor) or features.device.type != 'cpu' or
                    list(features.shape) != record['feature_shape'] or str(features.dtype) != record['feature_dtype']):
                raise ValueError('State feature shape/dtype/device descriptor mismatch')
            if (not isinstance(grid, torch.Tensor) or grid.device.type != 'cpu' or
                    list(grid.shape) != record['grid_shape'] or str(grid.dtype) != record['grid_dtype'] or
                    tuple(grid.shape) != (3,) or grid.dtype != torch.int64):
                raise ValueError('State grid shape/dtype/device descriptor mismatch')
            frames.append(EncodedFrame(record['frame_id'], record['timestamp'], features, grid, record['metadata']))
        if any(b.frame_id <= a.frame_id or b.timestamp <= a.timestamp for a,b in zip(frames, frames[1:])):
            raise ValueError('State retained frame IDs/timestamps are not strictly ordered')
        history = {name:state[name] for name in _HISTORY_FIELDS}
        if not count:
            if any(value is not None for value in history.values()):
                raise ValueError('Empty memory has nonempty observation history')
        else:
            first_id = _frame_id(history['first_observed_frame_id'])
            first_time = _timestamp(history['first_observed_timestamp'])
            last_id = _frame_id(history['last_observed_frame_id'])
            last_time = _timestamp(history['last_observed_timestamp'])
            if ((last_id, last_time) != (frames[-1].frame_id, frames[-1].timestamp) or
                    first_id > frames[0].frame_id or first_time > frames[0].timestamp or
                    last_id-first_id+1 < count):
                raise ValueError('State first/last observation history is inconsistent')
            if count <= memory.window_frames:
                if ((first_id, first_time) != (frames[0].frame_id, frames[0].timestamp) or
                        history['evicted_through_frame_id'] is not None or history['evicted_through_timestamp'] is not None):
                    raise ValueError('State claims eviction before its window was exceeded')
            else:
                evicted_id = _frame_id(history['evicted_through_frame_id'])
                evicted_time = _timestamp(history['evicted_through_timestamp'])
                if not (first_id <= evicted_id < frames[0].frame_id and first_time <= evicted_time < frames[0].timestamp):
                    raise ValueError('State eviction boundary is inconsistent with retained frames')
                if count-memory.window_frames == 1 and (evicted_id,evicted_time) != (first_id,first_time):
                    raise ValueError('State single-frame eviction history is inconsistent')
                if evicted_id-first_id+1 < count-memory.window_frames:
                    raise ValueError('State eviction count exceeds the possible observed IDs')
        memory._frames = deque(frames)
        memory._observed_count = count
        for name,value in history.items():
            setattr(memory, '_'+name, value)
        return memory

    def memory_byte_breakdown(self):
        state = self.state_dict(copy=False)
        feature_bytes = grid_bytes = 0
        for record in state['frames']:
            features, grid = record.pop('features'), record.pop('grid_thw')
            feature_bytes += features.numel()*features.element_size()
            grid_bytes += grid.numel()*grid.element_size()
        metadata_bytes = len(_json(state))
        return dict(feature_bytes=feature_bytes, grid_bytes=grid_bytes, metadata_bytes=metadata_bytes,
                    total_bytes=feature_bytes+grid_bytes+metadata_bytes)

    def memory_bytes(self):
        return self.memory_byte_breakdown()['total_bytes']
