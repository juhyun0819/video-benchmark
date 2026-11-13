"""Causal recent-frame selection from the existing verified Qwen feature archive.

The archive's source inventory supplies a small timeline index. Selection uses
the stored float timestamp convention, including fractional Day boundaries;
it does not equate seconds with global frame IDs. Only selected frames are
materialized. The frozen reader verifies their containing shards in full.
"""
from __future__ import annotations

from bisect import bisect_right
import copy
from fractions import Fraction
from pathlib import Path

import torch

from baselines.rekv_qwen25.rekv_qwen25.inputs import _header, read_feature_frames
from .memory import EncodedFrame, _positive_integer, _timestamp


class QwenFeatureArchive:
    def __init__(self, root):
        self.root = Path(root)
        self._header = _header(self.root)
        self._sources = []
        self._starts = []
        first = 0
        for source in self._header['inventory']['sources']:
            self._starts.append(first)
            self._sources.append(dict(first=first, count=source['expected_1fps_frames'],
                timestamp=Fraction(source['global_start_fraction'])))
            first += source['expected_1fps_frames']
        self._total_frames = first

    @property
    def header(self):
        """Owned copy of the pinned, verified archive header."""
        return copy.deepcopy(self._header)

    def _stored_timestamp(self, frame_id):
        source = self._sources[bisect_right(self._starts, frame_id)-1]
        return float(source['timestamp'] + frame_id-source['first'])

    def _eligible_end(self, cutoff):
        # Binary search uses exactly float(Fraction(global_start)+local_id),
        # matching the frozen reader. Fraction(str(cutoff)) floor arithmetic
        # would incorrectly exclude a frame at a rounded nondecimal boundary.
        lo, hi = 0, self._total_frames
        while lo < hi:
            middle = (lo+hi)//2
            if self._stored_timestamp(middle) <= cutoff:
                lo = middle+1
            else:
                hi = middle
        return lo

    def recent(self, cutoff_seconds, window_frames):
        cutoff = _timestamp(cutoff_seconds, 'cutoff_seconds')
        window = _positive_integer(window_frames, 'window_frames')
        if _header(self.root) != self._header:
            raise ValueError('Feature archive header/source identity changed after opening')
        end = self._eligible_end(cutoff)
        if not end:
            return []
        start = max(0, end-window)
        features, rows = read_feature_frames(self.root, start, end-start)
        if len(rows) != end-start or features.shape[0] != len(rows):
            raise ValueError('Verified feature reader returned incomplete frame coverage')
        result = []
        for offset, row in enumerate(rows):
            frame_id = start+offset
            timestamp = self._stored_timestamp(frame_id)
            if (row['frame_id'] != frame_id or row['global_frame_id'] != frame_id or
                    row['timestamp'] != timestamp or row['timestamp_seconds'] != timestamp or timestamp > cutoff or
                    row['visual_archive_id'] != self._header['archive_id']):
                raise ValueError('Selected frame violates archive identity, timeline, or causal cutoff')
            result.append(EncodedFrame(frame_id=frame_id, timestamp=timestamp,
                features=features[offset], grid_thw=torch.tensor(row['grid_thw'], dtype=torch.int64),
                metadata=row))
        if _header(self.root) != self._header:
            raise ValueError('Feature archive header/source identity changed during selection')
        return result
