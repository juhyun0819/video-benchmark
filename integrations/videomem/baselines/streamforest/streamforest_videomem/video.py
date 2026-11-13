"""Causal single-frame sampling with one owned RGB image at a time.

The iterator owns each yielded PIL image. Encode or copy it before advancing or
closing the iterator; both operations close the previous image. Decoder buffers
are separate from the bounded sampled-image count. No resize is done by default:
the native StreamForest visual encoder applies its own image preprocessing.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import math
from pathlib import Path

from .shared import load_shared

load_shared()
from local_videomem.data import _query_time, _seconds, _state, select_prefix


def time_key(value):
    """Chronological Day/order key for an explicit query or frame envelope."""
    if not isinstance(value, dict):
        raise ValueError('Time must be an explicit object')
    value = value.get('query_time', value.get('time', value))
    checked = _query_time(value)
    return checked['day_order'], Fraction(str(checked['seconds']))


def frame_key(value):
    """Use exact source PTS metadata instead of rounded presentation floats."""
    if not isinstance(value, dict):
        raise ValueError('Frame time must be an explicit object')
    metadata = value.get('metadata', value)
    if not isinstance(metadata, dict) or 'day_seconds' not in metadata:
        return time_key(value)
    checked = _query_time({'day_id': metadata['day_id'], 'day_order': metadata['day_order'],
                           'seconds': metadata['day_seconds']})
    seconds = Fraction(metadata.get('day_seconds_fraction', str(checked['seconds'])))
    if seconds < 0 or float(seconds) != checked['seconds']:
        raise ValueError('Frame exact PTS must match its nonnegative presentation time')
    if 'time' in value:
        envelope = _query_time(value['time'])
        if envelope != checked:
            raise ValueError('Frame time and source metadata disagree')
    return checked['day_order'], seconds


def _check_source(segment):
    path = Path(segment['path'])
    fingerprint = segment['source_fingerprint']
    if (not isinstance(fingerprint, str) or len(fingerprint) != 64
            or any(char not in '0123456789abcdef' for char in fingerprint)):
        raise ValueError(f'Invalid source SHA256 fingerprint: {path}')
    if _state(path) != segment['provenance']['source_state']:
        raise ValueError(f'Source changed since manifest preparation or during decoding: {path}')


def iter_frames(manifest, query_time, *, fps=1, max_side=None, stats=None):
    """Yield source-local samples strictly before the reviewed observation cutoff.

    Each source uses its own zero-origin sampling grid. Select the first actual
    presentation timestamp at/after a sampling target, never a frame at/after
    the cutoff. Frames are not duplicated when requested FPS exceeds source FPS.
    Original Day/file times are retained even across gaps and Day transitions.
    """
    rate = _seconds(fps, 'fps', positive=True)
    if max_side is not None and (type(max_side) is not int or max_side < 1):
        raise ValueError('max_side must be None or a positive integer')
    if stats is None:
        stats = {}
    if not isinstance(stats, dict):
        raise ValueError('stats must be a dictionary')
    for key in ('decoded_frames', 'sampled_frames', 'yielded_frames', 'max_live_sampled_frames'):
        stats[key] = 0
    segments = select_prefix(manifest, query_time)
    import av
    from PIL import Image

    frame_index = 0
    for segment in segments:
        _check_source(segment)
        path = Path(segment['path'])
        end = Fraction(segment['file_end_fraction'])
        day_start = Fraction(str(segment['day_start_seconds']))
        next_sample, previous_pts = Fraction(0), None
        image = None
        try:
            with av.open(str(path), mode='r') as container:
                if not container.streams.video:
                    raise ValueError(f'No video stream: {path}')
                stream = container.streams.video[0]
                stream.thread_type = 'SLICE'
                stream.codec_context.thread_count = 1
                origin = Fraction(stream.start_time or 0) * Fraction(stream.time_base)
                for decoded in container.decode(stream):
                    stats['decoded_frames'] += 1
                    if decoded.pts is None or decoded.time_base is None:
                        raise ValueError(f'Video frame has no exact PTS: {path}')
                    raw_pts = Fraction(decoded.pts) * Fraction(decoded.time_base)
                    actual = raw_pts - origin
                    if previous_pts is not None and actual < previous_pts:
                        raise ValueError(f'Video presentation timestamps are not ordered: {path}')
                    previous_pts = actual
                    if actual >= end:
                        break
                    if actual < next_sample:
                        continue
                    image = decoded.to_image()
                    if image.mode != 'RGB':
                        original = image
                        try:
                            image = original.convert('RGB')
                        finally:
                            original.close()
                    if max_side is not None and max(image.size) > max_side:
                        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                    day_seconds = day_start + actual
                    metadata = {
                        'position': frame_index, 'source_id': segment['source_id'], 'path': str(path),
                        'day_id': segment['day_id'], 'day_order': segment['day_order'],
                        'file_seconds': float(actual), 'day_seconds': float(day_seconds),
                        'file_seconds_fraction': str(actual), 'day_seconds_fraction': str(day_seconds),
                        'pts': decoded.pts, 'time_base': str(decoded.time_base),
                        'raw_pts_seconds_fraction': str(raw_pts),
                        'sample_target_seconds': float(next_sample),
                        'source_fingerprint': segment['source_fingerprint'],
                        'pixel_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                        'width': image.width, 'height': image.height,
                        'mode': 'RGB', 'decode_device': 'cpu',
                    }
                    item = {'frame': image, 'frame_index': frame_index,
                            'time': {'day_id': segment['day_id'], 'day_order': segment['day_order'],
                                     'seconds': float(day_seconds)},
                            'metadata': metadata}
                    stats['sampled_frames'] += 1
                    stats['yielded_frames'] += 1
                    stats['max_live_sampled_frames'] = 1
                    next_sample = Fraction(math.floor(actual * rate) + 1, 1) / rate
                    _check_source(segment)
                    try:
                        yield item
                    finally:
                        image.close()
                        image = None
                        _check_source(segment)
                    frame_index += 1
                    if next_sample >= end:
                        break
        finally:
            if image is not None:
                image.close()
            _check_source(segment)
