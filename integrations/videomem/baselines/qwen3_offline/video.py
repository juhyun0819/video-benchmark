"""Exact raw-frame indexing for a concatenated, equal-rate CFR video timeline.

No sampling policy lives here: callers pass the indices selected by the official
Qwen3 video processor. Only constant-rate, equal-size inputs are supported, and
selected decoded PTS must match the inventory's rational frame grid exactly.
"""
from bisect import bisect_right
from fractions import Fraction
import math
from pathlib import Path
import subprocess
import json


def _state(path):
    stat = Path(path).stat()
    return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
            'ctime_ns': stat.st_ctime_ns, 'inode': stat.st_ino, 'device': stat.st_dev}


def _positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _fraction(value, name, *, positive=False):
    if not isinstance(value, str):
        raise ValueError(f'{name} must be a fraction string')
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f'Invalid {name}') from exc
    if result < 0 or (positive and result == 0):
        raise ValueError(f'Invalid {name}')
    return result


def build_inventory(paths):
    """Probe ordered original files; no decoding, downsampling or re-encoding.

    All sources must have equal rational FPS, matching avg/r frame rates,
    dimensions, and exact frame-count/rate duration. Container/audio duration is
    never used as a video timeline boundary.
    """
    if not isinstance(paths, list) or not paths:
        raise ValueError('paths must be a nonempty ordered list')
    sources, offset = [], Fraction(0)
    for day, supplied in enumerate(paths, 1):
        path = Path(supplied).expanduser().resolve(strict=True)
        before = _state(path)
        if not path.is_file() or before['size'] <= 0:
            raise ValueError(f'Video must be a nonempty file: {path}')
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_streams',
             '-of', 'json', str(path)], capture_output=True, text=True,
            check=True, timeout=60,
        )
        try:
            stream = json.loads(result.stdout)['streams'][0]
            fps = Fraction(stream['avg_frame_rate'])
            rate = Fraction(stream['r_frame_rate'])
            frames = int(stream['nb_frames'])
            time_base = Fraction(stream['time_base'])
            duration = int(stream['duration_ts']) * time_base
            start_pts = int(stream['start_pts'])
            width, height = int(stream['width']), int(stream['height'])
        except (KeyError, IndexError, ValueError, TypeError, ZeroDivisionError) as exc:
            raise ValueError(f'Exact CFR stream metadata required: {path}') from exc
        if (fps <= 0 or rate != fps or frames <= 0 or time_base <= 0
                or duration != frames / fps or min(width, height) <= 0
                or (1 / fps / time_base).denominator != 1):
            raise ValueError(f'Unsupported CFR rate, duration or PTS grid: {path}')
        if _state(path) != before:
            raise ValueError(f'Source changed during inventory: {path}')
        sources.append({'path': str(path), 'day_id': f'Day{day}', 'day_order': day,
                        'fps': str(fps), 'num_frames': frames,
                        'duration_fraction': str(duration),
                        'global_start_fraction': str(offset), 'width': width,
                        'height': height, 'start_pts': start_pts,
                        'time_base': str(time_base), 'state': before})
        offset += duration
    inventory = {'sources': sources, 'fps': sources[0]['fps'],
                 'total_frames': sum(s['num_frames'] for s in sources),
                 'duration_fraction': str(offset)}
    validate_inventory(inventory)
    return inventory


def validate_inventory(inventory):
    """Validate cumulative CFR metadata and unchanged source files; return rate/duration."""
    if not isinstance(inventory, dict):
        raise ValueError('inventory must be an object')
    try:
        sources = inventory['sources']
        if not isinstance(sources, list) or not sources:
            raise ValueError('inventory must contain sources')
        fps = _fraction(inventory['fps'], 'fps', positive=True)
        offset, count, dimensions = Fraction(0), 0, None
        paths = set()
        for day, source in enumerate(sources, 1):
            if not isinstance(source, dict):
                raise ValueError('source must be an object')
            if source['day_id'] != f'Day{day}' or type(source['day_order']) is not int or source['day_order'] != day:
                raise ValueError('Source days must be consecutive in input order')
            if _fraction(source['fps'], 'source fps', positive=True) != fps:
                raise ValueError('All sources must have the same CFR rate')
            frames = _positive_integer(source['num_frames'], 'num_frames')
            size = (_positive_integer(source['width'], 'width'),
                    _positive_integer(source['height'], 'height'))
            if dimensions is not None and size != dimensions:
                raise ValueError('All source dimensions must match')
            dimensions = size
            if _fraction(source['duration_fraction'], 'duration', positive=True) != frames / fps:
                raise ValueError('CFR duration does not match frame count/rate')
            if _fraction(source['global_start_fraction'], 'global start') != offset:
                raise ValueError('Source boundaries must be exact cumulative durations')
            time_base = _fraction(source['time_base'], 'time_base', positive=True)
            if type(source['start_pts']) is not int or (1 / fps / time_base).denominator != 1:
                raise ValueError('Unsupported CFR PTS grid')
            path = Path(source['path'])
            if not path.is_absolute() or str(path) in paths:
                raise ValueError('Source paths must be distinct absolute paths')
            paths.add(str(path))
            if _state(path) != source['state']:
                raise ValueError(f'Source changed since inventory: {path}')
            offset += frames / fps
            count += frames
        if _positive_integer(inventory['total_frames'], 'total_frames') != count:
            raise ValueError('Inventory total frame count mismatch')
        if _fraction(inventory['duration_fraction'], 'total duration', positive=True) != offset:
            raise ValueError('Inventory total duration mismatch')
        return fps, offset
    except (KeyError, TypeError, OSError) as exc:
        raise ValueError(f'Invalid or unavailable inventory source: {exc}') from exc


def _cutoff(query_time, duration):
    if isinstance(query_time, bool) or not isinstance(query_time, (int, float, Fraction)) or not math.isfinite(query_time):
        raise ValueError('query_time must be finite global seconds')
    cutoff = Fraction(str(query_time))
    if cutoff < 0 or cutoff > duration:
        raise ValueError('query_time outside available video timeline')
    return cutoff


def prefix_frame_count(inventory, query_time):
    """Number of raw frames whose global presentation time is <= query_time."""
    fps, duration = validate_inventory(inventory)
    cutoff = _cutoff(query_time, duration)
    return min(inventory['total_frames'], math.floor(cutoff * fps) + 1)


def map_indices(inventory, indices, query_time):
    """Map official sampled raw ordinals to exact source and global times.

    Sorted duplicates are preserved, since an official sampler may repeat frames.
    Global frame_id is the original raw ordinal, NOT an old 1fps archive ID.
    """
    count = prefix_frame_count(inventory, query_time)
    if not isinstance(indices, list):
        raise ValueError('indices must be a list of integer raw frame ordinals')
    starts, total = [], 0
    for source in inventory['sources']:
        starts.append(total)
        total += source['num_frames']
    records, previous = [], -1
    for index in indices:
        if type(index) is not int or index < 0 or index >= count or index < previous:
            raise ValueError('Sampled indices must be sorted and inside the causal prefix')
        previous = index
        position = bisect_right(starts, index) - 1
        source = inventory['sources'][position]
        source_index = index - starts[position]
        stamp = source_index / Fraction(source['fps'])
        records.append({'path': source['path'], 'day_id': source['day_id'],
                        'day_order': source['day_order'], 'frame_id': index,
                        'source_frame_index': source_index,
                        'source_seconds_fraction': str(stamp),
                        'global_seconds_fraction': str(Fraction(source['global_start_fraction']) + stamp)})
    return records


def decode_indices(inventory, indices, query_time):
    """Seek/decode selected original RGB pixels, verifying exact presentation PTS.

    Keyframe preroll is decoded internally; only requested causal frames are
    materialized as RGB arrays. Codec reference frames are not model observations.
    """
    import av
    records = map_indices(inventory, indices, query_time)
    sources = {s['path']: s for s in inventory['sources']}
    arrays = []
    current_path, container, stream = None, None, None
    last_key, last_array = None, None
    try:
        for record in records:
            source = sources[record['path']]
            if record['path'] != current_path:
                if container is not None:
                    container.close()
                if _state(source['path']) != source['state']:
                    raise ValueError('Source changed before decoding')
                container = av.open(source['path'], mode='r')
                stream = container.streams.video[0]
                stream.codec_context.thread_count = 1
                stream.thread_type = 'SLICE'
                current_path = record['path']
                if Fraction(stream.time_base) != Fraction(source['time_base']) or stream.start_time != source['start_pts']:
                    raise ValueError('Decoded stream PTS metadata differs from inventory')
            key = (record['path'], record['source_frame_index'])
            if key == last_key:
                arrays.append(last_array.copy())
                continue
            expected = (source['start_pts'] * Fraction(source['time_base'])
                        + Fraction(record['source_seconds_fraction']))
            ticks = expected / Fraction(stream.time_base)
            if ticks.denominator != 1:
                raise ValueError('Selected frame does not lie on source PTS grid')
            container.seek(int(ticks), stream=stream, backward=True, any_frame=False)
            found = None
            previous = None
            for frame in container.decode(stream):
                if frame.pts is None or frame.time_base is None:
                    raise ValueError('Decoded frame has no exact PTS')
                actual = frame.pts * Fraction(frame.time_base)
                local = actual - source['start_pts'] * Fraction(source['time_base'])
                if local < 0 or (local * Fraction(source['fps'])).denominator != 1:
                    raise ValueError('Decoded frame violates CFR PTS grid')
                if previous is not None and actual <= previous:
                    raise ValueError('Decoded PTS are not strictly increasing')
                previous = actual
                if actual > expected:
                    break
                if actual == expected:
                    if (frame.width, frame.height) != (source['width'], source['height']):
                        raise ValueError('Decoded dimensions differ from inventory')
                    found = frame.to_ndarray(format='rgb24')
                    break
            if found is None:
                raise ValueError(f'Missing exact selected frame PTS: {key}')
            arrays.append(found)
            last_key, last_array = key, found
        validate_inventory(inventory)
        return arrays, records
    finally:
        if container is not None:
            container.close()
