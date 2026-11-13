"""Generic CFR source sampling and native raw decoding on the dataset clock."""
from fractions import Fraction
import itertools
import math
from pathlib import Path

from baselines.ours_offline.inputs import finite_time


def source_time(source, key):
    numeric = finite_time(source[key], key)
    return Fraction(source[key+'_fraction']) if key+'_fraction' in source else numeric


def sample_plan(sources, fps=1.0):
    rate = finite_time(fps, 'sampling fps')
    if rate <= 0:
        raise ValueError('sampling fps must be positive')
    sources = sorted(sources, key=lambda s: s['start_time'])
    result = []
    for pos, source in enumerate(sources):
        native = source_time(source, 'fps')
        if native <= 0 or rate > native:
            raise ValueError('sampling fps must not exceed any source native fps')
        if type(source['frames']) is not int or source['frames'] < 1:
            raise ValueError('Positive source frame count required')
        start = source_time(source, 'start_time')
        duration = min(source_time(source, 'duration'), Fraction(source['frames'], 1)/native)
        if pos+1 < len(sources):
            duration = min(duration, source_time(sources[pos+1], 'start_time') - start)
        if duration <= 0:
            raise ValueError('Source durations must be positive and start times distinct')
        for index in range(math.ceil(duration*rate)):
            frame_index = math.floor(Fraction(index, 1)*native/rate)
            seconds = Fraction(frame_index, 1)/native
            result.append({'source': source['path'], 'source_start_time': source['start_time'],
                'day': source.get('day'), 'source_frame_index': frame_index,
                'source_fps': float(native), 'requested_source_seconds': float(seconds),
                'requested_global_timestamp': float(start+seconds), 'global_frame_id': len(result)})
    if not result:
        raise ValueError('No source frames')
    return result


def decode_window(root, samples):
    """Seek once per source/chunk; preserve exact selected source frame identity.

    The verified index describes CFR inputs. A mismatched decoded timestamp is an
    error rather than inventing a frame index for a variable-rate stream.
    """
    import av
    frames, records = [], []
    for source, grouped in itertools.groupby(samples, key=lambda s: s['source']):
        group = list(grouped)
        pending = 0
        with av.open(str(Path(root)/source)) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 2
            origin = stream.start_time or 0
            container.seek(origin + int(group[0]['requested_source_seconds']/float(stream.time_base)),
                           stream=stream, backward=True, any_frame=False)
            previous = None

            def append(frame):
                nonlocal pending
                item = group[pending]
                seconds = float((frame.pts-origin)*stream.time_base)
                if abs(seconds-item['requested_source_seconds']) > 1e-6:
                    raise ValueError('Decoded frame does not match indexed CFR frame timestamp')
                timestamp = item['source_start_time']+seconds
                if timestamp > item['requested_global_timestamp']+1e-8:
                    raise ValueError('Future decoded frame')
                frames.append(frame.to_image())
                records.append({'stream_id': 'ours', 'day': item['day'], 'source_id': source,
                    'source_frame_index': item['source_frame_index'], 'source_timestamp': seconds,
                    'source_pts': frame.pts, 'time_base': str(stream.time_base),
                    'global_frame_id': item['global_frame_id'], 'global_timestamp': timestamp})
                pending += 1

            for frame in container.decode(stream):
                if frame.pts is None:
                    raise ValueError('Missing video PTS')
                seconds = float((frame.pts-origin)*stream.time_base)
                while pending < len(group) and seconds > group[pending]['requested_source_seconds']+1e-8:
                    if previous is None:
                        raise ValueError('Missing causal frame')
                    append(previous)
                if pending == len(group):
                    break
                previous = frame
            while pending < len(group):
                if previous is None:
                    raise ValueError('Empty video')
                append(previous)
    return frames, records
