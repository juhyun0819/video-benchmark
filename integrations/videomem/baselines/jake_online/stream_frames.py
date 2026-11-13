"""Chronological raw decoding at a fixed source-local sample rate."""
import itertools
import math
from pathlib import Path


def sample_plan(sources, fps=2):
    plan = []
    sources = sorted(sources, key=lambda s: s['start_time'])
    for index, s in enumerate(sources):
        # Some clips include a short tail overlapping the next clip. The
        # next source owns its start time; never ingest the overlap twice.
        duration = s['duration']
        if index + 1 < len(sources):
            duration = min(duration, sources[index + 1]['start_time'] - s['start_time'])
        if duration <= 0:
            raise ValueError('Sources must have distinct chronological start times')
        for i in range(math.ceil(duration * fps - 1e-9)):
            plan.append({'source': s['path'], 'source_start_time': s['start_time'],
                'day': s['day'], 'requested_source_seconds': i/fps,
                'requested_global_timestamp': s['start_time'] + i/fps})
    plan.sort(key=lambda r: r['requested_global_timestamp'])
    assert all(a['requested_global_timestamp'] < b['requested_global_timestamp'] for a,b in zip(plan, plan[1:]))
    for i, record in enumerate(plan): record['global_frame_id'] = i
    return plan


def decode_window(root, samples):
    import av
    frames, records = [], []
    for source, grouped in itertools.groupby(samples, key=lambda s: s['source']):
        group = list(grouped)
        pending = 0
        with av.open(str(Path(root) / source)) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 2
            origin = stream.start_time or 0
            container.seek(origin + int(group[0]['requested_source_seconds'] / float(stream.time_base)),
                           stream=stream, backward=True, any_frame=False)
            previous = None
            def append(frame):
                nonlocal pending
                item = group[pending]
                seconds = float((frame.pts-origin) * stream.time_base)
                assert seconds <= item['requested_source_seconds'] + 1e-8
                timestamp = item['source_start_time'] + seconds
                frames.append(frame.to_image())
                records.append({'stream_id':'A1_JAKE', 'day':item['day'], 'source_id':source,
                    'source_frame_index':round(seconds*20), 'source_timestamp':seconds,
                    'global_frame_id':item['global_frame_id'], 'global_timestamp':timestamp})
                pending += 1
            for frame in container.decode(stream):
                if frame.pts is None: raise ValueError('Missing video PTS')
                seconds = float((frame.pts-origin) * stream.time_base)
                while pending < len(group) and seconds > group[pending]['requested_source_seconds'] + 1e-8:
                    if previous is None: raise ValueError('Missing causal frame')
                    append(previous)
                if pending == len(group): break
                previous = frame
            while pending < len(group):
                if previous is None: raise ValueError('Empty video')
                append(previous)
    assert len(frames) == len(samples)
    assert all(a['global_timestamp'] < b['global_timestamp'] for a,b in zip(records, records[1:]))
    return frames, records
