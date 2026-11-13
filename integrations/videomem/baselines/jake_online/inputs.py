import hashlib
import json
import math
from pathlib import Path


def clock_seconds(value):
    value = str(value)
    if len(value) != 8 or not value.isdigit():
        raise ValueError('Expected HHMMSSFF')
    h, m, s, f = [int(value[i:i+2]) for i in range(0, 8, 2)]
    if h >= 24 or m >= 60 or s >= 60 or f >= 20:
        raise ValueError('Invalid 20fps timecode')
    return h * 3600 + m * 60 + s + f / 20


def global_time(day, value):
    day = int(str(day).removeprefix('DAY'))
    if not 1 <= day <= 7:
        raise ValueError('Invalid day')
    return (day - 1) * 86400 + clock_seconds(value)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def canonical_questions(raw):
    result = []
    for q in raw:
        choices = [q['choice_' + letter] for letter in 'abcd']
        answer = q['answer'].strip().upper()
        if answer not in 'ABCD' or len(answer) != 1:
            raise ValueError('Invalid ground truth')
        try:
            query_time = global_time(q['query_time']['date'], q['query_time']['time'])
            time_status = 'valid_20fps_timecode'
        except ValueError:
            query_time = None
            time_status = 'needs_review_invalid_20fps_timecode'
        result.append({'qa_id': str(q['ID']), 'day': int(q['query_time']['date'][3:]),
            'query_time': query_time, 'time_status': time_status,
            'original_query_time': q['query_time'], 'question': q['question'],
            'choices': choices, 'answer': answer, 'correct_idx': ord(answer) - 65,
            'need_audio': q['need_audio'], 'need_name': q['need_name'], 'type': q['type']})
    if len({q['qa_id'] for q in result}) != len(result):
        raise ValueError('Duplicate QA IDs')
    return sorted(result, key=lambda q: (q['day'], q['original_query_time']['time'], q['qa_id']))


def recent_samples(sources, cutoff, n=4, fps=1.0):
    """Latest N points on each clip's fixed sampling grid, including cutoff."""
    points = []
    for source in sources:
        start, duration = source['start_time'], source['duration']
        if start > cutoff:
            continue
        last = min(math.floor((cutoff - start) * fps + 1e-9),
                   math.ceil(duration * fps - 1e-9) - 1)
        for k in range(max(0, last - n + 1), last + 1):
            timestamp = start + k / fps
            if timestamp <= cutoff and k / fps < duration:
                points.append({'source': source['path'], 'day': source['day'],
                    'source_start_time': start, 'requested_source_seconds': k / fps,
                    'requested_global_timestamp': timestamp})
    return sorted(points, key=lambda p: (p['requested_global_timestamp'], p['source']))[-n:]


def decode_sample(root, record, cutoff):
    import av
    wanted = record['requested_source_seconds']
    chosen = None
    with av.open(str(Path(root) / record['source'])) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'SLICE'
        stream.codec_context.thread_count = 1
        origin = stream.start_time or 0
        container.seek(origin + int(wanted / float(stream.time_base)),
                       stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError('Frame without PTS')
            seconds = float((frame.pts - origin) * stream.time_base)
            if seconds > wanted + 1e-8:
                break
            chosen = (frame, seconds, frame.pts, str(stream.time_base))
    if chosen is None:
        raise ValueError('No causal decoded frame at requested timestamp')
    frame, seconds, pts, time_base = chosen
    image = frame.to_image()
    actual = record['source_start_time'] + seconds
    if actual > cutoff + 1e-8:
        raise ValueError('Future frame rejected')
    return image, dict(record, actual_source_seconds=seconds,
                       actual_global_timestamp=actual, source_pts=pts, time_base=time_base)
