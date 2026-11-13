"""Validated canonical inputs and bounded sampling on actual source clocks."""
from __future__ import annotations

import bisect
from fractions import Fraction
import json
import math
from pathlib import Path

import hashlib


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_time(value, name='timestamp'):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite nonnegative seconds')
    return Fraction(str(value))


def validate_sources(root, sources, check_files=True):
    root = Path(root).resolve()
    if not isinstance(sources, list) or not sources:
        raise ValueError('sources must be a nonempty list')
    paths, starts = set(), set()
    result = []
    for source in sources:
        row = dict(source)
        path = Path(row.get('path', ''))
        if str(path) == '.' or path.is_absolute() or '..' in path.parts or str(path) in paths:
            raise ValueError('Source paths must be unique relative paths inside root')
        absolute = (root / path).resolve()
        if not absolute.is_relative_to(root):
            raise ValueError('Source path escapes data root')
        paths.add(str(path))
        start = finite_time(row.get('start_time'), 'source start_time')
        duration = finite_time(row.get('duration'), 'source duration')
        for name, numeric in [('start_time', start), ('duration', duration)]:
            if name+'_fraction' in row and abs(Fraction(row[name+'_fraction'])-numeric) > Fraction(1,10**8):
                raise ValueError(f'{name} and exact fractional metadata disagree')
        fps = finite_time(row.get('fps'), 'source fps')
        if 'fps_fraction' in row:
            exact_fps = Fraction(row['fps_fraction'])
            if abs(exact_fps-fps) > Fraction(1,10**8):
                raise ValueError('fps and fps_fraction disagree')
            fps = exact_fps
        if duration <= 0 or fps <= 0 or type(row.get('frames')) is not int or row['frames'] < 1:
            raise ValueError('Positive duration, fps and frame count required')
        if abs(duration - Fraction(row['frames'], 1) / fps) > 1 / fps:
            raise ValueError('Source duration does not match its CFR frame count')
        if start in starts:
            raise ValueError('Sources cannot share a global start time')
        starts.add(start)
        day = row.get('day')
        if day is not None and (type(day) is not int or day < 1):
            raise ValueError('Optional source day must be a positive integer')
        if check_files:
            if not absolute.is_file():
                raise ValueError(f'Source is not a regular file: {path}')
            s = absolute.stat()
            state = [s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]
            if state != row.get('file_state'):
                raise ValueError(f'Source changed since canonical verification: {path}')
            if row.get('bytes', s.st_size) != s.st_size:
                raise ValueError(f'Source size mismatch: {path}')
        result.append(row)
    return sorted(result, key=lambda s: s['start_time'])


def load_inputs(root, canonical_path, sources_path, days=None):
    """Return (full manifest, verified sources, selected labeled questions)."""
    manifest = json.loads(Path(canonical_path).read_text())
    scorer = Path(__file__).parents[1] / 'streambench/scoring.py'
    if manifest.get('scorer_sha256') != sha256(scorer):
        raise ValueError('Canonical scorer hash does not match this checkout')
    if manifest.get('sources_sha256') is not None and manifest['sources_sha256'] != sha256(sources_path):
        raise ValueError('Canonical QA is bound to a different source index')
    sources = validate_sources(root, json.loads(Path(sources_path).read_text())['sources'])
    selected_days = None if days is None else set(days)
    if selected_days is not None and (not selected_days or any(type(d) is not int or d < 1 for d in selected_days)):
        raise ValueError('days must be a nonempty set of positive integers')
    # Prior-day sources remain available when evaluating a later-day subset.
    seen, questions = set(), []
    for q in manifest.get('questions', []):
        qa_id = q.get('qa_id')
        if not isinstance(qa_id, str) or not qa_id or qa_id in seen:
            raise ValueError('Canonical QA IDs must be unique nonempty strings')
        seen.add(qa_id)
        day = q.get('day')
        if day is not None and (type(day) is not int or day < 1):
            raise ValueError('Optional QA day must be a positive integer')
        if selected_days is not None and day not in selected_days:
            continue
        finite_time(q.get('query_time'), 'query_time')
        if not isinstance(q.get('question'), str) or not q['question'].strip():
            raise ValueError('Question text required')
        choices = q.get('choices')
        if not isinstance(choices, list) or not 2 <= len(choices) <= 26 or any(not isinstance(x, str) or not x.strip() for x in choices):
            raise ValueError('Multiple-choice text required')
        index = q.get('correct_idx')
        answer = q.get('answer')
        if index is None:
            matches = [i for i, text in enumerate(choices) if answer == text]
            if isinstance(answer, str) and len(answer.strip()) == 1 and 'A' <= answer.strip().upper() <= chr(64+len(choices)):
                index = ord(answer.strip().upper()) - 65
            elif len(matches) == 1:
                index = matches[0]
        if type(index) is not int or not 0 <= index < len(choices):
            raise ValueError('Canonical answer or correct_idx must identify an option')
        if answer is not None and answer not in (chr(65 + index), choices[index]):
            raise ValueError('Canonical answer and correct_idx disagree')
        questions.append(dict(q, correct_idx=index, answer=chr(65+index)))
    if not questions:
        raise ValueError('QA not supplied or no questions in requested day subset')
    return manifest, sources, sorted(questions, key=lambda q: (q['query_time'], q['qa_id']))


def _intervals(intervals, cutoff):
    if intervals is None:
        return [(Fraction(0), cutoff)]
    result = []
    for interval in intervals:
        if len(interval) != 2:
            raise ValueError('Each interval must be (global_start, global_end)')
        a, b = [finite_time(x, 'interval endpoint') for x in interval]
        if b < a:
            raise ValueError('Reversed interval')
        b = min(b, cutoff)
        if a <= b:
            result.append((a, b))
    merged = []
    for a, b in sorted(result):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def uniform_samples(sources, cutoff, max_frames, intervals=None):
    """Sample <=max_frames raw CFR positions uniformly over available frame count.

    Real day/gap timestamps are retained. Overlapping tails belong to the later
    source. Optional closed global intervals restrict evidence access; they are
    intersected with cutoff. This explicit benchmark sampler is not advertised
    as a model's untouched default sampling policy.
    """
    cutoff = finite_time(cutoff, 'cutoff')
    if type(max_frames) is not int or max_frames < 2:
        raise ValueError('max_frames must be an explicit integer >=2')
    sources = sorted(sources, key=lambda s: s['start_time'])
    intervals = _intervals(intervals, cutoff)
    segments, ends, total = [], [], 0
    for pos, source in enumerate(sources):
        start = Fraction(source['start_time_fraction']) if 'start_time_fraction' in source else finite_time(source['start_time'])
        duration = Fraction(source['duration_fraction']) if 'duration_fraction' in source else finite_time(source['duration'])
        fps = Fraction(source['fps_fraction']) if 'fps_fraction' in source else finite_time(source['fps'])
        if fps <= 0 or duration <= 0:
            raise ValueError('Positive source fps and duration required')
        if pos+1 < len(sources):
            next_source = sources[pos+1]
            next_start = Fraction(next_source['start_time_fraction']) if 'start_time_fraction' in next_source else finite_time(next_source['start_time'])
            duration = min(duration, next_start - start)
        if duration <= 0:
            raise ValueError('Source start times must be distinct')
        # A source's endpoint is exclusive, while query/interval cutoff is inclusive.
        last = min(source['frames']-1, math.ceil(duration * fps)-1, math.floor((cutoff-start)*fps))
        if last < 0:
            continue
        for a, b in intervals:
            first = max(0, math.ceil((a-start)*fps))
            final = min(last, math.floor((b-start)*fps))
            if first <= final:
                segments.append((source, first, final, fps))
                total += final-first+1
                ends.append(total)
    if total < 2:
        raise ValueError('Fewer than two causal raw frames are available')
    count = min(max_frames, total)
    positions = [round(Fraction(i*(total-1), count-1)) for i in range(count)]
    records = []
    for ordinal in positions:
        segment = bisect.bisect_right(ends, ordinal)
        source, first, _, fps = segments[segment]
        prior = ends[segment-1] if segment else 0
        frame = first + ordinal-prior
        seconds = Fraction(frame, 1)/fps
        source_start = Fraction(source['start_time_fraction']) if 'start_time_fraction' in source else finite_time(source['start_time'])
        global_seconds = source_start + seconds
        if global_seconds > cutoff:
            raise AssertionError('Sampler selected a future frame')
        records.append({'source':source['path'], 'day':source.get('day'),
            'source_start_time':source['start_time'], 'source_frame_index':frame,
            'requested_source_seconds':float(seconds),
            'requested_global_timestamp':float(global_seconds)})
    return records


def rate_uniform_samples(sources, cutoff, max_frames, *, sample_fps):
    """Uniformly cap a source-local rate grid, quantized to causal original frames.

    The published 0.5-fps/768-frame budget does not specify grid anchoring or
    rounding. These choices are explicit: anchor each source at its first frame,
    floor to its native CFR index, and uniformly subsample the complete prefix.
    """
    rate = finite_time(sample_fps, 'sample_fps')
    cutoff = finite_time(cutoff, 'cutoff')
    if rate <= 0 or type(max_frames) is not int or max_frames < 2:
        raise ValueError('Positive sample_fps and max_frames >=2 required')
    ordered = sorted(sources, key=lambda s:s['start_time'])
    segments, ends, total = [], [], 0
    for i, source in enumerate(ordered):
        start = Fraction(source.get('start_time_fraction', str(source['start_time'])))
        duration = Fraction(source.get('duration_fraction', str(source['duration'])))
        native = Fraction(source.get('fps_fraction', str(source['fps'])))
        if rate > native:
            raise ValueError('Sampling rate cannot exceed native source fps')
        if i+1 < len(ordered):
            following = ordered[i+1]
            duration = min(duration, Fraction(following.get('start_time_fraction', str(following['start_time'])))-start)
        if duration <= 0:
            raise ValueError('Positive distinct source spans required')
        last = min(math.ceil(duration*rate)-1,
                   math.ceil(Fraction(source['frames'])*rate/native)-1,
                   math.floor((cutoff-start)*rate))
        if last < 0:
            continue
        segments.append((source, start, native)); total += last+1; ends.append(total)
    if total < 2:
        raise ValueError('Fewer than two causal sampled frames are available')
    count = min(total, max_frames)
    positions = [round(Fraction(i*(total-1), count-1)) for i in range(count)]
    records = []
    for position in positions:
        segment = bisect.bisect_right(ends, position)
        source, start, native = segments[segment]
        candidate = position-(ends[segment-1] if segment else 0)
        frame = math.floor(Fraction(candidate)*native/rate)
        seconds = Fraction(frame)/native
        if start+seconds > cutoff:
            raise AssertionError('Future sampled frame')
        records.append({'source':source['path'],'day':source.get('day'),
            'source_start_time':source['start_time'],'source_frame_index':frame,
            'requested_source_seconds':float(seconds),'requested_global_timestamp':float(start+seconds),
            'sampling_fps':float(rate),'candidate_index':candidate})
    return records


def decode_samples(root, samples, cutoff, *, workers=1):
    """Decode selected original frames and verify actual causal, unique PTS."""
    if not samples:
        raise ValueError('No selected frames')
    if type(workers) is not int or workers < 1:
        raise ValueError('Decode workers must be a positive integer')
    if workers == 1:
        pairs = [decode_sample(root, record, cutoff) for record in samples]
    else:
        from concurrent.futures import ThreadPoolExecutor
        grouped = {}
        for index, record in enumerate(samples):
            grouped.setdefault(record['source'], []).append((index,record))
        def decode_group(group):
            indices, records = zip(*group)
            decoded = ([decode_sample(root,records[0],cutoff)] if len(records)==1
                       else decode_source_samples(root,records,cutoff))
            return list(zip(indices,decoded))
        # One container/codec thread per source, so densely sampled clips decode
        # once. Restore caller order before the unchanged PTS/uniqueness audit.
        with ThreadPoolExecutor(max_workers=min(workers, len(grouped))) as pool:
            decoded = [item for group in pool.map(decode_group,grouped.values()) for item in group]
        pairs = [pair for _,pair in sorted(decoded,key=lambda item:item[0])]
    frames, records = zip(*pairs)
    times = [r['actual_global_timestamp'] for r in records]
    if any(t > cutoff + 1e-8 for t in times) or any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError('Decoded PTS must be causal and strictly increasing')
    if len({(r['source'], r['source_pts']) for r in records}) != len(records):
        raise ValueError('Duplicate decoded frame')
    return list(frames), list(records)


def decode_source_samples(root, samples, cutoff):
    """Read several requested PTS from one source with the serial decoder rule."""
    import av
    if len({r['source'] for r in samples}) != 1:
        raise ValueError('Grouped decoder requires exactly one source')
    ordered=sorted(enumerate(samples),key=lambda item:item[1]['requested_source_seconds'])
    results={};position=0;chosen=None
    def emit():
        nonlocal position
        index,record=ordered[position]
        if chosen is None:raise ValueError('No causal decoded frame at requested timestamp')
        frame,seconds,pts,time_base=chosen
        actual=record['source_start_time']+seconds
        if actual>cutoff+1e-8:raise ValueError('Future frame rejected')
        results[index]=(frame.to_image(),dict(record,actual_source_seconds=seconds,
            actual_global_timestamp=actual,source_pts=pts,time_base=time_base))
        position+=1
    with av.open(str(Path(root)/samples[0]['source'])) as container:
        stream=container.streams.video[0]
        stream.thread_type='SLICE';stream.codec_context.thread_count=1
        origin=stream.start_time or 0
        container.seek(origin+int(ordered[0][1]['requested_source_seconds']/float(stream.time_base)),
                       stream=stream,backward=True,any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is None:raise ValueError('Frame without PTS')
            seconds=float((frame.pts-origin)*stream.time_base)
            while position<len(ordered) and seconds>ordered[position][1]['requested_source_seconds']+1e-8:
                emit()
            if position==len(ordered):break
            chosen=(frame,seconds,frame.pts,str(stream.time_base))
        while position<len(ordered):emit()
    return [results[i] for i in range(len(samples))]


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
