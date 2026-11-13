"""Single-pass 1 Hz source inventory and decoding, without raw frame files.

The decoder is the existing StreamForest causal single-frame iterator. Its
zero-origin sampling grid resets per source and its presentation timestamps are
mapped to one cumulative playback clock. Images are owned by the iterator and
closed on advance; callers must encode each image before requesting another.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import time

from baselines.streamforest.streamforest_videomem.video import iter_frames
from local_videomem.data import _probe, _state, prepare_manifest


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a finite positive number')
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a finite positive number')
    return Fraction(str(value))


def inspect_sources(paths, status_path=None, timeline_path=None):
    """Inspect sources in caller order, using actual probed video durations.

    With status_path, only individually SHA256-verified completed downloads are
    admitted for decoding. Their trusted download digest is reused, rather than
    rereading tens of GB; this trust decision is recorded, and current file stat
    is bound to the manifest and checked throughout decoding. Missing/pending
    sources can have *provisional* durations from timeline_path for planning.

    Without a download status, existing local files are hashed through the
    repository's prepare_manifest function. This mode is useful for independent
    local sources and tests and does not claim remote-source verification.
    """
    if not paths:
        raise ValueError('At least one video path is required')
    paths = [Path(p).expanduser().resolve() for p in paths]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate source paths would replay video frames')
    day_specs = []
    for order, path in enumerate(paths, 1):
        match = re.fullmatch(r'day([0-9]+)', path.stem, flags=re.I)
        day_number = int(match.group(1)) if match else order
        day = f'Day{day_number}' if match else f'Source{order}'
        day_specs.append((day_number, day))
    if len({day for _, day in day_specs}) != len(day_specs):
        raise ValueError('Source Day IDs must be distinct; duplicate Day basenames are ambiguous')
    status = json.loads(Path(status_path).read_text()) if status_path else None
    downloaded = {x['path']: x for x in status.get('files', [])} if status else {}
    timeline_days = {}
    if timeline_path:
        timeline = json.loads(Path(timeline_path).read_text())
        timeline_days = {int(x['day']): x for x in timeline.get('days', [])}

    sources = []
    elapsed = Fraction(0)
    all_durations_known = True
    for order, path in enumerate(paths, 1):
        day_number, day = day_specs[order - 1]
        base = {'path': str(path), 'day_id': day, 'day_order': order,
                'day_start_seconds': 0.0, 'timeline_status': 'direct_mp4'}
        expected = downloaded.get(path.name)
        verified = False
        remote_verified = False
        reason = None
        if status is not None:
            verified = bool(expected and expected.get('status') == 'verified'
                            and expected.get('sha256')
                            and expected.get('sha256') == expected.get('lfs_sha256')
                            and expected.get('local_path')
                            and Path(expected['local_path']).resolve() == path
                            and path.is_file()
                            and path.stat().st_size == expected.get('size'))
            if verified:
                before = _state(path)
                probe = _probe(path)
                if before != _state(path):
                    raise ValueError(f'Source changed during probing: {path}')
                digest = expected['sha256']
                if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                    raise ValueError(f'Invalid completed source digest: {path}')
                identity = {**base, 'source_fingerprint': digest}
                source = {**identity, **probe,
                          'day_end_seconds': probe['duration_seconds'],
                          'source_id': 'source_' + hashlib.sha256(
                              json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24],
                          'provenance': {'source_state': before,
                              'source_fingerprint_algorithm': 'sha256',
                              'fingerprint_reused_from_verified_download': True,
                              'fingerprint_rehashed_this_run': False,
                              'download_status_path': str(Path(status_path).resolve()),
                              'download_repo_id': status.get('repo_id'),
                              'download_revision': status.get('revision'),
                              'timeline_status': 'direct_mp4',
                              'time_basis': 'day_playback_seconds'}}
                remote_verified = True
            else:
                reason = 'download_not_individually_verified_or_file_mismatch'
        elif path.is_file():
            source = prepare_manifest([base])['videos'][0]
            source['provenance']['fingerprint_rehashed_this_run'] = True
            verified = True
        else:
            reason = 'missing_local_video'

        if not verified:
            provisional = timeline_days.get(day_number, {}).get('duration_seconds')
            duration = _number(provisional, 'timeline duration') if provisional is not None else None
            source = {**base, 'duration_seconds': float(duration) if duration else None,
                      'duration_fraction': str(duration) if duration else None,
                      'width': None, 'height': None,
                      'day_end_seconds': float(duration) if duration else None,
                      'unavailable_reason': reason,
                      'provenance': {'duration_basis': 'annotation_timeline_provisional'
                                     if duration else 'unknown'}}
        duration = Fraction(source['duration_fraction']) if source['duration_fraction'] else None
        source.update({'verified': verified, 'remote_source_verified': remote_verified,
                       'duration_provisional': not verified,
                       'global_start_seconds': float(elapsed) if all_durations_known else None,
                       'global_start_fraction': str(elapsed) if all_durations_known else None,
                       'expected_1fps_frames': math.ceil(duration) if duration else None})
        if duration is not None and all_durations_known:
            elapsed += duration
        else:
            all_durations_known = False
        source['global_end_seconds'] = float(elapsed) if all_durations_known else None
        sources.append(source)
    all_verified = all(s['verified'] for s in sources)
    return {'schema_version': 'feature_memory_source_inventory_v1', 'sources': sources,
            'fps': 1.0, 'sampling': 'first PTS at/after each per-source integer-second target',
            'time_basis': 'cumulative video playback duration, source order as supplied',
            'interval_convention': '[start,end)', 'audio_decoded': False,
            'raw_frames_saved': False, 'all_verified': all_verified,
            'counts_provisional': not all_verified,
            'count_basis': 'ceil(video duration * 1fps) per source; actual decode count checked at run end',
            'total_duration_seconds': float(elapsed) if all_durations_known else None,
            'total_duration_fraction': str(elapsed) if all_durations_known else None,
            'total_1fps_frames': sum(s['expected_1fps_frames'] for s in sources)
                                if all_durations_known else None,
            'status_path': str(Path(status_path).resolve()) if status_path else None,
            'decoder_reused_from': 'baselines/streamforest/streamforest_videomem/video.py:iter_frames'}


def iter_stream(inventory, max_seconds=None, stats=None, start_frame=0):
    """Yield RGB images exactly once, with independent decoder-pipeline timing.

    decode_seconds measures next() on the reused video iterator: video decoding,
    RGB conversion, timestamp checks and pixel hashing, excluding caller time
    and all model work. No encoder, questions, annotations, audio or raw-frame
    storage enter this path. Internal codec reorder buffers are not model input.
    """
    if stats is None:
        stats = {}
    if isinstance(start_frame, bool) or not isinstance(start_frame, int) or start_frame < 0:
        raise ValueError('start_frame must be a nonnegative integer')
    stats.update({'decoded_frames': 0, 'sampled_frames': 0, 'max_live_sampled_frames': 0,
                  'decode_seconds': 0.0, 'decode_tail_seconds': 0.0,
                  'source_open_count': 0, 'start_frame': start_frame,
                  'prefix_frames_decoded_skipped': 0, 'decode_prefix_skipped_seconds': 0.0,
                  'frames_yielded': 0,
                  'resume_sampling': 'decode prefix without encoding; preserves exact existing PTS sampling'})
    sources = inventory.get('sources')
    if not sources:
        raise ValueError('Inventory has no sources')
    cutoff = _number(max_seconds, 'max_seconds') if max_seconds is not None else None
    if cutoff is None:
        if not inventory.get('all_verified'):
            raise ValueError('Full-history run requires every selected source to be verified')
        cutoff = Fraction(inventory['total_duration_fraction'])
    selected = []
    elapsed = Fraction(0)
    for source in sources:
        if elapsed >= cutoff:
            break
        if not source.get('verified'):
            raise ValueError(f'Cannot decode an unverified source: {source["path"]}')
        duration = Fraction(source['duration_fraction'])
        selected.append(source)
        elapsed += duration
    if not selected:
        return
    if cutoff > elapsed:
        cutoff = elapsed
    last = selected[-1]
    last_start = Fraction(last['global_start_fraction'])
    local_cutoff = cutoff - last_start
    query = {'day_id': last['day_id'], 'day_order': last['day_order'], 'seconds': float(local_cutoff)}
    manifest = {'videos': selected}
    offsets = {s['day_id']: Fraction(s['global_start_fraction']) for s in selected}
    raw_stats = {}
    raw = iter_frames(manifest, query, fps=1, max_side=None, stats=raw_stats)
    previous = None
    opened_days = set()
    expected = sum(math.ceil(min(Fraction(s['duration_fraction']),
                                    max(Fraction(0), cutoff - Fraction(s['global_start_fraction']))))
                   for s in selected)
    stats['expected_1fps_frames'] = expected
    if start_frame > expected:
        raise ValueError('start_frame exceeds the selected stream length')
    try:
        while True:
            started = time.perf_counter()
            try:
                item = next(raw)
            except StopIteration:
                tail = time.perf_counter() - started
                stats['decode_seconds'] += tail
                stats['decode_tail_seconds'] = tail
                break
            decode_seconds = time.perf_counter() - started
            stats['decode_seconds'] += decode_seconds
            metadata = item['metadata']
            timestamp = offsets[metadata['day_id']] + Fraction(metadata['file_seconds_fraction'])
            if timestamp >= cutoff or (previous is not None and timestamp <= previous):
                raise ValueError('Noncausal or non-increasing cumulative sample timestamp')
            previous = timestamp
            opened_days.add(metadata['day_id'])
            stats.update({'decoded_frames': raw_stats['decoded_frames'],
                          'sampled_frames': raw_stats['sampled_frames'],
                          'max_live_sampled_frames': raw_stats['max_live_sampled_frames'],
                          'source_open_count': len(opened_days)})
            if item['frame_index'] < start_frame:
                stats['prefix_frames_decoded_skipped'] += 1
                stats['decode_prefix_skipped_seconds'] += decode_seconds
                continue
            stats['frames_yielded'] += 1
            yield {'frame': item['frame'], 'frame_id': item['frame_index'],
                   'timestamp': float(timestamp), 'timestamp_fraction': str(timestamp),
                   'day': metadata['day_id'], 'source_timestamp': metadata['file_seconds'],
                   'decode_seconds': decode_seconds, 'metadata': metadata}
        if stats['sampled_frames'] != expected:
            raise ValueError(f'Actual 1fps samples differ from inventory: {stats["sampled_frames"]}/{expected}; '
                             'source has gaps, low frame rate, or unexpected timing')
    finally:
        raw.close()
        stats['decoded_frames'] = raw_stats.get('decoded_frames', 0)
