"""Thirty-second, source-local media windows for EgoGPT caption generation.

Clips never cross a day boundary. Video sampling is exactly one original frame
per source-local second, without image resizing or a 16-frame cap. Audio remains
unpadded mono float32 at 16 kHz; the caller applies the native Whisper processor.
"""
from fractions import Fraction
import json
import hashlib
import os
from pathlib import Path
import tempfile
import math
import subprocess

import numpy as np

from baselines.qwen3_offline.video import map_indices, validate_inventory


AUDIO_SAMPLE_RATE = 16000
CLIP_SECONDS = 30


def _rate(inventory):
    fps, _ = validate_inventory(inventory)
    if fps.denominator != 1 or fps < 1:
        raise ValueError('Exact 1fps caption sampling requires an integer source FPS >= 1')
    return int(fps)


def _clip(source, raw_start, number, fps):
    start = Fraction(number * CLIP_SECONDS)
    end = min(start + CLIP_SECONDS, Fraction(source['duration_fraction']))
    if start < 0 or start >= end:
        raise ValueError('Clip number is outside its source')
    global_start = Fraction(source['global_start_fraction'])
    indices = [raw_start + int(start * fps) + second * fps
               for second in range(math.ceil(end - start))]
    return {'clip_id': f"{source['day_id'].lower()}_{number:06d}",
            'day_id': source['day_id'], 'day_order': source['day_order'],
            'path': source['path'], 'source_start_fraction': str(start),
            'source_end_fraction': str(end),
            'global_start_fraction': str(global_start + start),
            'global_end_fraction': str(global_start + end),
            'frame_indices': indices, 'duration_fraction': str(end - start)}


def build_clips(inventory):
    """Create complete 30-second coverage, including each day's partial final clip."""
    fps = _rate(inventory)
    clips, raw_start = [], 0
    for source in inventory['sources']:
        for number in range(math.ceil(Fraction(source['duration_fraction']) / CLIP_SECONDS)):
            clips.append(_clip(source, raw_start, number, fps))
        raw_start += source['num_frames']
    return clips


def _validated_clip(inventory, clip):
    fps = _rate(inventory)
    if not isinstance(clip, dict):
        raise ValueError('clip must be a manifest record')
    raw_start = 0
    for source in inventory['sources']:
        if source['day_id'] == clip.get('day_id'):
            try:
                start = Fraction(clip['source_start_fraction'])
                number = start / CLIP_SECONDS
                if number.denominator != 1:
                    raise ValueError('Clip start must lie on a source-local 30-second boundary')
                expected = _clip(source, raw_start, int(number), fps)
            except (KeyError, TypeError, ZeroDivisionError, ValueError) as exc:
                raise ValueError('Invalid clip boundary') from exc
            if any(clip.get(k) != v for k, v in expected.items()):
                raise ValueError('Clip differs from its exact source manifest window')
            if any(type(index) is not int for index in clip['frame_indices']):
                raise ValueError('Raw frame indices must be integers')
            return source, expected
        raw_start += source['num_frames']
    raise ValueError('Clip day is not in inventory')


def _audio_stream(source):
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries',
         'stream=index,sample_rate,channels,start_time', '-of', 'json', source['path']],
        capture_output=True, text=True, check=True, timeout=60,
    )
    streams = json.loads(result.stdout).get('streams', [])
    if not streams:
        raise ValueError(f"Missing audio stream for audiovisual caption: {source['path']}")
    video_origin = source['start_pts'] * Fraction(source['time_base'])
    audio_origin = Fraction(streams[0].get('start_time', '0'))
    if video_origin != audio_origin:
        raise ValueError('Audio/video stream origins differ; explicit alignment is required')
    return streams[0]


def _audio_recipe():
    import librosa
    import importlib.metadata
    import soxr
    import soundfile
    return {'loader': 'librosa.load', 'librosa_version': librosa.__version__,
            'versions': {name: importlib.metadata.version(name) for name in
                         ['numpy','scipy','numba','soxr','soundfile','audioread']},
            'libsoxr': soxr.__libsoxr_version__, 'libsndfile': soundfile.__libsndfile_version__,
            'ffmpeg_version': subprocess.check_output(['ffmpeg','-version'],text=True),
            'sample_rate': AUDIO_SAMPLE_RATE, 'mono': True, 'dtype': 'float32',
            'res_type': 'librosa.load default', 'scope': 'full source before source-local slicing'}


def _load_audio(source):
    import librosa
    stream = _audio_stream(source)
    waveform, rate = librosa.load(source['path'], sr=AUDIO_SAMPLE_RATE)
    if rate != AUDIO_SAMPLE_RATE or waveform.ndim != 1 or waveform.dtype != np.float32:
        raise ValueError('Native audio loader returned unexpected waveform format')
    if not waveform.size or not np.isfinite(waveform).all():
        raise ValueError('Decoded audio is empty or nonfinite')
    return waveform, stream


def _cache_state(path):
    stat = path.stat()
    return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
            'ctime_ns': stat.st_ctime_ns, 'inode': stat.st_ino}


def _cache_paths(directory, source):
    base = Path(directory) / source['day_id'].lower()
    return base.with_suffix('.npy'), base.with_suffix('.json')


def _read_cache(directory, source):
    data_path, receipt_path = _cache_paths(directory, source)
    try:
        receipt = json.loads(receipt_path.read_text())
        if (receipt['source_path'] != source['path'] or receipt['source_state'] != source['state']
                or receipt['recipe'] != _audio_recipe()
                or receipt['cache_state'] != _cache_state(data_path)):
            raise ValueError('Audio cache identity or file state changed')
        waveform = np.load(data_path, mmap_mode='r', allow_pickle=False)
        if (waveform.ndim != 1 or waveform.dtype != np.float32
                or waveform.size != receipt['sample_count']):
            raise ValueError('Audio cache payload does not match its receipt')
        return waveform, receipt
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError('Audio cache is missing or invalid') from exc


def _atomic_json(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def prepare_audio_cache(inventory, directory):
    """Cache native full-source librosa waveforms once, with atomic npy receipts.

    This is an audio preprocessing cache, not visual or model features. Existing
    valid cache artifacts are reused; changed source/runtime/payload is rejected.
    """
    _rate(inventory)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    receipts = []
    for source in inventory['sources']:
        data_path, receipt_path = _cache_paths(directory, source)
        if receipt_path.exists():
            waveform, receipt = _read_cache(directory, source)
            del waveform
            receipts.append(receipt)
            continue
        waveform, stream = _load_audio(source)
        validate_inventory(inventory)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile('wb', dir=directory, delete=False) as handle:
                temporary = Path(handle.name)
                np.save(handle, waveform, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, data_path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        sha256 = hashlib.sha256()
        with data_path.open('rb') as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                sha256.update(block)
        receipt = {'source_path': source['path'], 'source_state': source['state'],
                   'source_audio_stream': stream, 'recipe': _audio_recipe(),
                   'sample_count': int(waveform.size), 'cache_state': _cache_state(data_path),
                   'sha256': sha256.hexdigest()}
        del waveform
        _atomic_json(receipt_path, receipt)
        receipts.append(receipt)
    validate_inventory(inventory)
    return receipts


def decode_clip(inventory, clip, audio_cache_dir=None):
    """Return official Decord RGB frames and source-local native librosa audio.

    Audio is loaded/resampled from the full original source, exactly as the
    official adapter does; only its erroneous time-of-day slice is corrected to
    source-local clip seconds. Optional cached npy waveforms preserve those values.
    """
    from decord import VideoReader, cpu
    source, clip = _validated_clip(inventory, clip)
    start = Fraction(clip['source_start_fraction'])
    end = Fraction(clip['source_end_fraction'])
    duration = end - start
    cutoff = Fraction(clip['global_end_fraction']) - 1 / Fraction(source['fps'])
    frames_metadata = map_indices(inventory, clip['frame_indices'], cutoff)
    if any(record['path'] != source['path']
           or not (start <= Fraction(record['source_seconds_fraction']) < end)
           for record in frames_metadata):
        raise ValueError('Selected frame lies outside its caption clip')
    reader = VideoReader(source['path'], ctx=cpu(0), num_threads=1)
    if len(reader) != source['num_frames'] or abs(reader.get_avg_fps()-float(Fraction(source['fps']))) > 1e-6:
        raise ValueError('Decord source frame count/rate differs from inventory')
    indices = [record['source_frame_index'] for record in frames_metadata]
    timestamps = reader.get_frame_timestamp(indices)
    if hasattr(timestamps, 'asnumpy'):
        timestamps = timestamps.asnumpy()
    for record, stamp in zip(frames_metadata, timestamps):
        expected = float(Fraction(record['source_seconds_fraction']))
        # Decord exposes timestamp floats; use half one stream time-base tick.
        tolerance = max(1e-6, float(Fraction(source['time_base']))/2)
        if abs(float(stamp[0])-expected) > tolerance:
            raise ValueError('Decord selected PTS differs from the exact source frame grid')
        record['decoder_source_seconds'] = float(stamp[0])
    frames = reader.get_batch(indices).asnumpy()
    del reader
    if frames.shape != (len(indices), source['height'], source['width'], 3) or frames.dtype != np.uint8:
        raise ValueError('Decoded RGB dimensions or dtype differ from inventory')
    if audio_cache_dir is None:
        waveform, stream = _load_audio(source)
        cached = False
        cache_sha256 = None
    else:
        waveform, receipt = _read_cache(audio_cache_dir, source)
        stream = receipt['source_audio_stream']
        cached = True
        cache_sha256 = receipt['sha256']
    first = math.floor(start * AUDIO_SAMPLE_RATE)
    last = math.floor(end * AUDIO_SAMPLE_RATE)
    samples = np.array(waveform[first:last], dtype=np.float32, copy=True)
    full_source_audio_samples = int(waveform.size)
    del waveform
    if samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError('Caption clip audio is empty or nonfinite')
    validate_inventory(inventory)
    metadata = {'frames': frames_metadata,
                'source_start_fraction': str(start), 'source_end_fraction': str(end),
                'global_start_fraction': clip['global_start_fraction'],
                'global_end_fraction': clip['global_end_fraction'],
                'video_sampling': 'Decord CPU, source-local 1fps, [clip_start, clip_end), original RGB',
                'audio': {'sample_rate': AUDIO_SAMPLE_RATE, 'sample_count': int(samples.size),
                          'rms': float(np.sqrt(np.mean(samples.astype(np.float64)**2))),
                          'absolute_peak': float(np.max(np.abs(samples))),
                          'duration_fraction': str(Fraction(samples.size, AUDIO_SAMPLE_RATE)),
                          'requested_duration_fraction': str(duration),
                          'source_stream': stream, 'recipe': _audio_recipe(), 'cached': cached,
                          'cache_sha256': cache_sha256,
                          'strict_causal': False,
                          'full_source_sample_count': full_source_audio_samples,
                          'full_source_global_end_fraction': str(Fraction(source['global_start_fraction'])
                              + max(Fraction(source['duration_fraction']),
                                    Fraction(full_source_audio_samples, AUDIO_SAMPLE_RATE))),
                          'source_start_fraction': str(start), 'source_end_fraction': str(end),
                          'slice_samples': [first, last], 'channels': 1, 'whisper_padded': False}}
    return frames, samples, metadata
