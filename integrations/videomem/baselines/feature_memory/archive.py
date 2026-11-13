"""Stage 1: encode once and atomically publish checksummed feature shards.

Ready receipts are the commit boundary. Stage 2 never opens partial files and
reads each committed feature shard once, hashing bytes during that read.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import uuid

import numpy as np
import torch


class ArchiveError(RuntimeError):
    pass


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')


def _atomic_json(path, value):
    path = Path(path)
    data = _json_bytes(value)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('wb', buffering=0) as handle:
        handle.write(data)
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(data).hexdigest()


def _state(path):
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class ArchiveWriter:
    """Streaming writer: RAM holds metadata and one incoming frame, not a shard."""
    def __init__(self, directory, header, frames_per_shard=100):
        if sys.byteorder != 'little':
            raise ValueError('Feature archive currently requires a little-endian host')
        if type(frames_per_shard) is not int or frames_per_shard < 1:
            raise ValueError('frames_per_shard must be a positive integer')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if (self.directory / 'archive.json').exists():
            raise FileExistsError('Archive header already exists')
        self.header = dict(header)
        self.header.update(schema='feature_memory_archive_v1', archive_id=uuid.uuid4().hex,
                           dtype='bfloat16', byte_order='little', frames_per_shard=frames_per_shard,
                           feature_compression='none', raw_pixels_saved=False)
        self.header_hash = _atomic_json(self.directory / 'archive.json', self.header)
        self.frames_per_shard = frames_per_shard
        self.frame_shape = tuple(int(x) for x in self.header['frame_shape'])
        self.bytes_per_frame = int(np.prod(self.frame_shape)) * 2
        self.total_frames = self.payload_bytes = self.published_frames = self.shard_id = 0
        self.last_timestamp = -1.0
        self.frames = []
        self.handle = None
        self.shard_digest = None
        self.ordered_digest = hashlib.sha256()

    def append(self, features, metadata):
        if (features.device.type != 'cpu' or features.dtype != torch.bfloat16
                or tuple(features.shape) != self.frame_shape or not features.is_contiguous()):
            raise ArchiveError('Archive feature shape/dtype/device differs from fixed representation')
        row = dict(metadata)
        fid, stamp = row['frame_id'], float(row['timestamp'])
        if fid != self.total_frames or not np.isfinite(stamp) or stamp <= self.last_timestamp:
            raise ArchiveError('Archive frames must be contiguous and timestamp ordered')
        raw = memoryview(features.view(torch.uint8).numpy()).cast('B')
        digest = hashlib.sha256(raw).hexdigest()
        if row.get('shared_feature_sha256', digest) != digest:
            raise ArchiveError(f'Source feature digest mismatch at frame {fid}')
        row.update(shared_feature_sha256=digest, shape=list(features.shape), dtype='bfloat16',
                   feature_bytes=len(raw), offset=len(self.frames) * self.bytes_per_frame)
        _json_bytes(row)  # Reject tensors, NaN, or unserializable metadata before writing.
        if self.handle is None:
            self.partial_path = self.directory / f'features_{self.shard_id:08d}.bin.partial'
            self.handle = self.partial_path.open('xb', buffering=0)
            self.shard_digest = hashlib.sha256()
        written = 0
        while written < len(raw):
            count = self.handle.write(raw[written:])
            if not count:
                raise IOError('Short feature archive write')
            written += count
        self.shard_digest.update(raw)
        self.ordered_digest.update(bytes.fromhex(digest))
        self.frames.append(row)
        self.total_frames += 1
        self.payload_bytes += len(raw)
        self.last_timestamp = stamp
        if len(self.frames) == self.frames_per_shard:
            return self.publish()
        return None

    def publish(self):
        if not self.frames:
            return None
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        self.handle = None
        feature_name = f'features_{self.shard_id:08d}.bin'
        index_name = f'index_{self.shard_id:08d}.json'
        target = self.directory / feature_name
        if target.exists():
            raise FileExistsError(target)
        os.replace(self.partial_path, target)
        index = {'schema': 'feature_memory_archive_index_v1', 'archive_id': self.header['archive_id'],
                 'shard_id': self.shard_id, 'frame_shape': list(self.frame_shape),
                 'dtype': 'bfloat16', 'frames': self.frames}
        index_hash = _atomic_json(self.directory / index_name, index)
        receipt = {'schema': 'feature_memory_archive_ready_v1', 'archive_id': self.header['archive_id'],
                   'shard_id': self.shard_id, 'first_frame_id': self.frames[0]['frame_id'],
                   'frame_count': len(self.frames), 'feature_file': feature_name, 'index_file': index_name,
                   'feature_bytes': target.stat().st_size, 'feature_sha256': self.shard_digest.hexdigest(),
                   'index_sha256': index_hash, 'last_timestamp': self.frames[-1]['timestamp']}
        # Published last, after both payload and index are durable and renamed.
        _atomic_json(self.directory / f'ready_{self.shard_id:08d}.json', receipt)
        self.published_frames += len(self.frames)
        self.shard_id += 1
        self.frames = []
        self.shard_digest = None
        return receipt

    def complete(self, expected_frames, **extra):
        if self.total_frames != expected_frames:
            raise ArchiveError(f'Incomplete archive: {self.total_frames}/{expected_frames}')
        self.publish()
        receipt = {'schema': 'feature_memory_archive_complete_v1', 'archive_id': self.header['archive_id'],
                   'archive_header_sha256': self.header_hash, 'total_frames': self.total_frames,
                   'payload_bytes': self.payload_bytes, 'shards': self.shard_id,
                   'ordered_shared_features_sha256': self.ordered_digest.hexdigest(), **extra}
        _atomic_json(self.directory / 'complete.json', receipt)
        return receipt

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def _check_producer(directory):
    path = directory / 'status.json'
    if path.exists():
        status = _load(path)
        if status.get('status') == 'failed':
            raise ArchiveError('Archive producer failed: ' + str(status.get('error_type')) + ': ' + str(status.get('error', '')))
        pid = status.get('producer_pid')
        if pid and status.get('producer_host') == socket.gethostname() and status.get('status') not in ('completed',):
            try:
                os.kill(int(pid), 0)
            except ProcessLookupError as exc:
                raise ArchiveError('Archive producer exited before completion') from exc


def _wait_file(path, directory, wait):
    while not path.is_file():
        _check_producer(directory)
        if (directory / 'complete.json').exists():
            return False
        if not wait:
            return False
        time.sleep(.2)
    return True


def iter_archive(archive_dir, wait=True, max_frames=None):
    """Read committed shards once into owned CPU tensors, never partial files.

    The result's ``features`` has shape [frames,tokens,hidden]. max_frames limits
    yielded frames; a crossing shard's remainder is hashed through a small
    scratch buffer rather than exposed as future features to the consumer.
    """
    directory = Path(archive_dir)
    if max_frames is not None and (type(max_frames) is not int or max_frames < 0):
        raise ValueError('max_frames must be a nonnegative integer')
    if max_frames == 0:
        return
    if not _wait_file(directory / 'archive.json', directory, wait):
        return
    header = _load(directory / 'archive.json')
    if header.get('schema') != 'feature_memory_archive_v1' or header.get('dtype') != 'bfloat16':
        raise ArchiveError('Unsupported feature archive format')
    frame_shape = tuple(header['frame_shape'])
    frame_bytes = int(np.prod(frame_shape)) * 2
    count, shard_id, previous_time = 0, 0, -1.0
    while max_frames is None or count < max_frames:
        ready_path = directory / f'ready_{shard_id:08d}.json'
        if not _wait_file(ready_path, directory, wait):
            if (directory / 'complete.json').exists():
                complete = _load(directory / 'complete.json')
                if complete['total_frames'] != count:
                    raise ArchiveError('Completed archive is missing a committed shard')
            return
        started = time.perf_counter()
        ready = _load(ready_path)
        feature_path = directory / f'features_{shard_id:08d}.bin'
        index_path = directory / f'index_{shard_id:08d}.json'
        if (ready.get('archive_id') != header['archive_id'] or ready.get('shard_id') != shard_id
                or ready.get('first_frame_id') != count or ready.get('feature_file') != feature_path.name
                or ready.get('index_file') != index_path.name):
            raise ArchiveError('Ready receipt archive/shard/frame identity mismatch')
        index_raw = index_path.read_bytes()
        if hashlib.sha256(index_raw).hexdigest() != ready['index_sha256']:
            raise ArchiveError('Archive index SHA256 mismatch')
        index = json.loads(index_raw)
        frames = index['frames']
        if (index.get('archive_id') != header['archive_id'] or index.get('shard_id') != shard_id
                or tuple(index.get('frame_shape', [])) != frame_shape or index.get('dtype') != 'bfloat16'
                or len(frames) != ready['frame_count'] or not 0 < len(frames) <= header['frames_per_shard']):
            raise ArchiveError('Index structure differs from committed receipt')
        for number, frame in enumerate(frames):
            if (frame['frame_id'] != count + number or frame['offset'] != number * frame_bytes
                    or frame.get('feature_bytes') != frame_bytes or tuple(frame.get('shape', [])) != frame_shape
                    or frame.get('dtype') != 'bfloat16' or not np.isfinite(frame['timestamp'])
                    or frame['timestamp'] <= previous_time):
                raise ArchiveError('Index frame order, shape or timestamp is invalid')
            previous_time = frame['timestamp']
        take = min(len(frames), max_frames - count) if max_frames is not None else len(frames)
        expected_file_bytes = len(frames) * frame_bytes
        before = _state(feature_path)
        if before[2] != expected_file_bytes or ready['feature_bytes'] != expected_file_bytes:
            raise ArchiveError('Feature shard byte size mismatch')
        raw = torch.empty(take * frame_bytes, dtype=torch.uint8)
        view = memoryview(raw.numpy()).cast('B')
        digest = hashlib.sha256()
        with feature_path.open('rb', buffering=0) as handle:
            offset = 0
            while offset < len(view):
                chunk = view[offset:min(len(view), offset + 16 * 1024 * 1024)]
                n = handle.readinto(chunk)
                if not n:
                    raise ArchiveError('Truncated feature shard')
                digest.update(chunk[:n])
                offset += n
            remaining = expected_file_bytes - len(view)
            while remaining:
                chunk = handle.read(min(remaining, 16 * 1024 * 1024))
                if not chunk:
                    raise ArchiveError('Truncated feature shard tail')
                digest.update(chunk)
                remaining -= len(chunk)
            if handle.read(1):
                raise ArchiveError('Unexpected bytes after feature shard')
        if _state(feature_path) != before or digest.hexdigest() != ready['feature_sha256']:
            raise ArchiveError('Feature shard SHA256/identity mismatch')
        for number, frame in enumerate(frames[:take]):
            if hashlib.sha256(view[number * frame_bytes:(number + 1) * frame_bytes]).hexdigest() != frame['shared_feature_sha256']:
                raise ArchiveError('Per-frame feature SHA256 mismatch')
        features = raw.view(torch.bfloat16).reshape((take, *frame_shape))
        yield {'features': features, 'frames': frames[:take], 'shard_id': shard_id,
               'read_seconds': time.perf_counter() - started,
               'file_bytes_read': expected_file_bytes, 'payload_bytes_yielded': take * frame_bytes}
        count += take
        shard_id += 1
        del features, raw, view


def _seed_info(seed_run, plan, inventory):
    """Validate the frozen FIFO journal prefix; do not open stores for writing."""
    directory = Path(seed_run)
    status = _load(directory / 'status.json')
    if status.get('status') in ('running', 'initializing'):
        raise ArchiveError('Seed run must be stopped before its FIFO records are reused')
    old_inventory = _load(directory / 'inventory.json')
    old_sources = [x['source_fingerprint'] for x in old_inventory['sources']]
    if old_sources != [x['source_fingerprint'] for x in inventory['sources']]:
        raise ArchiveError('Seed videos differ from archive source inventory')
    old_plan = _load(directory / 'budget_plan.json')
    for key in ('dtype', 'hidden_dim', 'tokens_per_frame', 'post_merger_grid_hw'):
        if old_plan[key] != plan[key]:
            raise ArchiveError('Seed representation differs from archive plan: ' + key)
    config = _load(directory / 'run_config.json')
    current_encoder = hashlib.sha256(Path(__file__).with_name('encoder.py').read_bytes()).hexdigest()
    if config.get('source_sha256', {}).get('encoder.py') != current_encoder:
        raise ArchiveError('Seed encoder implementation differs from current encoder')
    journal = directory / 'frames.jsonl'
    before = _state(journal)
    count, last = 0, -1.0
    with journal.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row['frame_id'] != count or row['timestamp'] <= last or len(row['shared_feature_sha256']) != 64:
                raise ArchiveError('Seed journal is not a verified contiguous prefix')
            count += 1
            last = row['timestamp']
    if before != _state(journal):
        raise ArchiveError('Seed journal changed during inspection')
    use = min(count, plan['planned_frames'])
    return {'run': str(directory.resolve()), 'journal_frames': count, 'frames_to_reuse': use,
            'journal_state': before, 'encoder_implementation_sha256': current_encoder,
            'source': 'stopped FIFO/5pct retained records verified against committed frame hashes',
            'prior_uncommitted_prefetch_excluded': True}


def _iter_seed_frames(info, frame_shape):
    from .byte_memory import BASE_FIELDS
    directory = Path(info['run'])
    store = directory / 'live' / 'fifo' / '5pct'
    control_path = store / 'control.json'
    control_before = _state(control_path)
    control = _load(control_path)
    tokens, hidden = frame_shape
    if (control['dtype'] != 'bfloat16' or control['hidden_dim'] != hidden
            or control.get('extra_fields') or control['owner']['method'] != 'fifo'
            or control['owner']['fifo_cursor'] != 0
            or control['count'] < info['frames_to_reuse'] * tokens):
        raise ArchiveError('Seed FIFO no longer contains the complete unwrapped prefix')
    dtype = np.dtype([('features', '<u2', (hidden,)), *BASE_FIELDS], align=False)
    if dtype.itemsize != control['record_bytes']:
        raise ArchiveError('Seed record schema mismatch')
    anchors = {}
    with (directory / 'frames.jsonl').open() as journal:
        for frame_id in range(info['frames_to_reuse']):
            row = json.loads(next(journal))
            records = np.empty(tokens, dtype=dtype)
            copied = 0
            while copied < tokens:
                slot = frame_id * tokens + copied
                shard, within = divmod(slot, control['shard_records'])
                amount = min(tokens - copied, control['shard_records'] - within)
                path = store / f'records_{shard:08d}.bin'
                anchors.setdefault(path, _state(path))
                with path.open('rb', buffering=0) as handle:
                    handle.seek(within * dtype.itemsize)
                    view = memoryview(records[copied:copied + amount]).cast('B')
                    offset = 0
                    while offset < len(view):
                        n = handle.readinto(view[offset:])
                        if not n:
                            raise ArchiveError('Seed FIFO has a truncated committed prefix')
                        offset += n
                copied += amount
            if (not np.array_equal(records['global_token_id'], np.arange(frame_id * tokens, (frame_id + 1) * tokens))
                    or not np.all(records['frame_id'] == frame_id)
                    or not np.array_equal(records['token_index'], np.arange(tokens))
                    or not np.all(records['timestamp'] == row['timestamp'])):
                raise ArchiveError('Seed retained record identity differs from journal')
            features = torch.from_numpy(records['features'].copy()).view(torch.bfloat16)
            raw = memoryview(features.view(torch.uint8).numpy()).cast('B')
            if hashlib.sha256(raw).hexdigest() != row['shared_feature_sha256']:
                raise ArchiveError(f'Seed feature digest mismatch at frame {frame_id}')
            yield features, {**row, 'feature_origin': 'reused_seed_prefix', 'seed_run': info['run']}
    if _state(control_path) != control_before or _state(directory / 'frames.jsonl') != tuple(info['journal_state']):
        raise ArchiveError('Frozen seed state changed during reuse')
    if any(_state(path) != state for path, state in anchors.items()):
        raise ArchiveError('Seed feature file changed during reuse')


def build_archive(args, *, encoder_factory=None, feature_size_estimator=None):
    from .parallel import iter_encoded
    from .planning import make_plan, RATIOS
    from .video import inspect_sources, iter_stream
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    identity = {'producer_pid': os.getpid(), 'producer_host': socket.gethostname()}
    _atomic_json(output / 'status.json', {'status': 'initializing', **identity})
    writer = encoded = None
    seed_count = 0
    encoder_stats, decoder_stats = {}, {}
    try:
        inventory = inspect_sources(args.videos, status_path=getattr(args, 'dataset_status', None),
                                    timeline_path=getattr(args, 'timeline', None))
        plan = make_plan(inventory, args.model, max_seconds=getattr(args, 'max_seconds', None),
                         ratios=getattr(args, 'ratios', RATIOS))
        shape = (int(plan['tokens_per_frame']), plan['hidden_dim'])
        seed = _seed_info(args.seed_run, plan, inventory) if getattr(args, 'seed_run', None) else None
        model_path = Path(args.model).resolve()
        verification = model_path.parent / '.downloads' / model_path.name / 'verification.json'
        model_identity = {'path': str(model_path), 'config_sha256': hashlib.sha256((model_path / 'config.json').read_bytes()).hexdigest(),
                          'preprocessor_sha256': hashlib.sha256((model_path / 'preprocessor_config.json').read_bytes()).hexdigest()}
        if verification.exists():
            marker = _load(verification)
            model_identity.update(repo_id=marker.get('repo_id'), revision=marker.get('revision'),
                                  verification_manifest_sha256=hashlib.sha256(verification.read_bytes()).hexdigest())
        writer = ArchiveWriter(output, {'plan': plan, 'inventory': inventory, 'frame_shape': list(shape),
                               'model_identity': model_identity, 'seed': seed, **identity,
                               'created_at_utc': datetime.now(timezone.utc).isoformat(),
                               'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                 for p in sorted(Path(__file__).parent.glob('*.py'))}},
                               frames_per_shard=getattr(args, 'archive_frames_per_shard', 100))

        def report(phase, status='running', **extra):
            elapsed = time.perf_counter() - started
            value = {'status': status, 'phase': phase, **identity,
                     'frames_processed': writer.total_frames, 'frames_published': writer.published_frames,
                     'total_frames': plan['planned_frames'], 'seed_frames_reused': seed_count,
                     'num_encoder_calls': encoder_stats.get('num_encoder_calls', 0),
                     'new_frames_encoded': encoder_stats.get('num_frames_encoded', 0),
                     'payload_bytes': writer.payload_bytes, 'published_shards': writer.shard_id,
                     'last_timestamp': writer.last_timestamp, 'elapsed_seconds': elapsed,
                     'total_throughput': writer.total_frames / max(elapsed, 1e-9),
                     'encoder_stats': encoder_stats, 'decoder_stats': decoder_stats, **extra}
            _atomic_json(output / 'status.json', value)
            print(json.dumps({'event': 'archive_progress', 'phase': phase, 'frames': writer.total_frames,
                              'published': writer.published_frames, 'total': plan['planned_frames'],
                              'fps': value['total_throughput']}, allow_nan=False), flush=True)
            return value

        if seed:
            report('seed_reuse')
            for features, metadata in _iter_seed_frames(seed, shape):
                receipt = writer.append(features, metadata)
                seed_count += 1
                if receipt:
                    report('seed_reuse')
            if seed_count != seed['frames_to_reuse']:
                raise ArchiveError('Seed reuse count mismatch')
        if seed_count < plan['planned_frames']:
            stream = iter_stream(inventory, max_seconds=getattr(args, 'max_seconds', None),
                                 stats=decoder_stats, start_frame=seed_count)
            encoded = iter_encoded(stream, args.model, mode=getattr(args, 'mode', 'parallel'),
                        devices=getattr(args, 'devices', tuple(map(str, range(8)))),
                        chunk_frames=getattr(args, 'chunk_frames', 8),
                        max_inflight_chunks=getattr(args, 'max_inflight_chunks', 8),
                        max_queue_bytes=getattr(args, 'max_queue_bytes', 2 * 1024 ** 3),
                        cpu_threads=getattr(args, 'cpu_threads', 1), dtype='bfloat16', stats=encoder_stats,
                        encoder_factory=encoder_factory, feature_size_estimator=feature_size_estimator)
            report('encoding')
            for item in encoded:
                features = item['features']
                metadata = {k: v for k, v in item.items() if k != 'features'}
                metadata['feature_origin'] = 'new_encoder_forward'
                receipt = writer.append(features, metadata)
                if receipt or (writer.total_frames - seed_count) % 30 == 0:
                    report('encoding')
                del features, item
        new_count = plan['planned_frames'] - seed_count
        if (encoder_stats.get('num_encoder_calls', 0) != new_count
                or encoder_stats.get('num_frames_encoded', 0) != new_count):
            raise ArchiveError('New encoder call count differs from unseeded frames')
        if writer.payload_bytes != plan['planned_run_feature_bytes']:
            raise ArchiveError('Observed archive bytes differ from fixed feature plan')
        complete = writer.complete(plan['planned_frames'], seed_frames_reused=seed_count,
                                   new_encoder_calls=new_count, num_unique_frames=seed_count + new_count,
                                   encoder_stats=encoder_stats, decoder_stats=decoder_stats)
        report('completed', status='completed', complete=complete)
        return complete
    except BaseException as exc:
        _atomic_json(output / 'status.json', {'status': 'failed', 'phase': 'failed', **identity,
                     'frames_processed': writer.total_frames if writer else 0,
                     'frames_published': writer.published_frames if writer else 0,
                     'seed_frames_reused': seed_count, 'error_type': type(exc).__name__,
                     'error': str(exc)[:2000], 'elapsed_seconds': time.perf_counter() - started})
        raise
    finally:
        if encoded is not None:
            encoded.close()
        if writer is not None:
            writer.close()
