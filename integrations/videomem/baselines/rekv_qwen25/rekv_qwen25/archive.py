"""Atomic, lossless storage for ReKV's pre-RoPE K/V and optional resume state.

This is the NAS storage extension for ``ReKV (Qwen2.5-VL-7B adaptation)``;
it does not generate KV, implement retrieval, or change tensor values.

Metadata requires ``frame_ids`` and ``timestamps`` lists. Their order is the
actual causal input order. Chunk bounds and availability are derived from those
lists, never inferred from a fixed timestamp spacing or a fixed Day duration.
The caller must include checkpoint/revision, sampling, source/Day information,
position convention and attention configuration in config/metadata.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import torch
from safetensors.torch import load_file, save_file


BASELINE = 'ReKV (Qwen2.5-VL-7B adaptation)'
SCHEMA = 'rekv_qwen25_pre_rope_archive_v1'


def _json(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def file_sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def _tensor_sha256(tensor):
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def _fsync_dir(path):
    fd = os.open(str(path), os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _mkdir(path):
    path = Path(path)
    missing = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    path.mkdir(parents=True, exist_ok=True)
    for created in reversed(missing):
        _fsync_dir(created.parent)


def _atomic_json(path, value):
    path = Path(path)
    _mkdir(path.parent)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
    try:
        with tmp.open('xb') as stream:
            stream.write(_json(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    finally:
        if tmp.exists():
            tmp.unlink()


@contextmanager
def _lock(root, *, shared=False):
    if not shared:
        _mkdir(root)
    with (root / '.writer.lock').open('rb' if shared else 'a') as stream:
        fcntl.flock(stream, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        yield


@contextmanager
def _chunk_lock(root, chunk_id):
    """Own one chunk's payload without serializing other chunk writers.

    The lock order is chunk then root. Readers need only a root metadata
    snapshot because a ready chunk's payload/index are immutable.
    """
    directory = root / '.chunk_locks'
    _mkdir(directory)
    with (directory / f'chunk_{_chunk_id(chunk_id):08d}.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def _chunk_id(value):
    if type(value) is not int or value < 0:
        raise ValueError('chunk_id must be a nonnegative integer')
    return value


def _paths(root, chunk_id):
    name = f'chunk_{_chunk_id(chunk_id):08d}'
    return (root / 'shards' / f'{name}.safetensors',
            root / 'indices' / f'{name}.json',
            root / 'receipts' / f'{name}.ready.json')


def _normalize_config(config):
    config = copy.deepcopy(config)
    config.setdefault('baseline_name', BASELINE)
    config.setdefault('key_representation', 'pre_rope')
    config.setdefault('archive_schema', SCHEMA)
    if config['baseline_name'] != BASELINE or config['key_representation'] != 'pre_rope':
        raise ValueError('This archive requires the named Qwen2.5 ReKV adaptation and pre-RoPE keys')
    if config['archive_schema'] != SCHEMA:
        raise ValueError('Unsupported archive schema')
    _json(config)
    return config


def _normalize_metadata(chunk_id, metadata):
    metadata = copy.deepcopy(metadata)
    if 'chunk_id' in metadata and metadata['chunk_id'] != chunk_id:
        raise ValueError('Metadata chunk_id disagrees with the requested chunk')
    ids, times = metadata.get('frame_ids'), metadata.get('timestamps')
    if not isinstance(ids, list) or not ids or not all(type(i) is int and i >= 0 for i in ids):
        raise ValueError('metadata.frame_ids must be a nonempty list of nonnegative integers')
    if ids != list(range(ids[0], ids[0] + len(ids))):
        raise ValueError('Chunk contains missing, duplicate, or unordered frame IDs')
    if not isinstance(times, list) or len(times) != len(ids):
        raise ValueError('metadata.timestamps must have one timestamp per frame ID')
    if any(isinstance(t, bool) or not isinstance(t, (float, int)) or not math.isfinite(t) for t in times):
        raise ValueError('Frame timestamps must be finite numeric seconds')
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Frame timestamps must be strictly increasing within a chunk')
    derived = dict(chunk_id=chunk_id, frame_count=len(ids), first_frame_id=ids[0],
                   last_frame_id=ids[-1], chunk_start_timestamp=times[0],
                   chunk_end_timestamp=times[-1], availability_timestamp=max(times))
    for key, value in derived.items():
        if key in metadata and metadata[key] != value:
            raise ValueError(f'Metadata {key} disagrees with actual frame coverage')
    metadata.update(derived)
    _json(metadata)
    return metadata


def _read_header(root, expected_config_id=None):
    config = json.loads((root / 'config.json').read_text())
    if _normalize_config(config) != config:
        raise ValueError('Incomplete archive configuration')
    config_id = hashlib.sha256(_json(config)).hexdigest()
    if expected_config_id is not None and config_id != expected_config_id:
        raise ValueError('Unexpected model/config identifier')
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('schema') != SCHEMA or manifest.get('config_id') != config_id:
        raise ValueError('Manifest config identifier mismatch')
    return config, config_id, manifest


def _descriptor(metadata):
    return {k: metadata[k] for k in ('chunk_id', 'frame_count', 'first_frame_id',
            'last_frame_id', 'chunk_start_timestamp', 'chunk_end_timestamp', 'availability_timestamp')}


def _check_chunk_relationships(entries):
    """Frame IDs define continuity; timestamps need only increase, not step by one."""
    descriptors = sorted(entries, key=lambda d: d['chunk_id'])
    for previous, current in zip(descriptors, descriptors[1:]):
        if current['first_frame_id'] <= previous['last_frame_id']:
            raise ValueError('Committed chunks overlap or have reversed frame ranges')
        if current['chunk_start_timestamp'] <= previous['chunk_end_timestamp']:
            raise ValueError('Committed chunks overlap or reverse the global timeline')
        if current['chunk_id'] == previous['chunk_id'] + 1:
            if current['first_frame_id'] != previous['last_frame_id'] + 1:
                raise ValueError('Adjacent chunks have missing frame IDs')
    return descriptors


def _continuity(manifest):
    entries = [v for v in manifest['chunks'].values() if v['state'] == 'ready']
    ordered = _check_chunk_relationships(entries)
    expected = manifest.get('first_chunk_id', 0)
    prefix_chunks = prefix_frames = 0
    for entry in ordered:
        if entry['chunk_id'] != expected:
            break
        prefix_chunks += 1
        prefix_frames += entry['frame_count']
        expected += 1
    return dict(committed_chunk_count=len(ordered),
                committed_frame_count=sum(e['frame_count'] for e in ordered),
                contiguous_prefix_chunks=prefix_chunks, contiguous_prefix_frames=prefix_frames,
                has_chunk_gaps=prefix_chunks != len(ordered))


def _index(root, chunk_id, config_id, manifest):
    _, index_path, receipt_path = _paths(root, chunk_id)
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('schema') != SCHEMA or receipt.get('config_id') != config_id:
        raise ValueError('Receipt config identifier mismatch')
    if receipt.get('chunk_id') != chunk_id:
        raise ValueError('Receipt chunk identifier mismatch')
    entry = manifest['chunks'].get(str(chunk_id))
    if entry is None:
        raise ValueError('Ready chunk is absent from the manifest')
    if file_sha256(index_path) != receipt['index_sha256']:
        raise ValueError('Index checksum mismatch')
    index = json.loads(index_path.read_text())
    if index.get('config_id') != config_id or index.get('chunk_id') != chunk_id:
        raise ValueError('Index model/config or chunk identifier mismatch')
    metadata = _normalize_metadata(chunk_id, index['metadata'])
    if metadata != index['metadata']:
        raise ValueError('Incomplete chunk metadata')
    descriptor = _descriptor(metadata)
    if any(entry.get(k) != v or receipt.get(k) != v for k, v in descriptor.items()):
        raise ValueError('Manifest/receipt/index continuity disagreement')
    if entry['index_sha256'] != receipt['index_sha256'] or entry['payload_sha256'] != receipt['payload_sha256']:
        raise ValueError('Manifest/receipt checksum disagreement')
    return index, receipt


def _load_payload(root, chunk_id, index, receipt, *, cutoff_timestamp=None):
    """Load an immutable ready payload after its metadata snapshot is checked."""
    metadata = index['metadata']
    if cutoff_timestamp is not None:
        if not math.isfinite(cutoff_timestamp):
            raise ValueError('Cutoff timestamp must be finite')
        if metadata['availability_timestamp'] > cutoff_timestamp or metadata['chunk_end_timestamp'] > cutoff_timestamp:
            raise ValueError('Block is in the future of the requested causal cutoff')
    payload_path, _, _ = _paths(root, chunk_id)
    if payload_path.stat().st_size != receipt['serialized_bytes']:
        raise ValueError('Payload serialized byte count mismatch')
    if file_sha256(payload_path) != receipt['payload_sha256']:
        raise ValueError('Payload checksum mismatch')
    tensors = load_file(str(payload_path), device='cpu')
    if set(tensors) != set(index['tensors']):
        raise ValueError('Payload tensor keys disagree with the index')
    for name, tensor in tensors.items():
        spec = index['tensors'][name]
        if (list(tensor.shape), str(tensor.dtype), tensor.numel() * tensor.element_size()) != (
                spec['shape'], spec['dtype'], spec['bytes']):
            raise ValueError(f'Tensor dtype/shape/byte count mismatch: {name}')
        if _tensor_sha256(tensor) != spec['sha256']:
            raise ValueError(f'Tensor checksum mismatch: {name}')
    return tensors, metadata


def load_block(root, chunk_id, expected_config_id=None, *, cutoff_timestamp=None):
    """Verify a ready block without holding a global lock during payload I/O.

    A brief shared lock captures a consistent ready/index/manifest snapshot.
    Ready payloads and indices are immutable, including during repeated commits.
    The cutoff is checked before opening or hashing the payload.
    """
    root, chunk_id = Path(root), _chunk_id(chunk_id)
    with _lock(root, shared=True):
        _, config_id, manifest = _read_header(root, expected_config_id)
        index, receipt = _index(root, chunk_id, config_id, manifest)
    return _load_payload(root, chunk_id, index, receipt, cutoff_timestamp=cutoff_timestamp)


def _committed_blocks_unlocked(root, cutoff_timestamp=None, *, expected_config_id=None,
                              require_contiguous=False):
    """Return ordered, validated candidate descriptors, without loading KV.

    Only ready receipts count. A ``prepared`` manifest entry with no receipt is
    an interrupted partial write, not a completed block. Missing receipts for
    entries already marked ready are corruption. An optional contiguity check
    rejects gaps instead of silently treating a partial archive as complete.
    """
    root = Path(root)
    _, config_id, manifest = _read_header(root, expected_config_id)
    if cutoff_timestamp is not None and not math.isfinite(cutoff_timestamp):
        raise ValueError('Cutoff timestamp must be finite')
    result = []
    expected_paths = {_paths(root, int(key))[2] for key in manifest['chunks']}
    actual_paths = set((root / 'receipts').glob('chunk_*.ready.json'))
    if actual_paths - expected_paths:
        raise ValueError('Ready receipt missing from manifest')
    for key, entry in manifest['chunks'].items():
        chunk_id = _chunk_id(int(key))
        receipt_path = _paths(root, chunk_id)[2]
        if not receipt_path.exists():
            if entry['state'] == 'ready':
                raise ValueError('Committed block is missing its ready receipt')
            continue
        index, receipt = _index(root, chunk_id, config_id, manifest)
        descriptor = dict(_descriptor(index['metadata']), config_id=config_id,
                          payload_sha256=receipt['payload_sha256'],
                          serialized_bytes=receipt['serialized_bytes'])
        result.append(descriptor)
    result = _check_chunk_relationships(result)
    if require_contiguous:
        first = manifest.get('first_chunk_id', 0)
        if [d['chunk_id'] for d in result] != list(range(first, first + len(result))):
            raise ValueError('Archive contains missing chunk IDs')
    # Filter before consumers load any tensor payload, including resume tensors.
    if cutoff_timestamp is not None:
        result = [d for d in result if d['chunk_end_timestamp'] <= cutoff_timestamp
                  and d['availability_timestamp'] <= cutoff_timestamp]
    return result


def committed_blocks(root, cutoff_timestamp=None, *, expected_config_id=None,
                     require_contiguous=False):
    """Enumerate ready metadata under a consistent read lock; no KV is loaded.

    Timestamp filtering occurs before loading any tensor payload. Set
    ``require_contiguous=True`` to reject missing chunk IDs explicitly.
    """
    root = Path(root)
    with _lock(root, shared=True):
        return _committed_blocks_unlocked(root, cutoff_timestamp,
            expected_config_id=expected_config_id, require_contiguous=require_contiguous)


def _repair_prepared(root, config_id, manifest):
    changed = False
    for key, entry in manifest['chunks'].items():
        if entry['state'] == 'prepared' and _paths(root, int(key))[2].exists():
            _index(root, int(key), config_id, manifest)
            entry['state'] = 'ready'
            changed = True
    if changed:
        manifest['continuity'] = _continuity(manifest)
        _atomic_json(root / 'manifest.json', manifest)


def _initialize_manifest(root, config_id, config):
    if any((root / 'receipts').glob('*.ready.json')):
        raise ValueError('Manifest is missing for committed blocks')
    _atomic_json(root / 'manifest.json', dict(schema=SCHEMA,
        config_id=config_id, status='open', first_chunk_id=config.get('first_chunk_id', 0),
        chunks={}, continuity=dict(committed_chunk_count=0, committed_frame_count=0,
            contiguous_prefix_chunks=0, contiguous_prefix_frames=0, has_chunk_gaps=False)))


class KVArchiveWriter:
    """Serialize disjoint chunk blocks, atomically, with verified resume reuse.

    The manifest describes committed coverage; it never labels the entire
    benchmark complete merely because one writer exits. The coordinator owns
    whole-run completeness. Tensor names beginning with ``resume_state.`` may
    store the exact boundary state needed for context-preserving continuation.
    """

    def __init__(self, root, config):
        self.root = Path(root)
        self.config = _normalize_config(config)
        self.config_id = hashlib.sha256(_json(self.config)).hexdigest()
        with _lock(self.root):
            for name in ('shards', 'indices', 'receipts', 'benchmarks', 'logs', '.chunk_locks'):
                _mkdir(self.root / name)
            if (self.root / 'config.json').exists():
                if not (self.root / 'manifest.json').exists():
                    existing = json.loads((self.root / 'config.json').read_text())
                    if existing != self.config:
                        raise ValueError('Refusing archive resume with different model/config')
                    # Recover a crash during initial config/manifest bootstrap.
                    _initialize_manifest(self.root, self.config_id, self.config)
                existing, config_id, manifest = _read_header(self.root)
                if config_id != self.config_id or existing != self.config:
                    raise ValueError('Refusing archive resume with different model/config')
                _repair_prepared(self.root, config_id, manifest)
            else:
                if (self.root / 'manifest.json').exists() or any((self.root / 'receipts').glob('*.ready.json')):
                    raise ValueError('Archive configuration is missing; refusing to reinterpret existing blocks')
                _atomic_json(self.root / 'config.json', self.config)
                _initialize_manifest(self.root, self.config_id, self.config)

    def commit(self, chunk_id, tensors, metadata):
        chunk_id = _chunk_id(chunk_id)
        metadata = _normalize_metadata(chunk_id, metadata)
        if not isinstance(tensors, dict) or not tensors:
            raise ValueError('tensors must be a nonempty name-to-Tensor dictionary')
        if any(not isinstance(k, str) or not k or not isinstance(v, torch.Tensor)
               or v.layout != torch.strided for k, v in tensors.items()):
            raise ValueError('Only named dense tensors are supported')
        # Copy disjoint tensors to avoid safetensors aliasing restrictions. This
        # preserves exact bit patterns, including bfloat16 and position integers.
        cpu = {k: v.detach().cpu().contiguous().clone() for k, v in tensors.items()}
        specs = {k: dict(shape=list(v.shape), dtype=str(v.dtype),
                         bytes=v.numel() * v.element_size(), sha256=_tensor_sha256(v))
                 for k, v in cpu.items()}
        with _chunk_lock(self.root, chunk_id):
            payload_path, index_path, receipt_path = _paths(self.root, chunk_id)
            with _lock(self.root):
                _, config_id, manifest = _read_header(self.root, self.config_id)
                _repair_prepared(self.root, config_id, manifest)
                if receipt_path.exists():
                    index, receipt = _index(self.root, chunk_id, config_id, manifest)
                else:
                    entry = manifest['chunks'].get(str(chunk_id))
                    if entry is not None and entry['state'] == 'ready':
                        raise ValueError('Committed block is missing its ready receipt')
                    previous = [v for key, v in manifest['chunks'].items()
                                if int(key) != chunk_id and v['state'] == 'ready']
                    _check_chunk_relationships(previous + [_descriptor(metadata)])
                    receipt = None
            if receipt is not None:
                if index['metadata'] != metadata or index['tensors'] != specs:
                    raise ValueError('Chunk already committed with different metadata or tensor values')
                _load_payload(self.root, chunk_id, index, receipt)
                return dict(receipt, reused=True)
            # No root lock spans multi-GB serialization, fsync, or hashing.
            # The per-chunk lock prevents another writer from overwriting this
            # chunk while it is prepared, committed, or verified for reuse.
            temp = payload_path.with_name(payload_path.name + '.' + uuid.uuid4().hex + '.partial')
            try:
                save_file(cpu, str(temp), metadata={'schema': SCHEMA, 'config_id': config_id,
                                                   'chunk_id': str(chunk_id), 'keys': 'pre_rope'})
                with temp.open('rb') as stream:
                    os.fsync(stream.fileno())
                payload_hash = file_sha256(temp)
                size = temp.stat().st_size
                os.replace(temp, payload_path)
                _fsync_dir(payload_path.parent)
            finally:
                if temp.exists():
                    temp.unlink()
            index = dict(schema=SCHEMA, config_id=config_id, chunk_id=chunk_id,
                         metadata=metadata, tensors=specs)
            _atomic_json(index_path, index)
            index_hash = file_sha256(index_path)
            descriptor = _descriptor(metadata)
            receipt = dict(schema=SCHEMA, config_id=config_id, **descriptor,
                           payload_sha256=payload_hash, index_sha256=index_hash,
                           serialized_bytes=size,
                           tensor_payload_bytes=sum(s['bytes'] for s in specs.values()))
            with _lock(self.root):
                # Other chunks may have finished during payload serialization.
                # Revalidate against their committed coverage before publishing.
                _, _, manifest = _read_header(self.root, self.config_id)
                _repair_prepared(self.root, config_id, manifest)
                if receipt_path.exists():
                    raise ValueError('Chunk acquired a ready receipt outside its exclusive lock')
                entry = manifest['chunks'].get(str(chunk_id))
                if entry is not None and entry['state'] == 'ready':
                    raise ValueError('Committed block is missing its ready receipt')
                previous = [v for key, v in manifest['chunks'].items()
                            if int(key) != chunk_id and v['state'] == 'ready']
                _check_chunk_relationships(previous + [descriptor])
                manifest['chunks'][str(chunk_id)] = dict(descriptor, state='prepared',
                    payload_sha256=payload_hash, index_sha256=index_hash)
                _atomic_json(self.root / 'manifest.json', manifest)
                # This is the durable commit point. Neither partial payloads nor
                # a prepared manifest entry alone expose a block to readers.
                _atomic_json(receipt_path, receipt)
                manifest['chunks'][str(chunk_id)]['state'] = 'ready'
                manifest['continuity'] = _continuity(manifest)
                _atomic_json(self.root / 'manifest.json', manifest)
            return dict(receipt, reused=False)
