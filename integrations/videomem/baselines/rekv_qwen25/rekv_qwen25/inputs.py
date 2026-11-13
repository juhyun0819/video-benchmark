"""Verified inputs for the Qwen2.5 ReKV adaptation; no video/QA inference.

The existing visual archive is frame-as-image, post-spatial-merger BF16. It is
not a native paired-video feature archive. Reads may target a committed prefix
while its producer continues writing later shards.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ID = 'Qwen/Qwen2.5-VL-7B-Instruct'
REVISION = 'cc594898137f460bfe9f0759e9844b3ce807cfb5'
FRAME_SHAPE = (676, 3584)
FRAME_BYTES = math.prod(FRAME_SHAPE) * 2
WEIGHT_SHARDS = {f'model-{i:05d}-of-00005.safetensors' for i in range(1, 6)}
_VERIFIED_SHARDS = OrderedDict()


def _sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def _state(path):
    value = Path(path).stat()
    # Device IDs differ between mounts/hosts. The remaining NFS identity and
    # mutation fields are the portable materialized-file comparison contract.
    return (value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _portable_state(value):
    if isinstance(value, list) and len(value) == 5:
        return tuple(value[1:])
    if isinstance(value, dict):
        return tuple(value[k] for k in ('inode', 'size', 'mtime_ns', 'ctime_ns'))
    return None


def _safe_file(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError('Invalid archive/checkpoint relative filename')
    path = root / relative
    if '..' in Path(relative).parts or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Archive/checkpoint filename escapes its root')
    return path


def _verify_remote_hash(path, expected, size):
    if len(expected) == 64:
        hasher = hashlib.sha256()
        algorithm = 'sha256'
    elif len(expected) == 40:
        hasher = hashlib.sha1()
        hasher.update(f'blob {size}\0'.encode())
        algorithm = 'git_blob_sha1'
    else:
        raise ValueError('Unsupported checkpoint verification hash')
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            hasher.update(block)
    if hasher.hexdigest() != expected:
        raise ValueError(f'Checkpoint content hash mismatch: {path.name}')
    return algorithm


def validate_checkpoint(model_path, ledger_path):
    """Verify pinned identity and unchanged materialized states, rehashing changes.

    The ledger was produced by the earlier remote revision/LFS verification.
    Files whose inode, size, mtime and ctime match reuse that verification;
    changed files must match their original SHA-256 or Git-blob SHA-1. No ledger
    is modified. All five model shards and their index remain mandatory.
    """
    model_path, ledger_path = Path(model_path), Path(ledger_path)
    ledger_raw = ledger_path.read_bytes()
    ledger = json.loads(ledger_raw)
    if ledger.get('schema_version') != 'baseline_remote_verification_v1':
        raise ValueError('Unsupported checkpoint verification ledger')
    if (ledger.get('repo_id'), ledger.get('revision')) != (REPO_ID, REVISION):
        raise ValueError('Checkpoint repo/revision differs from the controlled Qwen baseline')
    if ledger.get('remote_revision_verified') is not True:
        raise ValueError('Ledger does not attest remote revision verification')
    records = ledger['verified_files']
    names = [item['path'] for item in records]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate checkpoint ledger entries')
    required = WEIGHT_SHARDS | {'config.json', 'model.safetensors.index.json',
        'preprocessor_config.json', 'tokenizer.json', 'tokenizer_config.json',
        'merges.txt', 'vocab.json'}
    if not required.issubset(names):
        raise ValueError('Ledger lacks mandatory Qwen model or tokenizer files')
    previous_states = ledger.get('materialized_file_states', {})
    changed, reused, omitted, current_states = [], [], [], {}
    for entry in records:
        relative = entry['path']
        path = _safe_file(model_path, relative)
        if not path.exists() and relative in {'.gitattributes', 'README.md'}:
            omitted.append(relative)
            continue
        before = _state(path)
        if before[1] != entry['bytes']:
            raise ValueError(f'Checkpoint file byte count mismatch: {relative}')
        if before == _portable_state(previous_states.get(relative)):
            reused.append(relative)
        else:
            algorithm = _verify_remote_hash(path, entry['hash'], entry['bytes'])
            changed.append(dict(path=relative, algorithm=algorithm))
        if _state(path) != before:
            raise ValueError(f'Checkpoint changed during verification: {relative}')
        current_states[relative] = dict(zip(('inode', 'size', 'mtime_ns', 'ctime_ns'), before))
    index = json.loads((model_path / 'model.safetensors.index.json').read_text())
    if set(index['weight_map'].values()) != WEIGHT_SHARDS:
        raise ValueError('Checkpoint index does not describe the pinned five-shard model')
    config = json.loads((model_path / 'config.json').read_text())
    geometry = (config.get('model_type'), config.get('hidden_size'),
                config.get('num_hidden_layers'), config.get('num_key_value_heads'))
    if geometry != ('qwen2_5_vl', 3584, 28, 4):
        raise ValueError('Checkpoint architecture differs from Qwen2.5-VL-7B')
    return dict(repo_id=REPO_ID, revision=REVISION, model_path=str(model_path.resolve()),
        verification_manifest=str(ledger_path.resolve()),
        verification_manifest_sha256=hashlib.sha256(ledger_raw).hexdigest(),
        config_sha256=_sha256(model_path/'config.json'),
        index_sha256=_sha256(model_path/'model.safetensors.index.json'),
        preprocessor_sha256=_sha256(model_path/'preprocessor_config.json'),
        verified_weight_shards=sorted(WEIGHT_SHARDS),
        reused_verified_file_states=reused, rehashed_files=changed,
        omitted_non_model_files=omitted, materialized_file_states=current_states,
        device_id_ignored_for_cross_host_mounts=True)


def _header(root):
    value = json.loads((root/'archive.json').read_text())
    identity = value['model_identity']
    if (identity.get('repo_id'), identity.get('revision')) != (REPO_ID, REVISION):
        raise ValueError('Visual archive checkpoint identity mismatch')
    if (value.get('schema'), value.get('dtype'), value.get('frame_shape'), value.get('byte_order')) != (
            'feature_memory_archive_v1', 'bfloat16', list(FRAME_SHAPE), 'little'):
        raise ValueError('Unsupported Qwen visual archive tensor contract')
    if sys.byteorder != 'little':
        raise ValueError('This reader requires little-endian tensor storage')
    if value.get('feature_compression', 'none') != 'none':
        raise ValueError('Compressed visual archives are unsupported')
    inventory = value['inventory']
    if inventory.get('fps') != 1.0 or inventory.get('interval_convention') != '[start,end)':
        raise ValueError('Visual archive sampling protocol differs from the controlled baseline')
    if type(value['frames_per_shard']) is not int or value['frames_per_shard'] < 1:
        raise ValueError('Invalid visual shard size')
    offset, count = Fraction(0), 0
    for source in inventory['sources']:
        duration = Fraction(source['duration_fraction'])
        if Fraction(source['global_start_fraction']) != offset or duration <= 0:
            raise ValueError('Invalid source timeline in visual archive inventory')
        if source['expected_1fps_frames'] != math.ceil(duration):
            raise ValueError('Source count disagrees with source-local 1fps sampling')
        # These recorded inputs are CFR at an integer frame rate. Thus each
        # integer-second target has an exact source PTS, including seed rows.
        rate = Fraction(source['avg_frame_rate'])
        if rate.denominator != 1 or rate < 1:
            raise ValueError('Cannot reconstruct exact source timestamps for noninteger source FPS')
        offset += duration
        count += source['expected_1fps_frames']
    if count != inventory['total_1fps_frames']:
        raise ValueError('Visual archive inventory count mismatch')
    return value


def _frame_envelope(inventory, frame_id):
    base = 0
    for source in inventory['sources']:
        count = source['expected_1fps_frames']
        if base <= frame_id < base + count:
            local = Fraction(frame_id-base)
            global_time = Fraction(source['global_start_fraction']) + local
            return dict(global_frame_id=frame_id, day=source['day_id'], day_id=source['day_id'],
                day_order=source['day_order'], source_id=source['source_id'],
                source_video=source['path'], path=source['path'],
                source_timestamp=float(local), source_timestamp_fraction=str(local),
                timestamp=float(global_time), timestamp_seconds=float(global_time),
                timestamp_fraction=str(global_time), source_sha256=source['source_fingerprint'])
        base += count
    raise ValueError('Requested frame lies outside the verified inventory')


def _shard(root, header, shard_id):
    ready_path = root/f'ready_{shard_id:08d}.json'
    if not ready_path.is_file():
        raise FileNotFoundError(f'Visual shard is not committed yet: {ready_path}')
    ready_raw = ready_path.read_bytes()
    ready = json.loads(ready_raw)
    if (ready.get('schema'), ready.get('archive_id'), ready.get('shard_id')) != (
            'feature_memory_archive_ready_v1', header['archive_id'], shard_id):
        raise ValueError('Visual shard receipt identity mismatch')
    feature_path = _safe_file(root, ready['feature_file'])
    index_path = _safe_file(root, ready['index_file'])
    if feature_path.name != f'features_{shard_id:08d}.bin' or index_path.name != f'index_{shard_id:08d}.json':
        raise ValueError('Visual shard receipt points to unexpected files')
    before = (_state(ready_path), _state(index_path), _state(feature_path))
    key = (str(root.resolve()), header['archive_id'], shard_id,
           hashlib.sha256(ready_raw).hexdigest(), before)
    if key in _VERIFIED_SHARDS:
        _VERIFIED_SHARDS.move_to_end(key)
        return feature_path, copy.deepcopy(_VERIFIED_SHARDS[key]), before
    if _sha256(index_path) != ready['index_sha256']:
        raise ValueError('Visual shard index checksum mismatch')
    index = json.loads(index_path.read_text())
    if (index.get('schema'), index.get('archive_id'), index.get('shard_id'),
        index.get('dtype'), index.get('frame_shape')) != (
            'feature_memory_archive_index_v1', header['archive_id'], shard_id,
            'bfloat16', list(FRAME_SHAPE)):
        raise ValueError('Visual shard index identity or tensor contract mismatch')
    rows = index['frames']
    first = shard_id * header['frames_per_shard']
    expected_count = min(header['frames_per_shard'], header['inventory']['total_1fps_frames']-first)
    if (ready['first_frame_id'], ready['frame_count'], len(rows)) != (first, expected_count, expected_count):
        raise ValueError('Visual shard receipt/index frame count mismatch')
    if ready['feature_bytes'] != expected_count * FRAME_BYTES or feature_path.stat().st_size != ready['feature_bytes']:
        raise ValueError('Visual shard payload byte count mismatch')
    for position, row in enumerate(rows):
        if row['frame_id'] != first+position:
            raise ValueError('Missing, duplicate, or unordered visual frame IDs')
        # Seed-prefix rows use `bytes` (and normally `feature_bytes`), while
        # native producer rows use `feature_bytes` alone. Both are explicit
        # byte-count declarations, not permissions to infer a missing size.
        # Check every declared alias so one correct field cannot hide a bad one.
        byte_fields = [name for name in ('bytes', 'feature_bytes') if name in row]
        if not byte_fields or any(type(row[name]) is not int or row[name] != FRAME_BYTES
                                  for name in byte_fields):
            raise ValueError('Visual frame byte count missing, inconsistent, or incompatible')
        if (type(row['offset']) is not int or
                (row['offset'], row['shape'], row['dtype']) != (
                    position*FRAME_BYTES, list(FRAME_SHAPE), 'bfloat16')):
            raise ValueError('Visual frame offset/size/shape/dtype mismatch')
        envelope = _frame_envelope(header['inventory'], row['frame_id'])
        for name, expected in envelope.items():
            if name in row and row[name] != expected:
                raise ValueError(f'Visual frame timeline/source mismatch: {name}')
        if row['timestamp'] != envelope['timestamp']:
            raise ValueError('Visual frame timestamp mismatch')
        encoder = row.get('encoder', {})
        if encoder.get('grid_thw') != [1, 52, 52] or encoder.get('tensor_shape') != list(FRAME_SHAPE):
            raise ValueError('Visual frame native image grid/merger contract mismatch')
        nested = row.get('metadata', {})
        for name, expected in [('day_id', envelope['day_id']), ('source_id', envelope['source_id']),
                ('file_seconds', envelope['source_timestamp']), ('source_fingerprint', envelope['source_sha256'])]:
            if name in nested and nested[name] != expected:
                raise ValueError(f'Visual frame nested metadata mismatch: {name}')
        row.update(envelope)
        row['visual_archive_id'] = header['archive_id']
        row['model_checkpoint'] = REPO_ID
        row['model_revision'] = REVISION
        row['grid_thw'] = [1, 52, 52]
        row['representation'] = 'qwen2_5_vl_post_spatial_merger_image_features'
    if ready['last_timestamp'] != rows[-1]['timestamp']:
        raise ValueError('Visual shard last timestamp mismatch')
    # This reads the complete containing shard solely for integrity; only the
    # requested frame slices are materialized or forwarded into the model.
    if _sha256(feature_path) != ready['feature_sha256']:
        raise ValueError('Visual shard payload checksum mismatch')
    if before != (_state(ready_path), _state(index_path), _state(feature_path)):
        raise ValueError('Committed visual shard changed during verification')
    _VERIFIED_SHARDS[key] = copy.deepcopy(rows)
    while len(_VERIFIED_SHARDS) > 32:
        _VERIFIED_SHARDS.popitem(last=False)
    return feature_path, rows, before


def read_feature_frames(root, start, count):
    """Return verified requested features and source-enriched frame envelopes.

    Missing ready receipts fail immediately. The reader never waits for or
    substitutes incomplete future shards. Per-frame hashes are checked for all
    returned values, even when an unchanged containing-shard hash is reused.
    """
    root = Path(root)
    if type(start) is not int or start < 0 or type(count) is not int or count < 0:
        raise ValueError('start/count must be nonnegative integers')
    header = _header(root)
    if start+count > header['inventory']['total_1fps_frames']:
        raise ValueError('Requested frame range exceeds the source inventory')
    if not count:
        return torch.empty((0, *FRAME_SHAPE), dtype=torch.bfloat16), []
    result = torch.empty((count, *FRAME_SHAPE), dtype=torch.bfloat16)
    output_rows = []
    size = header['frames_per_shard']
    for shard_id in range(start//size, (start+count-1)//size+1):
        feature_path, rows, before = _shard(root, header, shard_id)
        lo, hi = max(start, shard_id*size), min(start+count, shard_id*size+len(rows))
        with feature_path.open('rb') as stream:
            for frame_id in range(lo, hi):
                row = rows[frame_id-shard_id*size]
                stream.seek(row['offset'])
                raw = stream.read(FRAME_BYTES)
                if len(raw) != FRAME_BYTES or hashlib.sha256(raw).hexdigest() != row['shared_feature_sha256']:
                    raise ValueError('Visual frame checksum mismatch')
                values = torch.from_numpy(np.frombuffer(raw, dtype='<i2').copy()).view(torch.bfloat16).reshape(FRAME_SHAPE)
                result[frame_id-start].copy_(values)
                output_rows.append(copy.deepcopy(row))
        ready_path = root/f'ready_{shard_id:08d}.json'
        index_path = root/f'index_{shard_id:08d}.json'
        if before != (_state(ready_path), _state(index_path), _state(feature_path)):
            raise ValueError('Committed visual shard changed while reading frames')
    if [r['frame_id'] for r in output_rows] != list(range(start, start+count)):
        raise ValueError('Requested visual frame coverage mismatch')
    return result, output_rows
