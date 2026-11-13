"""Atomic retained-state checkpoints without pickle or feature re-encoding.

Only current retained state is written. Old checkpoints are evaluation artifacts
and are never consulted by an updating policy. Full tensor storages are streamed
once; views refer to a validated JSON index and retain their original aliases.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

import torch

from .budget import DTYPES, FORMAT, canonical_json, manifest_for, measure_state_bytes, pack_state


class CheckpointError(ValueError):
    pass


def _identity(path):
    s = Path(path).stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def _fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_all(handle, data):
    view = memoryview(data).cast('B')
    offset = 0
    while offset < len(view):
        n = handle.write(view[offset:])
        if not n:
            raise OSError('Short checkpoint write')
        offset += n


def _write_file(path, data):
    with Path(path).open('xb', buffering=0) as handle:
        _write_all(handle, data)
        os.fsync(handle.fileno())


def _budget(state):
    config = state['control'].get('config', {})
    value = config.get('hard_budget_bytes', config.get('budget_bytes'))
    if value is None or type(value) is not int or value <= 0:
        raise CheckpointError('Retained state must declare a positive hard_budget_bytes or budget_bytes')
    return value


def save_checkpoint(memory, directory):
    """Return an integrity manifest after a new directory is atomically committed.

    The caller must pause its updates during this synchronous operation. Writes
    use zero-copy storage views and at most16MiB chunks, not a16GB feature clone.
    The target must not exist; errors leave no apparently completed checkpoint.
    """
    destination = Path(directory)
    if destination.exists():
        raise FileExistsError(destination)
    state = memory.state_dict()
    accounting = measure_state_bytes(state)
    cap = _budget(state)
    if accounting['actual_memory_bytes'] > cap or accounting['serialized_bytes'] > cap:
        raise CheckpointError('Current retained state exceeds its hard memory budget')
    packed = pack_state(state)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.' + destination.name + '.partial-', dir=destination.parent))
    try:
        digest = hashlib.sha256()
        with (temporary / 'features.bin').open('xb', buffering=0) as handle:
            for storage in packed['storage_handles']:
                raw = torch.empty(0, dtype=torch.uint8).set_(storage, 0, (storage.nbytes(),), (1,))
                view = memoryview(raw.numpy()).cast('B')
                for lo in range(0, len(view), 16 * 1024 ** 2):
                    chunk = view[lo:lo + 16 * 1024 ** 2]
                    _write_all(handle, chunk)
                    digest.update(chunk)
                del view, raw
            os.fsync(handle.fileno())
        _write_file(temporary / 'state.json', packed['index_raw'])
        manifest = manifest_for(len(packed['index_raw']), packed['feature_bytes'], digest.hexdigest(),
                                hashlib.sha256(packed['index_raw']).hexdigest())
        _write_file(temporary / 'manifest.json', canonical_json(manifest))
        actual = sum(p.stat().st_size for p in temporary.iterdir())
        if actual != accounting['serialized_bytes'] or actual != manifest['total_file_bytes']:
            raise CheckpointError('Measured serialized size differs from budget accounting')
        # Detect metadata/index mutations if a caller violated the update barrier.
        if pack_state(memory.state_dict())['index_raw'] != packed['index_raw']:
            raise CheckpointError('Memory changed during checkpoint creation; pause updates while saving')
        _fsync_directory(temporary)
        if destination.exists():
            raise FileExistsError(destination)
        os.rename(temporary, destination)
        _fsync_directory(destination.parent)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise CheckpointError(name + ' must be a nonnegative integer')
    return value


def load_state(directory):
    """Validate hashes and layout, then read each storage once into owned CPU RAM."""
    directory = Path(directory)
    manifest_path, index_path, feature_path = [directory / x for x in ('manifest.json', 'state.json', 'features.bin')]
    before = {p.name: _identity(p) for p in (manifest_path, index_path, feature_path)}
    if before['manifest.json'][2] > 1024 * 1024:
        raise CheckpointError('Manifest is unexpectedly large')
    manifest_raw = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_raw)
        if (manifest['schema'] != FORMAT or manifest['byte_order'] != 'little'
                or manifest['features_file'] != 'features.bin' or manifest['index_file'] != 'state.json'):
            raise CheckpointError('Unsupported checkpoint format or paths')
        for name in ('features_bytes', 'index_bytes', 'total_file_bytes'):
            _integer(manifest[name], name)
        for name in ('features_sha256', 'index_sha256'):
            if not isinstance(manifest[name], str) or not re.fullmatch('[0-9a-f]{64}', manifest[name]):
                raise CheckpointError('Invalid SHA256 digest')
        if (before['features.bin'][2] != manifest['features_bytes']
                or before['state.json'][2] != manifest['index_bytes']
                or sum(x[2] for x in before.values()) != manifest['total_file_bytes']):
            raise CheckpointError('Checkpoint serialized byte sizes disagree')
        expected_manifest = manifest_for(manifest['index_bytes'], manifest['features_bytes'],
                                         manifest['features_sha256'], manifest['index_sha256'])
        if canonical_json(expected_manifest) != manifest_raw:
            raise CheckpointError('Manifest is not canonical or has unsupported fields')
        index_raw = index_path.read_bytes()
        if hashlib.sha256(index_raw).hexdigest() != manifest['index_sha256']:
            raise CheckpointError('Checkpoint index SHA256 mismatch')
        index = json.loads(index_raw)
        if index['schema'] != FORMAT or canonical_json(index) != index_raw:
            raise CheckpointError('Invalid checkpoint index encoding')
        offset = 0
        for storage in index['storages']:
            if _integer(storage['offset'], 'storage offset') != offset:
                raise CheckpointError('Storage offsets are not contiguous')
            offset += _integer(storage['nbytes'], 'storage nbytes')
        if offset != manifest['features_bytes']:
            raise CheckpointError('Storage table size differs from feature file')
        for item in index['tensors']:
            s = _integer(item['storage'], 'tensor storage')
            if s >= len(index['storages']) or item['dtype'] not in DTYPES:
                raise CheckpointError('Invalid tensor storage or dtype')
            shape, stride = item['shape'], item['stride']
            if not isinstance(shape, list) or not isinstance(stride, list) or len(shape) != len(stride):
                raise CheckpointError('Invalid tensor shape/stride')
            for number in shape + stride:
                _integer(number, 'tensor shape/stride')
            begin = _integer(item['storage_offset'], 'tensor storage offset')
            elements = math.prod(shape)
            end = begin if not elements else begin + 1 + sum((n - 1) * step for n, step in zip(shape, stride))
            itemsize = torch.empty((), dtype=DTYPES[item['dtype']]).element_size()
            if elements * itemsize > index['storages'][s]['nbytes']:
                raise CheckpointError('Expanded tensor logical size exceeds its backing storage')
            if end * itemsize > index['storages'][s]['nbytes']:
                raise CheckpointError('Tensor view extends beyond its storage')
        digest = hashlib.sha256()
        buffers = []
        with feature_path.open('rb', buffering=0) as handle:
            for storage in index['storages']:
                raw = torch.empty(storage['nbytes'], dtype=torch.uint8)
                view = memoryview(raw.numpy()).cast('B')
                offset = 0
                while offset < len(view):
                    chunk = view[offset:min(len(view), offset + 16 * 1024 ** 2)]
                    n = handle.readinto(chunk)
                    if not n:
                        raise CheckpointError('Truncated checkpoint features')
                    digest.update(chunk[:n])
                    offset += n
                buffers.append(raw)
                del view
            if handle.read(1):
                raise CheckpointError('Unexpected trailing feature bytes')
        if digest.hexdigest() != manifest['features_sha256']:
            raise CheckpointError('Checkpoint feature SHA256 mismatch')
        tensors = [torch.empty(0, dtype=DTYPES[item['dtype']]).set_(
                       buffers[item['storage']].untyped_storage(), item['storage_offset'],
                       tuple(item['shape']), tuple(item['stride'])) for item in index['tensors']]

        def decode(node):
            if not isinstance(node, dict) or len(node) != 1:
                raise CheckpointError('Invalid state tree node')
            kind, value = next(iter(node.items()))
            if kind == 'scalar':
                if value is None or type(value) in (str, int, bool) or (type(value) is float and math.isfinite(value)):
                    return value
            elif kind == 'tensor':
                if type(value) is int and 0 <= value < len(tensors):
                    return tensors[value]
            elif kind in ('list', 'tuple') and isinstance(value, list):
                items = [decode(x) for x in value]
                return tuple(items) if kind == 'tuple' else items
            elif kind == 'dict' and isinstance(value, list):
                result = {}
                for pair in value:
                    if not isinstance(pair, list) or len(pair) != 2 or type(pair[0]) is not str or pair[0] in result:
                        raise CheckpointError('Invalid or duplicate dictionary key')
                    result[pair[0]] = decode(pair[1])
                return result
            raise CheckpointError('Unsupported state tree node')

        state = decode(index['state'])
        accounting = measure_state_bytes(state)
        if accounting['serialized_bytes'] != manifest['total_file_bytes']:
            raise CheckpointError('Decoded state does not match canonical checkpoint byte accounting')
        if accounting['actual_memory_bytes'] > _budget(state):
            raise CheckpointError('Restored state exceeds its declared hard budget')
        if any(_identity(directory / name) != identity for name, identity in before.items()):
            raise CheckpointError('Checkpoint files changed during load')
        return state, manifest
    except CheckpointError:
        raise
    except (KeyError, TypeError, IndexError, json.JSONDecodeError, OverflowError, RuntimeError) as exc:
        raise CheckpointError('Malformed retained-state checkpoint: ' + str(exc)) from exc


def load_checkpoint(directory):
    from .memory import RoutineAwareMemory
    state, _ = load_state(directory)
    return RoutineAwareMemory.from_state_dict(state)


class RoutineSnapshotReader:
    """Query one retained snapshot directly, without replaying source frames."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.memory = load_checkpoint(directory)

    def stats(self):
        return self.memory.stats()

    def retrieve(self, query_features, k=5, as_of=None):
        from .retrieval import retrieve
        return retrieve(self.memory, query_features, k=k, as_of=as_of)

    def query(self, query_features, top_k=5, as_of=None):
        return self.retrieve(query_features, k=top_k, as_of=as_of)

    def iter_batches(self, max_tokens=1024):
        from .retrieval import iter_evidence_batches
        return iter_evidence_batches(self.memory, max_tokens=max_tokens)
