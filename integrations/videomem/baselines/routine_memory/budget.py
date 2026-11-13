"""Deterministic accounting for all retained routine state and its checkpoint.

The hard charge includes complete CPU tensor backing allocations, the exact
canonical checkpoint index/manifest size, and conservative deterministic Python
object reserves. Process RSS, allocator caches, model weights and bounded input
or serialization buffers are runtime costs, not additional retrievable memory.
"""
from __future__ import annotations

import json
import math
import sys

import torch

from baselines.streambench.accounting import measure_state


COMPONENTS = ('short_term', 'routine_memory', 'event_memory', 'control')
FORMAT = 'routine_memory_checkpoint_v1'
DTYPES = {str(getattr(torch, name)).removeprefix('torch.'): getattr(torch, name)
          for name in ('bfloat16', 'float16', 'float32', 'float64', 'int8', 'int16',
                       'int32', 'int64', 'uint8', 'bool')}


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def manifest_for(index_bytes, feature_bytes, feature_sha256='0' * 64, index_sha256='0' * 64):
    """Digest strings have fixed length, so size is exact before writing data."""
    value = {'schema': FORMAT, 'byte_order': 'little', 'features_file': 'features.bin',
             'index_file': 'state.json', 'features_bytes': feature_bytes,
             'index_bytes': index_bytes, 'features_sha256': feature_sha256,
             'index_sha256': index_sha256, 'total_file_bytes': 0}
    while True:
        total = feature_bytes + index_bytes + len(canonical_json(value))
        if value['total_file_bytes'] == total:
            return value
        value['total_file_bytes'] = total


def pack_state(state):
    """Make a small JSON index and zero-copy storage handles, never copy features.

    Normal tensor views and repeated tensor references preserve their full
    storage aliases. Distinct partially overlapping frombuffer allocations are
    rejected: the core owns compact CPU tensors and never creates such owners.
    """
    if not isinstance(state, dict) or set(state) != set(COMPONENTS):
        raise ValueError('State must contain exactly short_term, routine_memory, event_memory, control')
    if sys.byteorder != 'little':
        raise ValueError('Routine checkpoint currently requires a little-endian host')
    storages, tensors, storage_lookup, tensor_lookup = [], [], {}, {}
    storage_handles = []
    by_component = {name: [] for name in COMPONENTS}
    ancestors = set()
    offset = 0

    def encode(value, component):
        nonlocal offset
        if isinstance(value, torch.Tensor):
            if (value.device.type != 'cpu' or value.layout != torch.strided
                    or value.is_quantized or value.dtype not in DTYPES.values()
                    or value.is_conj() or value.is_neg()):
                raise TypeError('Retained tensors must have a supported ordinary CPU strided dtype')
            by_component[component].append(value)
            key = id(value)
            if key not in tensor_lookup:
                storage = value.untyped_storage()
                if value.numel() * value.element_size() > storage.nbytes():
                    raise ValueError('Expanded tensor views must be compacted before retention')
                storage_key = (storage.data_ptr(), storage.nbytes())
                if storage_key not in storage_lookup:
                    storage_lookup[storage_key] = len(storages)
                    storages.append({'offset': offset, 'nbytes': storage.nbytes()})
                    storage_handles.append(storage)
                    offset += storage.nbytes()
                tensor_lookup[key] = len(tensors)
                tensors.append({'storage': storage_lookup[storage_key],
                                'dtype': str(value.dtype).removeprefix('torch.'),
                                'shape': list(value.shape), 'stride': list(value.stride()),
                                'storage_offset': value.storage_offset()})
            return {'tensor': tensor_lookup[key]}
        if value is None or type(value) in (bool, int, str):
            return {'scalar': value}
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError('Persistent scalar metadata must be finite')
            return {'scalar': value}
        if type(value) not in (dict, list, tuple):
            raise TypeError('Unaccounted persistent state type: ' + type(value).__name__)
        if id(value) in ancestors:
            raise ValueError('Cyclic retained state is unsupported')
        ancestors.add(id(value))
        try:
            if isinstance(value, dict):
                if any(type(key) is not str for key in value):
                    raise TypeError('Persistent state dictionary keys must be strings')
                return {'dict': [[key, encode(value[key], component)] for key in sorted(value)]}
            return {'tuple' if isinstance(value, tuple) else 'list': [encode(x, component) for x in value]}
        finally:
            ancestors.remove(id(value))

    tree = {'dict': [[name, encode(state[name], name)] for name in COMPONENTS]}
    index = {'schema': FORMAT, 'state': tree, 'tensors': tensors, 'storages': storages}
    index_raw = canonical_json(index)
    names = ('visual_memory_bytes', 'kv_memory_bytes', 'text_memory_bytes', 'metadata_bytes')
    storage_counts = measure_state(dict(zip(names, (by_component[name] for name in COMPONENTS))))
    if storage_counts['persistent_total_bytes'] != offset:
        raise ValueError('Distinct partially overlapping backing owners are unsupported; use owned clones')
    return {'index': index, 'index_raw': index_raw, 'storage_handles': storage_handles,
            'feature_bytes': offset,
            'component_storage_bytes': dict(zip(COMPONENTS, (storage_counts[name] for name in names)))}


def _object_charge(value):
    """Stable reserves avoid eviction changing after JSON reconstructs containers.

    Count each reference conservatively, including shared immutable scalars.
    Actual CPython object sizes are reported separately; these reserves cover
    ordinary fresh core containers without depending on their allocation growth.
    Tensor payload is charged separately through the existing storage accountant.
    """
    if isinstance(value, torch.Tensor):
        return 384
    if value is None or type(value) in (bool, float):
        return 32
    if type(value) is int:
        return 64 + (abs(value).bit_length() + 7) // 8
    if type(value) is str:
        return 64 + 4 * len(value.encode('utf-8'))
    if isinstance(value, dict):
        return 256 + 64 * len(value) + sum(_object_charge(k) + _object_charge(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return 128 + 16 * len(value) + sum(_object_charge(x) for x in value)
    raise TypeError('Unsupported object in retained state')


def _python_size(value, seen):
    if id(value) in seen:
        return 0
    seen.add(id(value))
    total = sys.getsizeof(value)
    if isinstance(value, dict):
        total += sum(_python_size(k, seen) + _python_size(v, seen) for k, v in value.items())
    elif isinstance(value, (list, tuple)):
        total += sum(_python_size(x, seen) for x in value)
    return total


def measure_state_bytes(state):
    packed = pack_state(state)
    manifest = manifest_for(len(packed['index_raw']), packed['feature_bytes'])
    overhead = len(packed['index_raw']) + len(canonical_json(manifest))
    reserves = {name: _object_charge(state[name]) for name in COMPONENTS}
    # Include the top-level map itself and its key objects in metadata.
    reserves['control'] += 256 + 64 * len(COMPONENTS) + sum(_object_charge(x) for x in COMPONENTS)
    actual_python = _python_size(state, set())
    reserve_total = sum(reserves.values())
    if actual_python > reserve_total:
        raise ValueError('Persistent containers exceed deterministic object reserve; compact owned containers')
    charges = {name: packed['component_storage_bytes'][name] + reserves[name] for name in COMPONENTS}
    metadata = charges['control'] + overhead
    actual = sum(charges.values()) + overhead
    return {'short_term_bytes': charges['short_term'], 'routine_memory_bytes': charges['routine_memory'],
            'event_memory_bytes': charges['event_memory'], 'metadata_bytes': metadata,
            'actual_memory_bytes': actual, 'serialized_bytes': manifest['total_file_bytes'],
            'tensor_storage_bytes': packed['feature_bytes'],
            'canonical_metadata_and_index_bytes': overhead,
            'python_object_reserve_bytes': reserve_total,
            'python_object_bytes_estimate': actual_python,
            'resident_state_bytes_estimate': packed['feature_bytes'] + actual_python,
            'accounting_schema': 'full_cpu_tensor_storage_plus_canonical_checkpoint_plus_object_reserve_v1',
            'runtime_rss_and_allocator_caches_separate': True}
