"""Deterministic persistent payload bytes, separate from allocator/RSS overhead.

Tensors charge their entire underlying allocation, including unused view regions.
Aliased allocations are charged once across categories. CPU/GPU/disk copies are
separate. Metadata charges UTF-8 field names/text, 64-bit scalars (larger integers
as needed) and contents; Python allocator/container headers are runtime RAM,
reported separately. Unknown objects fail closed: adapters must expose state.
"""
from bisect import bisect_left
from collections import deque
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

COMPONENTS = ('visual_memory_bytes', 'kv_memory_bytes', 'text_memory_bytes',
              'fast_weight_bytes', 'metadata_bytes')


@dataclass(frozen=True)
class DiskState:
    """A retrievable file/directory, not merely an audit output path."""
    path: Path


def measure_state(components: dict[str, Any]) -> dict[str, int]:
    if not isinstance(components, dict) or set(components) - set(COMPONENTS):
        raise ValueError('persistent_state must map supported byte components to live values')
    import torch
    import numpy as np
    seen_objects, seen_files = set(), set()
    spans = {}

    def allocation(device, start, size):
        if size == 0:
            return 0
        if start == 0:
            raise ValueError('Persistent tensor has no materialized storage')
        ranges = spans.setdefault(str(device), [])
        end = start + size
        # Existing intervals are sorted and disjoint. Find the insertion point
        # in logarithmic comparisons, then inspect only intersecting/adjacent
        # intervals. Scanning every old allocation made per-frame accounting
        # quadratic in the number of retained native-memory tensors.
        left = bisect_left(ranges, (start,))
        if left and ranges[left - 1][1] >= start:
            left -= 1
        right = left
        overlap, lo, hi = 0, start, end
        while right < len(ranges) and ranges[right][0] <= hi:
            a, b = ranges[right]
            # Charge only intersection with the ORIGINAL new allocation;
            # the expanded union endpoints are used solely for coalescing.
            overlap += max(0, min(end, b) - max(start, a))
            lo, hi = min(lo, a), max(hi, b)
            right += 1
        ranges[left:right] = [(lo, hi)]
        return size - overlap

    def size(value):
        if value is None:
            return 0
        if isinstance(value, torch.Tensor):
            if value.layout != torch.strided:
                raise TypeError('Expose sparse/quantized auxiliary storage explicitly')
            storage = value.untyped_storage()
            return allocation(value.device, storage.data_ptr(), storage.nbytes())
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise TypeError('Object arrays must be exposed as explicit state')
            base = value
            while isinstance(base.base, np.ndarray):
                base = base.base
            if isinstance(base.base, torch.Tensor):
                return size(base.base)
            if base.base is not None:
                owner = base.base
                while isinstance(owner, memoryview):
                    owner = owner.obj
                try:
                    backing = np.frombuffer(owner, dtype=np.uint8)
                except (TypeError, ValueError) as exc:
                    raise TypeError('Expose the full owner of this NumPy storage explicitly') from exc
                return allocation('cpu', backing.__array_interface__['data'][0], backing.nbytes)
            return allocation('cpu', base.__array_interface__['data'][0], base.nbytes)
        if isinstance(value, (bool, np.bool_)):
            return 1
        if isinstance(value, (int, np.integer)):
            return max(8, (int(value).bit_length() + 8) // 8)
        if isinstance(value, (float, np.floating)):
            return 8
        if isinstance(value, str):
            return len(value.encode('utf-8'))
        if isinstance(value, (bytes, bytearray, memoryview)):
            if id(value) in seen_objects:
                return 0
            seen_objects.add(id(value))
            return value.nbytes if isinstance(value, memoryview) else len(value)
        if isinstance(value, (torch.dtype, torch.device, Path)):
            return len(str(value).encode('utf-8'))
        if id(value) in seen_objects:
            return 0
        seen_objects.add(id(value))
        if isinstance(value, DiskState):
            path = Path(value.path)
            if not path.exists():
                raise FileNotFoundError(path)
            paths = [path] if path.is_file() else sorted(p for p in path.rglob('*') if p.is_file())
            total = 0
            for p in paths:
                stat = p.stat()
                key = stat.st_dev, stat.st_ino
                if key not in seen_files:
                    seen_files.add(key)
                    total += stat.st_size
                total += len(str(p).encode('utf-8'))
            return total
        if isinstance(value, dict):
            return sum(size(k) + size(v) for k, v in value.items())
        if isinstance(value, (list, tuple, deque, set, frozenset)):
            return sum(size(v) for v in value)
        if is_dataclass(value) and not isinstance(value, type):
            return sum(size(f.name) + size(getattr(value, f.name)) for f in fields(value))
        raise TypeError(f'Unaccounted persistent object: {type(value).__module__}.{type(value).__name__}')

    result = {name: size(components.get(name)) for name in COMPONENTS}
    result['persistent_total_bytes'] = sum(result.values())
    return result
