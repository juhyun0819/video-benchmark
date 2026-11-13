"""Uncompressed feature serialization; archives are never read by policies."""
import json
import os
from pathlib import Path

import torch
from safetensors.torch import save_file


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class HistoryWriter:
    """Append row-major tensor bytes plus a per-frame seek index, in arrival order.

    features.bin is raw little-endian float16/bfloat16/float32 data (no pixels).
    index.jsonl supplies each frame's offset, dtype and shape. This avoids keeping
    an entire history/shard in RAM and makes serialization overhead measurable.
    This optional measurement archive is outside every bounded policy and must
    never be consulted when updating or querying those policies.
    """
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.data = (self.directory / 'features.bin').open('xb')
        self.index = (self.directory / 'index.jsonl').open('x', encoding='utf-8')
        self.payload_bytes = 0
        self.frames = 0
        write_json(self.directory / 'format.json', {
            'format': 'videomem_contiguous_features_v1', 'byte_order': 'little',
            'order': 'C', 'compression': 'none', 'raw_images_stored': False,
            'index': 'index.jsonl', 'payload': 'features.bin',
            'bounded_policies_can_read_archive': False})

    def append(self, features, frame_id, timestamp, metadata):
        if features.device.type != 'cpu' or not features.is_contiguous() or features.ndim != 2:
            raise ValueError('Archive needs a contiguous CPU [tokens, hidden] tensor')
        nbytes = features.numel() * features.element_size()
        payload = memoryview(features.view(torch.uint8).numpy()).cast('B')
        if self.data.write(payload) != nbytes:
            raise IOError('Incomplete feature archive write')
        row = {'frame_id': frame_id, 'timestamp': timestamp, 'offset': self.payload_bytes,
               'nbytes': nbytes, 'shape': list(features.shape),
               'dtype': str(features.dtype).removeprefix('torch.'),
               'grid_thw': metadata.get('grid_thw')}
        self.index.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
        self.payload_bytes += nbytes
        self.frames += 1

    def sizes(self):
        if not self.data.closed:
            self.data.flush()
            self.index.flush()
        files = {p.name: p.stat().st_size for p in self.directory.iterdir() if p.is_file()}
        return {'frames': self.frames, 'tensor_payload_bytes': self.payload_bytes,
                'serialized_file_bytes': sum(files.values()), 'files': files}

    def close(self):
        for handle in (self.data, self.index):
            if not handle.closed:
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()


def save_memory_snapshot(manager, directory, timestamp):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    snapshot = manager.snapshot()
    save_file(snapshot['tensors'], str(directory / 'features.safetensors'))
    row = dict(snapshot['metadata'])
    row.update(timestamp=timestamp, timestamp_hours=timestamp / 3600,
               feature_file='features.safetensors',
               feature_file_bytes=(directory / 'features.safetensors').stat().st_size,
               state=snapshot.get('state', {}))
    write_json(directory / 'metadata.json', row)
    # The small receipt is deliberately not included in its own byte accounting.
    sizes = {p.name: p.stat().st_size for p in directory.iterdir() if p.is_file()}
    row['snapshot_serialized_bytes'] = sum(sizes.values())
    return row
