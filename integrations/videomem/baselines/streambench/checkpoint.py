"""Atomic, hashed, local checkpoints preserving tensor storage aliasing."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_checkpoint(path, payload):
    import torch
    from .state import validate_checkpoint_state
    validate_checkpoint_state(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'Checkpoint already exists: {path}')
    temporary = Path(tempfile.mkdtemp(prefix='.' + path.name + '-', dir=path.parent))
    try:
        with (temporary/'state.pt').open('wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        write_json(temporary/'metadata.json', payload.get('metadata', {
            'adapter': payload.get('adapter'), 'memory_budget_bytes': payload.get('memory_budget_bytes')}))
        write_json(temporary/'integrity.json', {'schema_version': 1, 'state_sha256': file_digest(temporary/'state.pt'),
                                               'metadata_sha256': file_digest(temporary/'metadata.json')})
        temporary.rename(path)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return sum(p.stat().st_size for p in path.iterdir() if p.is_file())


def read_checkpoint(path):
    """Only load checkpoints produced locally/by a trusted producer (torch pickle)."""
    import torch
    path = Path(path)
    integrity = json.loads((path/'integrity.json').read_text())
    for name, key in [('state.pt','state_sha256'), ('metadata.json','metadata_sha256')]:
        if file_digest(path/name) != integrity[key]:
            raise ValueError(f'Checkpoint integrity mismatch: {name}')
    payload = torch.load(path/'state.pt', map_location='cpu', weights_only=False)
    if not isinstance(payload, dict) or payload.get('schema_version') != 1:
        raise ValueError('Unsupported checkpoint schema')
    if 'metadata' in payload and payload['metadata'] != json.loads((path/'metadata.json').read_text()):
        raise ValueError('Checkpoint metadata mismatch')
    return payload
