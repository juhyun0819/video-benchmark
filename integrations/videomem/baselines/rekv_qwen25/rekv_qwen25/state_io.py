"""Lossless JSON+safetensors representation of a completed ReKVSession state.

Use ``flat, descriptor = flatten_state(session.export_state())``. Merge ``flat``
with the archive block's raw K/V tensors, and store ``descriptor`` as metadata's
``resume_state``. After ``load_block``, call
``session.import_state(unflatten_state(tensors, metadata['resume_state']))``.
No pickle, Python class reconstruction, model calls, or GPU work is involved.
"""
from __future__ import annotations

import hashlib
import copy
import json
import math
import re

import torch

SCHEMA = 'rekv_session_state_tree_v1'


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def _hash(tensor):
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def flatten_state(state, prefix='resume_state'):
    """Return ``(CPU tensor dictionary, JSON descriptor)`` without value loss.

    Tuple/list distinctions and integer position/counter types are preserved.
    Both raw local/sink KV and the optional short-prefix native post-RoPE cache
    remain explicitly identified by their tree paths. Tensor aliases become
    independent copies, consistent with ``ReKVSession.export_state`` semantics.
    """
    if not isinstance(state, dict) or state.get('schema') != 1:
        raise ValueError('Expected a ReKVSession schema=1 exported state')
    if not isinstance(prefix, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', prefix):
        raise ValueError('State tensor prefix must be a simple identifier')
    flat, specs = {}, {}

    def visit(value, path):
        if isinstance(value, torch.Tensor):
            if value.layout != torch.strided:
                raise ValueError('Resume state contains a non-dense tensor')
            name = f'{prefix}.tensor_{len(flat):06d}'
            tensor = value.detach().cpu().contiguous().clone()
            flat[name] = tensor
            specs[name] = dict(path=path, shape=list(tensor.shape), dtype=str(tensor.dtype),
                bytes=tensor.numel()*tensor.element_size(), sha256=_hash(tensor))
            return dict(kind='tensor', name=name)
        if isinstance(value, dict):
            if any(not isinstance(k, str) for k in value):
                raise ValueError('Resume dictionary keys must be strings')
            return dict(kind='dict', items=[[k, visit(v, path+[k])] for k, v in value.items()])
        if isinstance(value, (list, tuple)):
            return dict(kind='tuple' if isinstance(value, tuple) else 'list',
                        items=[visit(v,path+[i]) for i,v in enumerate(value)])
        if value is None:
            return dict(kind='none')
        if type(value) in (str, int, bool):
            return dict(kind=type(value).__name__, value=value)
        if type(value) is float and math.isfinite(value):
            return dict(kind='float', value=value)
        raise ValueError(f'Unsupported resume-state value at {path}: {type(value).__name__}')

    tree = visit(state, [])
    descriptor = dict(schema=SCHEMA, tensor_prefix=prefix, tree=tree,
        tree_sha256=hashlib.sha256(_json(tree)).hexdigest(), tensors=specs,
        tensor_count=len(flat), tensor_payload_bytes=sum(s['bytes'] for s in specs.values()))
    _json(descriptor)
    return flat, descriptor


def unflatten_state(tensors, descriptor):
    """Validate and rebuild a CPU state accepted by ``ReKVSession.import_state``.

    Unrelated raw archive tensors are allowed. Missing or additional tensors in
    this descriptor's resume namespace, changed values, dtypes or shapes fail.
    The session itself subsequently checks policy/model/layer identity.
    """
    if descriptor.get('schema') != SCHEMA:
        raise ValueError('Unsupported serialized ReKVSession state schema')
    prefix = descriptor['tensor_prefix']
    if not isinstance(prefix, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', prefix):
        raise ValueError('Invalid state tensor namespace')
    tree, specs = descriptor['tree'], descriptor['tensors']
    if hashlib.sha256(_json(tree)).hexdigest() != descriptor['tree_sha256']:
        raise ValueError('Resume state tree checksum mismatch')
    actual = {name for name in tensors if name.startswith(prefix+'.')}
    if actual != set(specs) or len(specs) != descriptor['tensor_count']:
        raise ValueError('Resume state tensor coverage mismatch')
    if sum(s['bytes'] for s in specs.values()) != descriptor['tensor_payload_bytes']:
        raise ValueError('Resume state payload byte count mismatch')
    consumed = set()

    def visit(node, path):
        kind = node['kind']
        if kind == 'tensor':
            name = node['name']
            if name in consumed or name not in specs:
                raise ValueError('Duplicate or unknown resume tensor reference')
            spec, tensor = specs[name], tensors[name]
            if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
                raise ValueError('Invalid resume tensor object')
            observed = (list(tensor.shape), str(tensor.dtype), tensor.numel()*tensor.element_size())
            if observed != (spec['shape'], spec['dtype'], spec['bytes']) or spec['path'] != path:
                raise ValueError('Resume tensor path/shape/dtype/size mismatch')
            if _hash(tensor) != spec['sha256']:
                raise ValueError('Resume tensor checksum mismatch')
            consumed.add(name)
            return tensor.detach().cpu().contiguous().clone()
        if kind == 'dict':
            keys = [item[0] for item in node['items']]
            if any(not isinstance(k,str) for k in keys) or len(set(keys)) != len(keys):
                raise ValueError('Invalid/duplicate resume dictionary keys')
            return {key: visit(value,path+[key]) for key,value in node['items']}
        if kind in ('list','tuple'):
            values = [visit(value,path+[i]) for i,value in enumerate(node['items'])]
            return tuple(values) if kind == 'tuple' else values
        if kind == 'none':
            return None
        allowed = {'str':str, 'int':int, 'bool':bool, 'float':float}
        if kind not in allowed or type(node.get('value')) is not allowed[kind]:
            raise ValueError('Invalid resume scalar value/type')
        if kind == 'float' and not math.isfinite(node['value']):
            raise ValueError('Nonfinite resume scalar')
        return node['value']

    restored = visit(tree, [])
    if consumed != set(specs):
        raise ValueError('Unreferenced resume tensors')
    if not isinstance(restored, dict) or restored.get('schema') != 1:
        raise ValueError('Serialized state is not a ReKVSession export')
    return restored


def reconstruct_long_state(template_identity, chunk_tensors, sink_tensors, total_seen):
    """Rebuild exact long-context state from already archived raw KV and sinks.

    ``template_identity`` is ``ReKVSession._identity()`` or an exported state's
    identity fields. ``chunk_tensors`` contains ``position_ids`` [3,1,T] and
    ``layer_XX.key/value`` [1,H,T,D], covering the final T tokens in causal order.
    T must be at least W=n_local; concatenate preceding committed blocks first
    when the last archive block alone is shorter. ``sink_tensors`` contains the
    same layer key/value names for the original fixed initial prefix.

    The caller must verify both artifacts' archive/checkpoint identity before
    using them. This function copies the last W raw KV/positions and first
    n_init sink tokens; it never rotates keys or recomputes model outputs.
    Short prefixes (total_seen <= W) need their native post-RoPE cache and must
    instead use flatten_state/unflatten_state. No duplicate large resume-state
    payload needs to be stored for a sufficiently long archive block.
    """
    identity_fields = ('schema', 'n_init', 'n_local', 'layers', 'policy', 'upstream', 'geometry')
    if not isinstance(template_identity, dict) or any(k not in template_identity for k in identity_fields):
        raise ValueError('Missing ReKVSession identity fields')
    identity = {k: copy.deepcopy(template_identity[k]) for k in identity_fields}
    n_init, window, layers = identity['n_init'], identity['n_local'], identity['layers']
    if (identity['schema'] != 1 or type(window) is not int or window <= 0 or
            type(n_init) is not int or not 0 <= n_init <= window or
            type(layers) is not int or layers <= 0 or
            not isinstance(identity['geometry'], list) or len(identity['geometry']) != layers):
        raise ValueError('Invalid ReKVSession identity geometry/window')
    if type(total_seen) is not int or total_seen <= window:
        raise ValueError('Long-state reconstruction requires total_seen > n_local')
    if not isinstance(chunk_tensors, dict) or not isinstance(sink_tensors, dict):
        raise ValueError('Chunk and sink tensors must be dictionaries')
    positions = chunk_tensors.get('position_ids')
    if (not isinstance(positions, torch.Tensor) or positions.layout != torch.strided or
            positions.dtype != torch.int64 or positions.ndim != 3 or
            tuple(positions.shape[:2]) != (3, 1)):
        raise ValueError('Archive position_ids must be int64 [3,1,T]')
    count = positions.shape[-1]
    if count < window or count > total_seen:
        raise ValueError('Raw archive coverage must include at least the last n_local tokens')
    restored = []
    for layer, geometry in enumerate(identity['geometry']):
        heads, dim = geometry.get('kv_heads'), geometry.get('head_dim')
        if type(heads) is not int or heads <= 0 or type(dim) is not int or dim <= 0:
            raise ValueError('Invalid KV head geometry')
        current = {}
        for kind in ('key', 'value'):
            name = f'layer_{layer:02d}.{kind}'
            tensor = chunk_tensors.get(name)
            if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided or
                    tuple(tensor.shape) != (1, heads, count, dim) or
                    not tensor.is_floating_point() or str(tensor.dtype) != geometry.get('dtype')):
                raise ValueError(f'Raw archive tensor geometry/dtype mismatch: {name}')
            current[kind] = tensor[:, :, -window:].detach().cpu().contiguous().clone()
            if n_init:
                sink = sink_tensors.get(name)
                if (not isinstance(sink, torch.Tensor) or sink.layout != torch.strided or
                        sink.ndim != 4 or tuple(sink.shape[:2]) != (1, heads) or
                        sink.shape[2] < n_init or sink.shape[3] != dim or sink.dtype != tensor.dtype):
                    raise ValueError(f'Initial sink tensor geometry/dtype mismatch: {name}')
                current[f'init_{kind}'] = sink[:, :, :n_init].detach().cpu().contiguous().clone()
            else:
                current[f'init_{kind}'] = torch.empty((1, heads, 0, dim), dtype=tensor.dtype)
        # A separate owned position tensor per layer matches export_state.
        current['positions'] = positions[:, :, -window:].detach().cpu().contiguous().clone()
        restored.append(current)
    return dict(identity, total_seen=total_seen, states=restored, native_post_rope_cache=None)
