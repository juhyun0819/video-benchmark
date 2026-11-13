import copy
import json

import pytest
import torch

from rekv_qwen25.archive import KVArchiveWriter, load_block
from rekv_qwen25.state_io import flatten_state, unflatten_state, reconstruct_long_state


def state_fixture(short=True):
    key = torch.arange(24,dtype=torch.float32).reshape(1,2,3,4).to(torch.bfloat16)
    value = -key
    state = dict(schema=1, n_init=1, n_local=3, layers=1, total_seen=3 if short else 8,
        policy={'name':'synthetic', 'enabled':True, 'ratio':0.5}, upstream={'revision':'pinned'},
        geometry=[dict(query_heads=4,kv_heads=2,head_dim=4,mrope_section=[1,1],
                       rope_theta=1000000.0,dtype='torch.bfloat16')],
        states=[dict(key=key,value=value,positions=torch.arange(9).reshape(3,1,3),
                     init_key=key[:,:,:1],init_value=value[:,:,:1])],
        native_post_rope_cache=[(key+1,value)] if short else None)
    return state


def equal_tree(a,b):
    assert type(a) is type(b)
    if isinstance(a,torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a,b)
    elif isinstance(a,dict):
        assert list(a)==list(b)
        for name in a:equal_tree(a[name],b[name])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal_tree(x,y)
    else:assert a==b


@pytest.mark.parametrize('short',[False,True])
def test_atomic_archive_roundtrip_preserves_full_state(tmp_path,short):
    state=state_fixture(short)
    flat,descriptor=flatten_state(state)
    descriptor=json.loads(json.dumps(descriptor))
    flat['layer_00.key']=torch.ones((1,2,1,4),dtype=torch.bfloat16)
    writer=KVArchiveWriter(tmp_path,{'checkpoint':'test','revision':'fixed'})
    writer.commit(0,flat,dict(frame_ids=[0],timestamps=[0.0],resume_state=descriptor))
    loaded,metadata=load_block(tmp_path,0,writer.config_id,cutoff_timestamp=0.0)
    restored=unflatten_state(loaded,metadata['resume_state'])
    equal_tree(state,restored)
    if short:assert isinstance(restored['native_post_rope_cache'][0],tuple)


def test_empty_initial_state_roundtrip():
    state=state_fixture()
    state.update(total_seen=0,states=[None],native_post_rope_cache=None)
    flat,descriptor=flatten_state(state)
    assert flat=={}
    equal_tree(state,unflatten_state(flat,descriptor))


@pytest.mark.parametrize('mutation',['missing','extra','dtype','value','tree'])
def test_state_mutation_rejected(mutation):
    flat,descriptor=flatten_state(state_fixture())
    name=next(iter(flat))
    if mutation=='missing':del flat[name]
    elif mutation=='extra':flat['resume_state.unexpected']=torch.tensor(1)
    elif mutation=='dtype':flat[name]=flat[name].float()
    elif mutation=='value':flat[name].reshape(-1)[0]=99
    elif mutation=='tree':descriptor['tree']['items'][0][1]['value']=2
    with pytest.raises(ValueError):unflatten_state(flat,descriptor)


def test_flatten_does_not_alias_live_state_and_restored_state_is_owned():
    original=state_fixture()
    flat,descriptor=flatten_state(original)
    original['states'][0]['key'].fill_(999)
    restored=unflatten_state(flat,descriptor)
    assert restored['states'][0]['key'].reshape(-1)[0].item()==0
    restored['states'][0]['key'].fill_(555)
    assert unflatten_state(flat,descriptor)['states'][0]['key'].reshape(-1)[0].item()==0


def long_archive_fixture():
    state = state_fixture(short=False)
    layer = state['states'][0]
    # This archive block is longer than W, so reconstruction must take its tail.
    raw = {f'layer_00.{kind}': torch.cat((torch.full_like(layer[kind][:,:,:1], 99), layer[kind]), dim=2)
           for kind in ('key', 'value')}
    raw['position_ids'] = torch.cat((torch.full_like(layer['positions'][:,:,:1], -1), layer['positions']), dim=2)
    sink = {f'layer_00.{kind}': layer[f'init_{kind}'].clone() for kind in ('key', 'value')}
    return state, raw, sink


def test_reconstruct_long_state_exactly_without_duplicated_resume_payload():
    expected, raw, sink = long_archive_fixture()
    restored = reconstruct_long_state(expected, raw, sink, total_seen=expected['total_seen'])
    for key, value in expected.items():
        if key == 'states':
            for field, tensor in value[0].items():
                equal_tree(tensor, restored['states'][0][field])
        else:
            equal_tree(value, restored[key])
    raw['layer_00.key'].fill_(55)
    sink['layer_00.key'].fill_(66)
    equal_tree(expected['states'][0]['key'], restored['states'][0]['key'])
    equal_tree(expected['states'][0]['init_key'], restored['states'][0]['init_key'])


@pytest.mark.parametrize('mutation', ['short_total', 'insufficient_tail', 'bad_positions',
                                     'missing_sink', 'wrong_dtype', 'wrong_layer_shape'])
def test_reconstruct_long_state_rejects_invalid_context(mutation):
    expected, raw, sink = long_archive_fixture()
    total = expected['total_seen']
    if mutation == 'short_total': total = expected['n_local']
    elif mutation == 'insufficient_tail': raw['position_ids'] = raw['position_ids'][:,:,:2]
    elif mutation == 'bad_positions': raw['position_ids'] = raw['position_ids'].float()
    elif mutation == 'missing_sink': sink.clear()
    elif mutation == 'wrong_dtype': raw['layer_00.key'] = raw['layer_00.key'].float()
    elif mutation == 'wrong_layer_shape': raw['layer_00.value'] = raw['layer_00.value'][:,:,:2]
    with pytest.raises(ValueError): reconstruct_long_state(expected, raw, sink, total)


def test_zero_sink_reconstruction_needs_no_sink_payload():
    expected, raw, _ = long_archive_fixture()
    expected['n_init'] = 0
    restored = reconstruct_long_state(expected, raw, {}, expected['total_seen'])
    assert restored['states'][0]['init_key'].shape == (1, 2, 0, 4)
    assert restored['states'][0]['init_value'].dtype == torch.bfloat16
