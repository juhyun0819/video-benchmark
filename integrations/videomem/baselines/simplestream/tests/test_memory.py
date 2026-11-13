import copy
import json

import pytest
import torch

from baselines.simplestream.memory import EncodedFrame, RecentWindow


def frame(frame_id, timestamp=None):
    return EncodedFrame(frame_id, float(frame_id) if timestamp is None else timestamp,
        torch.full((2,4),float(frame_id),dtype=torch.bfloat16),
        torch.tensor([[1,2,4]],dtype=torch.int32),{'source':'video.mp4','nested':{'day':'Day1'}})


def test_latest_window_and_cutoff_do_not_reconstruct_evicted_history():
    memory=RecentWindow(2)
    memory.observe(frame(0));memory.observe(frame(1))
    assert [f.frame_id for f in memory.selected(0.5)]==[0]
    memory.observe(frame(2))
    assert [f.frame_id for f in memory.selected()]==[1,2]
    assert [f.frame_id for f in memory.selected(2)]==[1,2]
    assert memory.selected(-0.1)==[]
    with pytest.raises(ValueError,match='evicted'):
        memory.selected(1.5)
    assert memory.observed_count==3 and memory.window_frames==2


def test_owned_frames_selections_and_snapshots_cannot_mutate_memory():
    raw=torch.arange(8,dtype=torch.float32).reshape(2,4)
    meta={'source':{'name':'original'}}
    encoded=EncodedFrame(0,0.0,raw,torch.tensor([1,2,4]),meta)
    raw.fill_(90);meta['source']['name']='changed'
    assert encoded.features[0,0].item()==0 and encoded.metadata['source']['name']=='original'
    memory=RecentWindow(2);memory.observe(encoded)
    encoded.features.fill_(91);encoded.metadata['source']['name']='changed_again'
    selected=memory.selected();selected[0].features.fill_(92);selected[0].metadata.clear()
    snapshot=memory.state_dict();snapshot['frames'][0]['features'].fill_(93)
    snapshot['frames'][0]['metadata'].clear()
    actual=memory.selected()[0]
    assert actual.features[0,0].item()==0 and actual.metadata['source']['name']=='original'
    borrowed=memory.state_dict(copy=False)
    assert borrowed['frames'][0]['features'].data_ptr()==memory.state_dict(copy=False)['frames'][0]['features'].data_ptr()
    assert borrowed['frames'][0]['features'].data_ptr()!=memory.state_dict()['frames'][0]['features'].data_ptr()


def test_failed_observation_validates_before_eviction():
    memory=RecentWindow(1);memory.observe(frame(4,5))
    for invalid in [frame(4,6),frame(5,5),frame(3,7)]:
        with pytest.raises(ValueError):memory.observe(invalid)
    invalid=frame(5,6);invalid.features[0,0]=float('nan')
    with pytest.raises(ValueError,match='finite'):memory.observe(invalid)
    invalid=frame(5,6);invalid.metadata['question']='What happened?'
    with pytest.raises(ValueError,match='metadata'):memory.observe(invalid)
    assert memory.observed_count==1 and [f.frame_id for f in memory.selected()]==[4]


def test_roundtrip_retains_eviction_history_and_owned_tensors():
    memory=RecentWindow(2)
    for frame_id in (2,4,7):memory.observe(frame(frame_id,frame_id/2))
    state=memory.state_dict();restored=RecentWindow.from_state_dict(state)
    state['frames'][0]['features'].fill_(99)
    assert [f.frame_id for f in restored.selected()]==[4,7]
    assert restored.selected()[0].features[0,0].item()==4
    with pytest.raises(ValueError,match='evicted'):restored.selected(2)
    restored.observe(frame(8,4))
    assert [f.frame_id for f in restored.selected()]==[7,8]
    assert [f.frame_id for f in memory.selected()]==[4,7]
    empty=RecentWindow.from_state_dict(RecentWindow(3).state_dict())
    assert empty.selected()==[] and empty.observed_count==0


@pytest.mark.parametrize('mutation',['feature_shape','feature_dtype','grid_shape','grid_dtype',
                                     'count','history','eviction','qa_metadata'])
def test_state_contract_corruption_is_rejected(mutation):
    memory=RecentWindow(1);memory.observe(frame(0));memory.observe(frame(1))
    state=memory.state_dict();record=state['frames'][0]
    if mutation=='feature_shape':record['feature_shape']=[8,1]
    elif mutation=='feature_dtype':record['feature_dtype']='torch.float32'
    elif mutation=='grid_shape':record['grid_shape']=[1,3]
    elif mutation=='grid_dtype':record['grid_dtype']='torch.int32'
    elif mutation=='count':state['observed_count']=0
    elif mutation=='history':state['last_observed_frame_id']=2
    elif mutation=='eviction':state['evicted_through_timestamp']=2
    elif mutation=='qa_metadata':record['metadata']['nested']={'labels':['future_answer']}
    with pytest.raises(ValueError):RecentWindow.from_state_dict(state)


@pytest.mark.parametrize('value',[0,-1,True,1.5])
def test_invalid_window_rejected(value):
    with pytest.raises(ValueError):RecentWindow(value)


@pytest.mark.parametrize('metadata',[{'question_id':1},{'nested':{'answer':'yes'}},
                                   {'labels':[]},{'query_independent':False}])
def test_query_or_label_metadata_never_enters_memory(metadata):
    with pytest.raises(ValueError,match='metadata'):
        EncodedFrame(0,0,torch.ones((2,4)),torch.tensor([1,2,4]),metadata)


def test_grid_and_tensor_validation_and_byte_accounting():
    with pytest.raises(ValueError):EncodedFrame(0,0,torch.ones((2,4)),torch.tensor([1,0,4]),{})
    with pytest.raises(ValueError):EncodedFrame(0,0,torch.ones((2,4)),torch.tensor([1.,2.,4.]),{})
    with pytest.raises(ValueError):EncodedFrame(0,0,torch.ones((8,),dtype=torch.float32),torch.tensor([1,2,4]),{})
    memory=RecentWindow(2);memory.observe(frame(0))
    state=memory.state_dict(copy=False)
    for row in state['frames']:
        row.pop('features');row.pop('grid_thw')
    expected_metadata=len(json.dumps(state,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode())
    breakdown=memory.memory_byte_breakdown()
    assert breakdown==dict(feature_bytes=16,grid_bytes=24,metadata_bytes=expected_metadata,
                            total_bytes=16+24+expected_metadata)
    assert memory.memory_bytes()==breakdown['total_bytes']
    for cutoff in (float('nan'),float('inf'),True):
        with pytest.raises(ValueError):memory.selected(cutoff)
