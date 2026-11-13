"""Raw evidence coverage is separate from candidate identity and recurrence."""
from dataclasses import asdict
import math

import pytest
import torch

from baselines.routine_memory import RoutineAwareMemory, RoutineConfig
from baselines.routine_memory.budget import measure_state_bytes
from baselines.routine_memory.components import plain_copy


def config(**kwargs):
    return RoutineConfig(**{'budget_bytes':2_000_000, **kwargs})


def feature(activity=0, common=3., sign=1.):
    values = torch.zeros(256,32)
    values[:,0] = common
    values[:,activity+1] = 1.
    # Raw activity evidence differs; all16 bin means retain a common component.
    values[::2,activity+8] = sign*math.sqrt(10)
    values[1::2,activity+8] = -sign*math.sqrt(10)
    return values.to(torch.bfloat16)


def episode(memory, values, count=8, start=None):
    start = (memory.control['last_timestamp'] or 0)+1 if start is None else start
    return [memory.update(values,memory.control['total_frames_seen'],float(start+i))
            for i in range(count)]


def coverage(memory):
    return [e for e in memory.state_dict()['event_memory']['events']
            if e.get('retention_reason')=='anchor_change']


def same(left,right):
    if isinstance(left,torch.Tensor):
        assert torch.equal(left,right)
    elif isinstance(left,dict):
        assert set(left)==set(right)
        for key in left:same(left[key],right[key])
    elif isinstance(left,(list,tuple)):
        assert len(left)==len(right)
        for a,b in zip(left,right):same(a,b)
    else:
        assert left==right


def test_common_feature_abc_preserves_evidence_without_occurrence_inflation():
    memory=RoutineAwareMemory(config())
    for start,activity in [(0,0),(20,1),(100,0),(120,2),(200,0),(208,0),(216,1),(224,2)]:
        episode(memory,feature(activity),start=start)
    stats=memory.stats()
    assert stats['events']==7 and len(coverage(memory))==6
    assert stats['candidates']==1 and stats['routines']==stats['promotions']==0
    candidate=memory.state_dict()['event_memory']['candidates'][0]
    assert candidate['occurrence_count']==1 and len(candidate['intervals'])==1
    assert [e['evidence']['frame_id'] for e in coverage(memory)]==[10,18,26,34,50,58]


def test_coverage_pointer_does_not_replace_primary_or_churn_next_frame():
    memory=RoutineAwareMemory(config())
    episode(memory,feature(),start=0)
    primary=memory.control['pending']['event_id']
    decisions=episode(memory,feature(1),count=15)
    assert memory.control['pending']['event_id']==primary
    assert memory.control['pending']['anchor_event_id']!=primary
    assert len(coverage(memory))==1 and memory.stats()['events']==2
    assert sum(d['coverage_action']=='retained' for d in decisions)==1
    assert all(d['candidate_similarity'] is not None for d in decisions)


@pytest.mark.parametrize('noise_frames',[1,2])
def test_brief_raw_noise_does_not_create_coverage(noise_frames):
    memory=RoutineAwareMemory(config())
    episode(memory,feature(),start=0)
    episode(memory,feature(1),count=noise_frames)
    episode(memory,feature())
    assert memory.stats()['events']==1 and not coverage(memory)
    assert memory.control['pending']['anchor_change_count']==0


def test_gradual_drift_uses_fixed_event_anchor_not_candidate_ema():
    memory=RoutineAwareMemory(config())
    for index in range(400):
        angle=math.radians(150)*index/399
        value=torch.zeros(256,32)
        value[:,0]=math.cos(angle);value[:,1]=math.sin(angle)
        memory.update(value.to(torch.bfloat16),index,float(index))
    assert len(coverage(memory))>=3
    assert coverage(memory)[-1]['evidence']['frame_id']>250
    candidate=memory.state_dict()['event_memory']['candidates'][0]
    assert candidate['minimum_similarity']>.99 and candidate['occurrence_count']==1
    assert memory.stats()['promotions']==0


def test_pending_reset_recovers_latest_coverage_anchor():
    memory=RoutineAwareMemory(config())
    episode(memory,feature(),start=0)
    episode(memory,feature(1))
    anchor=memory.control['pending']['anchor_event_id']
    interruption=episode(memory,feature(3,common=0),count=1)[0]
    assert interruption['pending_reset_reason'] is not None
    episode(memory,feature(1),count=8)
    assert len(coverage(memory))==1
    assert memory.control['pending']['anchor_event_id']==anchor


def test_raw_evidence_exact_owned_and_causal():
    memory=RoutineAwareMemory(config())
    episode(memory,feature(),start=0)
    archive=torch.stack([feature(1) for _ in range(8)])
    for value in archive:episode(memory,value,count=1)
    event=coverage(memory)[0];evidence=event['evidence']
    assert evidence['frame_id']==10 and evidence['timestamp']==10
    assert event['start_frame_id']==8 and event['end_frame_id']==10
    assert torch.equal(evidence['features'],archive[2].index_select(0,evidence['token_indices']))
    assert evidence['features'].untyped_storage().data_ptr()!=archive.untyped_storage().data_ptr()
    assert evidence['features'].untyped_storage().nbytes()==evidence['features'].numel()*2
    preserved=evidence['features'].clone();archive.zero_()
    assert torch.equal(evidence['features'],preserved)


def test_coverage_survives_promotion_with_routine_attribution():
    memory=RoutineAwareMemory(config())
    for start in (0,100,200):
        episode(memory,feature(),count=4,start=start)
        episode(memory,feature(sign=-1),count=4,start=start+4)
        if start<200:episode(memory,feature(3,common=0),start=start+20)
    assert memory.stats()['promotions']==1
    events=coverage(memory)
    assert events and all(e['distinctive'] for e in events)
    assert all(e['candidate_id'] is None and e['routine_id']==1 for e in events)
    assert any(e['candidate_id'] is not None for e in memory.state_dict()['event_memory']['events'])
    RoutineAwareMemory.from_state_dict(memory.state_dict())


def test_confirmation_checkpoint_resume_is_exact():
    memory=RoutineAwareMemory(config())
    episode(memory,feature(),start=0)
    episode(memory,feature(1),count=2)
    assert memory.control['pending']['anchor_change_count']==2
    resumed=RoutineAwareMemory.from_state_dict(memory.state_dict())
    left=episode(memory,feature(1),count=1)[0]
    right=episode(resumed,feature(1),count=1)[0]
    left.pop('processing_seconds');right.pop('processing_seconds');same(left,right)
    same(memory.state_dict(),resumed.state_dict())
    assert len(coverage(memory))==1


def test_legacy_checkpoint_materializes_charged_defaults_and_rejects_tight_migration():
    memory=RoutineAwareMemory(config())
    episode(memory,feature(),start=0)
    old=plain_copy(memory.state_dict(),clone_tensors=True)
    old['control']['config'].pop('event_anchor_similarity')
    for key in RoutineAwareMemory._coverage_defaults():old['control']['pending'].pop(key)
    before=measure_state_bytes(old)['actual_memory_bytes']
    loaded=RoutineAwareMemory.from_state_dict(old)
    assert loaded.control['config']==asdict(loaded.config)
    assert loaded.config.event_anchor_similarity==.80
    assert loaded.stats()['actual_memory_bytes']>before
    for _ in range(5):old['control']['config']['budget_bytes']=measure_state_bytes(old)['actual_memory_bytes']
    assert measure_state_bytes(old)['actual_memory_bytes']<=old['control']['config']['budget_bytes']
    with pytest.raises(ValueError,match='budget'):
        RoutineAwareMemory.from_state_dict(old)


def test_budget_pressure_cleans_anchor_and_keeps_state_resumable():
    initial=RoutineAwareMemory(config(short_term_frames=2))
    episode(initial,feature(),start=0)
    cap=initial.stats()['actual_memory_bytes']+50_000
    memory=RoutineAwareMemory(config(short_term_frames=2,budget_bytes=cap))
    for index in range(60):
        memory.update(feature((index//4)%3),index,float(index))
        assert memory.stats()['actual_memory_bytes']<=cap
        assert memory.stats()['serialized_bytes']<=cap
        RoutineAwareMemory.from_state_dict(memory.state_dict())
    assert memory.stats()['event_evictions']>0


def test_pooled_instability_resets_confirmation_and_motion_does_not_inflate_occurrences():
    memory=RoutineAwareMemory(config())
    episode(memory,feature(),start=0)
    episode(memory,feature(1),count=2)
    assert memory.control['pending']['anchor_change_count']==2
    decision=episode(memory,feature(3,common=0),count=1)[0]
    assert not decision['stable'] and memory.control['pending']['anchor_change_count']==0
    episode(memory,feature())
    assert not coverage(memory)
    # Pooled-stable changing raw evidence may emit frequent coverage by design.
    for index in range(18):episode(memory,feature(1+index%2),count=1)
    assert coverage(memory)
    assert memory.state_dict()['event_memory']['candidates'][0]['occurrence_count']==1
    assert memory.stats()['promotions']==0


def test_coverage_callback_failure_rolls_back_confirmation_and_event():
    memory=RoutineAwareMemory(config())
    episode(memory,feature(),start=0)
    episode(memory,feature(1),count=2)
    before=plain_copy(memory.state_dict(),clone_tensors=True)
    def fail(evidence):
        evidence['features'].zero_()
        raise RuntimeError('coverage caption failed')
    with pytest.raises(RuntimeError,match='coverage caption failed'):
        memory.update(feature(1),10,10.,caption_callback=fail)
    same(before,memory.state_dict())
    assert episode(memory,feature(1),count=1)[0]['coverage_action']=='retained'
