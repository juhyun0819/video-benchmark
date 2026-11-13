"""Causal routing, independent recurrence, residual preservation and hard caps."""
from dataclasses import replace

import pytest
import torch

from baselines.routine_memory import RoutineAwareMemory, RoutineConfig
from baselines.routine_memory.components import FeatureAggregator, plain_copy


def feature(index=0, tokens=16, hidden=8):
    value = torch.zeros(tokens,hidden,dtype=torch.bfloat16)
    value[:,index] = 1
    return value


def config(**kwargs):
    values = dict(budget_bytes=2_000_000,short_term_frames=4,spatial_bins=4,
                  temporal_bins=4,evidence_tokens=16,occurrence_gap_seconds=20.)
    values.update(kwargs)
    return RoutineConfig(**values)


def same(left,right):
    if isinstance(left,torch.Tensor):
        assert torch.equal(left,right)
    elif isinstance(left,dict):
        assert set(left)==set(right)
        for key in left: same(left[key],right[key])
    elif isinstance(left,(tuple,list)):
        assert len(left)==len(right)
        for a,b in zip(left,right): same(a,b)
    else:
        assert left==right


def episode(memory,values,start,count=8):
    decisions=[]
    for offset in range(count):
        fid=memory.stats()['total_frames_seen']
        decisions.append(memory.update(values,fid,float(start+offset)))
    return decisions


def promoted(memory=None):
    memory = memory or RoutineAwareMemory(config())
    for start,which in [(0,0),(10,1),(40,0),(50,1),(80,0)]:
        episode(memory,feature(which),start)
    assert memory.stats()['promotions']==1
    return memory


def test_default_total_budget_and_invalid_configuration():
    assert RoutineConfig().budget_bytes==16_000_000_000
    with pytest.raises(ValueError,match='minimal'):
        RoutineAwareMemory(RoutineConfig(budget_bytes=1))
    with pytest.raises(ValueError): RoutineConfig(stability_frames=0)
    with pytest.raises(ValueError): RoutineConfig(occurrence_gap_seconds=float('nan'))


def test_short_term_advances_but_three_consistent_frames_are_required_for_event():
    memory=RoutineAwareMemory(config())
    for index in range(2):
        decision=memory.update(feature(),index,float(index))
        assert decision['case']=='C' and not decision['stable']
        assert memory.stats()['events']==memory.stats()['candidates']==0
    decision=memory.update(feature(),2,2.)
    assert decision['stable'] and memory.stats()['events']==memory.stats()['candidates']==1
    event=memory.state_dict()['event_memory']['events'][0]
    assert event['description'] is None
    assert event['start_timestamp']==0 and event['end_timestamp']==2
    assert event['evidence']['timestamp']==2
    episode(memory,feature(),3,8)
    frames=memory.state_dict()['short_term']['frames']
    assert len(frames)==4 and [row['frame_id'] for row in frames]==[7,8,9,10]
    assert memory.stats()['events']==1


def test_single_frame_noise_does_not_create_candidates_or_events():
    memory=RoutineAwareMemory(config())
    for index in range(20):
        memory.update(feature(index%2),index,float(index))
    assert memory.stats()['events']==memory.stats()['candidates']==memory.stats()['routines']==0


def test_continuous_activity_never_increments_occurrences_even_across_long_time_gaps():
    memory=RoutineAwareMemory(config(max_occurrence_intervals=2))
    for index in range(80):
        memory.update(feature(),index,float(index*100))
    candidate=memory.state_dict()['event_memory']['candidates'][0]
    assert candidate['occurrence_count']==1 and len(candidate['intervals'])==1
    assert candidate['last_seen']==7900 and memory.stats()['promotions']==0
    assert memory.stats()['events']==1


def test_three_separated_stable_visits_promote_release_common_events_and_keep_unrelated():
    memory=promoted()
    routine=memory.state_dict()['routine_memory']['routines'][0]
    assert routine['occurrence_count']==3
    assert len(routine['representatives'])==3
    assert [x['start_timestamp'] for x in routine['intervals']]==[0.,40.,80.]
    events=memory.state_dict()['event_memory']['events']
    assert len(events)==2
    assert all(event['candidate_id']!=routine['source_candidate_id'] for event in events)
    assert all(torch.equal(event['evidence']['features'],feature(1)) for event in events)
    previous=memory.stats()['events']
    assert all(d['case']=='A' for d in episode(memory,feature(),88))
    assert memory.stats()['events']==previous
    assert memory.state_dict()['routine_memory']['routines'][0]['occurrence_count']==3


def test_partial_match_preserves_exact_residual_tokens_and_only_updates_shared_prototype():
    memory=promoted()
    before=memory.state_dict()['routine_memory']['routines'][0]['spatial'][-1].clone()
    varied=feature();varied[12:]=feature(2)[12:]
    decisions=episode(memory,varied,88)
    assert all(row['case']=='B' for row in decisions)
    assert not decisions[0]['stable'] and not decisions[1]['stable'] and decisions[2]['stable']
    residual=[event for event in memory.state_dict()['event_memory']['events'] if event['routine_id'] is not None]
    assert len(residual)==1 and residual[0]['distinctive']
    evidence=residual[0]['evidence']
    assert evidence['token_indices'].tolist()==[12,13,14,15]
    assert torch.equal(evidence['features'],varied[12:])
    assert torch.equal(before,memory.state_dict()['routine_memory']['routines'][0]['spatial'][-1])


def test_candidate_deviation_survives_promotion_and_gets_routine_association():
    memory=RoutineAwareMemory(config(spatial_bins=16))
    varied=feature();varied[-1]=feature(2)[-1]
    for start,values in [(0,feature()),(10,feature(1)),(40,varied),(50,feature(1)),(80,feature())]:
        episode(memory,values,start)
    assert memory.stats()['promotions']==1
    distinctive=[event for event in memory.state_dict()['event_memory']['events'] if event['distinctive']]
    assert len(distinctive)==1
    event=distinctive[0]
    assert event['candidate_id'] is None and event['routine_id']==1
    assert event['evidence']['token_indices'].tolist()==[15]
    assert torch.equal(event['evidence']['features'],varied[-1:])


def test_order_shape_metadata_and_caption_failures_leave_state_unchanged():
    memory=RoutineAwareMemory(config())
    memory.update(feature(),0,0.)
    memory.update(feature(),1,1.)
    before=plain_copy(memory.state_dict(),clone_tensors=True)
    for value,fid,stamp,meta in [(feature(),1,2.,None),(feature(),2,1.,None),
                               (feature().float(),2,2.,None),(feature()[:8],2,2.,None),
                               (feature(),2,2.,{'question':'forbidden'})]:
        with pytest.raises(ValueError): memory.update(value,fid,stamp,meta)
        same(before,memory.state_dict())
    def broken_caption(evidence):
        evidence['features'].zero_()
        raise RuntimeError('caption unavailable')
    with pytest.raises(RuntimeError,match='caption unavailable'):
        memory.update(feature(),2,2.,caption_callback=broken_caption)
    same(before,memory.state_dict())
    memory.update(feature(),2,2.,caption_callback=lambda evidence:'Configured external caption')
    assert memory.state_dict()['event_memory']['events'][0]['description']=='Configured external caption'


def test_retained_tensors_own_compact_storage_and_do_not_alias_archive_views():
    archive=torch.stack([feature(i%8) for i in range(10)])
    memory=RoutineAwareMemory(config(evidence_tokens=5))
    memory.update(archive[3],0,0.)
    source_pointer=archive.untyped_storage().data_ptr()
    def check(value):
        if isinstance(value,torch.Tensor):
            assert value.untyped_storage().data_ptr()!=source_pointer
            assert value.untyped_storage().nbytes()==value.numel()*value.element_size()
        elif isinstance(value,dict):
            for item in value.values():check(item)
        elif isinstance(value,list):
            for item in value:check(item)
    check(memory.state_dict())
    before=next(memory.retained_evidence())['features'].clone()
    archive.zero_()
    assert torch.equal(next(memory.retained_evidence())['features'],before)


def test_pressure_eviction_is_deterministic_bounded_and_preserves_current_observation():
    roomy=RoutineAwareMemory(config(stability_frames=1))
    roomy.update(feature(),0,0.)
    cap=roomy.stats()['actual_memory_bytes']*2+10_000
    memory=RoutineAwareMemory(config(stability_frames=1,budget_bytes=cap))
    replay=RoutineAwareMemory(config(stability_frames=1,budget_bytes=cap))
    evictions=[]
    for index in range(40):
        decision=memory.update(feature(index%8),index,float(index))
        other=replay.update(feature(index%8),index,float(index))
        assert decision['audit']==other['audit']
        assert memory.stats()['actual_memory_bytes']<=cap
        assert memory.stats()['serialized_bytes']<=cap
        assert memory.state_dict()['short_term']['frames'][-1]['frame_id']==index
        evictions += [row for row in decision['audit'] if row['operation'].startswith('evict')]
    assert evictions
    same(memory.state_dict(),replay.state_dict())
    RoutineAwareMemory.from_state_dict(memory.state_dict())


def test_failure_when_minimum_input_does_not_fit_is_transactional():
    base=RoutineAwareMemory(config())
    cap=base.stats()['actual_memory_bytes']+100
    memory=RoutineAwareMemory(config(budget_bytes=cap))
    before=plain_copy(memory.state_dict(),clone_tensors=True)
    with pytest.raises(ValueError,match='Budget'):
        memory.update(feature(),0,0.)
    same(before,memory.state_dict())


def test_resume_exact_and_future_suffix_cannot_change_previous_state():
    original=RoutineAwareMemory(config())
    for start,which in [(0,0),(10,1),(40,0)]:episode(original,feature(which),start)
    snapshot=plain_copy(original.state_dict(),clone_tensors=True)
    resumed=RoutineAwareMemory.from_state_dict(snapshot)
    for start,which in [(50,1),(80,0)]:
        left=episode(original,feature(which),start)
        right=episode(resumed,feature(which),start)
        for a,b in zip(left,right):
            a.pop('processing_seconds');b.pop('processing_seconds');same(a,b)
    same(original.state_dict(),resumed.state_dict())
    assert snapshot['control']['last_timestamp']==47
    assert snapshot['control']['promotions']==0
    assert all(item['timestamp']<=47 for item in RoutineAwareMemory.from_state_dict(snapshot).retained_evidence())


def test_phase_sequence_keeps_order_instead_of_one_global_average():
    aggregator=FeatureAggregator();cfg=config()
    frames=[aggregator.frame(feature(index),index,float(index),{},cfg) for index in range(4)]
    temporal=aggregator.temporal(frames,cfg)
    reversed_temporal=aggregator.temporal(list(reversed(frames)),cfg)
    assert temporal.shape==(4,8)
    assert torch.equal(temporal.mean(0),reversed_temporal.mean(0))
    assert not torch.equal(temporal,reversed_temporal)


def test_record_cost_inspection_and_source_identity_are_accounted_without_cache():
    memory=RoutineAwareMemory(config())
    before=memory.stats()['actual_memory_bytes']
    memory.set_source_context({'archive_id':'abc','header_sha256':'1'*64})
    assert memory.stats()['actual_memory_bytes']>before
    episode(memory,feature(),0)
    state=memory.state_dict()
    report=memory.record_costs()
    assert {row['kind'] for row in report['records']}=={'short_term','event','candidate'}
    assert all(row['estimated_memory_bytes']>0 for row in report['records'])
    assert memory.state_dict() is state
    with pytest.raises(ValueError):memory.set_source_context({'archive_id':'different'})


def test_switching_candidate_resets_pending_occurrence_provenance_even_for_similar_frames():
    memory=RoutineAwareMemory(config())
    draft=plain_copy(memory.state_dict())
    aggregator=FeatureAggregator()
    first=aggregator.frame(feature(),0,100.,{},memory.config)
    memory._pending(draft,'C',None,{'id':1,'last_seen':0.},None,first,None,memory.config)
    draft['control']['pending']['event_id']=99
    second=aggregator.frame(feature(),1,101.,{},memory.config,first)
    pending,stable,reset_reason=memory._pending(draft,'C',None,{'id':2,'last_seen':50.},None,second,first,memory.config)
    assert reset_reason=='candidate_changed'
    assert pending['candidate_id']==2 and pending['count']==1 and not stable
    assert pending['start_frame_id']==1 and pending['start_timestamp']==101.
    assert pending['prior_last_seen']==50. and pending['event_id'] is None


def test_one_frame_local_candidate_noise_is_not_an_event_and_common_evidence_is_reused():
    memory=RoutineAwareMemory(config(spatial_bins=16))
    episode(memory,feature(),0)
    varied=feature();varied[-1]=feature(2)[-1]
    decision=memory.update(varied,8,8.)
    assert decision['case']=='C' and not decision['stable']
    assert memory.stats()['events']==1
    decisions=episode(memory,feature(),9,3)
    assert decisions[-1]['stable'] and memory.stats()['events']==1
    assert any(row['operation']=='reuse_common_candidate_evidence' for row in decisions[-1]['audit'])
    assert memory.state_dict()['event_memory']['candidates'][0]['occurrence_count']==1
    decisions=episode(memory,varied,12,3)
    assert not decisions[0]['stable'] and not decisions[1]['stable'] and decisions[2]['stable']
    assert memory.stats()['events']==2
    assert sum(event['distinctive'] for event in memory.state_dict()['event_memory']['events'])==1
