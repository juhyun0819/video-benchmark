"""The full stream defines fixed byte caps, independently of observed cutoffs."""
from decimal import Decimal, ROUND_FLOOR
from fractions import Fraction
import json
import math

import pytest
import torch

from baselines.feature_memory.byte_memory import ByteBudgetMemory, DiskTokenStore
from baselines.feature_memory.planning import make_plan


DURATIONS = ('39019.76','29440.24','34641.16','35013.88','31581.72','19372.2','32655.96')
FULL_FRAMES = 221728
BYTES_PER_FRAME = 676 * 3584 * 2
FULL_BYTES = 1074398101504


@pytest.fixture
def full_inventory():
    sources=[]
    offset=Fraction(0)
    for i,text in enumerate(DURATIONS,1):
        duration=Fraction(text)
        sources.append({'path':f'/test-only/day{i}.mp4','verified':True,
                        'width':720,'height':720,'duration_fraction':str(duration),
                        'global_start_fraction':str(offset)})
        offset+=duration
    return {'sources':sources,'total_duration_fraction':str(offset),
            'total_duration_seconds':float(offset),'total_1fps_frames':FULL_FRAMES,
            'all_verified':True,'counts_provisional':False}


@pytest.fixture
def native_processor_config(tmp_path):
    # Real Transformers image preprocessing, entirely local metadata and no
    # model weights. These are the measured 720-square checkpoint settings.
    from transformers import Qwen2VLImageProcessor
    directory=tmp_path/'processor_config'
    directory.mkdir()
    Qwen2VLImageProcessor(patch_size=14,merge_size=2,temporal_patch_size=2,
                         min_pixels=3136,max_pixels=12845056).save_pretrained(directory)
    (directory/'config.json').write_text(json.dumps({
        'model_type':'qwen2_5_vl','vision_config':{'out_hidden_size':3584}}))
    return directory


def expected_prefix_frames(seconds):
    stop=Fraction(str(seconds)) if seconds is not None else sum(map(Fraction,DURATIONS))
    elapsed=Fraction(0)
    count=0
    for duration in map(Fraction,DURATIONS):
        if elapsed>=stop:
            break
        count+=math.ceil(min(duration,stop-elapsed))
        elapsed+=duration
    return count


def test_full_stream_caps_unchanged_by_smoke_or_snapshot_horizon(full_inventory,native_processor_config):
    plans=[make_plan(full_inventory,native_processor_config,max_seconds=seconds)
           for seconds in [300,18000,36000,72000,144000,216000,None]]
    expected_caps=[53719905075,107439810150,268599525376,537199050752]
    reference_settings=plans[-1]['settings']
    assert plans[-1]['planned_frames']==FULL_FRAMES
    for seconds,plan in zip([300,18000,36000,72000,144000,216000,None],plans):
        assert plan['full_feature_bytes']==FULL_BYTES
        assert plan['full_feature_bytes']//BYTES_PER_FRAME==FULL_FRAMES
        assert plan['reference_frames']==FULL_FRAMES
        assert [s['hard_budget_bytes'] for s in plan['settings']]==expected_caps
        assert plan['settings']==reference_settings
        assert plan['planned_frames']==expected_prefix_frames(seconds)
        assert plan['planned_run_feature_bytes']==plan['planned_frames']*BYTES_PER_FRAME
        assert plan['tokens_per_frame']==676
        assert plan['post_merger_grid_hw']==[26,26]
    assert plans[0]['planned_frames']==300
    assert plans[0]['planned_run_feature_bytes']==1453670400
    assert plans[0]['planned_run_feature_bytes']<plans[0]['full_feature_bytes']


def test_caps_use_decimal_floor_and_metadata_reduces_usable_tokens(full_inventory,native_processor_config,tmp_path):
    ratios=[Decimal('0.05'),Decimal('0.10'),Decimal('0.25'),Decimal('0.50'),Decimal('0.333333333333333333')]
    plan=make_plan(full_inventory,native_processor_config,max_seconds=300,ratios=ratios)
    for ratio,setting in zip(ratios,plan['settings']):
        exact=int((Decimal(FULL_BYTES)*ratio).to_integral_value(rounding=ROUND_FLOOR))
        assert setting['hard_budget_bytes']==exact
        assert setting['target_visual_tokens']==exact//7168
        assert setting['hard_budget_GB']==exact/1e9
        assert setting['approximate_frame_equivalent']==pytest.approx(setting['target_visual_tokens']/676)
    store=DiskTokenStore(tmp_path/'huge_cap_no_preallocation',plan['settings'][0]['hard_budget_bytes'],hidden_dim=3584)
    try:
        assert store.capacity==(plan['settings'][0]['hard_budget_bytes']-4096)//7200
        assert store.capacity<plan['settings'][0]['target_visual_tokens']
        assert store.stats()['persistent_disk_bytes']<4096
        assert store.stats()['actual_memory_bytes']<4096
    finally:
        store.close()


def test_effective_retention_changes_while_absolute_cap_stays_fixed(tmp_path):
    cap=4096+17*(3*2+32)
    memory=ByteBudgetMemory('fifo',cap,.05,hidden_dim=3,tokens_per_frame=7,directory=tmp_path/'live')
    history=[]
    try:
        initial=memory.stats()
        assert initial['effective_retention_ratio']==0
        assert initial['payload_retention_ratio']==0
        for i in range(8):
            memory.update(torch.ones(7,3,dtype=torch.bfloat16),i,float(i))
            stats=memory.stats()
            assert stats['hard_budget_bytes']==cap
            assert stats['retention_ratio']==.05
            assert stats['actual_memory_bytes']<=cap
            assert stats['effective_retention_ratio']==stats['actual_memory_bytes']/stats['cumulative_feature_bytes']
            assert stats['payload_retention_ratio']==stats['feature_payload_bytes']/stats['cumulative_feature_bytes']
            assert stats['budget_utilization']==stats['actual_memory_bytes']/cap
            history.append(stats)
        assert history[0]['payload_retention_ratio']==1
        assert history[-1]['payload_retention_ratio']==pytest.approx(17/56)
        assert history[-1]['effective_retention_ratio']<history[0]['effective_retention_ratio']
        assert history[-1]['actual_memory_bytes']>history[0]['actual_memory_bytes']
    finally:
        memory.close()
