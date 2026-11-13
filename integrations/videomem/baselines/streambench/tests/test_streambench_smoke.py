"""Synthetic features exercise real native memory policies and common runner."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from baselines.streambench.runner import observe_stream, evaluate_checkpoint
from baselines.streambench.checkpoint import read_checkpoint
from test_streambench_native import adapter, frames, timestamps, assert_tree_equal, SF_CONFIG

OPTIONS = {'fifo': {'capacity_grids':2}, 'reservoir': {'capacity_grids':2,'seed':19},
           'fluxmem_streaming': {'short_frames':2,'medium_frames':0,'long_frames':2},
           'fluxmem_qwen3vl': {'short_frames':2,'medium_frames':0,'long_frames':2},
           'streamforest': {'memory_config':SF_CONFIG,'hidden_size':8,'num_attention_heads':2}}


def stream(method, days=(1,2)):
    start=0
    for day,count in [(1,13),(2,19)]:
        if day in days:
            for offset in range(0,count,5):
                n=min(5,count-offset)
                yield frames(start+offset,n,tokens=16 if method=='streamforest' else 4),timestamps(start+offset,n,day=day)
        start+=count


@pytest.mark.parametrize('name', list(OPTIONS))
def test_common_runner_native_day_resume_and_immutable_qa(name,tmp_path):
    method=adapter(name,budget=65536,**OPTIONS[name])
    report=observe_stream(method,stream(name),tmp_path/'continuous',stream_id='subject',manifest_digest='synthetic',method_config=method.config)
    assert report['maximum_persistent_memory_bytes']<=65536
    assert report['days_observed']==[1,2]
    assert report['frames_seen_total']==32
    assert method.compression_count>0 or method.eviction_count>0
    assert len((tmp_path/'continuous/observed_frames.jsonl').read_text().splitlines())==7
    state=method.snapshot()
    answers=evaluate_checkpoint(method,tmp_path/'continuous/checkpoint_D1_D2',[
        {'question_id':'a','source_day':1,'question':'what?'},
        {'question_id':'b','source_day':1,'question':'what?'}])
    assert answers[0]['answer']==answers[1]['answer']
    assert answers[0]['query_read_bytes']>0
    assert_tree_equal(method.state_dict(),state['method'])
    script='''import runpy,sys
sys.path.insert(0,'baselines/streambench/tests')
from pathlib import Path
from baselines.streambench.runner import observe_stream
n=runpy.run_path('baselines/streambench/tests/test_streambench_native.py')
s=runpy.run_path('baselines/streambench/tests/test_streambench_smoke.py')
name=sys.argv[1]; root=Path(sys.argv[2])
m=n['adapter'](name,budget=65536,**s['OPTIONS'][name])
observe_stream(m,s['stream'](name,(2,)),root/'resumed',stream_id='subject',manifest_digest='synthetic',method_config=m.config,resume_checkpoint=root/'continuous/checkpoint_D1')
'''
    subprocess.run([sys.executable,'-c',script,name,str(tmp_path)],check=True,cwd=Path(__file__).parents[3])
    resumed=read_checkpoint(tmp_path/'resumed/checkpoint_D1_D2')
    continuous=read_checkpoint(tmp_path/'continuous/checkpoint_D1_D2')
    assert_tree_equal(resumed['snapshot']['method'],continuous['snapshot']['method'])
    assert resumed['metadata']==continuous['metadata']
