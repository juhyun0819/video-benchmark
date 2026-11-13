import json
from pathlib import Path
import sys
import time

import pytest
import torch

from rekv_qwen25.archive import KVArchiveWriter
from rekv_qwen25 import coordinator as module


@pytest.mark.parametrize('total,workers', [(0,4),(1,4),(99,4),(100,4),(101,4),(221728,4),(221728,8)])
def test_balanced_partition_covers_exactly_without_empty_workers(total, workers):
    ranges = module.balanced_partition(total, workers)
    assert sum(end-start for start,end in ranges) == total
    assert all(start < end for start,end in ranges)
    assert len(ranges) <= workers
    if ranges:
        assert ranges[0][0] == 0 and ranges[-1][1] == total
        assert all(a[1] == b[0] for a,b in zip(ranges,ranges[1:]))
        assert all(start%100 == 0 for start,_ in ranges)
        counts = [(end-start+99)//100 for start,end in ranges]
        assert max(counts)-min(counts) <= 1


def test_scaling_plan_never_assigns_unused_gpus(tmp_path):
    stages = module.scaling_plan(tmp_path)
    assert [s['gpu_count'] for s in stages] == [1,2,4]
    assert [[w['gpu'] for w in stage['workers']] for stage in stages] == [[0],[0,1],[0,1,2,3]]
    for stage in stages:
        assert sorted((w['start_frame'],w['end_frame']) for w in stage['workers']) == [
            (100*(i+1),100*(i+2)) for i in range(stage['gpu_count'])]
        assert all(w['start_frame'] < 620 for w in stage['workers'])
    assert [(w['gpu'],w['start_frame']) for w in stages[1]['workers']]==[(0,200),(1,100)]


def configuration(tmp_path):
    attention = module.sha256_file(module.ATTENTION_PATH)
    backbone = dict(repo_id=module.CHECKPOINT,revision=module.REVISION)
    policy = {'test_policy':'exact'}
    initial = tmp_path/'initial_state'
    writer = KVArchiveWriter(initial,dict(backbone=backbone,policy=policy,n_init=17,n_local=15000,
        input_protocol=module.PREFIX_PROTOCOL,attention_sha256=attention))
    writer.commit(0,{'synthetic':torch.tensor(1)},dict(frame_ids=[0],timestamps=[0.0]))
    report_path = tmp_path/'halo.json'
    report = dict(status='passed',halo_frames=620,n_init=17,n_local=15000,chunk_frames=1,
        absolute_positions=True,fixed_original_per_layer_sinks=True,policy=policy,checkpoint=backbone,
        adaptation_sha256=attention,boundary_frame=24,verification_frames=1,
        checks=[dict(frame_id=644,all_kv_exact=True,bounded_state_exact=True,logits={'exact':True})])
    module.write_json(report_path,report)
    config = dict(backbone=backbone,policy=policy,n_init=17,n_local=15000,chunk_frames=1,
        frames_per_block=100,total_frames=221728,prefix_protocol=module.PREFIX_PROTOCOL,
        feature_archive_id='fixed-feature-id',initial_state_config_id=writer.config_id,
        halo_validation=dict(status='passed',report_path=str(report_path),
                             report_sha256=module.sha256_file(report_path),halo_frames=620,
                             attention_sha256=attention))
    config_path=tmp_path/'run_config.json'
    module.write_json(config_path,config)
    return config_path,initial,report_path


def test_config_validation_binds_proof_code_initial_and_canonical_id(tmp_path):
    path,initial,_ = configuration(tmp_path)
    config, validation = module.validate_config(path,initial)
    assert validation['config_id'] == KVArchiveWriter(tmp_path/'new_archive',config).config_id
    assert validation['config_file_sha256'] == module.sha256_file(path)


@pytest.mark.parametrize('mutation',['proof_bytes','attention_hash','false_exact','prefix_identity','protocol'])
def test_bad_proof_or_changed_configuration_rejected(tmp_path,mutation):
    path,initial,report_path=configuration(tmp_path)
    config=json.loads(path.read_text())
    report=json.loads(report_path.read_text())
    if mutation=='proof_bytes': report['new_field']='modified'
    elif mutation=='attention_hash': config['halo_validation']['attention_sha256']='incorrect'
    elif mutation=='false_exact': report['checks'][0]['logits']['exact']=False
    elif mutation=='prefix_identity':config['initial_state_config_id']='incorrect'
    elif mutation=='protocol':config['chunk_frames']=2
    module.write_json(report_path,report)
    if mutation=='false_exact':config['halo_validation']['report_sha256']=module.sha256_file(report_path)
    module.write_json(path,config)
    with pytest.raises(ValueError):module.validate_config(path,initial)


FAKE_WORKER = r'''
import argparse,json,os,time
from pathlib import Path
p=argparse.ArgumentParser()
for field in ('archive-root','config','worker-id','start-frame','end-frame','barrier-dir'):
    p.add_argument('--'+field,required=True)
a,_=p.parse_known_args()
root=Path(a.archive_root);barrier=Path(a.barrier_dir);config=json.loads(Path(a.config).read_text())
worker=a.worker_id;start=int(a.start_frame);end=int(a.end_frame)
def write(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value));temporary.replace(path)
if worker=='gpu1':time.sleep(0.05)
write(barrier/(worker+'.ready.json'),dict(worker_id=worker,start_frame=start,end_frame=end,
    config_id=config['test_config_id'],pid=os.getpid()))
while not (barrier/'start.json').exists():time.sleep(0.01)
message=json.loads((barrier/'start.json').read_text())
assert all((barrier/(name+'.ready.json')).exists() for name in message['workers'])
(root/(worker+'.started')).write_text('only after every worker was ready')
if config.get('fail_worker')==worker:raise SystemExit(7)
if config.get('fail_worker'):time.sleep(30)
time.sleep(0.02)
write(root/'logs'/('worker_'+worker+'.report.json'),dict(status='complete',worker_id=worker,
    start_frame=start,end_frame=end,config_id=config['test_config_id'],resumed_frames=0,
    committed_frames=end-start,warmup_frames=start,measured_seconds=0.1,read_seconds=0.01,
    infer_seconds=0.05,atomic_nas_seconds=0.03,serialized_bytes=2000,peak_vram_bytes=100,
    commit_intervals=[dict(chunk_id=start//100,started_time=1,finished_time=1.03,serialized_bytes=2000)]))
'''


def fake_commands(tmp_path,monkeypatch,fail_worker=None):
    script=tmp_path/'cpu_worker.py';script.write_text(FAKE_WORKER)
    config_path=tmp_path/'test_config.json'
    module.write_json(config_path,dict(test_config_id='verified-test-id',fail_worker=fail_worker))
    original=module.worker_command
    def command(*args,**kwargs):
        real=original(*args,**kwargs)
        return [sys.executable,str(script),*real[3:]]
    monkeypatch.setattr(module,'worker_command',command)
    return config_path


def test_cpu_workers_barrier_and_aggregate_report(tmp_path,monkeypatch):
    config=fake_commands(tmp_path,monkeypatch)
    stage=module.scaling_plan(tmp_path)[1]
    result=module.run_stage(stage,config,'verified-test-id',tmp_path/'initial',
                           poll_seconds=0.01,heartbeat_seconds=0.02)
    assert result['committed_frames']==200 and result['serialized_bytes']==4000
    assert result['steady_fps']==2000 and result['assigned_gpus']==[0,1]
    assert result['full_run_started'] is False
    root=Path(stage['archive_root'])
    assert (root/'gpu0.started').exists() and (root/'gpu1.started').exists()
    assert json.loads((root/'status.json').read_text())['status']=='complete'
    with pytest.raises(ValueError,match='reuse'):
        module.run_stage(stage,config,'verified-test-id',tmp_path/'initial')


def test_one_failed_cpu_worker_stops_its_stage_peer(tmp_path,monkeypatch):
    config=fake_commands(tmp_path,monkeypatch,fail_worker='gpu0')
    stage=module.scaling_plan(tmp_path)[1]
    began=time.monotonic()
    with pytest.raises(RuntimeError,match='exited with 7'):
        module.run_stage(stage,config,'verified-test-id',tmp_path/'initial',
                         poll_seconds=0.01,heartbeat_seconds=0.02)
    assert time.monotonic()-began < 5
    status=json.loads((Path(stage['archive_root'])/'status.json').read_text())
    assert status['status']=='failed'
    assert all(worker['returncode'] is not None for worker in status['workers'])


def test_report_cannot_hide_partial_outputs_or_skipped_warmup():
    worker=dict(worker_id='gpu0',start_frame=100,end_frame=200)
    report=dict(status='complete',config_id='id',**worker,committed_frames=99,resumed_frames=0)
    with pytest.raises(ValueError,match='coverage'):
        module.validate_worker_report(report,worker,'id',benchmark=True)
    report.update(committed_frames=100,warmup_frames=0)
    with pytest.raises(ValueError,match='full earlier prefix'):
        module.validate_worker_report(report,worker,'id',benchmark=True)


def test_atomic_commit_interval_union_handles_overlap_and_disjoint_periods():
    intervals=[dict(started_time=start,finished_time=end) for start,end in [(1,3),(2,4),(6,8),(6.5,7)]]
    assert module.interval_union_seconds(intervals)==5
    assert module.interval_union_seconds([])==0
    with pytest.raises(ValueError):module.interval_union_seconds([dict(started_time=3,finished_time=2)])


def test_shared_chunk_tensor_identity_crosses_gpu_assignments_and_rejects_changes(tmp_path):
    stages=module.scaling_plan(tmp_path)[:2]
    reports=[]
    for stage in stages:
        writer=KVArchiveWriter(stage['archive_root'],{'checkpoint':'synthetic'})
        for worker in stage['workers']:
            start=worker['start_frame']
            writer.commit(start//100,{'layer_00.key':torch.tensor([start],dtype=torch.bfloat16)},
                          dict(frame_ids=list(range(start,start+100)),timestamps=list(range(start,start+100))))
        reports.append(dict(stage,config_id=writer.config_id))
    comparison=module.compare_shared_chunks(*reports)
    assert len(comparison)==1 and comparison[0]['reference_gpu']==0 and comparison[0]['candidate_gpu']==1
    # Create a separately valid archive with changed values; this is a fidelity
    # mismatch, rather than an ordinary corrupt receipt or index.
    changed=dict(reports[1],archive_root=str(tmp_path/'different'))
    writer=KVArchiveWriter(changed['archive_root'],{'checkpoint':'synthetic'})
    for worker in changed['workers']:
        start=worker['start_frame']
        writer.commit(start//100,{'layer_00.key':torch.tensor([-start],dtype=torch.bfloat16)},
                      dict(frame_ids=list(range(start,start+100)),timestamps=list(range(start,start+100))))
    with pytest.raises(ValueError,match='Cross-GPU tensor identity mismatch'):
        module.compare_shared_chunks(reports[0],changed)
