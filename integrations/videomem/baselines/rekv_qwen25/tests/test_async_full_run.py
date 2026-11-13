import copy
import json

import pytest

from rekv_qwen25 import async_full_run, full_run


def fixture_report(tmp_path):
    validation = dict(config_path=str(tmp_path/'config.json'), config_id='example', config_file_sha256='hash')
    row = dict(status='passed', all_tensor_sha256_exact=True, all_tensor_geometry_exact=True,
               metadata_equivalent_except_config=True, metadata_exact=True)
    worker=dict(writer_drained=True,writer_exitcode=0,writer_cuda_initialized=False,
                compute_write_overlap_seconds=10,overlapped_observation_calls=20)
    report = dict(schema='rekv_async_scaling_v1', status='complete',
        writer_mode='cpu_process_overlap_v1', max_pending_blocks=1, validation=validation,
        stages=[dict(status='complete',gpu_count=n,config_id='example',workers=[worker]*n) for n in (2,4)],
        cross_pipeline_fidelity=dict(status='passed',comparisons=[row]*12),
        cross_gpu_fidelity=dict(status='passed',comparisons=[row]*4))
    path=tmp_path/'benchmarks/async_scaling_report.json';path.parent.mkdir()
    path.write_text(json.dumps(report))
    config=dict(writer_mode='cpu_process_overlap_v1',max_pending_blocks=1)
    return config,validation,report,path


def test_requires_actual_complete_pipeline_report(tmp_path):
    config,validation,_,path=fixture_report(tmp_path)
    proof=async_full_run.validate_pipeline_report(config,validation)
    assert proof['report_path']==str(path) and len(proof['report_sha256'])==64


@pytest.mark.parametrize('field', ['status','config','stage','pipeline_kv','gpu_metadata','missing_comparison','no_overlap','undrained'])
def test_rejects_incomplete_or_changed_pipeline_evidence(tmp_path,field):
    config,validation,report,path=fixture_report(tmp_path)
    report=copy.deepcopy(report)
    if field=='status':report['status']='running'
    if field=='config':report['validation']['config_id']='other'
    if field=='stage':report['stages'].pop()
    if field=='pipeline_kv':report['cross_pipeline_fidelity']['comparisons'][0]['all_tensor_sha256_exact']=False
    if field=='gpu_metadata':report['cross_gpu_fidelity']['comparisons'][0]['metadata_exact']=False
    if field=='missing_comparison':report['cross_pipeline_fidelity']['comparisons'].pop()
    if field=='no_overlap':report['stages'][0]['workers'][0]['compute_write_overlap_seconds']=0
    if field=='undrained':report['stages'][0]['workers'][0]['writer_drained']=False
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):async_full_run.validate_pipeline_report(config,validation)


def test_supervision_scope_restores_worker_selector_after_failure():
    original=full_run.worker_command
    worker=dict(worker_id='gpu0',gpu=0,start_frame=0,end_frame=100)
    with pytest.raises(RuntimeError):
        with async_full_run.asynchronous_supervision():
            command=full_run.worker_command(worker,'/archive','/config','/initial')
            assert command[command.index('-m')+1]=='rekv_qwen25.async_runner'
            raise RuntimeError('interrupted')
    assert full_run.worker_command is original
