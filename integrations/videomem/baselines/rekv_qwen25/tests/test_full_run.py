import json
from pathlib import Path
import sys
import time

import pytest
import torch

from rekv_qwen25 import archive
from rekv_qwen25 import full_run as module
from rekv_qwen25.coordinator import PREFIX_PROTOCOL, config_id, sha256_file, write_json


def test_production_plan_balances_only_explicit_selected_gpus():
    workers=module.production_plan(221728,[3,1,0,2])
    assert [worker['gpu'] for worker in workers]==[3,1,0,2]
    assert workers[0]['start_frame']==0 and workers[-1]['end_frame']==221728
    assert all(a['end_frame']==b['start_frame'] for a,b in zip(workers,workers[1:]))
    assert len(module.production_plan(1,[0,1,2,3]))==1
    for gpus in ([],[0,0],[8],[-1],[True]):
        with pytest.raises(ValueError):module.production_plan(100,gpus)


def sparse_metadata_fixture(tmp_path):
    # Metadata-only audit intentionally must not read the payload bytes. Sparse
    # fixtures represent correctly sized immutable files without writing 100 MB.
    config=dict(total_frames=3,frames_per_block=2,initial_state_config_id='fixed-prefix',
                feature_archive_id='fixed-features',prefix_protocol=PREFIX_PROTOCOL)
    writer=archive.KVArchiveWriter(tmp_path,config)
    manifest=json.loads((tmp_path/'manifest.json').read_text())
    for chunk,start,end in ((0,0,2),(1,2,3)):
        count=end-start;tokens=count*678
        metadata=archive._normalize_metadata(chunk,dict(frame_ids=list(range(start,end)),
            timestamps=list(range(start,end)),config_id=writer.config_id,next_frame_id=end,
            total_seen=17+end*678,initial_state_config_id='fixed-prefix',feature_archive_id='fixed-features',
            key_representation='pre_rope',prefix_protocol=PREFIX_PROTOCOL,token_count=tokens,
            tokens_per_frame=678,query_independent=True,resume_strategy='last_raw_kv_window_plus_canonical_init',
            frames=[dict(frame_id=frame,timestamp_seconds=frame,token_start=offset*678,token_end=(offset+1)*678)
                    for offset,frame in enumerate(range(start,end))]))
        specs={f'layer_{layer:02d}.{kind}':dict(shape=[1,4,tokens,128],dtype='torch.bfloat16',
                bytes=4*tokens*128*2,sha256='0'*64) for layer in range(28) for kind in ('key','value')}
        specs['position_ids']=dict(shape=[3,1,tokens],dtype='torch.int64',bytes=3*tokens*8,sha256='0'*64)
        specs['input_ids']=dict(shape=[1,tokens],dtype='torch.int64',bytes=tokens*8,sha256='0'*64)
        size=sum(spec['bytes'] for spec in specs.values())+4096
        payload,index_path,receipt_path=archive._paths(tmp_path,chunk)
        with payload.open('wb') as stream:stream.truncate(size)
        index=dict(schema=archive.SCHEMA,config_id=writer.config_id,chunk_id=chunk,metadata=metadata,tensors=specs)
        write_json(index_path,index)
        descriptor=archive._descriptor(metadata)
        receipt=dict(schema=archive.SCHEMA,config_id=writer.config_id,**descriptor,payload_sha256='0'*64,
                     index_sha256=sha256_file(index_path),serialized_bytes=size)
        write_json(receipt_path,receipt)
        manifest['chunks'][str(chunk)]=dict(descriptor,state='ready',payload_sha256='0'*64,
                                          index_sha256=receipt['index_sha256'])
    manifest['continuity']=archive._continuity(manifest)
    write_json(tmp_path/'manifest.json',manifest)
    return config


def test_final_audit_exact_coverage_and_sizes_without_reading_payloads(tmp_path,monkeypatch):
    config=sparse_metadata_fixture(tmp_path)
    original=archive.file_sha256
    def metadata_only(path):
        assert Path(path).suffix!='.safetensors','Final audit must not rehash all payload bytes'
        return original(path)
    monkeypatch.setattr(archive,'file_sha256',metadata_only)
    result=module.final_audit(tmp_path,config)
    assert result['frame_count']==3 and result['chunk_count']==2
    assert result['payload_hashes_recomputed'] is False
    assert result['tensor_geometry_verified'] is True


@pytest.mark.parametrize('mutation',['missing_receipt','truncate','wrong_tensor_shape','wrong_next_frame'])
def test_final_audit_refuses_missing_corrupt_or_incompatible_outputs(tmp_path,mutation):
    config=sparse_metadata_fixture(tmp_path)
    payload,index_path,receipt_path=archive._paths(tmp_path,1)
    if mutation=='missing_receipt':receipt_path.unlink()
    elif mutation=='truncate':
        with payload.open('r+b') as stream:stream.truncate(payload.stat().st_size-1)
    else:
        index=json.loads(index_path.read_text())
        if mutation=='wrong_tensor_shape':index['tensors']['layer_00.key']['shape'][-1]=127
        else:index['metadata']['next_frame_id']=2
        write_json(index_path,index)
        receipt=json.loads(receipt_path.read_text());receipt['index_sha256']=sha256_file(index_path)
        write_json(receipt_path,receipt)
        manifest=json.loads((tmp_path/'manifest.json').read_text())
        manifest['chunks']['1']['index_sha256']=receipt['index_sha256']
        write_json(tmp_path/'manifest.json',manifest)
    with pytest.raises(ValueError):module.final_audit(tmp_path,config)


CPU_WORKER=r'''
import argparse,hashlib,json,time
from pathlib import Path
p=argparse.ArgumentParser()
for field in ('archive-root','config','worker-id','start-frame','end-frame'):
    p.add_argument('--'+field,required=True)
a,other=p.parse_known_args()
assert '--wait-for-features' in other and '--barrier-dir' not in other
root=Path(a.archive_root);config=json.loads(Path(a.config).read_text())
if config.get('test_fail_worker')==a.worker_id:raise SystemExit(9)
if config.get('test_fail_worker'):time.sleep(30)
config.setdefault('baseline_name','ReKV (Qwen2.5-VL-7B adaptation)')
config.setdefault('key_representation','pre_rope')
config.setdefault('archive_schema','rekv_qwen25_pre_rope_archive_v1')
identifier=hashlib.sha256((json.dumps(config,sort_keys=True,separators=(',',':'),ensure_ascii=False)+'\n').encode()).hexdigest()
report=dict(status='complete',worker_id=a.worker_id,start_frame=int(a.start_frame),end_frame=int(a.end_frame),
    config_id=identifier,committed_frames=0,resumed_frames=int(a.end_frame)-int(a.start_frame),
    measured_seconds=0,read_seconds=0,infer_seconds=0,atomic_nas_seconds=0,serialized_bytes=0,peak_vram_bytes=0)
path=root/'logs'/('worker_'+a.worker_id+'.report.json')
temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(report));temporary.replace(path)
'''


def mock_supervisor_inputs(tmp_path,monkeypatch,fail=False):
    config=dict(total_frames=200,frames_per_block=100,test_fail_worker='full_gpu0' if fail else None)
    config_path=tmp_path/'configuration.json';write_json(config_path,config)
    root=tmp_path/'output';writer=archive.KVArchiveWriter(root,config)
    for chunk in range(2):
        writer.commit(chunk,{'tiny':torch.tensor(chunk)},dict(frame_ids=list(range(chunk*100,(chunk+1)*100)),
                                                           timestamps=list(range(chunk*100,(chunk+1)*100))))
    validation=dict(config_id=writer.config_id,config_file_sha256=sha256_file(config_path))
    monkeypatch.setattr(module,'validate_config',lambda *_:(config,validation))
    monkeypatch.setattr(module,'final_audit',lambda *_:dict(status='passed',frame_count=200,chunk_count=2))
    script=tmp_path/'cpu_worker.py';script.write_text(CPU_WORKER)
    original=module.worker_command
    def command(*args,**kwargs):
        real=original(*args,**kwargs)
        return [sys.executable,str(script),*real[3:]]
    monkeypatch.setattr(module,'worker_command',command)
    return root,config_path


def test_supervisor_completes_only_after_workers_and_final_audit(tmp_path,monkeypatch):
    root,config=mock_supervisor_inputs(tmp_path,monkeypatch)
    result=module.run_full(root,config,[0,1],poll_seconds=0.01,heartbeat_seconds=0.02)
    assert result['status']=='complete' and result['all_workers_exited_successfully']
    assert json.loads((root/'manifest.json').read_text())['status']=='complete'
    assert json.loads((root/'logs'/'full-status.json').read_text())['status']=='complete'


def test_supervisor_failure_stops_peers_and_leaves_archive_resumable(tmp_path,monkeypatch):
    root,config=mock_supervisor_inputs(tmp_path,monkeypatch,fail=True)
    began=time.monotonic()
    with pytest.raises(RuntimeError,match='exited with 9'):
        module.run_full(root,config,[0,1],poll_seconds=0.01,heartbeat_seconds=0.02)
    assert time.monotonic()-began < 5
    result=json.loads((root/'logs'/'full-status.json').read_text())
    assert result['status']=='failed' and result['resumable']
    assert all(worker['returncode'] is not None for worker in result['workers'])
    assert json.loads((root/'manifest.json').read_text())['status']=='open'
