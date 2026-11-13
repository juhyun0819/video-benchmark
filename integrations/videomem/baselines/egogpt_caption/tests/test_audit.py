import pytest

from baselines.egogpt_caption import audit, media
from baselines.egogpt_caption.archive import atomic_json, object_hash, save_record


@pytest.fixture
def archive(tmp_path, monkeypatch):
    clips = [{'clip_id':f'day1_{i:06d}','frame_indices':list(range(30)),
              'global_end_fraction':str(30*(i+1))} for i in range(2)]
    monkeypatch.setattr(media,'build_clips',lambda inventory: clips)
    spec = {'inventory':{}}
    job_hash = object_hash(spec)
    atomic_json(tmp_path/'job.json',{'job_hash':job_hash,**spec})
    atomic_json(tmp_path/'manifest.json',{'job_hash':job_hash,'clips':clips})
    provenance = {'model':'native'}
    provenance_hash = object_hash(provenance)
    atomic_json(tmp_path/'provenance'/f'{provenance_hash}.json',provenance)
    record = {'clip_id':clips[0]['clip_id'],'clip':clips[0],'job_hash':job_hash,
              'status':'complete','caption':'I look around.',
              'generated_token_ids':[1,9],'eos_token_id':9,'provenance_hash':provenance_hash,
              'pixel_shape':[30,3,384,384],'speech_shape':[1,3000,128],'speech_lengths':[3000],
              'model_dtype':'torch.float16','speech_dtype':'torch.float16','strict_causal':False,
              'conservative_available_at_global_seconds_fraction':'60',
              'media':{'frames':list(range(30)),'audio':{'sample_rate':16000,'sample_count':480000}}}
    save_record(tmp_path,record)
    return tmp_path,record,clips


def test_full_audit_counts_actual_inventory_instead_of_worker_status(archive):
    root,record,clips = archive
    atomic_json(root/'status/worker_00.json',{'status':'complete'})
    result = audit.inspect(root)
    assert result['status']=='incomplete' and result['completed']==1 and result['expected']==2
    assert result['missing_clip_ids']==[clips[1]['clip_id']]


def test_smoke_requires_full_frame_count_and_native_tensor_shapes(archive):
    root,record,clips = archive
    assert audit.inspect(root,[record['clip_id']],smoke=True)['status']=='complete'
    record['pixel_shape'][0]=16
    save_record(root,record)
    with pytest.raises(ValueError,match='Smoke protocol mismatch'):
        audit.inspect(root,[record['clip_id']],smoke=True)


def test_truncated_manifest_cannot_falsely_pass_audit(archive):
    root,record,clips = archive
    atomic_json(root/'manifest.json',{'job_hash':record['job_hash'],'clips':clips[:1]})
    with pytest.raises(ValueError,match='exact source inventory'):
        audit.inspect(root)


def test_audit_accepts_official_length_stop_and_reports_it_separately(archive):
    root,record,clips=archive
    length_record={**record,'clip_id':clips[1]['clip_id'],'clip':clips[1],
                   'generated_token_ids':[3]*4096,'max_new_tokens':4096,
                   'generation_returned_normally':True,'finish_reason':'length','truncated':True}
    save_record(root,length_record)
    result=audit.inspect(root,smoke=True)
    assert result['status']=='complete' and result['completed']==2
    assert result['finish_reason_counts']=={'eos':1,'length':1}
    assert result['length_stopped_clip_ids']==[clips[1]['clip_id']]


def test_audit_does_not_count_short_output_claiming_a_length_stop(archive):
    root,record,clips=archive
    short_record={**record,'clip_id':clips[1]['clip_id'],'clip':clips[1],
                  'generated_token_ids':[3],'max_new_tokens':4096,
                  'generation_returned_normally':True,'finish_reason':'length','truncated':True}
    save_record(root,short_record)
    result=audit.inspect(root)
    assert result['status']=='incomplete' and result['completed']==1
    assert result['missing_clip_ids']==[clips[1]['clip_id']]
    assert result['finish_reason_counts']=={'eos':1,'length':0}
    assert result['length_stopped_clip_ids']==[]
