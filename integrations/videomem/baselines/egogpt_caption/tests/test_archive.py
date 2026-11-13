import json
from pathlib import Path
import pytest
from baselines.egogpt_caption import archive


def record():
    return {'clip_id':'day1_000000','job_hash':'job','status':'complete','caption':'I see a cup.',
            'generated_token_ids':[1,2],'eos_token_id':2,'clip':{'global_end_fraction':'30'}}


def test_atomic_write_and_valid_resume(tmp_path):
    r=record();archive.save_record(tmp_path,r)
    assert archive.load_complete(tmp_path,r['clip_id'],'job')==r
    assert not list(tmp_path.rglob('*.tmp'))


def test_corrupt_caption_cannot_resume(tmp_path):
    r=record();archive.save_record(tmp_path,r)
    p=tmp_path/'captions'/f"{r['clip_id']}.json"
    p.write_text(p.read_text().replace('cup','dog'))
    with pytest.raises(ValueError,match='hash'): archive.load_complete(tmp_path,r['clip_id'],'job')


def test_different_job_cannot_resume(tmp_path):
    r=record();archive.save_record(tmp_path,r)
    with pytest.raises(ValueError,match='job'): archive.load_complete(tmp_path,r['clip_id'],'other')


def test_incomplete_output_never_counts_as_done(tmp_path):
    r=record();r['status']='incomplete';archive.save_record(tmp_path,r)
    assert archive.load_complete(tmp_path,r['clip_id'],'job') is None


def test_missing_receipt_does_not_count_complete(tmp_path):
    r=record();archive.save_record(tmp_path,r)
    (tmp_path/'receipts'/f"{r['clip_id']}.json").unlink()
    assert archive.load_complete(tmp_path,r['clip_id'],'job') is None


def test_completion_requires_nonempty_text_and_terminal_eos():
    assert archive.is_complete('hello',[3,2],2)
    assert not archive.is_complete('hello',[3],2)
    assert not archive.is_complete('',[2],2)


def test_resume_rechecks_eos_instead_of_trusting_status(tmp_path):
    r=record();r['generated_token_ids']=[1];archive.save_record(tmp_path,r)
    assert archive.load_complete(tmp_path,r['clip_id'],'job') is None


def test_interrupted_retry_leaves_no_stale_receipt(tmp_path, monkeypatch):
    r=record();r['status']='incomplete';archive.save_record(tmp_path,r)
    r['caption']='I continue walking.';r['status']='complete'
    original=archive.atomic_json
    def interrupted(path, value):
        if Path(path).parent.name=='receipts': raise OSError('simulated interruption')
        return original(path,value)
    monkeypatch.setattr(archive,'atomic_json',interrupted)
    with pytest.raises(OSError):archive.save_record(tmp_path,r)
    assert archive.load_complete(tmp_path,r['clip_id'],'job') is None


def length_record():
    return {**record(),'generated_token_ids':[3]*4096,'max_new_tokens':4096,
            'generation_returned_normally':True,'finish_reason':'length','truncated':True}


def test_official_length_bound_is_completed_but_remains_reported_as_truncated(tmp_path):
    r=length_record()
    kwargs={'max_new_tokens':4096,'generation_returned_normally':True}
    assert archive.finish_reason(r['generated_token_ids'],2,**kwargs)=='length'
    assert archive.is_complete(r['caption'],r['generated_token_ids'],2,**kwargs)
    archive.save_record(tmp_path,r)
    assert archive.load_complete(tmp_path,r['clip_id'],'job')==r


@pytest.mark.parametrize('count,cap,normal',[
    (4095,4096,True),(4097,4096,True),(128,128,True),(4096,128,True),
    (4096,None,True),(4096,'4096',True),(4096,4096.0,True),
    (4096,4096,False),(4096,4096,1),
])
def test_non_eos_completion_requires_exact_official_cap_and_explicit_normal_return(count,cap,normal):
    assert not archive.is_complete('I look around.',[3]*count,2,
                                   max_new_tokens=cap,generation_returned_normally=normal)


@pytest.mark.parametrize('bad_token',[None,True,-1,3.0,2])
def test_length_completion_rejects_invalid_tokens_or_an_earlier_eos(bad_token):
    tokens=[3]*4096;tokens[0]=bad_token
    assert not archive.is_complete('I look around.',tokens,2,
                                   max_new_tokens=4096,generation_returned_normally=True)


@pytest.mark.parametrize('field',[
    'max_new_tokens','generation_returned_normally','finish_reason','truncated',
])
def test_length_resume_requires_explicit_termination_evidence(tmp_path,field):
    r=length_record();del r[field]
    archive.save_record(tmp_path,r)
    assert archive.load_complete(tmp_path,r['clip_id'],'job') is None


@pytest.mark.parametrize('changes',[
    {'generated_token_ids':[3]}, {'finish_reason':'eos'}, {'truncated':False},
    {'generation_returned_normally':False}, {'max_new_tokens':128}, {'caption':' '},
])
def test_length_resume_rechecks_tokens_and_metadata_instead_of_trusting_status(tmp_path,changes):
    r={**length_record(),**changes};archive.save_record(tmp_path,r)
    assert archive.load_complete(tmp_path,r['clip_id'],'job') is None


def test_eos_at_official_length_bound_is_eos_not_truncation(tmp_path):
    r={**length_record(),'generated_token_ids':[3]*4095+[2],
       'finish_reason':'eos','truncated':False}
    assert archive.finish_reason(r['generated_token_ids'],[2,9],max_new_tokens=4096,
                                 generation_returned_normally=True)=='eos'
    archive.save_record(tmp_path,r)
    assert archive.load_complete(tmp_path,r['clip_id'],'job')==r


@pytest.mark.parametrize('changes',[
    {'finish_reason':'length'}, {'finish_reason':'incomplete'}, {'truncated':True},
    {'generation_returned_normally':False}, {'max_new_tokens':128},
])
def test_eos_resume_rejects_explicit_inconsistent_termination_metadata(tmp_path,changes):
    r={**record(),**changes};archive.save_record(tmp_path,r)
    assert archive.load_complete(tmp_path,r['clip_id'],'job') is None
