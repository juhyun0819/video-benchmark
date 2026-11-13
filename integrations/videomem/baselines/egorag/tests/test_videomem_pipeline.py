import copy
import json
from pathlib import Path
import pytest
from videomem_adapter.cli import smoke, evaluate, run_rows
from videomem_adapter.common import public_question, read, digest
from videomem_adapter.pipeline import Pipeline
from videomem_adapter.media import prepare_manifest


@pytest.fixture(scope='module')
def ready(tmp_path_factory):
    work=tmp_path_factory.mktemp('pipeline')
    smoke(work)
    return work,read(work/'config.json'),read(work/'manifest.json')


def q(seconds=2,text='What color was the scene?'):
    return {'question_id':'q','question':text,'choices':['red','blue'],
            'query_time':{'day_id':'Day1','day_order':1,'seconds':seconds}}


def pipe(ready):
    work,config,manifest=ready
    return Pipeline(config,manifest,work/'run')


def test_temporal_visual_and_resume_end_to_end(ready):
    result=read(ready[0]/'smoke_result.json')
    assert result['status']=='passed' and result['test_only']
    assert result['new_model_calls_on_second_pass']==0
    assert len(result['resume']['reused'])==2
    assert result['answer']['prediction']=='A'


@pytest.mark.parametrize('key',['answer','target_time','reason','keywords','correct_idx','original_qa'])
def test_label_fields_are_not_in_public_schema(key):
    row={**q(),key:'SECRET'}
    assert 'SECRET' not in json.dumps(public_question(row))


def test_missing_query_time_not_inferred_from_evidence():
    row=q(); row.pop('query_time')
    row['target_time']={'day_id':'Day1','seconds':2}
    with pytest.raises(ValueError,match='Explicit query_time'): public_question(row)


def test_straddling_clip_is_excluded(ready):
    p=pipe(ready)
    result=p.query(q(1.9))
    assert result['retrieved']==[] and result['status']=='no_observed_evidence'
    assert p.answer(result)['prediction'] is None


def test_same_bucket_future_caption_not_summarized(ready):
    p=pipe(ready)
    future=p.summaries(q(4)['query_time'])
    early=p.summaries(q(2)['query_time'])
    for level,rows in early.items():
        assert rows and all(r['day_end_seconds']<=2 for r in rows)
        assert all('blue' not in r['text'].lower() for r in rows)
        assert not {r['summary_id'] for r in rows}&{r['summary_id'] for r in future[level]}


def test_tampered_future_retrieval_rejected_at_answer(ready):
    p=pipe(ready); result=p.query(q())
    result['retrieved'][0]['clip']['file_end_seconds']=4
    with pytest.raises(RuntimeError): p.answer(result)


def test_recency_branch_and_upper_bound_clamp(ready):
    p=pipe(ready)
    result=p.query(q(2,'What color was the last scene?'))
    assert result['retrieved']
    assert result['selected_range'][-1]<=2
    assert all(h['clip']['day_end_seconds']<=2 for h in result['retrieved'])


def test_index_isolated_when_manifest_changes(ready,tmp_path):
    # Old extra sentence must not contaminate current same-interval caption.
    p=Pipeline(ready[1],ready[2],tmp_path/'isolated_index'); p.create()
    old_id=p.index.collection.get()['ids'][0]
    meta=p.index.collection.get()['metadatas'][0]
    p.index.collection.upsert(ids=[old_id.rsplit('_',1)[0]+'_99'],documents=['OLD_MANIFEST_SECRET'],
        metadatas=[meta],embeddings=p.embeddings.embed(['OLD_MANIFEST_SECRET']))
    manifest=copy.deepcopy(p.manifest)
    manifest['provenance_test_revision']='new_manifest'
    new=Pipeline(p.config,manifest,p.work)
    assert new.index_namespace!=p.index_namespace
    created=new.create()
    assert len(created['reused'])==2 and new.models.calls==0
    retrieved=new.query(q())
    answer=new.answer(retrieved)
    assert 'OLD_MANIFEST_SECRET' not in json.dumps(answer)


def test_prompt_and_model_changes_do_not_inherit_cache(ready):
    p=pipe(ready); config=copy.deepcopy(p.config)
    config['models']['answerer']['revision']='different'
    newer=Pipeline(config,p.manifest,p.work)
    assert p.namespace!=newer.namespace


def test_failed_query_not_coerced_to_prediction(ready):
    p=pipe(ready)
    result={'status':'execution_error','question':q(),'error_type':'ValueError','prediction':None}
    answer=p.answer(result)
    assert answer['status']=='execution_error' and answer['prediction'] is None
    evaluation=evaluate([answer],[{'question_id':'q','answer':'A'}])
    assert evaluation['total']==1 and evaluation['correct']==0 and evaluation['unparsed_or_failed']==1


def test_actual_answer_wire_has_images_and_no_eval_fields(ready):
    rows=[json.loads(x) for x in (ready[0]/'run/model_calls.jsonl').read_text().splitlines()]
    answers=[x for x in rows if x['role']=='answerer']
    assert answers and all(x['images'] for x in answers)
    assert all(f['file_seconds']<2 for f in answers[0]['images'])
    assert 'FORBIDDEN_' not in json.dumps(rows)


def test_wrong_day_mapping_rejected(ready):
    row=q(); row['query_time']['day_order']=2
    with pytest.raises(ValueError,match='Day ID/order'): pipe(ready).query(row)
