import copy
import pytest
from baselines.qwen3_offline import run


def inventory():
    return {'sources':[{'day_id':'Day1','day_order':1,'global_start_fraction':'0',
                        'duration_fraction':'10'},
                       {'day_id':'Day2','day_order':2,'global_start_fraction':'10',
                        'duration_fraction':'10'}]}


def question():
    return {'question_id':'q1','question':'What happened?', 'choices':['A','B'],
            'query_time':{'day_id':'Day2','day_order':2,'seconds':2},
            'answer':'SECRET', 'evidence':'SECRET'}


def test_public_question_strips_labels_and_preserves_explicit_day_cutoff():
    q=run.validate_questions([question()],inventory())[0]
    assert q['global_seconds']==12
    assert 'SECRET' not in str(q)
    assert q['choices']==['A','B']


@pytest.mark.parametrize('change',[{'query_time':None},{'query_time':{'day_id':'Day2','day_order':1,'seconds':2}},
                                  {'query_time':{'day_id':'Day2','day_order':2,'seconds':11}},
                                  {'query_time':{'day_id':'Day2','day_order':2,'seconds':True}}])
def test_reject_unreviewed_or_invalid_cutoff(change):
    q=question();q.update(change)
    with pytest.raises(ValueError): run.validate_questions([q],inventory())


def test_duplicate_questions_rejected():
    with pytest.raises(ValueError): run.validate_questions([question(),question()],inventory())


def test_prompt_has_only_public_fields():
    q=run.validate_questions([question()],inventory())[0]
    m=run.messages(q)
    assert m[0]['content'][0]=={'type':'video'}
    assert 'SECRET' not in str(m)
    assert 'A. A' in str(m)


def test_processor_gets_original_indices_and_no_second_sampling():
    class Processor:
        def apply_chat_template(self,ms,**kw): return 'formatted'
        def __call__(self,**kw): return kw
    q=run.validate_questions([question()],inventory())[0]
    result=run.process_frames(Processor(),['rgb1','rgb2'],[0,25],26,'25',q)
    assert result['do_sample_frames'] is False
    assert result['video_metadata'][0]['frames_indices']==[0,25]
    assert result['video_metadata'][0]['fps']==25
    assert result['video_metadata'][0]['total_num_frames']==26
    assert 'size' not in result
    assert result['videos']==[['rgb1','rgb2']]
