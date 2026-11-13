import copy
import json
from pathlib import Path
import pytest
from src.common import probe, run
from src.clip_manager import create_clip, thumbnail, ClipStore
from src.dataset_adapters import adapter
from src.routine_generator import build, change, TEMPLATE
from src.stream_generator import generate, manifest
from src.qa_generator import generate_qa
from src.validation import validate

@pytest.fixture
def clips(tmp_path):
    path = tmp_path/'source.mp4'
    run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=160x90:rate=12:duration=2','-c:v','libx264','-threads','1',str(path)])
    result=[]
    for i, slot in enumerate(TEMPLATE):
        c=create_clip({'path':str(path),'source_dataset':'TEST_ONLY'},0,0.5,slot['activity'],verified=True,evidence='scripted TEST ONLY fixture')
        c['clip_id']=f'test_{i}'
        result.append(c)
    return result

def test_pipeline_seven_days(clips,tmp_path):
    plan=build(clips)
    assert plan==build(clips)
    data=generate(plan,clips,tmp_path/'render',True,width=160,height=90,fps=12)
    assert len(data['events'])==35
    assert data['duration']==17.5
    assert len(list((tmp_path/'render').glob('day_*.mp4')))==7
    assert probe(tmp_path/'render/full_stream.mp4')['duration']==pytest.approx(17.5,abs=.1)
    qa=generate_qa(data)
    assert {q['question_type'] for q in qa}=={'episodic_recall','routine_frequency','temporal_order','cross_day_comparison'}
    assert validate(data,qa)['valid']
    bad=copy.deepcopy(qa)
    bad[0]['answer']='hallucination'
    assert not validate(data,bad)['valid']

def test_metadata_browser(clips,tmp_path):
    store=ClipStore(tmp_path/'clips.json')
    store.add(clips[0])
    with pytest.raises(ValueError): store.add(clips[0])
    store.export_csv(tmp_path/'clips.csv')
    thumbnail(clips[0],tmp_path/'thumb.jpg')
    assert (tmp_path/'thumb.jpg').stat().st_size>0
    (tmp_path/'broken.mp4').write_text('broken')
    rows,errors=adapter(tmp_path,'TEST_ONLY').discover()
    assert len(rows)==1 and len(errors)==1

def test_invalid_and_unverified(clips,tmp_path):
    with pytest.raises(ValueError): create_clip({'path':clips[0]['source_video'],'source_dataset':'TEST_ONLY'},1,99,'reading')
    with pytest.raises(ValueError): build(clips,persona_id='../bad')
    with pytest.raises(ValueError): build([],days=7)
    plan=build(clips,days=1)
    changed=change(plan,plan['events'][0]['event_id'],clips[-1],'episodic_deviation')
    assert changed['events'][0]['activity']=='reading'
    assert plan['events'][0]['activity']=='wake_up'
    with pytest.raises(NotImplementedError): change(plan,plan['events'][0]['event_id'],clips[0],'cross_day_state_change')
    unverified=copy.deepcopy(clips)
    unverified[0]['verified']=False
    data=generate(plan,unverified,tmp_path/'manifest')
    assert not (tmp_path/'manifest/full_stream.mp4').exists()
    assert all(q['question_type']!='routine_frequency' for q in generate_qa(data))
    data['events'][0]['end_time']=99
    assert not validate(data)['valid']

def test_audio_and_missing_source(clips,tmp_path):
    data=generate(build(clips,days=1),clips,tmp_path/'audio',True,width=160,height=90,fps=12,remove_audio=False)
    assert probe(tmp_path/'audio/full_stream.mp4')['has_audio']
    broken=copy.deepcopy(clips)
    broken[0]['source_video']=str(tmp_path/'missing.mp4')
    with pytest.raises(ValueError): generate(build(broken,days=1),broken,tmp_path/'broken')
    with pytest.raises(ValueError): generate(build(clips,days=1),clips,Path(clips[0]['source_video']).parent)

def test_ui():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py').run(timeout=30)
    assert not app.exception
    assert len(app.tabs)==3


def test_manifest_rejects_mislabel_and_day(clips):
    plan=build(clips,days=1)
    plan['events'][0]['activity']='invented'
    with pytest.raises(ValueError): manifest(plan,clips)
    plan=build(clips,days=1)
    plan['events'][0]['synthetic_day']=10
    with pytest.raises(ValueError): manifest(plan,clips)
