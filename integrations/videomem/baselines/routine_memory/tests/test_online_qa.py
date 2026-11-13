import torch
from types import SimpleNamespace

from baselines.routine_memory import RoutineAwareMemory
from baselines.routine_memory.online_qa import public_question, absolute_time, select_evidence, sparse_prompt


def test_public_boundary_and_absolute_clock():
    raw={'qa_id':'1','day':2,'query_time':86500.,'question':'Where?','choices':['x','y'],
         'answer':'A','keywords':['secret'],'correct_idx':0}
    assert set(public_question(raw))=={'qa_id','day','query_time','question','choices'}
    assert absolute_time(raw)==86500.
    import pytest
    with pytest.raises(ValueError,match='canonical absolute'):
        absolute_time({**raw,'query_time':100.})


def test_readout_is_bounded_owned_and_causal():
    import pytest
    memory=RoutineAwareMemory()
    for i in range(4):
        memory.update(torch.ones(16,8,dtype=torch.bfloat16),i,float(i))
    rows=select_evidence(memory,9,3.)
    assert sum(len(r['features']) for r in rows)==9
    assert all(r['timestamp']<=3 for r in rows)
    original=next(memory.retained_evidence())['features'].clone()
    rows[0]['features'].zero_()
    assert torch.equal(original,next(memory.retained_evidence())['features'])
    with pytest.raises(ValueError,match='Future'):
        select_evidence(memory,9,0.)


def test_sparse_positions_equal_native_qwen_for_complete_grid():
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
    class Tokens:
        def encode(self,text,add_special_tokens=False):
            import re
            special={'<|vision_start|>':10,'<|vision_end|>':11}
            return [special.get(piece,1) for piece in re.split('(<\\|vision_start\\|>|<\\|vision_end\\|>)',text) if piece]
    config=SimpleNamespace(image_token_id=12,video_token_id=13,vision_start_token_id=10,
                           vision_config=SimpleNamespace(spatial_merge_size=2,tokens_per_second=2))
    question={'day':1,'query_time':20.,'question':'Where?','choices':['x','y']}
    records=[{'features':torch.ones(4,8),'token_indices':torch.arange(4),'frame_id':0,'timestamp':10.}]
    ids,positions,replacements,next_pos=sparse_prompt(Tokens(),config,question,records,grid_width=2)
    native,_=Qwen2_5_VLModel.get_rope_index(SimpleNamespace(config=config),ids,
                                         image_grid_thw=torch.tensor([[1,4,4]]))
    assert torch.equal(positions,native)
    assert next_pos==int(positions.max())+1
    begin,features=replacements[0]
    assert ids[0,begin:begin+4].tolist()==[12]*4
    assert features.shape==(4,8)
    sparse_records=[{**records[0],'features':records[0]['features'][:1],
                     'token_indices':torch.tensor([0])}]
    sparse_ids,sparse_positions,_,_=sparse_prompt(Tokens(),config,question,sparse_records,grid_width=2)
    keep=torch.ones(ids.shape[1],dtype=torch.bool)
    keep[begin+1:begin+4]=False
    assert torch.equal(sparse_ids,ids[:,keep])
    assert torch.equal(sparse_positions,positions[:,:,keep])


def test_online_run_logs_query_frozen_counts_bytes_and_correctness(tmp_path,monkeypatch):
    import csv
    import hashlib
    import json
    import queue
    from baselines.routine_memory import online_qa
    from baselines.feature_memory import archive
    model=tmp_path/'model'; model.mkdir()
    for name in ('config.json','preprocessor_config.json'):
        (model/name).write_text('{}')
    identity={'participant':'A1_JAKE','independent_image_encoding':True,
              'checkpoint_repo_id':'Qwen/Qwen2.5-VL-7B-Instruct',
              'model_config_sha256':hashlib.sha256(b'{}').hexdigest(),
              'processor_config_sha256':hashlib.sha256(b'{}').hexdigest()}
    data=tmp_path/'archive';data.mkdir()
    (data/'archive.json').write_text(json.dumps({'archive_id':'test','frame_shape':[2500,3584],
                                               'dtype':'bfloat16','identity':identity}))
    qa={'qa_id':'1','day':1,'query_time':.5,'question':'Where?','choices':['x','y'],
        'answer':'A','correct_idx':0,'need_audio':False}
    qa_path=tmp_path/'qa.json'
    qa_path.write_text(json.dumps({'questions':[qa,{**qa,'qa_id':'2','need_audio':True}]}))
    frames=[{'frame_id':i,'timestamp':float(i),'source_id':'source','day':'DAY1',
             'shared_feature_sha256':'1'*64} for i in range(2)]
    monkeypatch.setattr(archive,'iter_archive',lambda *a,**k:iter([{
        'features':torch.ones(2,2500,3584,dtype=torch.bfloat16),'frames':frames,'read_seconds':.1}]))
    class Context:
        def __init__(self):self.number=0
        def Queue(self,maxsize=0):
            self.number+=1
            if self.number==2:
                self.results=queue.Queue();return self.results
            context=self
            class Jobs:
                def put(self,task,timeout=None):
                    assert set(task['question'])=={'qa_id','day','query_time','question','choices'}
                    assert all(row['timestamp']<=.5 for row in task['evidence'])
                    context.results.put({'event':'prediction','qa_id':'1','gpu':2,'answer':'A','status':'ok'})
            return Jobs()
        def Process(self,target,args):
            context=self
            class Process:
                exitcode=0
                def start(self):context.results.put({'event':'ready','gpu':args[0]})
                def is_alive(self):return False
                def join(self,timeout=None):pass
            return Process()
    monkeypatch.setattr(online_qa.mp,'get_context',lambda name:Context())
    output=tmp_path/'run'
    args=SimpleNamespace(archive=str(data),qa=str(qa_path),model=str(model),output=str(output),
                         gpus=[2,3],budget_bytes=16_000_000_000,qa_read_tokens=32,
                         max_new_tokens=8,max_questions=0,cpu_threads=1)
    online_qa.run(args)
    status=json.loads((output/'status.json').read_text())
    prediction=json.loads((output/'predictions.jsonl').read_text())
    table=list(csv.DictReader((output/'predictions.csv').open()))
    assert status['phase']=='complete' and status['qa_completed']==1 and status['accuracy']==1.
    assert prediction['correct'] and prediction['ground_truth']=='A'
    assert prediction['memory_at_query']['total_frames_seen']==1
    for key in ('routines','events','event_to_routine_promotions','short_term_bytes',
                'routine_memory_bytes','event_memory_bytes','metadata_bytes','actual_memory_bytes'):
        assert key in prediction['memory_at_query'] and table[0][key]!=''
    assert table[0]['correct']=='True'
    assert not (output/'checkpoint_final').exists()
