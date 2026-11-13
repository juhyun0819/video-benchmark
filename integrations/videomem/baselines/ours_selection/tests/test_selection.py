import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import unittest
import tempfile
from baselines.ours_selection.captions import CaptionManifest
from baselines.ours_selection.selection import select_ids,evaluate_question
from baselines.ours_selection.archive import SelectionArchive,SCHEMA,candidate_plan,prepare_archive
from baselines.ours_offline.inputs import sha256


class Archive:
    identity='archive-hash'
    times=[0.,1.,2.,3.]
    features=np.array([[0.,1.],[1.,1.],[2.,1.],[3.,1.]],dtype=np.float32)
    records=[{'global_frame_id':i,'actual_global_timestamp':float(i)} for i in range(4)]
    def eligible(self,cutoff):return [i for i,t in enumerate(self.times) if t<=cutoff]
    def verify_selected_sources(self,root,ids):pass


def question(cutoff=1.):
    return {'qa_id':'Q','query_time':cutoff,'question':'Visible object?',
            'choices':['alpha','beta','gamma','delta'],'answer':'C','correct_idx':2,
            'reason':'SECRET_RATIONALE','target_time':'SECRET_TARGET'}


class Scorer:
    def score(self,features,text):
        assert text=='Visible object?'
        assert features[:,0].tolist()==[0.,1.]
        return [.1,.9]


class Backend:
    backbone='qwen3_vl'
    called=False
    def answer(self,frames,timestamps,text,choices):
        self.called=True
        assert text=='Visible object?' and choices==['alpha','beta','gamma','delta']
        assert [t['actual_global_timestamp'] for t in timestamps]==[0.,1.]
        return {'prediction':'C','raw_output':'C','complete':True,'generated_token_ids':[1]}


def decode(root,record,cutoff):
    assert record['actual_global_timestamp']<=cutoff
    return object(),dict(record)


def check_aks_restricts_before_scoring_and_preserves_choice_labels():
    backend=Backend()
    row=evaluate_question('.',Archive(),question(),method='aks',scorer=Scorer(),
        parameters={'max_num_frames':2,'all_depth':1},backend=backend,max_frames=2,decoder=decode)
    assert row['status']=='ok' and row['correct'] is True and backend.called
    assert row['selected_frame_ids']==[0,1]
    assert row['selection']['candidate_count']==2
    assert 'SECRET' not in row['raw_output']


def check_decode_future_frame_is_rejected_before_backend():
    def future(root,record,cutoff):return object(),dict(record,actual_global_timestamp=100.)
    backend=Backend()
    row=evaluate_question('.',Archive(),question(),method='aks',scorer=Scorer(),
        parameters={'max_num_frames':2,'all_depth':1},backend=backend,max_frames=2,decoder=future)
    assert row['status']=='error' and not backend.called and not row['correct']


def caption_file(tmp_path,updates=None):
    rows=[{'global_frame_id':i,'global_timestamp':float(i),'observed_until':float(i),
           'available_at':float(i),'text':f'caption-{i}'} for i in range(4)]
    for i,update in (updates or {}).items():rows[i].update(update)
    p=tmp_path/'captions.json';p.write_text(json.dumps({'schema':'ours_videotree_captions_v1',
        'archive_sha256':'archive-hash','producer':{'model':'external-verified-captioner','question_conditioned':False},'captions':rows}))
    return p


def check_videotree_excludes_caption_interval_extending_after_query(tmp_path):
    archive=Archive();captions=CaptionManifest(caption_file(tmp_path,{1:{'observed_until':5.,'available_at':5.}}),archive)
    class Relevance:
        def score(self,text,choices,caps):
            assert caps==['caption-0','caption-2']
            assert choices==question()['choices']
            return [1,3]
    def tree(features,texts,score,**kwargs):
        assert features[:,0].tolist()==[0.,2.]
        assert score(texts)==[1,3]
        return [1],{}
    ids,meta=select_ids(archive,question(2.),method='videotree',scorer=Relevance(),parameters={},
        captions=captions,tree_builder=tree)
    assert ids==[2] and meta['candidate_count']==2


def check_caption_misalignment_or_unknown_evidence_rejected(tmp_path,changes):
    with unittest.TestCase().assertRaises(ValueError):CaptionManifest(caption_file(tmp_path,changes),Archive())


def check_tree_budget_overflow_is_explicit_not_silent_truncation(tmp_path):
    captions=CaptionManifest(caption_file(tmp_path),Archive())
    backend=Backend()
    row=evaluate_question('.',Archive(),question(2.),method='videotree',scorer=object(),parameters={},
        backend=backend,max_frames=1,captions=captions,decoder=decode,tree_builder=lambda *a,**kw:([0,1],{}))
    assert row['status']=='error' and 'no silent truncation' in row['error'] and not backend.called


def check_archive_integrity_and_midrun_mutation(tmp_path):
    source=tmp_path/'clip.mp4';source.write_bytes(b'fixture');s=source.stat()
    rows=[{'path':'clip.mp4','day':1,'fps':20,'file_state':[s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]}]
    index=tmp_path/'sources.json';index.write_text(json.dumps({'sources':rows}))
    folder=tmp_path/'archive';folder.mkdir();np.save(folder/'features.npy',np.ones((1,2),dtype=np.float32))
    (folder/'frames.jsonl').write_text(json.dumps({'global_frame_id':0,'source':'clip.mp4','actual_global_timestamp':0.})+'\n')
    metadata={'schema':SCHEMA,'method':'aks','status':'complete','query_conditioned':False,
        'source_index_sha256':sha256(index),'source_index':rows,'feature_shape':[1,2],'feature_dtype':'float32',
        'frames_sha256':sha256(folder/'frames.jsonl'),'features_sha256':sha256(folder/'features.npy')}
    (folder/'metadata.json').write_text(json.dumps(metadata))
    archive=SelectionArchive(folder,tmp_path,index)
    assert archive.eligible(0.)==[0]
    archive.verify_selected_sources(tmp_path,[0])
    (folder/'frames.jsonl').write_text('{}\n')
    with unittest.TestCase().assertRaisesRegex(ValueError,'changed'):archive.eligible(1.)
    with unittest.TestCase().assertRaisesRegex(ValueError,'integrity'):SelectionArchive(folder,tmp_path,index)


class SelectionTests(unittest.TestCase):
    def test_arbitrary_fps_and_optional_days_preserve_gaps(self):
        sources=[{'path':'a','start_time':0.,'duration':2.,'fps':25.,'frames':50},
                 {'path':'b','start_time':10.,'duration':1.,'fps':30.,'frames':30}]
        plan=candidate_plan(sources,fps=2.)
        self.assertEqual([x['requested_global_timestamp'] for x in plan],[0.,.5,1.,1.5,10.,10.5])
        self.assertTrue(all(x['day'] is None for x in plan))

    def test_exact_fraction_metadata_avoids_extra_boundary_candidate(self):
        sources=[{'path':'fractional','start_time':0.,'start_time_fraction':'0',
            'duration':5/6,'duration_fraction':'5/6','fps':12.,'fps_fraction':'12','frames':11}]
        plan=candidate_plan(sources,fps=6.)
        self.assertEqual(len(plan),5)
        self.assertEqual(plan[-1]['requested_global_timestamp'],2/3)

    def test_incomplete_answer_is_incorrect_with_explicit_reason(self):
        class IncompleteBackend(Backend):
            def answer(self,*args):
                return {'prediction':'C','raw_output':'C','complete':False}
        row=evaluate_question('.',Archive(),question(),method='aks',scorer=Scorer(),
            parameters={'max_num_frames':2,'all_depth':1},backend=IncompleteBackend(),max_frames=2,decoder=decode)
        self.assertEqual(row['status'],'incomplete')
        self.assertFalse(row['correct'])
        self.assertIn('incomplete',row['error'])

    def test_single_selected_frame_is_rejected_without_padding(self):
        backend=Backend()
        class OneScorer:
            def score(self,features,text):return [1.]*len(features)
        row=evaluate_question('.',Archive(),question(0.),method='aks',scorer=OneScorer(),
            parameters={'max_num_frames':2,'all_depth':1},backend=backend,max_frames=2,decoder=decode)
        self.assertEqual(row['status'],'error')
        self.assertIn('no duplicate padding',row['error'])
        self.assertFalse(backend.called)

    def test_prepare_without_qa_and_bfloat16_exact_storage(self):
        import torch
        from unittest.mock import patch
        from baselines.ours_selection.run import readiness
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'video.mp4';source.write_bytes(b'fixture')
            s=source.stat()
            rows=[{'path':'video.mp4','start_time':0.,'duration':.1,'fps':30.,'frames':3,
                   'file_state':[s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]}]
            index=root/'sources.json';index.write_text(json.dumps({'sources':rows}))
            class Encoder:
                identity={'test_only':True}
                def encode_images(self,images):
                    return torch.tensor([[.5,1.5]]*len(images),dtype=torch.bfloat16)
            def decoded(root,sample,cutoff):
                return object(),dict(sample,actual_global_timestamp=sample['requested_global_timestamp'])
            archive_path=root/'features'
            with patch('baselines.ours_selection.archive.decode_sample',side_effect=decoded):
                metadata=prepare_archive(root,index,archive_path,'aks',Encoder(),fps=30.,batch_size=2)
            self.assertEqual(metadata['encoder_feature_dtype'],'torch.bfloat16')
            self.assertEqual(metadata['feature_dtype'],'float32')
            archive=SelectionArchive(archive_path,root,index)
            self.assertEqual(archive.features.tolist(),[[.5,1.5]]*3)
            status=readiness(SimpleNamespace(root=root,sources=index,archive=archive_path,canonical=None,captions=None))
            self.assertEqual(status['status'],'awaiting_QA')
            self.assertFalse(status['model_loaded'])
            self.assertEqual(status['selection_features'],'ready')

    def test_aks_causal(self):
        check_aks_restricts_before_scoring_and_preserves_choice_labels()
    def test_future_decode(self):
        check_decode_future_frame_is_rejected_before_backend()
    def test_caption_causality(self):
        with tempfile.TemporaryDirectory() as root:
            check_videotree_excludes_caption_interval_extending_after_query(Path(root))
    def test_caption_validation(self):
        for changes in [{1:{'global_timestamp':99.}},{1:{'observed_until':5.,'available_at':1.}},
                        {1:{'global_frame_id':0}},{1:{'text':''}}]:
            with self.subTest(changes=changes),tempfile.TemporaryDirectory() as root:
                check_caption_misalignment_or_unknown_evidence_rejected(Path(root),changes)
    def test_tree_budget(self):
        with tempfile.TemporaryDirectory() as root:
            check_tree_budget_overflow_is_explicit_not_silent_truncation(Path(root))
    def test_archive_mutation(self):
        with tempfile.TemporaryDirectory() as root:
            check_archive_integrity_and_midrun_mutation(Path(root))


if __name__=='__main__':
    unittest.main()
