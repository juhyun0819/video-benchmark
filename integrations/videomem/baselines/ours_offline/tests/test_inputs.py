import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from baselines.ours_offline.inputs import uniform_samples, decode_samples, load_inputs, validate_sources
from baselines.ours_offline.inputs import sha256
from baselines.ours_offline.run import select_questions, evaluate_question


def source(path, day, start, frames, fps):
    return {'path':path,'day':day,'start_time':start,'frames':frames,'fps':fps,'duration':frames/fps}


class ClockAndCoverageTests(unittest.TestCase):
    def test_arbitrary_fps_and_optional_day(self):
        row={'path':'a','start_time':100,'frames':10,'fps':25,'duration':.4}
        selected=uniform_samples([row],100.2,20)
        self.assertEqual(len(selected),6)
        self.assertAlmostEqual(selected[-1]['requested_global_timestamp'],100.2)
        self.assertIsNone(selected[-1]['day'])
        validate_sources('/unused',[dict(row,day=12)],check_files=False)

    def test_preserves_real_day_gap_and_inclusive_cutoff(self):
        rows=[source('one',1,.4,4,2),source('two',2,86401.2,4,2)]
        selected=uniform_samples(rows,86401.7,6)
        self.assertEqual([x['requested_global_timestamp'] for x in selected],[.4,.9,1.4,1.9,86401.2,86401.7])
        self.assertEqual([x['day'] for x in selected],[1,1,1,1,2,2])

    def test_overlap_owned_by_later_source(self):
        rows=[source('a',1,0,602,20),source('b',1,30,20,20)]
        selected=uniform_samples(rows,30.05,1000)
        at_boundary=[x for x in selected if x['requested_global_timestamp']>=30]
        self.assertEqual([x['source'] for x in at_boundary],['b','b'])
        self.assertEqual(len(selected),602)

    def test_evidence_intervals_are_union_then_causally_clipped(self):
        selected=uniform_samples([source('a',1,0,100,1)],20,100,
                                 intervals=[(5,9),(8,12),(18,30)])
        self.assertEqual([x['requested_global_timestamp'] for x in selected],list(range(5,13))+[18,19,20])

    def test_sparse_sampling_does_not_materialize_raw_stream(self):
        selected=uniform_samples([source('a',1,0,20*86400,20)],86399.95,8)
        self.assertEqual(len(selected),8)
        self.assertEqual(selected[0]['source_frame_index'],0)
        self.assertEqual(selected[-1]['source_frame_index'],20*86400-1)
        self.assertEqual(len({x['source_frame_index'] for x in selected}),8)

    def test_unknown_or_duplicate_qa_subset_rejected(self):
        questions=[{'qa_id':str(i)} for i in range(6)]
        self.assertEqual([q['qa_id'] for q in select_questions(questions,shard=1,shards=2)],['1','3','5'])
        for ids in [['missing'],['1','1']]:
            with self.assertRaises(ValueError):select_questions(questions,qa_ids=ids)

    def test_source_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):validate_sources(root,[source('../escape',1,0,4,1)],check_files=False)

    def test_changed_source_and_scorer_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);video=root/'clip';video.write_bytes(b'media')
            row=source('clip',1,0,4,1);s=video.stat();row['file_state']=[s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
            q={'qa_id':'a','day':1,'query_time':2,'question':'Q','choices':['x','y'],'answer':'A','correct_idx':0}
            scorer=Path(__file__).parents[2]/'streambench/scoring.py'
            manifest={'scorer_sha256':sha256(scorer),'questions':[q]}
            m=root/'qa.json';m.write_text(json.dumps(manifest));index=root/'sources.json';index.write_text(json.dumps({'sources':[row]}))
            self.assertEqual(load_inputs(root,m,index,[1])[2][0]['qa_id'],'a')
            manifest['scorer_sha256']='bad';m.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'scorer'):load_inputs(root,m,index,[1])
            manifest['scorer_sha256']=sha256(scorer);m.write_text(json.dumps(manifest));video.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Source changed'):load_inputs(root,m,index,[1])

    def test_decoder_rejects_duplicate_actual_pts(self):
        with patch('baselines.ours_offline.inputs.decode_sample',return_value=(object(),{'actual_global_timestamp':1,'source':'a','source_pts':20})):
            with self.assertRaisesRegex(ValueError,'strictly increasing'):
                decode_samples('/unused',[{},{}],2)


class PublicInputTests(unittest.TestCase):
    def test_truncated_correct_letter_is_not_counted_as_success(self):
        class Backend:
            backbone='qwen3_vl'
            def answer(self,*args):return {'prediction':'A','raw_output':'A','complete':False}
        q={'qa_id':'a','query_time':10,'question':'Q','choices':['one','two'],'answer':'A','correct_idx':0}
        result=evaluate_question('/unused',[],q,Backend(),2,selector=lambda *a:[],
            decoder=lambda *a:([object(),object()],[{'actual_global_timestamp':1},{'actual_global_timestamp':2}]))
        self.assertEqual(result['status'],'incomplete');self.assertFalse(result['correct'])

    def test_label_stays_outside_backend_and_common_score_is_used(self):
        calls=[]
        class Backend:
            backbone='qwen3_vl'
            def answer(self,*args):
                calls.append(args)
                return {'prediction':'B','raw_output':'B','complete':True}
        q={'qa_id':'a','query_time':10,'question':'Question','choices':['one','two'],
           'answer':'B','correct_idx':1,'target_time':'GT_SECRET','reason':'GT_SECRET'}
        select=lambda *args:[{'requested_global_timestamp':1}]
        decode=lambda *args:([object(),object()],[{'actual_global_timestamp':1},{'actual_global_timestamp':2}])
        result=evaluate_question('/unused',[],q,Backend(),2,decoder=decode,selector=select)
        self.assertEqual(calls[0][2:],('Question',['one','two']))
        self.assertEqual(result['status'],'ok');self.assertTrue(result['correct'])
        self.assertNotIn('GT_SECRET',str(calls))

    def test_future_decoded_input_never_reaches_model(self):
        class Backend:
            backbone='llava_onevision'
            def answer(self,*args):raise AssertionError('must not run')
        q={'qa_id':'a','query_time':10,'question':'Q','choices':['one','two'],'answer':'A','correct_idx':0}
        result=evaluate_question('/unused',[],q,Backend(),2,selector=lambda *a:[],
            decoder=lambda *a:([object()],[{'actual_global_timestamp':11}]))
        self.assertEqual(result['status'],'error');self.assertIn('future',result['error'])
