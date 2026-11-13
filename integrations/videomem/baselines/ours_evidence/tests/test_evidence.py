import unittest
import json
from pathlib import Path
import tempfile
from baselines.ours_evidence.inputs import EvidenceError, evidence_clips, evidence_samples, reference_samples, load_evidence
from baselines.ours_offline.inputs import sha256
from baselines.ours_evidence.tests.egolife_fixture import evidence_points


def source(path, start, duration, day=1):
    return {'path':path,'start_time':start,'duration':duration,'day':day,'fps':20,'frames':round(duration*20)}


class EvidenceTests(unittest.TestCase):
    def test_concatenated_dates_are_separate_points(self):
        points = evidence_points({'date':'DAY1','time':'11360904DAY2_10453312DAY3_11235811'})
        self.assertEqual(points,[41769.2,125133.6,213838.55])

    def test_list_duplicates_case_and_frame_overflow(self):
        points = evidence_points({'date':'Day1','time_list':['12002750','12002750','12002919']})
        self.assertEqual(points,[43229.5,43229.95])

    def test_ambiguous_encoding_is_not_guessed(self):
        for target in [{'date':'DAY1','time':'12000000-13000000'},
                       {'date':'DAY1','time':'12000000','time_list':['12000000']},
                       {'date':'DAY1','time_list':[]}, {'date':'DAY1','time':'25000000'}]:
            with self.assertRaises(EvidenceError):evidence_points(target)

    def test_clip_end_and_overlap_select_next_clip(self):
        sources=[source('a',0,30.1),source('b',30,30)]
        self.assertEqual([x['path'] for x in evidence_clips(sources,[30],35)],['b'])
        clips=evidence_clips(sources,[29.9,30],35)
        self.assertEqual([x['duration'] for x in clips],[30,30])

    def test_missing_or_future_evidence_rejects_whole_selection(self):
        sources=[source('a',0,10),source('b',20,10)]
        for points in [[5,15],[5,25]]:
            with self.assertRaises(EvidenceError):evidence_clips(sources,points,22)

    def test_multi_clip_sampling_covers_each_and_respects_query_cutoff(self):
        sources=[source('a',0,10),source('unrelated',20,10),source('b',40,10)]
        clips,samples=evidence_samples(sources,[5,41],42,8)
        self.assertEqual([x['path'] for x in clips],['a','b'])
        self.assertEqual({x['source'] for x in samples},{'a','b'})
        self.assertEqual(len(samples),8)
        self.assertTrue(all(x['requested_global_timestamp']<=42 for x in samples))
        with self.assertRaises(EvidenceError):evidence_samples(sources,[5,41],42,3)

    def test_generic_clip_and_half_open_interval_inputs(self):
        sources=[source('day1.mp4',0,10),source('day2.mp4',10,10)]
        for item in sources:
            item.pop('day')
            item.update(fps=25, frames=250)
        clips,samples=reference_samples(sources,{'clips':['day2.mp4']},12,8)
        self.assertEqual([clip['path'] for clip in clips],['day2.mp4'])
        self.assertTrue(all(10<=sample['requested_global_timestamp']<=12 for sample in samples))
        clips,samples=reference_samples(sources,{'intervals':[{'start_time':9,'end_time':11}]},12,16)
        self.assertEqual({sample['source'] for sample in samples},{'day1.mp4','day2.mp4'})
        self.assertTrue(all(9<=sample['requested_global_timestamp']<11 for sample in samples))

    def test_interval_gaps_future_and_malformed_evidence_fail(self):
        sources=[source('a',0,10),source('b',20,10)]
        for evidence in [
            {'intervals':[{'start_time':9,'end_time':21}]},
            {'intervals':[{'start_time':20,'end_time':23}]},
            {'intervals':[None]}, {'clips':['missing']},
            {'intervals':[{'start_time':True,'end_time':5}]},
            {'clips':['a'],'points':[1]}]:
            with self.subTest(evidence=evidence), self.assertRaises(EvidenceError):
                reference_samples(sources,evidence,22,16)

    def test_sidecar_is_bound_to_manifest_and_unique_known_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path=Path(directory)/'questions.json'
            manifest={'questions':[{'qa_id':'q'}]}
            manifest_path.write_text(json.dumps(manifest))
            evidence_path=Path(directory)/'evidence.json'
            entry={'qa_id':'q','evidence':{'points':[1]}}
            document={'qa_manifest_sha256':sha256(manifest_path),'entries':[entry]}
            evidence_path.write_text(json.dumps(document))
            self.assertEqual(load_evidence(evidence_path,manifest,manifest_path),{'q':{'points':[1]}})
            for entries in [[entry,entry],[{'qa_id':'unknown'}],[None]]:
                evidence_path.write_text(json.dumps(dict(document,entries=entries)))
                with self.assertRaises(EvidenceError):load_evidence(evidence_path,manifest,manifest_path)
            evidence_path.write_text(json.dumps(dict(document,qa_manifest_sha256='0'*64)))
            with self.assertRaises(EvidenceError):load_evidence(evidence_path,manifest,manifest_path)

class ModelBoundaryTests(unittest.TestCase):
    def test_labels_and_evidence_metadata_are_not_model_arguments(self):
        from baselines.ours_evidence.evaluate import plan_question,answer_question
        q={'qa_id':'q','query_time':8,'question':'Who?','choices':['x','y'],'answer':'B','correct_idx':1}
        annotation={'points':[5]}
        record=plan_question(q,annotation,[source('a',0,10)],4)
        class Backend:
            def answer(self,frames,timestamps,question,choices):
                assert question=='Who?' and choices==['x','y']
                assert all('GT' not in t and 'evidence_points' not in t for t in timestamps)
                return {'prediction':'B','raw_output':'B','complete':True}
        def decode(root,samples,cutoff):
            return [object() for s in samples],[dict(s,actual_global_timestamp=s['requested_global_timestamp']) for s in samples]
        result=answer_question(q,record,'unused',Backend(),decode)
        self.assertTrue(result['correct']);self.assertEqual(result['status'],'ok')

    def test_future_decoder_output_never_reaches_backend(self):
        from baselines.ours_evidence.evaluate import plan_question,answer_question
        q={'qa_id':'q','query_time':8,'question':'Q','choices':['x','y'],'answer':'B','correct_idx':1}
        record=plan_question(q,{'points':[5]},[source('a',0,10)],4)
        class Backend:
            def answer(self,*args):raise AssertionError('Must not call model')
        result=answer_question(q,record,'unused',Backend(),lambda *args:([object()],[{'actual_global_timestamp':9}]))
        self.assertEqual(result['status'],'error');self.assertIn('Future evidence frame',result['error'])


if __name__=='__main__':unittest.main()
