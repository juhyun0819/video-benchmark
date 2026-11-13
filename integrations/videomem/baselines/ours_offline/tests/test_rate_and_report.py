import unittest
from baselines.ours_offline.inputs import rate_uniform_samples
from baselines.ours_offline.report import performance_breakdown


def source(path,start,duration,fps=20):
    return {'path':path,'start_time':start,'duration':duration,'fps':fps,'frames':int(duration*fps)}


class RateTests(unittest.TestCase):
    def test_half_fps_grid_before_cap_and_causal_end(self):
        samples=rate_uniform_samples([source('a',100,10)],107.9,768,sample_fps=.5)
        self.assertEqual([s['requested_global_timestamp'] for s in samples],[100,102,104,106])
        self.assertEqual([s['source_frame_index'] for s in samples],[0,40,80,120])

    def test_cap_spans_complete_history_and_preserves_gap(self):
        samples=rate_uniform_samples([source('a',0,10),source('b',100,10)],109,4,sample_fps=.5)
        self.assertEqual([s['requested_global_timestamp'] for s in samples],[0,6,102,108])
        self.assertEqual(len(samples),4)

    def test_overlap_and_fractional_cfr_remain_unique_causal(self):
        samples=rate_uniform_samples([source('a',0,4.2,5),source('b',4,4,5)],7.5,768,sample_fps=2)
        times=[s['requested_global_timestamp'] for s in samples]
        self.assertTrue(all(a<b for a,b in zip(times,times[1:])))
        self.assertTrue(all(t<=7.5 for t in times))
        self.assertEqual([s['source'] for s in samples if s['requested_global_timestamp']>=4],['b']*8)

    def test_invalid_rate_rejected(self):
        for rate in [0,-1,True,float('nan'),21]:
            with self.assertRaises(ValueError):rate_uniform_samples([source('a',0,10)],9,768,sample_fps=rate)


class AudioReportTests(unittest.TestCase):
    def test_unknown_flags_and_incomplete_answers_are_not_silently_dropped(self):
        qs=[{'qa_id':str(i),'query_time':10,'answer':'A','correct_idx':0,'choices':['one','two'],'need_audio':flag}
            for i,flag in enumerate([False,True,None,'false'])]
        rows=[{'qa_id':str(i),'query_time':10,'GT':'A','prediction':'A','status':status}
              for i,status in enumerate(['ok','incomplete','error'])]
        out=performance_breakdown(qs,rows)
        self.assertEqual(out['all']['expected'],4)
        self.assertEqual(out['all']['completed'],3)
        self.assertEqual(out['all']['correct'],1)
        self.assertEqual(out['no_audio_required']['expected'],1)
        self.assertEqual(out['no_audio_required']['accuracy_over_completed'],1.)
        self.assertEqual(out['audio_required']['correct'],0)
        self.assertEqual(out['audio_requirement_unknown']['expected'],2)
        self.assertFalse(out['all']['coverage_complete'])
        rows.append(dict(rows[0]))
        with self.assertRaises(ValueError):performance_breakdown(qs,rows)

    def test_supervisor_missing_rows_are_not_counted_as_completed(self):
        q={'qa_id':'a','query_time':10,'answer':'A','choices':['one','two'],'need_audio':False}
        out=performance_breakdown([q],[{'qa_id':'a','supervisor_generated':True}])
        self.assertEqual(out['no_audio_required']['completed'],0)
        self.assertIsNone(out['no_audio_required']['accuracy_over_completed'])
        self.assertEqual(out['no_audio_required']['accuracy_over_expected'],0.)

    def test_index_only_canonical_qa_is_supported(self):
        q={'qa_id':'a','query_time':10,'correct_idx':1,'choices':['one','two'],'need_audio':False}
        r={'qa_id':'a','query_time':10,'GT':'B','prediction':'B','status':'ok'}
        self.assertEqual(performance_breakdown([q],[r])['no_audio_required']['correct'],1)
