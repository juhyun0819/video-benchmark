import unittest
from .inputs import clock_seconds, canonical_questions, recent_samples


class InputTests(unittest.TestCase):
    def test_overlapping_clip_tail_belongs_to_next_source(self):
        from .stream_frames import sample_plan
        sources = [{'path':'a','day':1,'start_time':0,'duration':30.1},
                   {'path':'b','day':1,'start_time':30,'duration':1}]
        plan = sample_plan(sources, 2)
        self.assertEqual(len(plan), 62)
        self.assertEqual(plan[60]['source'], 'b')
        self.assertEqual(plan[60]['requested_global_timestamp'], 30)

    def test_two_fps_plan_preserves_day_boundary(self):
        from .stream_frames import sample_plan
        sources=[{'path':'day1','day':1,'start_time':86399,'duration':1},
                 {'path':'day2','day':2,'start_time':86400,'duration':1}]
        plan=sample_plan(sources,2)
        self.assertEqual([r['global_frame_id'] for r in plan],[0,1,2,3])
        self.assertEqual([r['requested_global_timestamp'] for r in plan],[86399,86399.5,86400,86400.5])

    def test_native_20fps_fraction(self):
        assert clock_seconds('11094208') + 17.6 == clock_seconds('11100000')
        assert clock_seconds('11210217') == 11*3600+21*60+2.85


    def test_ambiguous_suffix_rejected(self):
        with self.assertRaises(ValueError):
            clock_seconds('12002750')


    def test_causal_selection_at_boundary_and_gap(self):
        sources = [{'path': 'a', 'day': 1, 'start_time': 0, 'duration': 3},
                   {'path': 'b', 'day': 1, 'start_time': 10, 'duration': 3}]
        assert [p['requested_global_timestamp'] for p in recent_samples(sources, 10)] == [0, 1, 2, 10]
        assert [p['requested_global_timestamp'] for p in recent_samples(sources, 9)] == [0, 1, 2]
        assert [p['requested_global_timestamp'] for p in recent_samples(sources, 11.5)] == [1, 2, 10, 11]


    def test_no_sampling_at_exclusive_clip_end(self):
        sources = [{'path': 'a', 'day': 1, 'start_time': 0.4, 'duration': 2.0}]
        assert [p['requested_global_timestamp'] for p in recent_samples(sources, 2.4)] == [0.4, 1.4]


    def test_no_target_time_or_rationale_in_common_model_inputs(self):
        q = {'ID': '1', 'query_time': {'date': 'DAY1', 'time': '11210217'},
             'question': 'Q', 'choice_a': 'a', 'choice_b': 'b', 'choice_c': 'c', 'choice_d': 'd',
             'answer': 'B', 'need_audio': False, 'need_name': True, 'type': 'test',
             'target_time': 'secret', 'reason': 'secret'}
        result = canonical_questions([q])[0]
        assert 'target_time' not in result and 'reason' not in result
        assert result['correct_idx'] == 1
        q['query_time']['time'] = '12002750'
        result = canonical_questions([q])[0]
        assert result['query_time'] is None and 'needs_review' in result['time_status']
