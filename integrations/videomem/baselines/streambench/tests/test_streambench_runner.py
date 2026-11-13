import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from streambench_fixture import CounterMemory
try:
    from baselines.streambench.runner import observe_stream, evaluate_checkpoint
    from baselines.streambench.checkpoint import read_checkpoint
except ImportError:
    observe_stream = evaluate_checkpoint = read_checkpoint = None


def chunks(days=(1,2)):
    for day in days:
        stamps = [dict(stream_id='test', day=day, global_frame_id=(day-1)*2+i,
                       source_timestamp=float(i), global_timestamp=float((day-1)*2+i),
                       source_id=f'day{day}', source_frame_index=i) for i in range(2)]
        yield [torch.tensor([day]), torch.tensor([day+1])], stamps

class RunnerTests(unittest.TestCase):
    def test_cumulative_day_checkpoints_and_independent_qa(self):
        self.assertIsNotNone(observe_stream, 'continuous runner must exist')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            method = CounterMemory(65536)
            report = observe_stream(method, chunks(), root, stream_id='test', manifest_digest='abc', method_config={'method':'fixture'})
            self.assertEqual(report['days_observed'], [1,2])
            ckpt = root/'checkpoint_D1_D2'
            state=read_checkpoint(ckpt)
            self.assertEqual(state['metadata']['frames_seen_total'], 4)
            self.assertEqual(state['metadata']['days_observed'], [1,2])
            questions=[{'question_id':'a','source_day':1,'question':'sum','answer':'131'},
                       {'question_id':'b','source_day':1,'question':'sum','answer':'131'}]
            original=method.snapshot()
            rows=evaluate_checkpoint(method, ckpt, questions)
            self.assertEqual(rows[0]['answer'], rows[1]['answer'])
            self.assertEqual(rows[0]['days_after_event'], 1)
            self.assertEqual(method.snapshot()['method'], original['method'])
            self.assertTrue(torch.equal(method.protocol_state['rng']['torch'], original['protocol']['rng']['torch']))
            self.assertTrue(rows[0]['correct'])
            self.assertEqual(len((root/'chunks.jsonl').read_text().splitlines()), 2)

    def test_resume_in_fresh_process_equals_uninterrupted(self):
        self.assertIsNotNone(observe_stream)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            method=CounterMemory(65536)
            observe_stream(method,chunks((1,)),root/'first',stream_id='test',manifest_digest='abc',method_config={'method':'fixture'})
            code='''from pathlib import Path
from streambench_fixture import CounterMemory
from test_streambench_runner import chunks
from baselines.streambench.runner import observe_stream
import sys
root=Path(sys.argv[1])
observe_stream(CounterMemory(65536),chunks((2,)),root/'resumed',stream_id='test',manifest_digest='abc',method_config={'method':'fixture'},resume_checkpoint=root/'first/checkpoint_D1')
'''
            env=dict(os.environ, PYTHONPATH=str(Path(__file__).parent)+os.pathsep+str(Path(__file__).parents[3]))
            subprocess.run([sys.executable,'-c',code,str(root)],check=True,env=env,capture_output=True,text=True)
            observe_stream(CounterMemory(65536),chunks(),root/'continuous',stream_id='test',manifest_digest='abc',method_config={'method':'fixture'})
            a=read_checkpoint(root/'resumed/checkpoint_D1_D2')
            b=read_checkpoint(root/'continuous/checkpoint_D1_D2')
            self.assertEqual(a['snapshot']['method'], b['snapshot']['method'])
            self.assertEqual(a['metadata'], b['metadata'])

    def test_resume_provenance_mismatch_and_qa_failure_restore(self):
        self.assertIsNotNone(observe_stream)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            method=CounterMemory(65536)
            observe_stream(method,chunks((1,)),root/'run',stream_id='test',manifest_digest='abc',method_config={'method':'fixture'})
            with self.assertRaisesRegex(ValueError,'manifest'):
                observe_stream(CounterMemory(65536),chunks((2,)),root/'bad',stream_id='test',manifest_digest='changed',method_config={'method':'fixture'},resume_checkpoint=root/'run/checkpoint_D1')
            original=method.snapshot()
            with self.assertRaisesRegex(RuntimeError,'intentional'):
                evaluate_checkpoint(method,root/'run/checkpoint_D1',[{'question_id':'a','source_day':1,'question':'fail'}])
            self.assertEqual(method.snapshot()['method'],original['method'])
            self.assertTrue(torch.equal(method.protocol_state['rng']['torch'], original['protocol']['rng']['torch']))

    def test_reject_nonchronological_chunks(self):
        self.assertIsNotNone(observe_stream)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                observe_stream(CounterMemory(65536),chunks((2,1)),Path(folder),stream_id='test',manifest_digest='abc',method_config={'method':'fixture'})

if __name__=='__main__': unittest.main()

class RNGAndProtocolTests(unittest.TestCase):
    def test_qa_restores_randomness_and_labels_are_not_passed(self):
        import random
        from baselines.streambench.runner import capture_rng
        class RandomMemory(CounterMemory):
            def answer(self, question, choices=None):
                self.calls+=1
                return {'answer':str((random.random(),torch.rand(1).item())), 'query_read_bytes':0}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            method=RandomMemory(65536)
            observe_stream(method,chunks((1,)),root,stream_id='test',manifest_digest='abc',method_config={'method':'fixture'})
            before=capture_rng()
            answers=evaluate_checkpoint(method,root/'checkpoint_D1',[
                {'question_id':'a','source_day':1,'question':'random','answer':'secret label'},
                {'question_id':'b','source_day':1,'question':'random','answer':'another secret label'}])
            self.assertEqual(answers[0]['answer'],answers[1]['answer'])
            after=capture_rng()
            self.assertEqual(before['python'],after['python'])
            self.assertTrue(torch.equal(before['torch'],after['torch']))

    def test_initial_protocol_rng_is_charged_and_unsupported_logged(self):
        from baselines.streambench.method import UnsupportedBudget
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with self.assertRaises(UnsupportedBudget):
                observe_stream(CounterMemory(1024),chunks(),root,stream_id='test',manifest_digest='abc',method_config={'method':'fixture'})
            report=json.loads((root/'summary.json').read_text())
            self.assertEqual(report['status'],'unsupported_under_budget')
            self.assertFalse((root/'checkpoint_D1').exists())

class NoLeakTests(unittest.TestCase):
    def test_runner_rejects_labels_even_for_direct_api_callers(self):
        data=list(chunks((1,)))
        data[0][1][0]['answer']='private ground truth'
        with tempfile.TemporaryDirectory() as folder:
            method=CounterMemory(65536)
            with self.assertRaisesRegex(ValueError,'public'):
                observe_stream(method,iter(data),Path(folder),stream_id='test',manifest_digest='abc',method_config={'method':'fixture'})
            self.assertEqual(method.values,[])

class IndependentStreamTests(unittest.TestCase):
    def test_new_independent_stream_resets_a_reused_lazy_adapter(self):
        class LazyMemory(CounterMemory):
            def initialize(self):
                if not hasattr(self,'values'):super().initialize()
            def reset(self):
                self.protocol_state={}
                self.values=[];self.calls=0
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);method=LazyMemory(65536)
            observe_stream(method,chunks((1,)),root/'first',stream_id='test',manifest_digest='abc',method_config={'method':'fixture'})
            from copy import deepcopy
            expected=deepcopy(method.state_dict())
            observe_stream(method,chunks((1,)),root/'second',stream_id='test',manifest_digest='abc',method_config={'method':'fixture'})
            self.assertEqual(method.state_dict(),expected)
