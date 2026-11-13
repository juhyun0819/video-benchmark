import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from .runner import run_stream, load_inputs
from baselines.ours_offline.inputs import sha256
from .stream_frames import sample_plan


class FakeNative:
    """Deliberately mutates QA state, proving the runner restores its branch."""
    def __init__(self, fail_answer=False):
        self.ids = []
        self.protocol_state = {}
        self.memory = SimpleNamespace(frames_seen=0)
        self.last_timestamp = None
        self.fail_answer = fail_answer
    def initialize(self): pass
    def observe(self, frames, timestamps):
        for meta in timestamps:
            self.ids.append(meta['global_frame_id'])
            self.memory.frames_seen += 1
            self.last_timestamp = dict(meta)
    def finalize_chunk(self): pass
    def enforce_budget(self): return {'persistent_total_bytes': len(self.ids) * 8}
    def state_dict(self):return {'ids': self.ids, 'frames_seen': self.memory.frames_seen, 'last': self.last_timestamp}
    def snapshot(self):return copy.deepcopy({'method': self.state_dict(), 'protocol': self.protocol_state})
    def restore(self, saved):
        state = copy.deepcopy(saved['method'])
        self.ids, self.memory.frames_seen, self.last_timestamp = state['ids'], state['frames_seen'], state['last']
        self.protocol_state = copy.deepcopy(saved['protocol'])
    def save_checkpoint(self, path):
        path.mkdir();(path / 'state.json').write_text(json.dumps(self.snapshot()))
    def load_checkpoint(self, path):self.restore(json.loads((path / 'state.json').read_text()))
    def answer(self, question, choices):
        self.ids.append('QA_MUTATION')
        if self.fail_answer:raise RuntimeError('fixture failure')
        return {'answer': 'A', 'retrieved_frame_ids': [x for x in self.ids if isinstance(x, int)]}


def decode(root, samples):
    return [None] * len(samples), [{'global_frame_id': s['global_frame_id'],
        'global_timestamp': s['requested_global_timestamp'], 'day': s['day']} for s in samples]


def fixtures():
    sources = [{'path': 'd1', 'day': 1, 'start_time': 10, 'duration': 2, 'fps': 24, 'frames': 48},
               {'path': 'd2', 'day': 2, 'start_time': 86410, 'duration': 2, 'fps': 24, 'frames': 48}]
    questions = [{'qa_id': 'a', 'day': 1, 'query_time': 10.5, 'question': 'q1', 'choices': ['yes', 'no'], 'answer': 'A'},
                 {'qa_id': 'b', 'day': 2, 'query_time': 86410, 'question': 'q2', 'choices': ['yes', 'no'], 'answer': 'A'}]
    return sources, questions


class ContinuousTests(unittest.TestCase):
    def test_sampling_cannot_exceed_raw_frame_rate_or_use_boolean(self):
        sources, questions = fixtures()
        with tempfile.TemporaryDirectory() as tmp:
            for fps in [25, True, float('nan')]:
                with self.assertRaisesRegex(ValueError, 'fps'):
                    run_stream(FakeNative(), root=tmp, questions=questions, sources=sources,
                        identity={}, output=Path(tmp) / 'run', decoder=decode, fps=fps)

    def test_causal_boundaries_and_day_transition_without_reset(self):
        sources, questions = fixtures()
        with tempfile.TemporaryDirectory() as tmp:
            method = FakeNative();out = Path(tmp) / 'run'
            status = run_stream(method, root=tmp, questions=questions, sources=sources,
                identity={'test': 'fixture'}, output=out, decoder=decode, chunk_frames=8)
            rows = [json.loads(l) for l in (out / 'predictions.jsonl').read_text().splitlines()]
            self.assertEqual(status['status'], 'complete')
            self.assertEqual(method.ids, [0, 1, 2, 3])  # Includes Day2 tail after last QA.
            self.assertEqual(rows[0]['actual_frame_timestamps'], [10.0])
            self.assertEqual(rows[1]['actual_frame_timestamps'], [10.0, 11.0, 86410.0])
            self.assertTrue(all(r['qa_state_restored'] for r in rows))
            self.assertTrue(all(not r['native_answer_state_unchanged'] for r in rows))
            self.assertTrue(all(r['correct'] for r in rows))

    def test_checkpoint_resume_matches_continuous_and_preserves_predictions(self):
        sources, questions = fixtures()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);full = FakeNative();split = FakeNative();resumed = FakeNative()
            common = dict(root=root, questions=questions, sources=sources, identity={'source': 'fixed'}, decoder=decode)
            run_stream(full, output=root / 'full', **common)
            run_stream(split, output=root / 'part', max_questions=1, **common)
            run_stream(resumed, output=root / 'resumed', resume=root / 'part/checkpoint', **common)
            self.assertEqual(full.state_dict(), resumed.state_dict())
            rows = [json.loads(l) for l in (root / 'resumed/predictions.jsonl').read_text().splitlines()]
            self.assertEqual([r['qa_id'] for r in rows], ['a', 'b'])
            self.assertEqual(rows[0], json.loads((root / 'part/predictions.jsonl').read_text()))

    def test_resume_rejects_changed_identity(self):
        sources, questions = fixtures()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            common = dict(root=root, questions=questions, sources=sources, decoder=decode)
            run_stream(FakeNative(), identity={'source': 'a'}, output=root / 'part', max_questions=1, **common)
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                run_stream(FakeNative(), identity={'source': 'b'}, output=root / 'bad', resume=root / 'part/checkpoint', **common)
            self.assertEqual(json.loads((root / 'bad/status.json').read_text())['status'], 'failed')

    def test_resume_rejects_edited_prior_score(self):
        sources, questions = fixtures()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            common = dict(root=root, questions=questions, sources=sources, decoder=decode, identity={'source': 'same'})
            run_stream(FakeNative(), output=root / 'part', max_questions=1, **common)
            path = root / 'part/predictions.jsonl'
            row = json.loads(path.read_text());row['correct'] = False
            path.write_text(json.dumps(row) + '\n')
            with self.assertRaisesRegex(ValueError, 'prediction row differs'):
                run_stream(FakeNative(), output=root / 'bad', resume=root / 'part/checkpoint', **common)

    def test_qa_exception_restores_state_and_writes_error(self):
        sources, questions = fixtures()
        with tempfile.TemporaryDirectory() as tmp:
            method = FakeNative(fail_answer=True);out = Path(tmp) / 'run'
            with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                run_stream(method, root=tmp, questions=questions, sources=sources,
                    identity={}, output=out, decoder=decode)
            self.assertEqual(method.ids, [0])
            row = json.loads((out / 'predictions.jsonl').read_text())
            self.assertEqual(row['status'], 'error')
            self.assertEqual(json.loads((out / 'status.json').read_text())['completed_qa'], 0)

    def test_future_decoded_frame_is_rejected_before_observation(self):
        sources, questions = fixtures()
        def future(root, samples):
            frames, meta = decode(root, samples)
            meta[0]['global_timestamp'] += 1
            return frames, meta
        with tempfile.TemporaryDirectory() as tmp:
            method = FakeNative()
            with self.assertRaisesRegex(ValueError, 'future frame'):
                run_stream(method, root=tmp, questions=questions, sources=sources,
                    identity={}, output=Path(tmp) / 'run', decoder=future)
            self.assertEqual(method.ids, [])

    def test_verified_input_loader_retains_earlier_days_for_later_qa(self):
        sources, questions = fixtures()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for source in sources:
                source['fps'] = 24
                path = root / source['path'];path.write_bytes(b'video fixture')
                st = path.stat();source['file_state'] = [st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]
            manifest = root / 'manifest.json';index = root / 'sources.json'
            scorer = Path(__file__).parents[1] / 'streambench/scoring.py'
            manifest.write_text(json.dumps({'scorer_sha256': sha256(scorer), 'questions': questions}))
            index.write_text(json.dumps({'sources': sources}))
            selected, retained, _ = load_inputs(root, manifest, index, [2])
            self.assertEqual([q['qa_id'] for q in selected], ['b'])
            self.assertEqual([s['day'] for s in retained], [1, 2])
            (root / 'd1').write_bytes(b'changed source')
            with self.assertRaisesRegex(ValueError, 'Source changed'):
                load_inputs(root, manifest, index, [2])

    def test_generic_timeline_without_day_and_mixed_native_rates(self):
        sources = [{'path': 'first', 'start_time': 12.5, 'duration': 1, 'fps': 24, 'frames': 24},
                   {'path': 'second', 'start_time': 50, 'duration': 1, 'fps': 30, 'frames': 30}]
        plan = sample_plan(sources, fps=4)
        self.assertEqual([r['source_frame_index'] for r in plan], [0, 6, 12, 18, 0, 7, 15, 22])
        self.assertEqual(plan[5]['requested_global_timestamp'], 50+7/30)
        questions = [{'qa_id': 'one', 'query_time': 50.24, 'question': 'What?', 'choices': ['a', 'b'], 'answer': 'A'}]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'run';method = FakeNative()
            run_stream(method, root=tmp, questions=questions, sources=sources, identity={}, output=out,
                       decoder=decode, fps=4)
            row = json.loads((out/'predictions.jsonl').read_text())
            self.assertEqual(row['actual_frame_timestamps'], [r['requested_global_timestamp'] for r in plan[:6]])
            self.assertEqual(method.ids, list(range(8)))

    def test_missing_qa_preflight_does_not_touch_gpu_or_load_model(self):
        from unittest.mock import patch
        from contextlib import redirect_stdout
        import io
        from .__main__ import main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);video = root/'clip';video.write_bytes(b'fixture')
            st = video.stat()
            source = {'path': 'clip', 'start_time': 7, 'duration': 1, 'fps': 30, 'frames': 30,
                      'file_state': [st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]}
            index = root/'sources.json';index.write_text(json.dumps({'sources': [source]}))
            config = root/'config.json';config.write_text(json.dumps({'method': 'cambrian_s',
                'model_path': str(root/'missing-model'), 'upstream_source': str(root/'missing-upstream'),
                'memory_budget_bytes': 1024}))
            with patch('subprocess.check_output', side_effect=AssertionError('Must not touch GPU')), redirect_stdout(io.StringIO()):
                result = main(['--root', str(root), '--sources', str(index), '--config', str(config),
                               '--output', str(root/'plan')])
            self.assertEqual(result['status'], 'awaiting_qa')
            self.assertFalse(result['model_loaded'])
            self.assertFalse(result['stream_observed'])
            self.assertFalse(result['checkpoint_created'])

    def test_raw_decoder_preserves_fractional_fps_frame_identity(self):
        try:
            import av
            from PIL import Image
        except ImportError:
            self.skipTest('PyAV/Pillow required for CPU raw-video fixture')
        from fractions import Fraction
        from .stream_frames import decode_window
        rate = Fraction(30000, 1001)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with av.open(str(root/'clip.mp4'), mode='w') as container:
                stream = container.add_stream('mpeg4', rate=rate)
                stream.width = stream.height = 16
                stream.pix_fmt = 'yuv420p'
                for i in range(30):
                    frame = av.VideoFrame.from_image(Image.new('RGB', (16, 16), (i*8, 0, 0)))
                    for packet in stream.encode(frame):container.mux(packet)
                for packet in stream.encode():container.mux(packet)
            sources = [{'path': 'clip.mp4', 'start_time': 12.5, 'duration': float(30/rate),
                'fps': float(rate), 'fps_fraction': str(rate), 'frames': 30}]
            samples = sample_plan(sources, fps=4)
            frames, records = decode_window(root, samples)
            self.assertEqual(len(frames), 5)
            self.assertEqual([r['source_frame_index'] for r in records], [0, 7, 14, 22, 29])
            for sample, record in zip(samples, records):
                self.assertAlmostEqual(record['global_timestamp'], sample['requested_global_timestamp'])
                self.assertLessEqual(record['global_timestamp'], sample['requested_global_timestamp']+1e-8)


if __name__ == '__main__':unittest.main()
