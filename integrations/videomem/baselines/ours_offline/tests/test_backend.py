import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from baselines.ours_offline.backend import NativeVideoBackend, prepare_video_inputs, verify_model_ledger

REVISION='1'*40


class CaptureProcessor:
    video_processor=SimpleNamespace(patch_size=16,merge_size=2,temporal_patch_size=2,
                                    size={'shortest_edge':131072,'longest_edge':786432})
    def apply_chat_template(self,messages,**kwargs):
        self.messages=messages;return 'native-template'
    def __call__(self,**kwargs):
        self.kwargs=kwargs;return kwargs


class NativeInputTests(unittest.TestCase):
    def setUp(self):
        self.frames=[np.zeros((4,4,3),dtype=np.uint8),np.ones((4,4,3),dtype=np.uint8)]

    def test_qwen_preserves_absolute_gap_metadata_and_video_sequence(self):
        p=CaptureProcessor()
        result=prepare_video_inputs(p,'qwen3_vl',self.frames,[10.4,86405.1],'Q',['a','b'])
        self.assertEqual(result['videos_kwargs']['video_metadata'][0]['frames_indices'],[10.4,86405.1])
        self.assertEqual(result['videos_kwargs']['video_metadata'][0]['fps'],1.)
        self.assertFalse(result['videos_kwargs']['do_sample_frames'])
        self.assertEqual(result['videos'][0].shape,(2,4,4,3))
        self.assertEqual(p.messages[0]['content'][0],{'type':'video'})
        self.assertNotIn('images',result)

    def test_llava_uses_video_processor_without_qwen_only_metadata(self):
        p=CaptureProcessor();result=prepare_video_inputs(p,'llava_onevision',self.frames,[1,2],'Q',['a','b'])
        self.assertFalse(result['do_sample_frames']);self.assertNotIn('video_metadata',result)
        with self.assertRaises(ValueError):prepare_video_inputs(p,'llava_onevision',self.frames,[1,2],'Q',['a','b'],max_pixels=4096)

    def test_repeated_time_or_implicitly_resized_frames_rejected(self):
        with self.assertRaises(ValueError):prepare_video_inputs(CaptureProcessor(),'qwen3_vl',self.frames,[1,1],'Q',['a','b'])
        with self.assertRaises(ValueError):prepare_video_inputs(CaptureProcessor(),'qwen3_vl',[self.frames[0],np.zeros((2,2,3),dtype=np.uint8)],[1,2],'Q',['a','b'])

    def test_qwen_budget_routes_to_video_size_and_respects_patch_minimum(self):
        result=prepare_video_inputs(CaptureProcessor(),'qwen3_vl',self.frames,[1,2],'Q',['a','b'],max_pixels=4096)
        self.assertEqual(result['videos_kwargs']['size'],{'shortest_edge':4096,'longest_edge':4096})
        self.assertNotIn('max_pixels',result)
        with self.assertRaisesRegex(ValueError,'minimum padded video patch budget'):
            prepare_video_inputs(CaptureProcessor(),'qwen3_vl',self.frames,[1,2],'Q',['a','b'],max_pixels=1024)

    def test_native_qwen_video_processor_honors_explicit_budget(self):
        try:
            from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor,Qwen3VLProcessorKwargs
            from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
        except ImportError:
            self.skipTest('Optional pinned Transformers CPU processor environment unavailable')
        frames=[np.zeros((64,64,3),dtype=np.uint8)]*2
        result=prepare_video_inputs(CaptureProcessor(),'qwen3_vl',frames,[1,2],'Q',['a','b'],max_pixels=4096)
        native=Qwen3VLProcessor.__new__(Qwen3VLProcessor)
        routed=native._merge_kwargs(Qwen3VLProcessorKwargs,tokenizer_init_kwargs={},
            **{k:v for k,v in result.items() if k not in ('text','videos')})
        self.assertEqual(routed['videos_kwargs']['size'],{'shortest_edge':4096,'longest_edge':4096})
        self.assertFalse(routed['videos_kwargs']['do_sample_frames'])
        self.assertEqual(routed['videos_kwargs']['video_metadata'][0]['frames_indices'],[1.,2.])
        output=Qwen3VLVideoProcessor()(videos=result['videos'],**routed['videos_kwargs'])
        self.assertEqual(output['video_grid_thw'].tolist(),[[1,2,2]])
        self.assertEqual(output['video_metadata'][0].frames_indices,[1.,2.])

    def test_version_pin_fails_before_loading_weights(self):
        b=NativeVideoBackend({'backbone':'qwen3_vl','model_path':'unused','model_revision':REVISION})
        with patch('baselines.ours_offline.backend.importlib.metadata.version',return_value='wrong'):
            with self.assertRaisesRegex(RuntimeError,'transformers'):b.initialize()


class LedgerTests(unittest.TestCase):
    def fixture(self,directory):
        root=Path(directory)/'model';root.mkdir()
        (root/'config.json').write_text(json.dumps({'model_type':'qwen3_vl'}));(root/'model.safetensors').write_bytes(b'fixture-not-real-weights')
        entries=[{'path':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in root.iterdir()]
        ledger=Path(directory)/'ledger.json';ledger.write_text(json.dumps({'status':'verified','repo_id':'Qwen/Qwen3-VL-8B-Instruct','model_path':str(root),'revision':REVISION,'verified_files':entries}))
        return root,ledger

    def test_rehash_and_unverified_artifact_detection(self):
        with tempfile.TemporaryDirectory() as d:
            root,ledger=self.fixture(d)
            self.assertEqual(verify_model_ledger(root,ledger,'qwen3_vl',REVISION)['verified_files_count'],2)
            (root/'tokenizer_config.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Unverified'):verify_model_ledger(root,ledger,'qwen3_vl',REVISION)

    def test_modified_weight_and_wrong_model_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root,ledger=self.fixture(d)
            with self.assertRaisesRegex(ValueError,'mismatch'):verify_model_ledger(root,ledger,'llava_onevision',REVISION)
            path=root/'model.safetensors';data=path.read_bytes();path.write_bytes(b'X'+data[1:])
            with self.assertRaisesRegex(ValueError,'hash changed'):verify_model_ledger(root,ledger,'qwen3_vl',REVISION)

    def test_native_chat_templates_must_be_covered_by_ledger(self):
        for relative in ('chat_template.jinja','chat_templates/tool_use.jinja'):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as d:
                root,ledger=self.fixture(d)
                path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text('{{ messages }}')
                with self.assertRaisesRegex(ValueError,'Unverified'):
                    verify_model_ledger(root,ledger,'qwen3_vl',REVISION)
                data=json.loads(ledger.read_text())
                data['verified_files'].append({'path':relative,'bytes':path.stat().st_size,
                    'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
                ledger.write_text(json.dumps(data))
                self.assertEqual(verify_model_ledger(root,ledger,'qwen3_vl',REVISION)['verified_files_count'],3)
