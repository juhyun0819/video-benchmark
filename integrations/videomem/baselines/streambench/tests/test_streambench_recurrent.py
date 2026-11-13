"""CPU state/policy tests; these do not claim pretrained model accuracy."""
import copy
import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
from transformers.cache_utils import DynamicCache, DynamicSlidingWindowLayer

from baselines.streambench.adapters.streamttt import (
    StreamTTTMethod, pack_cache, unpack_cache, released_memory_floor, move_tree,
)
from baselines.streambench.adapters.cambrian_s import RecallMemory, CambrianSMethod
from baselines.streambench.method import UnsupportedBudget


def kv(value, length=6):
    return [(torch.full((1, 1, length, 2), float(value)),
             torch.full((1, 1, length, 2), float(value + 1)))]


class RecurrentStateTests(unittest.TestCase):
    def test_device_transfer_preserves_shared_prefix_tensor(self):
        prefix = torch.ones(3)
        moved = move_tree({'runtime': prefix, 'long_term': prefix}, 'meta')
        self.assertIs(moved['runtime'], moved['long_term'])

    def test_streamttt_budget_rejects_before_loading_weights(self):
        method = StreamTTTMethod({'memory_budget_bytes': 512 * 2**20})
        with self.assertRaises(UnsupportedBudget) as ctx:
            method.initialize()
        self.assertGreaterEqual(ctx.exception.minimum_memory_bytes, 675 * 2**20)
        self.assertEqual(released_memory_floor()['fast_weights_and_momentum_bytes'], 707788800)

    def test_sliding_cache_restore_keeps_seen_length_and_physical_tail(self):
        cache = DynamicCache()
        cache.layers = [DynamicSlidingWindowLayer(sliding_window=4)]
        cache.update(torch.arange(12.).reshape(1, 1, 6, 2), torch.ones(1, 1, 6, 2), 0)
        saved = pack_cache(cache)
        restored = unpack_cache(copy.deepcopy(saved), device='cpu')
        self.assertEqual(restored.get_seq_length(), 6)
        self.assertEqual(restored.layers[0].keys.shape[-2], 3)
        for value in [cache, restored]:
            value.update(torch.full((1, 1, 1, 2), 42.), torch.zeros(1, 1, 1, 2), 0)
        self.assertTrue(torch.equal(cache.layers[0].keys, restored.layers[0].keys))
        self.assertEqual(restored.layers[0].get_mask_sizes(torch.tensor([7])), (4, 4))

    def test_all_twelve_ttt_buffers_and_pending_frame_survive_snapshot(self):
        method = StreamTTTMethod({'memory_budget_bytes': 2**30})
        state = method.state_dict()
        state['states'] = [tuple(torch.full((2, 3), float(i)) for i in range(12))]
        state['pending_frames'] = [torch.zeros(3, 2, 2, dtype=torch.uint8)]
        state['pending_timestamps'] = [{'day': 1, 'global_frame_id': 8, 'global_timestamp': 8.}]
        state['token_offset'] = 77
        state['running_pos_max'] = torch.tensor([[[55]]])
        method.load_state_dict(state)
        snap = method.snapshot()
        method._states[0][9].fill_(99)
        method.restore(snap)
        self.assertEqual(method.state_dict()['states'][0][9][0, 0].item(), 9)
        self.assertEqual(method.state_dict()['token_offset'], 77)
        self.assertEqual(len(method.state_dict()['pending_frames']), 1)
        self.assertGreaterEqual(method.memory_bytes()['fast_weight_bytes'], 12 * 2 * 3 * 4)

    def test_streamttt_freeform_is_explicitly_unsupported(self):
        method = StreamTTTMethod({'memory_budget_bytes': 2**30})
        with self.assertRaises(NotImplementedError):
            method.answer('What happened?')

    def test_recall_promotes_oldest_sensory_and_pools_only_low_surprise(self):
        memory = RecallMemory(1, sensory_window_size=1, surprise_threshold=.5,
                              compression_downsample_ratio=2, consolidation_mem_budget=100)
        memory.initialize(kv(0, 2))
        memory.add_frame(kv(10), .1, 10, (2, 2))
        memory.add_frame(kv(20), .9, 11, (2, 2))
        self.assertEqual(memory.runtime[0]['frame_ids'], [[], [11]])
        self.assertEqual(memory.long_term[0]['frame_ids'], [[], [10]])
        self.assertEqual(memory.long_term[0]['lengths'], [2, 1])
        self.assertEqual(memory.long_term[0]['key_states'][1][0, 0, 0, 0].item(), 10)
        self.assertEqual(memory.compression_count, 1)

    def test_recall_drops_least_surprising_and_keeps_prefix(self):
        memory = RecallMemory(1, sensory_window_size=1, surprise_threshold=.5,
                              compression_downsample_ratio=1, consolidation_mem_budget=15)
        memory.initialize(kv(0, 2))
        for fid, score in [(1, .8), (2, .1), (3, .9), (4, .2)]:
            memory.add_frame(kv(fid), score, fid, (2, 2))
        self.assertEqual(memory.long_term[0]['frame_ids'], [[], [1], [3]])
        self.assertEqual(memory.eviction_count, 1)

    def test_recall_merge_preserves_provenance_and_native_average(self):
        memory = RecallMemory(1, sensory_window_size=1, surprise_threshold=.5,
                              compression_downsample_ratio=1, consolidation_method='drop_merge',
                              consolidation_mem_budget=9)
        memory.initialize(kv(0, 2))
        for fid, score in [(1, .8), (2, .9), (3, .1)]:
            memory.add_frame(kv(fid * 10), score, fid, (2, 2))
        # Upstream drop_merge merges first, then always drops one if triggered.
        self.assertEqual(memory.long_term[0]['frame_ids'], [[]])
        self.assertEqual(memory.merge_count, 1)
        self.assertEqual(memory.eviction_count, 1)

    def test_recall_resume_preserves_predictor_and_byte_identical_updates(self):
        memory = RecallMemory(1, sensory_window_size=2, compression_downsample_ratio=1,
                              consolidation_mem_budget=14)
        memory.initialize(kv(0, 2))
        memory.add_frame(kv(1), .8, 1, (2, 2))
        memory.prediction = torch.arange(18.).reshape(1, 6, 3)
        with tempfile.TemporaryDirectory() as tmp:
            torch.save(memory.state_dict(), tmp + '/day1.pt')
            restored = RecallMemory(1, sensory_window_size=2, compression_downsample_ratio=1,
                                    consolidation_mem_budget=14)
            restored.load_state_dict(torch.load(tmp + '/day1.pt', weights_only=False))
        for target in [memory, restored]:
            target.add_frame(kv(2), .2, 2, (2, 2))
            target.add_frame(kv(3), .9, 3, (2, 2))
        self.assertTrue(torch.equal(memory.prediction, restored.prediction))
        self.assertEqual(memory.runtime[0]['frame_ids'], restored.runtime[0]['frame_ids'])
        self.assertEqual(memory.long_term[0]['frame_ids'], restored.long_term[0]['frame_ids'])
        for a, b in zip(memory.runtime[0]['key_states'], restored.runtime[0]['key_states']):
            self.assertTrue(torch.equal(a, b))

    def test_qa_flush_is_a_copy_and_does_not_duplicate_live_sensory(self):
        memory = RecallMemory(1, sensory_window_size=2, compression_downsample_ratio=1,
                              consolidation_mem_budget=100)
        memory.initialize(kv(0, 2))
        memory.add_frame(kv(1), .8, 1, (2, 2))
        flushed = memory.for_query()
        self.assertEqual(memory.long_term[0]['frame_ids'], [[]])
        self.assertEqual(flushed.long_term[0]['frame_ids'], [[], [1]])
        flushed.long_term[0]['lengths'].append(99)
        self.assertEqual(memory.long_term[0]['lengths'], [2])

    def test_cambrian_budget_eviction_retains_sensory_and_prediction(self):
        method = CambrianSMethod({'memory_budget_bytes': 3000, 'sensory_window_size': 1,
                                 'compression_downsample_ratio': 1})
        method.memory = RecallMemory(1, sensory_window_size=1, compression_downsample_ratio=1,
                                     consolidation_mem_budget=10000)
        method.memory.initialize(kv(0, 2))
        for frame in range(10):
            method.memory.add_frame(kv(frame, 72), .1 * frame, frame, (8, 8))
        method.memory.prediction = torch.ones(1, 72, 2)
        method.finalize_chunk()
        self.assertLessEqual(method.memory_bytes()['persistent_total_bytes'], 3000)
        self.assertEqual(method.memory.runtime[0]['frame_ids'][-1], [9])
        self.assertEqual(method.memory.prediction.sum().item(), 144)

    def test_cambrian_reenforces_when_protocol_metadata_grows_at_day_boundary(self):
        method = CambrianSMethod({'memory_budget_bytes': 5000, 'sensory_window_size': 1,
                                 'compression_downsample_ratio': 1})
        method.memory = RecallMemory(1, sensory_window_size=1, compression_downsample_ratio=1,
                                     consolidation_mem_budget=10000)
        method.memory.initialize(kv(0, 2))
        for frame in range(10):
            method.memory.add_frame(kv(frame, 72), .1 * frame, frame, (8, 8))
        method.memory.prediction = torch.ones(1, 72, 2)
        method.finalize_chunk()
        old_evictions = method.eviction_count
        method.protocol_state['new_day_metadata'] = 'x' * 1500
        method.enforce_budget()
        self.assertLessEqual(method.memory_bytes()['persistent_total_bytes'], 5000)
        self.assertGreater(method.eviction_count, old_evictions)

    def test_recall_resume_in_new_python_process(self):
        memory = RecallMemory(1, sensory_window_size=1, compression_downsample_ratio=1,
                              consolidation_mem_budget=15)
        memory.initialize(kv(0, 2))
        memory.add_frame(kv(1), .8, 1, (2, 2))
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = str(Path(tmp) / 'state.pt')
            torch.save(memory.state_dict(), checkpoint)
            script = '''
import sys, torch
from baselines.streambench.adapters.cambrian_s import RecallMemory
state = torch.load(sys.argv[1], weights_only=False)
memory = RecallMemory(1, sensory_window_size=1, compression_downsample_ratio=1, consolidation_mem_budget=15)
memory.load_state_dict(state)
keys = torch.full((1, 1, 6, 2), 2.)
memory.add_frame([(keys, keys + 1)], .2, 2, (2, 2))
torch.save(memory.state_dict(), sys.argv[1])
'''
            subprocess.run([sys.executable, '-c', script, checkpoint], check=True,
                           cwd=Path(__file__).resolve().parents[3], capture_output=True)
            resumed = torch.load(checkpoint, weights_only=False)
        memory.add_frame(kv(2), .2, 2, (2, 2))
        self.assertEqual(resumed['long_term'][0]['frame_ids'], [[], [1]])
        self.assertEqual(resumed['runtime'][0]['frame_ids'], [[], [2]])
        self.assertTrue(torch.equal(resumed['runtime'][0]['key_states'][-1], memory.runtime[0]['key_states'][-1]))

    @unittest.skipUnless(os.environ.get('STREAMBENCH_STREAMTTT_SOURCE'), 'Optional pinned upstream source not configured')
    def test_official_fast_weight_partial_update_survives_serialization(self):
        # Execute the actual upstream definitions, avoiding the unrelated broken
        # torchvision import in this host. No replacement recurrence or norm.
        import math
        import transformers
        from einops import rearrange
        from typing import Optional
        from baselines.streambench.adapters.streamttt import checked_source, SOURCE_REVISION
        source = checked_source(os.environ['STREAMBENCH_STREAMTTT_SOURCE'], SOURCE_REVISION, 'streamttt')
        namespace = {'torch': torch, 'nn': torch.nn, 'F': torch.nn.functional,
                     'math': math, 'Optional': Optional, 'rearrange': rearrange, 'causal_conv1d_fn': None}
        norm_path = Path(transformers.__file__).parent / 'models/qwen3_next/modeling_qwen3_next.py'
        norm = next(node for node in ast.parse(norm_path.read_text()).body
                    if isinstance(node, ast.ClassDef) and node.name == 'Qwen3NextRMSNormGated')
        exec(compile(ast.Module(body=[norm], type_ignores=[]), str(norm_path), 'exec'), namespace)
        fast_path = source / 'streamttt/models/fast_weight_block.py'
        definitions = [node for node in ast.parse(fast_path.read_text()).body
                       if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(fast_path), 'exec'), namespace)
        config = SimpleNamespace(hidden_size=8, num_fw_heads=2, num_fw_kv_heads=2,
                                 lact_chunk_size=4, use_residual=True, hidden_act='silu',
                                 ttt_weight_decay='headwise', rms_norm_eps=1e-6)
        torch.manual_seed(5)
        block = namespace['FastWeightBlock'](config).eval()
        block._init_weights()
        first, second = torch.randn(1, 3, 8), torch.randn(1, 2, 8)
        with torch.no_grad():
            _, first_state = block(first)
            self.assertEqual(len(first_state), 12)
            self.assertEqual(first_state[4].shape[-1], 3)
            with tempfile.TemporaryDirectory() as tmp:
                torch.save(first_state, tmp + '/state.pt')
                restored = torch.load(tmp + '/state.pt', weights_only=False)
            expected_output, expected_state = block(second, prev_states=first_state)
            actual_output, actual_state = block(second, prev_states=restored)
        self.assertTrue(torch.equal(expected_output, actual_output))
        for expected, actual in zip(expected_state, actual_state):
            self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(actual_state[4].shape[-1], 1)

    @unittest.skipUnless(os.environ.get('STREAMBENCH_STREAMTTT_SOURCE'), 'Optional pinned upstream source not configured')
    def test_official_multimodal_model_cache_and_ttt_resume_together(self):
        from transformers import Qwen3VLConfig
        from baselines.streambench.adapters.streamttt import checked_source, SOURCE_REVISION
        checked_source(os.environ['STREAMBENCH_STREAMTTT_SOURCE'], SOURCE_REVISION, 'streamttt')
        from streamttt.models.modeling_streamttt_qwen3vl import StreamTTTQwen3VLForConditionalGeneration
        config = Qwen3VLConfig(
            text_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                             num_attention_heads=2, num_key_value_heads=1, head_dim=8, vocab_size=64,
                             sliding_window=4, num_fw_heads=2, num_fw_kv_heads=2, lact_chunk_size=4,
                             use_residual=True, ttt_weight_decay='headwise', ttt_momentum='headwise',
                             output_states=True, rope_scaling={'rope_type': 'default',
                                 'mrope_section': [1, 1, 2], 'mrope_interleaved': True}),
            vision_config=dict(hidden_size=16, out_hidden_size=16, intermediate_size=32,
                               depth=2, num_heads=2, patch_size=2, temporal_patch_size=2,
                               spatial_merge_size=2, deepstack_visual_indexes=[0, 1], num_position_embeddings=16),
            image_token_id=60, video_token_id=61, vision_start_token_id=58, vision_end_token_id=59)
        config._attn_implementation = 'eager'
        torch.manual_seed(9)
        model = StreamTTTQwen3VLForConditionalGeneration(config).eval()
        model._custom_init()
        with torch.inference_mode():
            first = model(input_ids=torch.tensor([[1, 2, 3]]),
                          position_ids=torch.arange(3).reshape(1, 1, -1).expand(3, 1, -1),
                          cache_position=torch.arange(3), use_cache=True)
            with tempfile.TemporaryDirectory() as tmp:
                torch.save({'cache': pack_cache(first.past_key_values), 'states': first.states}, tmp + '/state.pt')
                saved = torch.load(tmp + '/state.pt', weights_only=False)
            inputs = dict(input_ids=torch.tensor([[58, 61, 61, 61, 61, 59]]),
                          pixel_values_videos=torch.randn(16, 24), video_grid_thw=torch.tensor([[1, 4, 4]]),
                          position_ids=torch.arange(3, 9).reshape(1, 1, -1).expand(3, 1, -1),
                          cache_position=torch.arange(3, 9), use_cache=True)
            direct = model(**inputs, past_key_values=first.past_key_values, states=first.states)
            resumed = model(**inputs, past_key_values=unpack_cache(saved['cache']), states=saved['states'])
        self.assertTrue(torch.equal(direct.logits, resumed.logits))
        self.assertEqual(resumed.past_key_values.get_seq_length(), 9)
        for layer_a, layer_b in zip(direct.states, resumed.states):
            for a, b in zip(layer_a, layer_b):
                self.assertTrue(torch.equal(a, b))


if __name__ == '__main__':
    unittest.main()

class LocalOnlyLoadingTests(unittest.TestCase):
    def test_cambrian_missing_offline_mode_fails_before_native_builder(self):
        from unittest.mock import patch
        from huggingface_hub import constants
        method=CambrianSMethod({'memory_budget_bytes':1024**3,'upstream_source':'unused','local_files_only':True})
        with patch('baselines.streambench.adapters.cambrian_s.checked_source',return_value=Path('/unused')):
            with patch.object(constants,'HF_HUB_OFFLINE',False):
                with self.assertRaisesRegex(RuntimeError,'HF_HUB_OFFLINE'):
                    method.initialize()
