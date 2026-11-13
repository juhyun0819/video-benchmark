# Portions adapted from Cambrian-S/lmms-eval, Copyright (c) 2025 LMMs-Lab.
# Licensed under Apache-2.0; see ../vendor/cambrian_s/LICENSE and
# ../vendor/cambrian_s/LMMS_EVAL_LICENSE. Changes: incremental lifecycle,
# complete memory serialization/accounting, byte cap, and QA isolation.
"""Cambrian-S-7B-LFP + native Recall Memory, incrementally driven.

RecallMemory adapts the Apache-2.0 implementation in cambrian-mllm/cambrian-s,
lmms-eval/lmms_eval/models/simple/cambrians_vsr.py, revision below. Frozen
encoder/LLM/LFP and query retrieval use the official checkout, unchanged.
"""
from copy import deepcopy
import importlib
import importlib.util
import math
from pathlib import Path
import types

import torch
import torch.nn.functional as F

from baselines.streambench.method import StreamingMemoryMethod, UnsupportedBudget
from .streamttt import checked_source, move_tree

SOURCE_REVISION = '058b1b4c96846a229d1c9bb69da6ecf5ceee4a6b'
MODEL_ID = 'nyu-visionx/Cambrian-S-7B-LFP'
MODEL_REVISION = 'd04f69849c1ddbf7e62a7790f2461ef549d4df8e'
FIELDS = ('key_states', 'value_states', 'modalities', 'lengths', 'surprising_scores', 'frame_ids')


def _empty_layer():
    return {key: [] for key in FIELDS}


class RecallMemory:
    """Native sensory FIFO, surprise pooling, consolidation, and provenance.

    The five upstream fields retain their original semantics. frame_ids adds
    accounted benchmark provenance; prediction is the next-frame LFP output.
    """
    def __init__(self, num_layers, sensory_window_size=32, surprise_threshold=.35,
                 compression_downsample_ratio=2, consolidation_method='drop',
                 consolidation_mem_budget=16384):
        if sensory_window_size < 1 or compression_downsample_ratio < 1 or consolidation_mem_budget < 1:
            raise ValueError('Finite positive sensory, pooling, and consolidation capacities are required')
        if consolidation_method not in {'drop', 'drop_merge'}:
            raise ValueError('Only official drop/drop_merge consolidation is supported')
        self.num_layers = num_layers
        self.sensory_window_size = sensory_window_size
        self.surprise_threshold = surprise_threshold
        self.compression_downsample_ratio = compression_downsample_ratio
        self.consolidation_method = consolidation_method
        self.consolidation_mem_budget = consolidation_mem_budget
        self.runtime = [_empty_layer() for _ in range(num_layers)]
        self.long_term = [_empty_layer() for _ in range(num_layers)]
        self.prediction = None
        self.grid_shape = None
        self.frames_seen = 0
        self.compression_count = self.merge_count = self.eviction_count = 0
        self.events = []

    def initialize(self, prefix_kv):
        if len(prefix_kv) != self.num_layers:
            raise ValueError('Prefix KV layer count mismatch')
        for index, (keys, values) in enumerate(prefix_kv):
            for cache in (self.runtime, self.long_term):
                cache[index]['key_states'].append(keys)
                cache[index]['value_states'].append(values)
                cache[index]['modalities'].append('T')
                cache[index]['lengths'].append(keys.size(2))
                cache[index]['surprising_scores'].append(1.)
                cache[index]['frame_ids'].append([])

    @staticmethod
    def _pop(layer, index):
        return {key: layer[key].pop(index) for key in FIELDS}

    @staticmethod
    def _append(layer, entry):
        for key in FIELDS:
            layer[key].append(entry[key])

    def _compress(self, entry, layer_index):
        ratio = self.compression_downsample_ratio
        if ratio > 1 and entry['surprising_scores'] < self.surprise_threshold:
            height, width = self.grid_shape
            for key in ('key_states', 'value_states'):
                tensor = entry[key]
                shape = tensor.shape
                tensor = tensor.flatten(0, 1).unflatten(1, (height, width + 1)).permute(0, 3, 1, 2)
                tensor = F.avg_pool2d(tensor, kernel_size=ratio, stride=ratio)
                entry[key] = tensor.flatten(2, 3).unflatten(0, shape[:2]).permute(0, 1, 3, 2).contiguous()
            entry['lengths'] = entry['key_states'].size(2)
            if layer_index == 0:
                self.compression_count += 1
                self.events.append({'compression_event': True, 'frame_ids': entry['frame_ids'][:],
                                    'memory_decision': 'pool_low_surprise'})
        return entry

    def _merge_at(self, layer, index, layer_index):
        for key in ('key_states', 'value_states', 'surprising_scores'):
            layer[key][index] = (layer[key][index] + layer[key][index + 1]) / 2.
        layer['frame_ids'][index] = layer['frame_ids'][index] + layer['frame_ids'][index + 1]
        self._pop(layer, index + 1)
        if layer_index == 0:
            self.merge_count += 1
            self.events.append({'compression_event': True, 'memory_decision': 'merge_adjacent_surprising',
                                'frame_ids': layer['frame_ids'][index][:]})

    def _can_merge(self, layer, index):
        return (index > 0 and index < len(layer['lengths']) - 1
                and layer['surprising_scores'][index] >= self.surprise_threshold
                and layer['surprising_scores'][index + 1] >= self.surprise_threshold)

    def _merge_all(self, layer, layer_index):
        index = 1
        while index < len(layer['lengths']) - 1:
            if self._can_merge(layer, index):
                self._merge_at(layer, index, layer_index)
            else:
                index += 1

    def _drop_one(self, layer, layer_index):
        if len(layer['lengths']) < 2:
            return False
        # Python min, like np.argmin upstream, takes the earliest tie.
        index = min(range(1, len(layer['lengths'])), key=lambda i: layer['surprising_scores'][i])
        entry = self._pop(layer, index)
        if layer_index == 0:
            self.eviction_count += 1
            self.events.append({'eviction_event': True, 'memory_decision': 'drop_least_surprising',
                                'frame_ids': entry['frame_ids'][:],
                                'frame_surprise_score': entry['surprising_scores']})
        # Native code indexes the gap with argmin relative to scores[1:].
        gap_index = index - 1
        if self.consolidation_method == 'drop_merge' and self._can_merge(layer, gap_index):
            self._merge_at(layer, gap_index, layer_index)
        return True

    def _consolidate(self, layer_index):
        layer = self.long_term[layer_index]
        if sum(layer['lengths']) <= self.consolidation_mem_budget:
            return
        if self.consolidation_method == 'drop_merge':
            self._merge_all(layer, layer_index)
        # Preserve upstream's always-drop-on-trigger and strict '<' stopping
        # condition, including after merge_all has brought it below capacity.
        while self._drop_one(layer, layer_index):
            if sum(layer['lengths']) < self.consolidation_mem_budget:
                break

    def add_frame(self, frame_kv, score, frame_id, grid_shape):
        if len(frame_kv) != self.num_layers:
            raise ValueError('Frame KV layer count mismatch')
        if self.grid_shape is not None and tuple(grid_shape) != self.grid_shape:
            raise ValueError('Recall Memory requires a stable unpadded frame grid; use one consistent aspect ratio')
        self.grid_shape = tuple(grid_shape)
        for index, (keys, values) in enumerate(frame_kv):
            entry = {'key_states': keys, 'value_states': values, 'modalities': 'I',
                     'lengths': keys.size(2), 'surprising_scores': float(score), 'frame_ids': [frame_id]}
            self._append(self.runtime[index], entry)
            if len(self.runtime[index]['lengths']) > self.sensory_window_size + 1:
                old = self._pop(self.runtime[index], 1)
                self._append(self.long_term[index], self._compress(old, index))
                self._consolidate(index)
        self.frames_seen += 1
        self.events.append({'frame_id': frame_id, 'frame_surprise_score': float(score),
                            'memory_decision': 'sensory'})

    def evict_for_bytes(self):
        """One native consolidation step across layers; never discard sensory."""
        if not self.long_term or len(self.long_term[0]['lengths']) < 2:
            return False
        for index, layer in enumerate(self.long_term):
            if self.consolidation_method == 'drop_merge':
                self._merge_all(layer, index)
            self._drop_one(layer, index)
        return True

    def for_query(self):
        query = deepcopy(self)
        query.events = []
        for index, layer in enumerate(query.runtime):
            for item in range(1, len(layer['lengths'])):
                entry = {key: layer[key][item] for key in FIELDS}
                query._append(query.long_term[index], query._compress(entry, index))
                query._consolidate(index)
        return query

    def state_dict(self):
        # Audit output is drained after each chunk and never used by memory or
        # QA. It is not part of the scientific persistent memory/checkpoint.
        return {key: value for key, value in vars(self).items() if key != 'events'}

    def load_state_dict(self, state):
        identity_fields = ('num_layers', 'sensory_window_size', 'surprise_threshold',
                           'compression_downsample_ratio', 'consolidation_method', 'consolidation_mem_budget')
        if any(getattr(self, key) != state[key] for key in identity_fields):
            raise ValueError('Recall Memory policy differs from checkpoint')
        self.__dict__.update(state)
        self.events = []


class CambrianSMethod(StreamingMemoryMethod):
    def __init__(self, config):
        super().__init__(int(config['memory_budget_bytes']))
        self.config = dict(config)
        self.policy = {key: config.get(key, default) for key, default in
                       [('sensory_window_size', 32), ('surprise_threshold', .35),
                        ('compression_downsample_ratio', 2), ('consolidation_method', 'drop'),
                        ('consolidation_mem_budget', 16384)]}
        self.identity = {'model': str(config.get('model_path', MODEL_ID)),
                         'model_revision': config.get('model_revision', MODEL_REVISION),
                         'source_revision': SOURCE_REVISION, 'policy': self.policy,
                         'retrieval_topk': int(config.get('retrieval_topk', 32)),
                         'miv_token_len': 64, 'si_token_len': 729}
        self.memory = RecallMemory(28, **self.policy)
        self._model = self._tokenizer = self._processors = None
        self._conversation = self._tokenizer_image_token = self._unpad = None
        self.last_timestamp = None
        self.minimum_memory_bytes = None

    def initialize(self):
        if self._model is not None:
            return
        if self.config.get('local_files_only', True):
            from huggingface_hub import constants as hub_constants
            if not hub_constants.HF_HUB_OFFLINE:
                raise RuntimeError('Local-only Cambrian loading requires HF_HUB_OFFLINE=1 before process startup; '
                                   'the native builder also loads a separate SigLIP vision tower')
        source = checked_source(self.config.get('upstream_source'), SOURCE_REVISION, 'cambrian')
        try:
            builder = importlib.import_module('cambrian.model.builder')
            conv = importlib.import_module('cambrian.conversation')
            utils = importlib.import_module('cambrian.mm_utils')
            arch = importlib.import_module('cambrian.model.cambrian_arch')
            path = self.identity['model']
            if not Path(path).is_dir():
                from huggingface_hub import snapshot_download
                path = snapshot_download(path, revision=self.identity['model_revision'],
                                         local_files_only=self.config.get('local_files_only', True))
            device = self.config.get('device', 'cuda:0')
            self._tokenizer, self._model, self._processors, _ = builder.load_pretrained_model(
                path, None, Path(path).name, device_map=device, device=device)
            self._model.eval()
            cfg = self._model.config
            expected = {'num_hidden_layers': 28, 'hidden_size': 3584, 'num_key_value_heads': 4,
                        'nfp_head': True, 'miv_token_len': 64}
            if any(getattr(cfg, key) != value for key, value in expected.items()):
                raise ValueError('Checkpoint differs from the audited Cambrian-S-7B-LFP release')
            spec = importlib.util.spec_from_file_location('streambench_cambrian_native_patch',
                                                          source / 'lmms-eval/qwen2_monkey_patch.py')
            patch = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(patch)
            for layer in self._model.model.layers:
                layer.self_attn.__class__ = patch.Qwen2SdpaAttention
            # Equivalent native patch, scoped to this model rather than globally
            # replacing the class method for every imported Cambrian instance.
            self._model.model.forward = types.MethodType(patch.cambrian_qwen2_forward, self._model.model)
            self._conversation = conv.conv_templates['qwen_2']
            self._tokenizer_image_token = utils.tokenizer_image_token
            self._unpad = arch.unpad_image
        except Exception as exc:
            self._model = None
            raise RuntimeError('Could not load official Cambrian-S; use its isolated transformers==4.37.0 '
                               'environment and cache the pinned LFP checkpoint') from exc
        if not self.memory.runtime[0]['lengths']:
            ids = self._prompt_ids('')
            marker = torch.where(ids[0] == -200)[0][0]
            with torch.inference_mode():
                out = self._forward(input_ids=ids[:, :marker])
            self.memory.initialize(out.past_key_values)
        else:
            self.memory.load_state_dict(move_tree(self.memory.state_dict(), self._model.device))
        self.enforce_budget()

    def _prompt_ids(self, question):
        conversation = self._conversation.copy()
        conversation.append_message(conversation.roles[0], '<image>\n' + question)
        conversation.append_message(conversation.roles[1], None)
        return self._tokenizer_image_token(conversation.get_prompt(), self._tokenizer, -200,
                                           return_tensors='pt').unsqueeze(0).to(self._model.device)

    def _forward(self, input_ids=None, inputs_embeds=None, past_key_values=None):
        return self._model(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=None,
                           position_ids=None, past_key_values=past_key_values, use_cache=True,
                           output_attentions=False, output_hidden_states=True, return_dict=True)

    def _encode_frame(self, frame):
        if isinstance(frame, torch.Tensor):
            from PIL import Image
            array = frame.detach().cpu()
            if array.ndim == 3 and array.shape[0] == 3:
                array = array.permute(1, 2, 0)
            frame = Image.fromarray(array.numpy().astype('uint8'), 'RGB')
        pixels = self._processors[0].preprocess([frame], return_tensors='pt')['pixel_values']
        pixels = pixels.to(device=self._model.device, dtype=torch.float16)
        vit = self._model.encode_images([pixels])[0]
        projected = self._model.get_model().mm_projector(vit)
        side = math.isqrt(vit.shape[1])
        if side * side != vit.shape[1]:
            raise ValueError('Unexpected non-square SigLIP token grid')
        def resize(value):
            value = value.unflatten(1, (side, side)).permute(0, 3, 1, 2)
            value = F.interpolate(value, size=(8, 8), mode='bilinear', align_corners=False)
            return self._unpad(value.permute(0, 2, 3, 1), frame.size)
        return resize(projected), resize(vit)

    def observe(self, frames, timestamps):
        if len(frames) != len(timestamps):
            raise ValueError('Each frame requires one timestamp')
        self.initialize()
        with torch.inference_mode():
            for frame, timestamp in zip(frames, timestamps):
                projected, target = self._encode_frame(frame)
                grid = tuple(projected.shape[1:3])
                if self.memory.grid_shape is not None and self.memory.grid_shape != grid:
                    raise ValueError('Input aspect ratio changed the native Recall Memory pooling grid')
                if self.memory.prediction is None:
                    score = 1.
                else:
                    pred = self.memory.prediction.unflatten(1, (target.size(1), target.size(2) + 1))[:, :, :-1]
                    score = 1 - torch.cosine_similarity(pred.flatten(1, 2), target.flatten(1, 2), dim=-1).mean(1).item()
                newline = self._model.model.image_newline[None, None, None, :].expand(*projected.shape[:2], 1, -1)
                embeddings = torch.cat((projected, newline), dim=2).flatten(1, 2)
                past = [(torch.cat(layer['key_states'], dim=2), torch.cat(layer['value_states'], dim=2))
                        for layer in self.memory.runtime]
                out = self._forward(inputs_embeds=embeddings, past_key_values=past)
                self.memory.prediction = self._model.model.nfp_head(out.hidden_states).detach().clone()
                fresh = [(keys[..., old[0].shape[2]:, :].clone(), values[..., old[1].shape[2]:, :].clone())
                         for old, (keys, values) in zip(past, out.past_key_values)]
                if self.minimum_memory_bytes is None:
                    prefix_bytes = sum(layer['key_states'][0].numel() * layer['key_states'][0].element_size()
                                       + layer['value_states'][0].numel() * layer['value_states'][0].element_size()
                                       for layer in self.memory.runtime)
                    frame_bytes = sum(key.numel() * key.element_size() + val.numel() * val.element_size()
                                      for key, val in fresh)
                    self.minimum_memory_bytes = (prefix_bytes + self.policy['sensory_window_size'] * frame_bytes
                                                 + self.memory.prediction.numel() * self.memory.prediction.element_size())
                    if self.memory_budget_bytes < self.minimum_memory_bytes:
                        raise UnsupportedBudget('Configured Cambrian sensory KV and next-frame predictor exceed B',
                                                memory_budget_bytes=self.memory_budget_bytes,
                                                minimum_memory_bytes=self.minimum_memory_bytes)
                self.memory.add_frame(fresh, score, int(timestamp['global_frame_id']), grid)
                self.last_timestamp = dict(timestamp)
                self._enforce_native_budget()

    def _enforce_native_budget(self):
        # Chunk audit events are output-only and drained by the common runner.
        while self.memory_bytes()['persistent_total_bytes'] > self.memory_budget_bytes:
            if not self.memory.evict_for_bytes():
                self.minimum_memory_bytes = self.memory_bytes()['persistent_total_bytes']
                return super().enforce_budget()
        return super().enforce_budget()

    def enforce_budget(self):
        # The common runner updates cumulative day/RNG metadata after finalize.
        # Give the same native policy a chance to make room at that boundary.
        return self._enforce_native_budget()

    def finalize_chunk(self):
        return self._enforce_native_budget()

    def drain_events(self):
        events, self.memory.events = self.memory.events, []
        return events

    @property
    def compression_count(self):
        return self.memory.compression_count + self.memory.merge_count

    @property
    def eviction_count(self):
        return self.memory.eviction_count

    def persistent_state(self):
        caches = (self.memory.runtime, self.memory.long_term)
        return {'kv_memory_bytes': [[(layer['key_states'], layer['value_states']) for layer in cache] for cache in caches],
                'visual_memory_bytes': self.memory.prediction,
                'metadata_bytes': {'identity': self.identity, 'last_timestamp': self.last_timestamp,
                    'caches': [[{k: v for k, v in layer.items() if k not in {'key_states', 'value_states'}}
                                for layer in cache] for cache in caches],
                    'memory': {k: v for k, v in self.memory.state_dict().items()
                               if k not in {'runtime', 'long_term', 'prediction'}}}}

    def state_dict(self):
        return {'identity': self.identity, 'memory': self.memory.state_dict(),
                'last_timestamp': self.last_timestamp, 'minimum_memory_bytes': self.minimum_memory_bytes}

    def load_state_dict(self, state):
        if state['identity'] != self.identity:
            raise ValueError('Cambrian checkpoint model/policy identity differs')
        device = self._model.device if self._model is not None else 'cpu'
        self.memory = RecallMemory(state['memory']['num_layers'], **self.policy)
        self.memory.load_state_dict(move_tree(state['memory'], device))
        self.last_timestamp = state['last_timestamp']
        self.minimum_memory_bytes = state['minimum_memory_bytes']

    def answer(self, question, choices=None):
        self.initialize()
        if not self.memory.frames_seen:
            raise ValueError('Observe at least one frame before QA')
        if choices:
            values = choices.items() if isinstance(choices, dict) else zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ', choices)
            question += '\n' + '\n'.join(f'{key}. {value}' for key, value in values)
            question += "\nAnswer with the best option's letter."
        query = self.memory.for_query()
        past = [(torch.cat(layer['key_states'], dim=2), torch.cat(layer['value_states'], dim=2))
                for layer in query.long_term]
        retrieval_topk = self.identity['retrieval_topk']
        # Native attention owns ranking and selection. Hooks compute the same
        # already-produced query/key scores only to report selected provenance.
        retrieved = {}
        handles = []
        try:
            if retrieval_topk > 1:
                for index, layer in enumerate(self._model.model.layers):
                    attention = layer.self_attn
                    attention.use_retrieval = True
                    attention.retrieval_topk = retrieval_topk
                    attention.cache_modalities = query.long_term[index]['modalities'][:]
                    attention.cache_lengths = query.long_term[index]['lengths'][:]
                    def trace(module, args, kwargs, i=index):
                        if not hasattr(module, 'use_retrieval'):
                            return
                        hidden = kwargs.get('hidden_states', args[0] if args else None)
                        q = module.q_proj(hidden).view(1, hidden.shape[1], module.num_heads, module.head_dim).transpose(1, 2)
                        keys = query.long_term[i]['key_states']
                        from transformers.models.qwen2.modeling_qwen2 import repeat_kv
                        representations = torch.cat([repeat_kv(key, module.num_key_value_groups).mean(2).flatten(1, 2)
                                                     for key in keys], dim=0)
                        scores = F.cosine_similarity(q.mean(2).flatten(1, 2), representations, dim=-1)
                        mask = scores.new_tensor([int(m == 'I') for m in query.long_term[i]['modalities']])
                        chosen = torch.topk(scores * mask, min(retrieval_topk, len(keys))).indices.tolist()
                        retrieved[i] = [j for j in range(len(keys)) if j in chosen or query.long_term[i]['modalities'][j] == 'T']
                    handles.append(attention.register_forward_pre_hook(trace, with_kwargs=True))
            ids = self._prompt_ids(question)
            marker = torch.where(ids[0] == -200)[0][0]
            with torch.inference_mode():
                out = self._forward(input_ids=ids[:, marker + 1:], past_key_values=past)
                pred = out.logits[:, -1, :].argmax(-1)
                # Match native greedy decoding including pad and repetition penalty.
                generated = torch.cat((torch.full_like(pred[:, None], self._tokenizer.pad_token_id), pred[:, None]), dim=1)
                for _ in range(int(self.config.get('max_new_tokens', 16)) - 1):
                    if pred.item() == self._tokenizer.eos_token_id:
                        break
                    out = self._forward(input_ids=generated[:, -1:], past_key_values=out.past_key_values)
                    logits = out.logits[:, -1, :]
                    scores = torch.gather(logits, 1, generated)
                    logits.scatter_(1, generated, torch.where(scores < 0, scores * 1.1, scores / 1.1))
                    pred = logits.argmax(-1)
                    generated = torch.cat((generated, pred[:, None]), dim=1)
            if retrieval_topk <= 1:
                retrieved = {i: list(range(len(layer['lengths']))) for i, layer in enumerate(query.long_term)}
            selected_ids = sorted({fid for i, selected in retrieved.items() for j in selected
                                   for fid in query.long_term[i]['frame_ids'][j]})
            selected_bytes = sum((query.long_term[i]['key_states'][j].numel()
                                  + query.long_term[i]['value_states'][j].numel())
                                 * query.long_term[i]['key_states'][j].element_size()
                                 for i, selected in retrieved.items() for j in selected)
            # Ranking scans every long-term key. Report this separately from
            # the smaller KV payload retained for answer attention.
            scanned_keys = sum(key.numel() * key.element_size() for layer in query.long_term for key in layer['key_states'])
            return {'answer': self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip(),
                    # Native construction concatenates all stored K and V
                    # before ranking, so R is not merely selected attention KV.
                    'query_read_bytes': self.memory_bytes()['kv_memory_bytes'],
                    'retrieved_attention_bytes': selected_bytes,
                    'retrieval_scan_key_bytes': scanned_keys, 'retrieved_frame_ids': selected_ids,
                    'retrieved_items': selected_ids,
                    'num_retrieved_items': len(selected_ids),
                    'num_kv_positions_used': sum(query.long_term[i]['lengths'][j] for i, js in retrieved.items() for j in js)}
        finally:
            for handle in handles:
                handle.remove()
            for layer in self._model.model.layers:
                for key in ('use_retrieval', 'retrieval_topk', 'cache_modalities', 'cache_lengths'):
                    if hasattr(layer.self_attn, key):
                        delattr(layer.self_attn, key)

    def reset(self):
        self.protocol_state = {}
        self.memory = RecallMemory(28, **self.policy)
        self.last_timestamp = None
        self.minimum_memory_bytes = None
        # Prefix is rebuilt on the first subsequent observe/QA.
        if self._model is not None:
            ids = self._prompt_ids('')
            marker = torch.where(ids[0] == -200)[0][0]
            with torch.inference_mode():
                self.memory.initialize(self._forward(input_ids=ids[:, :marker]).past_key_values)


def create_method(config):
    return CambrianSMethod(config)
