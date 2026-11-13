"""Official StreamTTT recurrent-state adapter (no replacement memory model).

The native model and processor are optional dependencies loaded only on initialize.
See docs/streambench_recurrent.md for revisions, state layout, and limitations.
"""
from copy import deepcopy
from pathlib import Path
import importlib
import subprocess
import sys

import torch

from baselines.streambench.method import StreamingMemoryMethod, UnsupportedBudget

SOURCE_REVISION = '96aed21cc4d78dc7e1787a53a0a95691f3cfd07a'
MODEL_ID = 'zeyun-zhong/StreamTTT-4B'
MODEL_REVISION = '58f05be5c71549d6c4826325e6d454d2b0b59228'


def released_memory_floor():
    """Analytical BF16, batch-one floor from the pinned checkpoint config.

    Not a measurement of a loaded 4B model. Partial buffers, KV and metadata
    are additional. The trained fast-weight dimensions must not be reduced.
    """
    weights_and_momentum = 36 * 6 * 4 * 640 * 640 * 2
    prefix = 36 * (3 * 2560) * 3 * 2
    partial = 36 * 1023 * (2 * 4 * 640 * 2 + 3 * 4 * 4)
    return {'fast_weights_and_momentum_bytes': weights_and_momentum,
            'convolution_prefix_bytes': prefix,
            'minimum_tensor_bytes': weights_and_momentum + prefix,
            'maximum_partial_buffer_bytes': partial,
            'kv_bytes_per_position': 36 * 2 * 8 * 128 * 2,
            'basis': 'analytical_from_pinned_config_not_runtime_measurement'}


def checked_source(path, revision, package):
    if not path:
        raise RuntimeError(f'{package} requires an explicit upstream_source checkout at {revision}')
    path = Path(path).expanduser().resolve()
    try:
        head = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'],
                              text=True, capture_output=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f'{package} upstream_source is not an accessible Git checkout: {path}') from exc
    if head != revision:
        raise ValueError(f'{package} source revision {head} does not match audited {revision}')
    clean = subprocess.run(['git', '-C', str(path), 'diff', '--quiet', 'HEAD', '--'], capture_output=True)
    if clean.returncode:
        raise ValueError(f'{package} audited upstream checkout has modified tracked files')
    existing = sys.modules.get(package)
    if existing is not None and getattr(existing, '__file__', None):
        if not Path(existing.__file__).resolve().is_relative_to(path):
            raise RuntimeError(f'{package} already imported from another checkout; use a separate process')
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


def move_tree(value, device, _memo=None):
    # Prefix KV tensors occur in both Cambrian sensory and long-term lists.
    # Moving the same tensor twice must not create two physical GPU copies.
    if _memo is None:
        _memo = {}
    if id(value) in _memo:
        return _memo[id(value)]
    if isinstance(value, torch.Tensor):
        result = value.to(device)
        _memo[id(value)] = result
        return result
    if isinstance(value, dict):
        result = {k: move_tree(v, device, _memo) for k, v in value.items()}
        _memo[id(value)] = result
        return result
    if isinstance(value, tuple):
        result = tuple(move_tree(v, device, _memo) for v in value)
        _memo[id(value)] = result
        return result
    if isinstance(value, list):
        result = [move_tree(v, device, _memo) for v in value]
        _memo[id(value)] = result
        return result
    return value


def pack_cache(cache):
    """Expose every continuation field of the pinned DynamicCache layer types."""
    if cache is None:
        return None
    if getattr(cache, 'offloading', False):
        raise NotImplementedError('StreamTTT cache offloading is not audited')
    layers = []
    allowed = {'keys', 'values', 'is_initialized', 'dtype', 'device',
               'sliding_window', 'cumulative_length'}
    for layer in cache.layers:
        name = type(layer).__name__
        if name not in {'DynamicLayer', 'DynamicSlidingWindowLayer'}:
            raise TypeError(f'Unaudited cache layer {name}')
        unknown = set(vars(layer)) - allowed
        if unknown:
            raise TypeError(f'Unaccounted cache fields: {sorted(unknown)}')
        layers.append({'type': name, 'fields': dict(vars(layer))})
    return {'layers': layers}


def unpack_cache(state, device=None):
    if state is None:
        return None
    from transformers.cache_utils import DynamicCache, DynamicLayer, DynamicSlidingWindowLayer
    cache = DynamicCache()
    cache.layers = []
    for entry in state['layers']:
        fields = dict(entry['fields'])
        if entry['type'] == 'DynamicSlidingWindowLayer':
            layer = DynamicSlidingWindowLayer(fields['sliding_window'])
        elif entry['type'] == 'DynamicLayer':
            layer = DynamicLayer()
        else:
            raise ValueError(f"Unknown cache layer {entry['type']}")
        if device is not None:
            fields = move_tree(fields, device)
            if 'device' in fields:
                fields['device'] = torch.device(device)
        layer.__dict__.update(fields)
        cache.layers.append(layer)
    return cache


def compact_states(states):
    # Native split() buffers can retain an entire forward's allocation. Copy
    # their payload, without changing values or the delayed-update schedule.
    if states is None:
        return None
    result = []
    for layer in states:
        if layer is None:
            result.append(None)
            continue
        if len(layer) != 12:
            raise ValueError('Pinned StreamTTT requires all 12 recurrent fields per TTT layer')
        result.append(tuple(t.detach().clone() if t is not None else None for t in layer))
    return result


class StreamTTTMethod(StreamingMemoryMethod):
    def __init__(self, config):
        super().__init__(int(config['memory_budget_bytes']))
        self.config = dict(config)
        self.recent_tokens = int(config.get('recent_tokens', 4096))
        self.window_frames = int(config.get('window_frames', 16))
        if self.recent_tokens < 2 or self.window_frames < 2 or self.window_frames % 2:
            raise ValueError('recent_tokens must be >=2; window_frames must be a positive multiple of 2')
        self.identity = {'source_revision': SOURCE_REVISION,
                         'model': str(config.get('model_path', MODEL_ID)),
                         'model_revision': config.get('model_revision', MODEL_REVISION),
                         'recent_tokens': self.recent_tokens, 'window_frames': self.window_frames,
                         'processor_kwargs': dict(config.get('processor_kwargs', {}))}
        self.minimum_memory_bytes = released_memory_floor()['minimum_tensor_bytes']
        self._model = self._processor = self._windowing = None
        self._clear_state()

    def _clear_state(self):
        self._cache = None
        self._states = None
        self._pending_frames = []
        self._pending_timestamps = []
        self._token_offset = 0
        self._running_pos_max = None
        self._rope_deltas = None
        self.frames_seen = 0
        self.eviction_count = 0

    def initialize(self):
        if self.memory_budget_bytes < self.minimum_memory_bytes:
            raise UnsupportedBudget('StreamTTT fixed fast weights, momentum and convolution prefix exceed B',
                                    memory_budget_bytes=self.memory_budget_bytes,
                                    minimum_memory_bytes=self.minimum_memory_bytes)
        if self._model is not None:
            return
        checked_source(self.config.get('upstream_source'), SOURCE_REVISION, 'streamttt')
        try:
            from transformers import AutoConfig, AutoProcessor
            native = importlib.import_module('streamttt.models.modeling_streamttt_qwen3vl')
            self._windowing = importlib.import_module('streamttt.streaming.windowing')
            path = self.identity['model']
            kwargs = {'revision': self.identity['model_revision'],
                      'local_files_only': self.config.get('local_files_only', True)}
            cfg = AutoConfig.from_pretrained(path, **kwargs)
            tc = cfg.text_config
            expected = {'num_hidden_layers': 36, 'hidden_size': 2560, 'num_fw_heads': 4,
                        'num_fw_kv_heads': 4, 'lact_chunk_size': 1024, 'inter_multi': 1.0,
                        'ttt_momentum': 'headwise', 'ttt_weight_decay': 'headwise'}
            if any(getattr(tc, k) != v for k, v in expected.items()):
                raise ValueError('Checkpoint architecture differs from the audited StreamTTT-4B release')
            tc.sliding_window = self.recent_tokens
            tc.output_states = True
            self._model = native.StreamTTTQwen3VLForConditionalGeneration.from_pretrained(
                path, config=cfg, torch_dtype=torch.bfloat16,
                attn_implementation=self.config.get('attn_implementation', 'flash_attention_2'), **kwargs
            ).eval().to(self.config.get('device', 'cuda'))
            self._processor = AutoProcessor.from_pretrained(path, **kwargs)
            # Also support load_checkpoint() before initialize()/answer().
            self.load_state_dict(self.state_dict())
        except Exception as exc:
            self._model = None
            raise RuntimeError('Could not load the official StreamTTT backend; install its pinned environment '
                               'and cache the pinned checkpoint (no substitute backend is used)') from exc

    def _clear_model_references(self):
        if self._model is None:
            return
        language = self._model.model.language_model
        language._states = [None] * len(language.layers)
        for layer in language.layers:
            if hasattr(layer.self_attn, '_last_states'):
                layer.self_attn._last_states = None
        self._model.model.rope_deltas = self._rope_deltas

    def _forward(self, inputs):
        device = self._model.device
        inputs = move_tree(inputs, device)
        ids = inputs['input_ids']
        local_mask = torch.ones_like(ids)
        positions, self._running_pos_max = self._windowing.recompute_single_window_position(
            self._model, ids, local_mask, inputs.get('video_grid_thw'),
            inputs.get('second_per_grid_ts'), self._model.config.image_token_id,
            self._model.config.video_token_id, self._running_pos_max)
        n = ids.shape[1]
        previous_cached = self._cache.layers[0].keys.shape[-2] if self._cache is not None else 0
        # An unpadded single stream is all-valid. None avoids storing/allocating
        # a mask growing with seven days while the native causal mask uses the
        # cache's cumulative_length and global cache_position.
        inputs.update(attention_mask=None, position_ids=positions, use_cache=True,
                      cache_position=torch.arange(self._token_offset, self._token_offset + n, device=device),
                      past_key_values=self._cache, states=self._states, logits_to_keep=1)
        with torch.inference_mode():
            out = self._model(**inputs)
        self._cache = out.past_key_values
        self._states = compact_states(out.states)
        self._rope_deltas = getattr(out, 'rope_deltas', None)
        self._token_offset += n
        # Clone even tails already below L: DynamicSlidingWindowLayer returns
        # views into a larger allocation, which otherwise remains persistent.
        for layer in self._cache.layers:
            layer.keys = layer.keys[..., -self.recent_tokens:, :].detach().clone()
            layer.values = layer.values[..., -self.recent_tokens:, :].detach().clone()
        self.eviction_count += max(0, previous_cached + n - self._cache.layers[0].keys.shape[-2])
        logits = out.logits[:, -1, :].detach()
        self._clear_model_references()
        return logits

    def _observe_window(self, frames, timestamps):
        video = torch.stack(frames)
        # Qwen's timestamps are frame_indices/fps. A 1 Hz virtual timebase
        # expresses the manifest's global seconds without resampling frames.
        metadata = {'fps': 1., 'frames_indices': [float(t['global_timestamp']) for t in timestamps],
                    'total_num_frames': max(1, int(float(timestamps[-1]['global_timestamp'])) + 1),
                    'video_backend': 'streambench_manifest'}
        kwargs = dict(self.identity['processor_kwargs'])
        forbidden = {'do_sample_frames', 'fps', 'num_frames', 'max_frames', 'video_metadata'} & kwargs.keys()
        if forbidden:
            raise ValueError(f'Processor may not resample the shared manifest: {sorted(forbidden)}')
        inputs = dict(self._processor(text='<|video_pad|>', videos=[video], video_metadata=[metadata],
                                      do_sample_frames=False, return_tensors='pt', **kwargs))
        if self._token_offset == 0:
            prefix, _ = self._windowing.build_prompt_ids(self._processor, '', 'cpu')
            inputs['input_ids'] = torch.cat((prefix, inputs['input_ids']), dim=1)
        allowed = {'input_ids', 'pixel_values_videos', 'video_grid_thw', 'second_per_grid_ts'}
        inputs = {k: v for k, v in inputs.items() if k in allowed}
        inputs['pixel_values_videos'] = inputs['pixel_values_videos'].to(torch.bfloat16)
        self._forward(inputs)

    def observe(self, frames, timestamps):
        if len(frames) != len(timestamps):
            raise ValueError('Each frame requires one timestamp record')
        self.initialize()
        import numpy as np
        for frame, timestamp in zip(frames, timestamps):
            if isinstance(frame, torch.Tensor):
                tensor = frame.detach().cpu().clone()
            else:
                tensor = torch.from_numpy(np.array(frame, copy=True)).permute(2, 0, 1)
            self._pending_frames.append(tensor)
            self._pending_timestamps.append(dict(timestamp))
            self.frames_seen += 1
            if len(self._pending_frames) == self.window_frames:
                self._observe_window(self._pending_frames, self._pending_timestamps)
                self._pending_frames, self._pending_timestamps = [], []

    def finalize_chunk(self):
        count = len(self._pending_frames) // 2 * 2
        if count:
            self._observe_window(self._pending_frames[:count], self._pending_timestamps[:count])
            self._pending_frames = self._pending_frames[count:]
            self._pending_timestamps = self._pending_timestamps[count:]
        return self.enforce_budget()

    def persistent_state(self):
        return {'kv_memory_bytes': pack_cache(self._cache), 'fast_weight_bytes': self._states,
                'visual_memory_bytes': self._pending_frames,
                'metadata_bytes': {'identity': self.identity, 'pending_timestamps': self._pending_timestamps,
                                   'token_offset': self._token_offset, 'running_pos_max': self._running_pos_max,
                                   'rope_deltas': self._rope_deltas, 'frames_seen': self.frames_seen,
                                   'eviction_count': self.eviction_count}}

    def state_dict(self):
        return {'identity': self.identity, 'cache': pack_cache(self._cache), 'states': self._states,
                'pending_frames': self._pending_frames, 'pending_timestamps': self._pending_timestamps,
                'token_offset': self._token_offset, 'running_pos_max': self._running_pos_max,
                'rope_deltas': self._rope_deltas, 'frames_seen': self.frames_seen,
                'eviction_count': self.eviction_count}

    def load_state_dict(self, state):
        if state['identity'] != self.identity:
            raise ValueError('StreamTTT checkpoint model/processor/window identity differs')
        device = self._model.device if self._model is not None else 'cpu'
        self._cache = unpack_cache(state['cache'], device=device)
        self._states = move_tree(state['states'], device)
        self._pending_frames = move_tree(state['pending_frames'], 'cpu')
        self._pending_timestamps = state['pending_timestamps']
        self._token_offset = state['token_offset']
        self._running_pos_max = move_tree(state['running_pos_max'], device)
        self._rope_deltas = move_tree(state['rope_deltas'], device)
        self.frames_seen = state['frames_seen']
        self.eviction_count = state['eviction_count']
        self._clear_model_references()

    def answer(self, question, choices=None):
        if not choices:
            raise NotImplementedError('Official StreamTTT streaming evaluation supports MC logits only; '
                                      'free-form streaming generation is not released')
        self.initialize()
        if self.frames_seen == 0:
            raise ValueError('Observe at least one frame before QA')
        state = self.snapshot()
        report = self.memory_bytes()
        padding = int(len(self._pending_frames) % 2)
        try:
            if self._pending_frames:
                self._observe_window(self._pending_frames, self._pending_timestamps)
                self._pending_frames, self._pending_timestamps = [], []
            labels = list(choices) if isinstance(choices, dict) else [chr(65 + i) for i in range(len(choices))]
            texts = list(choices.values()) if isinstance(choices, dict) else list(choices)
            prompt = (question + '\nOptions:\n' + '\n'.join(f'{k}. {v}' for k, v in zip(labels, texts))
                      + "\nOnly give the best option's letter directly.")
            _, ids = self._windowing.build_prompt_ids(self._processor, prompt, 'cpu', answer_prefix='Answer:')
            logits = self._forward({'input_ids': ids})
            token_ids = [self._processor.tokenizer(': ' + label).input_ids[-1] for label in labels]
            selected = logits[0, token_ids].argmax().item()
            positions = sum(layer.keys.shape[-2] for layer in self._cache.layers)
            return {'answer': labels[selected],
                    'query_read_bytes': report['kv_memory_bytes'] + report['fast_weight_bytes'] + report['visual_memory_bytes'],
                    'query_read_bytes_is_upper_bound': True,
                    'num_kv_positions_used': positions, 'num_retrieved_items': None,
                    'qa_temporal_padding_frames': padding, 'read_accounting': 'full_recurrent_state_and_recent_kv'}
        finally:
            self.restore(state)

    def reset(self):
        self.protocol_state = {}
        self._clear_state()
        self._clear_model_references()


def create_method(config):
    return StreamTTTMethod(config)
