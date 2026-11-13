"""Lazy official Hugging Face raw-video inference, without feature/KV reuse."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re

from .inputs import sha256

BACKBONES = {
    'qwen3_vl': {'repo_id':'Qwen/Qwen3-VL-8B-Instruct', 'model_type':'qwen3_vl'},
    'llava_onevision': {'repo_id':'llava-hf/llava-onevision-qwen2-7b-ov-hf', 'model_type':'llava_onevision'},
}


def verify_model_ledger(model_path, ledger_path, backbone, revision):
    """Rehash trusted local HF artifact inventory; never fetch model files.

    A ledger must itself come from a trusted download/verification workflow.
    Local hashes establish identity to that ledger, not remote authenticity.
    """
    if backbone not in BACKBONES or not re.fullmatch(r'[0-9a-fA-F]{40}', revision or ''):
        raise ValueError('Known backbone and immutable 40-character model revision required')
    root = Path(model_path).resolve()
    ledger = json.loads(Path(ledger_path).read_text())
    if ledger.get('status') != 'verified' or ledger.get('repo_id') != BACKBONES[backbone]['repo_id'] or ledger.get('revision') != revision:
        raise ValueError('Model ledger status/repository/revision mismatch')
    if Path(ledger.get('model_path', '')).resolve() != root:
        raise ValueError('Model ledger belongs to a different local directory')
    verified, total = set(), 0
    entries = ledger.get('verified_files')
    if not isinstance(entries, list) or not entries:
        raise ValueError('Empty model verification ledger')
    for record in entries:
        relative = Path(record.get('path', ''))
        if relative.is_absolute() or '..' in relative.parts or str(relative) in verified:
            raise ValueError('Duplicate or unsafe model artifact path')
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Missing or escaped model artifact')
        size = path.stat().st_size
        if size != record.get('bytes'):
            raise ValueError(f'Artifact size changed: {relative}')
        expected = record.get('sha256') or record.get('hash')
        if not isinstance(expected, str) or len(expected) not in (40,64):
            raise ValueError('Artifact digest must be SHA256 or Git blob SHA1')
        digest = hashlib.sha256() if len(expected) == 64 else hashlib.sha1()
        if len(expected) == 40:
            digest.update(f'blob {size}\0'.encode())
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024**2), b''):
                digest.update(block)
        if digest.hexdigest() != expected.lower():
            raise ValueError(f'Artifact hash changed: {relative}')
        verified.add(str(relative)); total += size
    # All files potentially consumed by the local native loader must be covered.
    artifacts = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
                 and not any(part.startswith('.') for part in p.relative_to(root).parts)
                 and p.suffix in {'.json','.safetensors','.bin','.model','.txt','.tiktoken','.jinja'}}
    if not artifacts <= verified or 'config.json' not in verified:
        raise ValueError(f'Unverified model-loader artifacts: {sorted(artifacts-verified)}')
    weights = sorted(p for p in verified if p.endswith(('.safetensors','.bin')))
    if not weights:
        raise ValueError('Ledger contains no model weights')
    for name in verified:
        if name.endswith('.index.json'):
            mapping = json.loads((root/name).read_text()).get('weight_map', {})
            if not mapping or not set(mapping.values()) <= verified:
                raise ValueError('Model weight index references unverified shards')
    config = json.loads((root/'config.json').read_text())
    if config.get('model_type') != BACKBONES[backbone]['model_type']:
        raise ValueError('Local model config has wrong backbone family')
    return {'repo_id':ledger['repo_id'], 'revision':revision,
            'ledger_sha256':sha256(ledger_path), 'verified_files_count':len(verified),
            'verified_bytes':total, 'verified_weight_files':weights}


def frame_times(timestamps):
    result = []
    for record in timestamps:
        value = record.get('actual_global_timestamp',record.get('global_timestamp')) if isinstance(record,dict) else record
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value < 0:
            raise ValueError('Each frame needs a finite global timestamp')
        result.append(float(value))
    if any(a >= b for a,b in zip(result,result[1:])):
        raise ValueError('Frame timestamps must be strictly chronological')
    return result


def public_messages(question, choices):
    if not isinstance(question,str) or not question.strip():
        raise ValueError('Question must be nonempty text')
    if not isinstance(choices,(list,tuple)) or not 2 <= len(choices) <= 26 or any(not isinstance(x,str) or not x.strip() for x in choices):
        raise ValueError('Expected multiple-choice option strings')
    text = question + '\nOptions:\n' + '\n'.join(f'{chr(65+i)}. {x}' for i,x in enumerate(choices))
    text += '\nAnswer with only the capital letter of the correct choice.'
    return [{'role':'user','content':[{'type':'video'},{'type':'text','text':text}]}]


def prepare_video_inputs(processor, backbone, frames, timestamps, question, choices, max_pixels=None):
    """Use each family's native video path after one explicit external sampling."""
    import numpy as np
    if backbone not in BACKBONES or len(frames) != len(timestamps) or len(frames) < 2:
        raise ValueError('Known backbone and at least two timestamped frames required')
    times = frame_times(timestamps)
    arrays = [np.asarray(frame.convert('RGB') if hasattr(frame,'convert') else frame) for frame in frames]
    if any(x.ndim != 3 or x.shape[-1] != 3 or x.dtype != np.uint8 for x in arrays):
        raise ValueError('Expected unnormalized RGB uint8 original frames')
    if len({x.shape for x in arrays}) != 1:
        raise ValueError('Native single-video inputs require consistent frame dimensions; no hidden resize is applied')
    prompt = processor.apply_chat_template(public_messages(question,choices), tokenize=False, add_generation_prompt=True)
    kwargs = {'text':[prompt], 'videos':[np.stack(arrays)], 'return_tensors':'pt', 'do_sample_frames':False}
    if backbone == 'qwen3_vl':
        # Virtual 1-Hz indices encode the exact wall-clock frame timestamps.
        # Native temporal pairs, DeepStack, spatial resizing and RoPE remain intact.
        # A structured modality dictionary must contain every video option:
        # HF ignores flat video options once videos_kwargs is supplied.
        kwargs.pop('do_sample_frames')
        kwargs['videos_kwargs'] = {'do_sample_frames':False,
            'video_metadata':[{'fps':1.0, 'frames_indices':times,
            'total_num_frames':max(1,math.floor(times[-1])+1),'video_backend':'ours_raw_pts'}]}
        if max_pixels is not None:
            if type(max_pixels) is not int or max_pixels < 1024:
                raise ValueError('Explicit Qwen max_pixels must be >=1024')
            # HF 4.57.6 flat max_pixels is image-only. Video smart_resize uses
            # size as a clip-wide T*H*W pixel budget, not a per-frame edge size.
            video_processor = processor.video_processor
            factor = video_processor.patch_size * video_processor.merge_size
            temporal = video_processor.temporal_patch_size
            padded_frames = math.ceil(len(frames) / temporal) * temporal
            if max_pixels < padded_frames * factor**2:
                raise ValueError('Qwen max_pixels is below the minimum padded video patch budget')
            minimum = int(video_processor.size['shortest_edge'])
            if minimum < 1:
                raise ValueError('Invalid native Qwen minimum video pixel budget')
            kwargs['videos_kwargs']['size'] = {
                'shortest_edge':min(minimum,max_pixels), 'longest_edge':max_pixels}
    elif max_pixels is not None:
        raise ValueError('max_pixels is Qwen-specific; preserve the native LLaVA video processor')
    return processor(**kwargs)


class NativeVideoBackend:
    def __init__(self, config):
        self.config = dict(config)
        self.backbone = self.config.get('backbone')
        if self.backbone not in BACKBONES:
            raise ValueError('backbone must be qwen3_vl or llava_onevision')
        if not self.config.get('model_path') or not re.fullmatch(r'[0-9a-fA-F]{40}', self.config.get('model_revision','')):
            raise ValueError('Local model_path and immutable model_revision required')
        self.config.setdefault('device','cuda:0')
        self.config.setdefault('dtype','bfloat16')
        self.config.setdefault('attn_implementation','sdpa')
        self.config.setdefault('max_new_tokens',256)
        self.config.setdefault('transformers_version','4.57.6')
        if type(self.config['max_new_tokens']) is not int or self.config['max_new_tokens'] < 1:
            raise ValueError('max_new_tokens must be positive')
        if self.backbone != 'qwen3_vl' and (self.config.get('max_pixels') is not None or self.config.get('channels_last_3d')):
            raise ValueError('Qwen-only processor/Conv3D settings supplied to LLaVA')
        self.model = self.processor = None

    def initialize(self):
        if self.model is not None:
            return
        if importlib.metadata.version('transformers') != self.config['transformers_version']:
            raise RuntimeError(f"Use explicitly configured transformers=={self.config['transformers_version']}")
        import torch
        from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration, LlavaOnevisionForConditionalGeneration
        path = self.config['model_path']
        native_config = AutoConfig.from_pretrained(path,local_files_only=True,trust_remote_code=False)
        if native_config.model_type != BACKBONES[self.backbone]['model_type']:
            raise ValueError('Configured native model family mismatch')
        self.processor = AutoProcessor.from_pretrained(path,local_files_only=True,trust_remote_code=False)
        cls = Qwen3VLForConditionalGeneration if self.backbone == 'qwen3_vl' else LlavaOnevisionForConditionalGeneration
        self.model = cls.from_pretrained(path,local_files_only=True,trust_remote_code=False,
            dtype=getattr(torch,self.config['dtype']), attn_implementation=self.config['attn_implementation'],
            device_map={'':self.config['device']}).eval()
        if self.config.get('channels_last_3d'):
            layer = self.model.model.visual.patch_embed.proj
            layer.to(memory_format=torch.channels_last_3d)
            layer.register_forward_pre_hook(lambda module,values:(values[0].contiguous(memory_format=torch.channels_last_3d),))

    def answer(self, frames, timestamps, question, choices):
        import torch
        self.initialize()
        inputs = prepare_video_inputs(self.processor,self.backbone,frames,timestamps,question,choices,self.config.get('max_pixels'))
        inputs = inputs.to(self.config['device'])
        for key,value in inputs.items():
            if isinstance(value,torch.Tensor) and value.is_floating_point():
                inputs[key] = value.to(self.model.dtype)
        metadata = {k:inputs[k].tolist() for k in ('video_grid_thw','image_sizes') if k in inputs}
        prefix = inputs['input_ids'].shape[1]
        if self.backbone == 'qwen3_vl':
            self.model.model.rope_deltas = None
        with torch.inference_mode():
            generated = self.model.generate(**inputs,do_sample=False,num_beams=1,
                max_new_tokens=self.config['max_new_tokens'],use_cache=True)
        tokens = generated[0,prefix:].tolist()
        text = self.processor.batch_decode([tokens],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos,list) else [eos]
        complete = bool(text.strip() and tokens and tokens[-1] in eos)
        return {'prediction':text.strip(),'raw_output':text,'generated_token_ids':tokens,
                'complete':complete,'input_tokens':prefix,'model_inputs_metadata':metadata}
