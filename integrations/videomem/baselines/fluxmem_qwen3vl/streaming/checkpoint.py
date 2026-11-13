"""Verify the local Qwen3-VL backbone without altering checkpoint files."""
import hashlib
import json
from pathlib import Path

MODEL_ID = 'Qwen/Qwen3-VL-8B-Instruct'
REVISION = '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'


def verify_weights(model_path, marker_path):
    root = Path(model_path).resolve()
    marker = json.loads(Path(marker_path).read_text())
    if marker['repo_id'] != MODEL_ID or marker['revision'] != REVISION:
        raise ValueError('Checkpoint identity is not the pinned Qwen3-VL-8B-Instruct')
    verified = set()
    for item in marker['verified_files']:
        path = (root/item['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Verification path escapes checkpoint directory')
        if path.stat().st_size != item['bytes']:
            raise RuntimeError('Checkpoint byte count changed: '+item['path'])
        h = hashlib.sha256() if len(item['hash']) == 64 else hashlib.sha1()
        if len(item['hash']) == 40:
            h.update(b'blob '+str(item['bytes']).encode()+b'\0')
        with path.open('rb') as handle:
            for chunk in iter(lambda:handle.read(8*1024*1024),b''):
                h.update(chunk)
        if h.hexdigest() != item['hash']:
            raise RuntimeError('Checkpoint hash changed: '+item['path'])
        verified.add(item['path'])
    config = json.loads((root/'config.json').read_text())
    if config['model_type'] != 'qwen3_vl' or config['architectures'] != ['Qwen3VLForConditionalGeneration']:
        raise ValueError('Wrong checkpoint architecture')
    index = json.loads((root/'model.safetensors.index.json').read_text())
    required = {'config.json','generation_config.json','chat_template.json','tokenizer.json',
                'tokenizer_config.json','video_preprocessor_config.json','preprocessor_config.json',
                'model.safetensors.index.json', *index['weight_map'].values()}
    if not required <= verified:
        raise ValueError('Verification record does not cover all loaded model files')
    return marker
