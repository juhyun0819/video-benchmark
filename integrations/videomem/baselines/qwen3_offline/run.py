"""Prepare, inspect or run the pinned HF Qwen3-VL raw-video baseline."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time

from baselines.fluxmem_qwen3vl.streaming.checkpoint import MODEL_ID, REVISION, verify_weights

TRANSFORMERS_VERSION = '4.57.3'


def _seconds(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('query_time.seconds must be a finite nonnegative number')
    return Fraction(str(value))


def validate_questions(rows, inventory):
    """Whitelist public fields. Observation cutoffs must be supplied explicitly."""
    if not isinstance(rows, list) or not rows:
        raise ValueError('questions must be a nonempty JSON list with reviewed query_time')
    result, seen = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Each question must be an object')
        for key in ('question_id', 'question'):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f'{key} must be a nonempty string')
        if row['question_id'] in seen:
            raise ValueError('Duplicate question_id')
        seen.add(row['question_id'])
        choices = row.get('choices', [])
        if not isinstance(choices, list) or len(choices) > 26 or any(not isinstance(x, str) or not x.strip() for x in choices):
            raise ValueError('choices must contain at most 26 nonempty strings')
        cutoff = row.get('query_time')
        if not isinstance(cutoff, dict) or type(cutoff.get('day_order')) is not int:
            raise ValueError('Explicit query_time {day_id, day_order, seconds} is required')
        matches = [s for s in inventory['sources'] if s['day_id'] == cutoff.get('day_id') and s['day_order'] == cutoff['day_order']]
        if len(matches) != 1:
            raise ValueError('Unknown or inconsistent query day')
        source = matches[0]
        seconds = _seconds(cutoff.get('seconds'))
        if seconds > Fraction(source['duration_fraction']):
            raise ValueError('query_time is beyond the source duration')
        global_seconds = Fraction(source['global_start_fraction']) + seconds
        result.append({'question_id': row['question_id'], 'question': row['question'],
                       'choices': list(choices), 'query_time': {k:cutoff[k] for k in ('day_id','day_order','seconds')},
                       'global_seconds': global_seconds})
    return result


def messages(question):
    text = question['question']
    if question['choices']:
        text += '\n' + '\n'.join(f'{chr(65+i)}. {c}' for i,c in enumerate(question['choices']))
        text += '\nAnswer with the option letter only.'
    return [{'role':'user', 'content':[{'type':'video'}, {'type':'text','text':text}]}]


def process_frames(processor, frames, indices, prefix_count, fps, question):
    """Sampling has happened once on the bounded prefix; retain original time IDs."""
    prompt = processor.apply_chat_template(messages(question), tokenize=False, add_generation_prompt=True)
    return processor(text=[prompt], videos=[frames], do_sample_frames=False,
                     video_metadata=[{'total_num_frames':prefix_count, 'fps':float(Fraction(fps)),
                                      'frames_indices':list(indices)}], return_tensors='pt')


def sample_prefix(processor, inventory, query_time):
    from transformers.video_utils import VideoMetadata
    from .video import prefix_frame_count, map_indices
    count = prefix_frame_count(inventory, query_time)
    if count < 2:
        raise ValueError('Official video preprocessing requires at least two available frames')
    metadata = VideoMetadata(total_num_frames=count, fps=float(Fraction(inventory['fps'])))
    indices = processor.video_processor.sample_frames(metadata).tolist()
    if len(indices) < 2:
        raise ValueError('Official sampler returned fewer than two frames')
    records = map_indices(inventory, indices, query_time)
    return count, indices, records


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   default=lambda x: str(x) if isinstance(x,Fraction) else _unserializable(x))+'\n')
    temporary.replace(path)


def _unserializable(value):
    raise TypeError(f'Cannot serialize {type(value)}')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare', help='Inspect an ordered list of CFR videos; no model or encoding')
    prep.add_argument('--sources', required=True, help='JSON list of absolute video paths, in day order')
    prep.add_argument('--output', required=True, help='New inventory JSON path')
    for command in ('plan', 'run'):
        p = sub.add_parser(command)
        for name in ('inventory','questions','model','output'):
            p.add_argument('--'+name, required=True)
        if command == 'run':
            p.add_argument('--verification', required=True)
            p.add_argument('--gpu', required=True, type=int, help='Physical GPU index')
            p.add_argument('--max-new-tokens', type=int, default=192)
    args = parser.parse_args(argv)
    from .video import build_inventory, validate_inventory, prefix_frame_count, decode_indices
    if args.command == 'prepare':
        if Path(args.output).exists():
            raise FileExistsError(args.output)
        paths = json.loads(Path(args.sources).read_text())
        value = build_inventory(paths)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, value)
        return 0
    inventory = json.loads(Path(args.inventory).read_text())
    validate_inventory(inventory)
    questions = validate_questions(json.loads(Path(args.questions).read_text()), inventory)
    for q in questions:
        if prefix_frame_count(inventory,q['global_seconds']) < 2:
            raise ValueError('Query prefix has fewer than two frames')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if importlib.metadata.version('transformers') != TRANSFORMERS_VERSION:
        raise RuntimeError(f'Use transformers=={TRANSFORMERS_VERSION}; do not silently change the baseline')
    marker, gpu = None, None
    if args.command == 'run':
        if args.max_new_tokens < 1:
            raise ValueError('max_new_tokens must be positive')
        # Reuse the existing full hash verifier and explicit physical GPU selector.
        marker = verify_weights(args.model, args.verification)
        from baselines.fifo_qwen3vl.fifo_videomem.run import bind_gpu
        gpu = bind_gpu(args.gpu)
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    plans = []
    for q in questions:
        count,indices,records = sample_prefix(processor,inventory,q['global_seconds'])
        plans.append({'question':q,'prefix_frame_count':count,'sampled_indices':indices,'frames':records})
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'plan.json', plans)
    source_root = Path(__file__).parent
    provenance = {'model_id':MODEL_ID,'revision':REVISION,'checkpoint_verified':marker is not None,
                  'checkpoint_verification':marker,'gpu':gpu,'mode':args.command,
                  'torch':torch.__version__,'transformers':TRANSFORMERS_VERSION,'cuda':torch.version.cuda,
                  'av':importlib.metadata.version('av'),'qwen_vl_utils_used':False,
                  'dtype':'bfloat16','attention':'sdpa','sampling':'HF Qwen3VLVideoProcessor.sample_frames defaults',
                  'sampler_config':processor.video_processor.to_dict(),
                  'cutoff':'original frame presentation time <= query_time; concatenated CFR days',
                  'inventory_sha256':digest(args.inventory),'questions_sha256':digest(args.questions),
                  'source_code_sha256':{p.name:digest(p) for p in source_root.glob('*.py')},
                  'max_new_tokens':getattr(args,'max_new_tokens',None),'do_sample':False,'num_beams':1,
                  'feature_archive':False, 'persistent_kv':False}
    write_json(output/'provenance.json',provenance)
    if args.command == 'plan':
        write_json(output/'status.json',{'status':'planned','questions':len(plans),'inference_performed':False})
        return 0
    write_json(output/'status.json',{'status':'loading'})
    predictions = []
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, local_files_only=True,
                  dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'':'cuda:0'}).eval()
        for plan in plans:
            q = plan['question']
            write_json(output/'status.json',{'status':'running','question_id':q['question_id'],'completed':len(predictions)})
            started = time.monotonic()
            frames, records = decode_indices(inventory,plan['sampled_indices'],q['global_seconds'])
            batch = process_frames(processor,frames,plan['sampled_indices'],plan['prefix_frame_count'],inventory['fps'],q).to('cuda:0')
            grid = batch['video_grid_thw'].tolist()
            # Fresh prefill and official multimodal reconstruction on every query.
            model.model.rope_deltas = None
            with torch.inference_mode():
                tokens = model.generate(**batch, do_sample=False, num_beams=1,
                                        max_new_tokens=args.max_new_tokens, use_cache=True)
            generated = tokens[0,batch['input_ids'].shape[1]:].tolist()
            answer = processor.tokenizer.decode(generated,skip_special_tokens=True).strip()
            eos = model.generation_config.eos_token_id
            eos = eos if isinstance(eos,list) else [eos]
            complete = bool(answer and generated and generated[-1] in eos)
            predictions.append({'question_id':q['question_id'],'query_time':q['query_time'],
                                'answer':answer,'complete':complete,'generated_token_ids':generated,
                                'video_grid_thw':grid,'sampled_frames':records,'elapsed_seconds':time.monotonic()-started})
            write_json(output/'predictions.json',predictions)
            del frames,batch,tokens
        status = 'complete' if all(p['complete'] for p in predictions) else 'incomplete'
        write_json(output/'status.json',{'status':status,'questions':len(predictions)})
        return 0 if status == 'complete' else 2
    except Exception as exc:
        write_json(output/'status.json',{'status':'error','error':str(exc),'completed':len(predictions)})
        raise
