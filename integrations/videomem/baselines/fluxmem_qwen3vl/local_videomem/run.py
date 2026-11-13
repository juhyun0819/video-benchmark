"""Inference-only VideoMem bridge to the unmodified official FluxMem model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _git_metadata(*arguments):
    """Export metadata is optional for source ZIPs and an unborn checkout."""
    try:
        return subprocess.check_output(
            ['git', *arguments], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(row, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def bind_gpu(index):
    if index not in (4, 5):
        raise ValueError('Only physical GPUs 4 and 5 are authorized')
    raw = subprocess.check_output(['nvidia-smi', f'--id={index}',
        '--query-gpu=index,uuid,name,memory.used', '--format=csv,noheader,nounits'], text=True)
    parts = [s.strip() for s in raw.strip().split(',')]
    if int(parts[0]) != index or not parts[1].startswith('GPU-'):
        raise RuntimeError('Cannot verify physical GPU identity')
    if int(parts[3]) > 1024:
        raise RuntimeError('Selected GPU is occupied; no existing process will be stopped')
    os.environ['CUDA_VISIBLE_DEVICES'] = parts[1]
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    return {'physical_index': index, 'uuid': parts[1], 'name': parts[2]}


def verify_weights(model_path, marker_path):
    marker = json.loads(Path(marker_path).read_text())
    if marker['repo_id'] != 'Qwen/Qwen2.5-VL-7B-Instruct':
        raise ValueError('FluxMem requires the official Qwen2.5-VL checkpoint')
    for item in marker['verified_files']:
        path = model_path/item['path']
        h = hashlib.sha256() if len(item['hash']) == 64 else hashlib.sha1()
        if len(item['hash']) == 40:
            h.update(b'blob '+str(item['bytes']).encode()+b'\0')
        if path.stat().st_size != item['bytes']:
            raise RuntimeError('Checkpoint byte count changed: '+item['path'])
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024), b''): h.update(chunk)
        if h.hexdigest() != item['hash']:
            raise RuntimeError('Checkpoint hash changed: '+item['path'])
    return marker


def instrument(memory, records):
    """Observe original calls and returns; leave pruning and embeddings intact."""
    original = memory.process_memory_streaming
    adjacent = memory._apply_adjacent_pruning
    spatial = memory._long_term_memory_merge_per_frames
    counters = {'temporal_calls': 0, 'spatial_calls': 0}
    def temporal(*a, **kw):
        counters['temporal_calls'] += 1
        return adjacent(*a, **kw)
    def long_term(*a, **kw):
        counters['spatial_calls'] += 1
        return spatial(*a, **kw)
    def measured(*a, **kw):
        from qwen2_5_vl_fluxmem.utils import scan_visual_indices
        counters.update(temporal_calls=0, spatial_calls=0)
        before = kw['hidden_states'].shape[1]
        visual = scan_visual_indices(kw['input_ids'][0], memory.vision_start_token_id, memory.vision_end_token_id)
        started = time.monotonic()
        result = original(*a, **kw)
        kept = set(result[4][0].detach().cpu().tolist())
        visual_set = set(visual.detach().cpu().tolist())
        if not (set(range(before))-visual_set).issubset(kept):
            raise RuntimeError('Compression unexpectedly removed nonvisual prompt tokens')
        records.append({'input_tokens': before, 'retained_tokens': result[0].shape[1],
                        'dropped_tokens': before-result[0].shape[1],
                        'visual_tokens_before':len(visual_set),'visual_tokens_after':len(visual_set & kept),
                        'seconds': time.monotonic()-started, **counters})
        return result
    memory._apply_adjacent_pruning = temporal
    memory._long_term_memory_merge_per_frames = long_term
    memory.process_memory_streaming = measured


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--questions', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model', default='/home/kai/models/Qwen2.5-VL-7B-Instruct')
    p.add_argument('--verification', default='/home/kai/models/.downloads/qwen25vl_7b_fluxmem/verification.json')
    p.add_argument('--gpu', type=int, choices=(4, 5), default=4)
    p.add_argument('--fluxmem', choices=('on','off'), default='on')
    p.add_argument('--fps', type=float, default=1)
    p.add_argument('--max-frames', type=int, default=512)
    p.add_argument('--max-side', type=int, default=448)
    p.add_argument('--max-pixels', type=int, default=256*28*28)
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--short-frames', type=int, default=8)
    p.add_argument('--medium-frames', type=int, default=64)
    args = p.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'provenance.json').exists() or (output/'predictions.json').exists():
        raise FileExistsError('Preserve previous results: choose a new output directory')
    gpu = bind_gpu(args.gpu)
    write(output/'status.json', {'phase':'verifying_weights','gpu':gpu})
    weights = verify_weights(Path(args.model), args.verification)

    import torch
    import transformers
    from qwen2_5_vl_fluxmem import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
    from qwen_vl_utils_fluxmem import process_vision_info
    from local_videomem.data import decode_prefix, public_question
    torch.set_num_threads(4)
    torch.manual_seed(0)
    if torch.cuda.device_count() != 1:
        raise RuntimeError('GPU isolation failed')
    manifest = json.loads(Path(args.manifest).read_text())
    questions = json.loads(Path(args.questions).read_text())
    if not isinstance(questions, list): raise ValueError('Questions must be a JSON list')
    ids = [public_question(q)['question_id'] for q in questions]
    if len(ids) != len(set(ids)): raise ValueError('Duplicate question IDs')
    source_hashes = {str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                     for base in ['models/qwen2-5-vl','models/qwen-vl-utils','local_videomem']
                     for f in (ROOT/base).rglob('*.py')}
    provenance = {'upstream_commit':'6fdbf65b6f2af1a94ca41d542be5be86a565d338', 'repository_commit': _git_metadata('rev-parse','HEAD'),
        'source_hashes':source_hashes,'model_revision':weights['revision'],'gpu':gpu,
        'manifest_sha256':hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        'torch':torch.__version__,'transformers':transformers.__version__,
        'settings':vars(args),'training':False,'dtype':'bfloat16','attention':'sdpa',
        'input_representation':'native_video_from_actual_sampled_frames',
        'timeline_note':'Native video positions use a compacted playback sequence; exact original file/Day PTS are logged.'}
    write(output/'provenance.json', provenance)
    write(output/'status.json', {'phase':'loading_model','gpu':gpu})
    started = time.monotonic()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda:0'}).eval()
    model.requires_grad_(False)
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model,local_files_only=True)
    memory_events = []
    instrument(model.model.fluxmem, memory_events)
    load_seconds = time.monotonic()-started
    rows = []
    for ordinal, original in enumerate(questions):
        q = public_question(original)
        case_started = time.monotonic()
        write(output/'status.json',{'phase':'decoding','completed':ordinal,'total':len(questions),'question_id':q['question_id']})
        row = {'question':q,'status':'execution_error','prediction':None}
        decoded = None
        try:
            decoded = decode_prefix(manifest,q,fps=args.fps,max_frames=args.max_frames,max_side=args.max_side)
            frames = decoded['frames']
            if not frames: raise ValueError('No frames available before the question cutoff')
            # Filenames can contain human activity labels; keep them in audit
            # records only, never in the model prompt.
            labels = {'query_time':q['query_time'],
                      'segments':[{k:v for k,v in s.items() if k in
                          ('source_id','day_id','day_order','file_start_seconds','file_end_seconds',
                           'day_start_seconds','day_end_seconds')} for s in decoded['segments']],
                      'sampled_frames':[ {k:v for k,v in f.items() if k in ('day_id','file_seconds','day_seconds')}
                                         for f in decoded['frame_metadata']]}
            prompt = 'Use only the observed video. Question: '+q['question']
            if q.get('choices'):
                prompt += '\n'+'\n'.join(f'{chr(65+i)}. {s}' for i,s in enumerate(q['choices']))+'\nAnswer with the option letter only.'
            else:
                prompt += '\nAnswer in at most three sentences; do not invent unseen details.'
            prompt += '\nPlayback mapping (not recording wall-clock): '+json.dumps(labels,ensure_ascii=False)
            messages=[{'role':'system','content':'You are a helpful assistant.'},
                      {'role':'user','content':[{'type':'text','text':prompt},
                          {'type':'video','video':frames,'fps':args.fps,'max_pixels':args.max_pixels}]}]
            text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            image_inputs,video_inputs,video_kw=process_vision_info(messages,return_video_kwargs=True)
            inputs=processor(text=[text],images=image_inputs,videos=video_inputs,padding=True,
                             return_tensors='pt',**video_kw).to('cuda:0')
            write(output/f'wire_{ordinal:04d}.json',{'question':q,'prompt':prompt,
                  'frames':decoded['frame_metadata'],'segments':decoded['segments'],
                  'tensor_shapes':{k:list(v.shape) for k,v in inputs.items() if hasattr(v,'shape')}})
            write(output/'status.json',{'phase':'generating','completed':ordinal,'total':len(questions)})
            memory_events.clear()
            torch.cuda.reset_peak_memory_stats()
            infer_started=time.monotonic()
            with torch.inference_mode():
                generated=model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=False,
                    use_fluxmem=args.fluxmem=='on',short_frames=args.short_frames,medium_frames=args.medium_frames)
            torch.cuda.synchronize()
            tail=generated[0,inputs.input_ids.shape[1]:]
            prediction=processor.decode(tail,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            eos=model.generation_config.eos_token_id
            eos=set(eos if isinstance(eos,list) else [eos])
            truncated=len(tail)>=args.max_new_tokens and int(tail[-1]) not in eos
            row.update(status='truncated' if truncated else 'complete',prediction=prediction,
                input_tokens=inputs.input_ids.shape[1],output_tokens=len(tail),frames=len(frames),
                generation_seconds=time.monotonic()-infer_started,
                peak_gpu_bytes=torch.cuda.max_memory_allocated(),memory_events=list(memory_events))
            if not prediction.strip():
                raise RuntimeError('Model produced an empty answer')
            if args.fluxmem=='on' and not memory_events:
                raise RuntimeError('FluxMem was requested but its compression path was not called')
            if args.fluxmem=='off' and memory_events:
                raise RuntimeError('Disabled control unexpectedly called FluxMem')
        except Exception as exc:
            row.update(status='execution_error',error_type=type(exc).__name__,error=str(exc)[:1800])
        finally:
            if decoded:
                for frame in decoded['frames']: frame.close()
        row['wall_seconds']=time.monotonic()-case_started
        rows.append(row)
        write(output/'predictions.json',rows)
    report={'phase':'finished','total':len(rows),'completed':sum(r['status']=='complete' for r in rows),
            'technical_failures':sum(r['status']!='complete' for r in rows),'training':False,
            'model_load_seconds':load_seconds,'predictions':str(output/'predictions.json')}
    write(output/'status.json',report)
    print(json.dumps(report))
    if report['technical_failures']: raise SystemExit(1)


if __name__=='__main__': main()
