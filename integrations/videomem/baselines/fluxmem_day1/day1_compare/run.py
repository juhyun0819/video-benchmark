"""Frozen Day1, same video/question, official FluxMem on/off inference.

No labels or annotations are opened. Preprocessing is shared across questions;
the official visual encoder and FluxMem prefill are still executed by generate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'models/qwen2-5-vl/src'),
               str(ROOT/'models/qwen-vl-utils/src')]
from local_videomem.run import bind_gpu, instrument, verify_weights, write


def _git_metadata(*arguments):
    """Export metadata is optional for source ZIPs and an unborn checkout."""
    try:
        return subprocess.check_output(
            ['git', *arguments], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def public_prompt(row, *, duration, frames):
    """Explicit whitelist; private additions to source rows cannot enter text."""
    aliases = []
    source = row['input_aliases']
    if isinstance(source, dict):
        source = [dict(value, alias=key) for key, value in source.items()]
    for alias in source:
        aliases.append({key: alias[key] for key in
                        ('alias', 'day_start_seconds', 'day_end_seconds',
                         'global_start_seconds', 'global_end_seconds') if key in alias})
    text = (
        f'This is an offline memory question asked after Day1 ends at {duration:.3f} playback seconds. '
        f'The video contains {frames} chronologically ordered, uniformly spaced observations across Day1. '
        'There are gaps between observations. Do not treat them as continuous dense video. '
        'Times in the question and options refer to Day1 playback time, not recording wall-clock time. '
        'The first sampled observation is at half a sampling interval after Day1 start. '
        'The input alias "video 1" refers to its specified interval within Day1, not the entire day. '
        'Answer about the requested interval using the observed video. '
        'This is a full-day adaptation: later observations are available.\n'
        'Input aliases: '+json.dumps(aliases, ensure_ascii=False)+'\n'
        'Question: '+row['question']+'\n'
    )
    for i, choice in enumerate(row['choices']):
        value = choice if isinstance(choice, str) else json.dumps(choice, ensure_ascii=False)
        text += f'{chr(65+i)}. {value}\n'
    return text+'Return only one option letter (A, B, C, D, or E).'


def assemble_inputs(processor, prompt, visual, fps):
    """Mirror official processor placeholder expansion using cached pixel tensors.

    Tested against the complete official processor. No embedding cache, label,
    retrieval, caption, or textual summary is inserted.
    """
    messages = [{'role':'system', 'content':'You are a helpful assistant.'},
                {'role':'user','content':[{'type':'text','text':prompt}, {'type':'video'}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if text.count(processor.video_token) != 1:
        raise ValueError('Expected exactly one video placeholder')
    grid = visual['video_grid_thw']
    if tuple(grid.shape) != (1,3):
        raise ValueError('Expected one chronological Day1 video tensor')
    visual_count = int(grid[0].prod()) // processor.image_processor.merge_size**2
    text = text.replace(processor.video_token, processor.video_token*visual_count, 1)
    inputs = dict(processor.tokenizer([text], padding=True, return_tensors='pt'))
    inputs.update(visual)
    inputs['second_per_grid_ts'] = [processor.image_processor.temporal_patch_size/fps]
    return inputs


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(__import__('torch').uint8).numpy().tobytes()).hexdigest()


def honor_full_attention(model):
    """Released SDPA mask ignores use_sliding_window=False above 32768 tokens.

    Keep the checkpoint's declared full-attention semantics in BOTH arms.
    This is a logged runtime config correction, not a weight/source edit.
    """
    original = model.config.sliding_window
    if model.config.use_sliding_window:
        raise ValueError('The frozen comparison expects a full-attention checkpoint')
    model.config.sliding_window = None
    model.model.config.sliding_window = None
    return {'checkpoint_use_sliding_window':False,'checkpoint_sliding_window':original,
            'effective_sliding_window':None,
            'reason':'Prevent released SDPA mask from applying an unintended window to the uncompressed arm'}


def main(*, allowed_gpus=(4, 5)):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, help='Prepared artifact directory, not a QA label file')
    p.add_argument('--output', required=True)
    p.add_argument('--gpu', type=int, choices=allowed_gpus, required=True)
    p.add_argument('--fluxmem', choices=('on','off'), required=True)
    p.add_argument('--model', default='/home/kai/models/Qwen2.5-VL-7B-Instruct')
    p.add_argument('--verification', default='/home/kai/models/.downloads/qwen25vl_7b_fluxmem/verification.json')
    p.add_argument('--max-new-tokens', type=int, default=16)
    p.add_argument('--short-frames', type=int, default=8)
    p.add_argument('--medium-frames', type=int, default=512)
    p.add_argument('--video-chunk-t', type=int, default=1)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'provenance.json').exists() or (out/'predictions.json').exists():
        raise FileExistsError('Use a fresh output directory; results are immutable')
    gpu = bind_gpu(args.gpu)
    write(out/'status.json', {'phase':'verifying_weights', 'gpu':gpu})
    weights = verify_weights(Path(args.model), args.verification)
    import torch
    import transformers
    from PIL import Image
    from qwen2_5_vl_fluxmem import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
    torch.set_num_threads(4)
    torch.manual_seed(0)
    if torch.cuda.device_count() != 1:
        raise RuntimeError('GPU isolation failed')
    base = Path(args.input)
    if not (base/'completed.json').exists():
        raise ValueError('Frame and question preparation is not complete')
    public_path = base/'public_questions.json'
    frame_path = base/'source_frames_manifest.json'
    questions = json.loads(public_path.read_text())
    manifest = json.loads(frame_path.read_text())
    frames = manifest['frames']
    # End cutoff is public and independent of any answer or evidence interval.
    cutoffs = {float(q['query_time_seconds']) for q in questions}
    if len(cutoffs) != 1:
        raise ValueError('This arm requires one fixed end-of-day cutoff')
    duration = cutoffs.pop()
    if any(float(f['actual_global_seconds']) >= duration for f in frames):
        raise ValueError('A frame exceeds the observation cutoff')
    ids = [q['question_id'] for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate questions')
    fps = len(frames)/duration
    source_hashes = {str(f.relative_to(ROOT)):sha(f)
                     for d in ('models/qwen2-5-vl','day1_compare','local_videomem')
                     for f in (ROOT/d).rglob('*.py')}
    provenance = {'settings':vars(args), 'gpu':gpu, 'training':False,
        'upstream_commit':'6fdbf65b6f2af1a94ca41d542be5be86a565d338', 'repository_commit': _git_metadata('rev-parse','HEAD'),
        'model_revision':weights['revision'], 'torch':str(torch.__version__),
        'transformers':transformers.__version__, 'source_hashes':source_hashes,
        'public_questions_sha256':sha(public_path), 'frames_manifest_sha256':sha(frame_path),
        'effective_fps':fps, 'frames':len(frames), 'dtype':'bfloat16', 'attention':'sdpa',
        'decoder':'greedy', 'video_chunk_t':args.video_chunk_t,
        'checkpoint':'unfinetuned Qwen2.5-VL-7B-Instruct',
        'method':'unchanged released FluxMem prefill' if args.fluxmem=='on' else 'same patched model with FluxMem disabled',
        'temporal_positions':'native uniform-grid elapsed seconds; midpoint offset stated in prompt; exact PTS audit only',
        'no_label_file_opened':True}
    write(out/'provenance.json', provenance)
    write(out/'status.json', {'phase':'loading_model','gpu':gpu})
    started = time.monotonic()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'':'cuda:0'}).eval()
    model.requires_grad_(False)
    attention_correction = honor_full_attention(model)
    write(out/'attention_runtime.json',attention_correction)
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model, local_files_only=True)
    events = []
    instrument(model.model.fluxmem, events)
    images = []
    write(out/'status.json', {'phase':'preprocessing_shared_video','frames':len(frames)})
    try:
        for frame in frames:
            path = Path(frame['path'])
            if sha(path) != frame['sha256']:
                raise ValueError('Frame cache hash mismatch')
            with Image.open(path) as image:
                if image.size != (224,224):
                    raise ValueError('Frozen paired protocol requires 224x224 frames')
                images.append(image.convert('RGB'))
        visual = processor.image_processor(images=None, videos=[images], return_tensors='pt',
                                          min_pixels=224*224, max_pixels=224*224)
    finally:
        for image in images: image.close()
    video_hashes = {k:tensor_hash(v) for k,v in visual.items()}
    write(out/'processed_video.json', {'hashes':video_hashes,
        'shapes':{k:list(v.shape) for k,v in visual.items()}, 'effective_fps':fps})
    load_and_preprocess_seconds = time.monotonic()-started
    rows = []
    for ordinal, q in enumerate(questions):
        events.clear()
        prompt = public_prompt(q,duration=duration,frames=len(frames))
        inputs = assemble_inputs(processor,prompt,visual,fps)
        wire = {'question_id':q['question_id'], 'prompt':prompt,
                'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                'input_ids_sha256':tensor_hash(inputs['input_ids']),
                'processed_video_hashes':video_hashes,
                'second_per_grid_ts':inputs['second_per_grid_ts'],
                'input_tokens':inputs['input_ids'].shape[1]}
        write(out/f'wire_{ordinal:04d}.json',wire)
        inputs = {k:v.to('cuda:0') if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
        write(out/'status.json', {'phase':'answering','arm':args.fluxmem,'completed':ordinal,
                                 'total':len(questions),'question_id':q['question_id']})
        print(json.dumps({'phase':'answering','arm':args.fluxmem,'completed':ordinal,'total':len(questions)}),flush=True)
        row = {'question_id':q['question_id'],'status':'execution_error','prediction':None}
        case_started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        try:
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                    use_fluxmem=args.fluxmem=='on', short_frames=args.short_frames,
                    medium_frames=args.medium_frames, video_chunk_t=args.video_chunk_t)
            torch.cuda.synchronize()
            tail = generated[0,inputs['input_ids'].shape[1]:]
            prediction = processor.decode(tail,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            eos = model.generation_config.eos_token_id
            eos = set(eos if isinstance(eos,list) else [eos])
            complete = len(tail)>0 and int(tail[-1]) in eos and bool(prediction.strip())
            if bool(events) != (args.fluxmem=='on'):
                raise RuntimeError('Observed compression path disagrees with declared arm')
            row.update(status='complete' if complete else 'truncated', prediction=prediction,
                output_token_ids=tail.tolist(),output_tokens=len(tail),input_tokens=wire['input_tokens'],
                memory_events=list(events),peak_gpu_bytes=torch.cuda.max_memory_allocated())
            del generated, tail
        except Exception as exc:
            row.update(error_type=type(exc).__name__,error=str(exc)[:1800],memory_events=list(events))
        finally:
            del inputs
            torch.cuda.empty_cache()
        row['generation_seconds'] = time.monotonic()-case_started
        rows.append(row)
        write(out/'predictions.json', rows)
        print(json.dumps({'question_id':row['question_id'],'status':row['status'],
                          'prediction':row['prediction'],'seconds':row['generation_seconds']}),flush=True)
        # Preserve failure and stop this arm. Do not reroll favorable answers.
        if row['status']=='execution_error':
            break
    report={'phase':'finished' if len(rows)==len(questions) else 'stopped_on_error',
        'arm':args.fluxmem, 'total':len(questions),'attempted':len(rows),
        'completed':sum(r['status']=='complete' for r in rows),
        'load_and_preprocess_seconds':load_and_preprocess_seconds,
        'wall_seconds':time.monotonic()-started}
    write(out/'status.json',report)
    print(json.dumps(report),flush=True)
    if report['completed']!=len(questions): raise SystemExit(1)


if __name__=='__main__': main()
