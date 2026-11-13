"""Run the native StreamForest online memory with explicit experiment settings."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

from .shared import BASELINE_ROOT, REPOSITORY_ROOT, SHARED_ROOT, load_shared


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='JSON defaults; explicit CLI arguments override them')
    for name in ('manifest', 'questions', 'output-dir', 'model-path', 'verification-file', 'memory-config'):
        parser.add_argument('--' + name)
    parser.add_argument('--gpu', type=int, help='Explicit physical GPU index')
    parser.add_argument('--fps', type=float, default=1.0)
    parser.add_argument('--max-side', type=int, default=None, help='Optional resize before native SigLip processing')
    parser.add_argument('--max-new-tokens', type=int, default=192)
    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.config:
        try:
            defaults = json.loads(Path(preliminary.config).expanduser().read_text())
        except (OSError, ValueError) as exc:
            parser.error(f'Cannot read config: {exc}')
        allowed = {action.dest for action in parser._actions} - {'help', 'config'}
        if not isinstance(defaults, dict) or set(defaults) - allowed:
            parser.error('Config must be an object containing only supported settings')
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    for name in ('manifest', 'questions', 'output_dir', 'model_path', 'verification_file', 'memory_config'):
        value = getattr(args, name)
        if not isinstance(value, str) or not value.strip():
            parser.error(f'{name} must be explicitly supplied; no memory budget is chosen by default')
        setattr(args, name, str(Path(value).expanduser()))
    if type(args.gpu) is not int or args.gpu < 0:
        parser.error('gpu must be an explicit nonnegative physical GPU index')
    if (isinstance(args.fps, bool) or not isinstance(args.fps, (int, float))
            or not math.isfinite(args.fps) or args.fps <= 0):
        parser.error('fps must be positive and finite')
    if args.max_side is not None and (type(args.max_side) is not int or args.max_side < 1):
        parser.error('max_side must be null or a positive integer')
    if type(args.max_new_tokens) is not int or args.max_new_tokens < 1:
        parser.error('max_new_tokens must be a positive integer')
    return args


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def bind_gpu(index):
    """Bind the requested physical GPU by UUID before importing Torch."""
    if type(index) is not int or index < 0:
        raise ValueError('gpu must be a nonnegative physical index')
    raw = subprocess.check_output(
        ['nvidia-smi', f'--id={index}', '--query-gpu=index,uuid,name,memory.used',
         '--format=csv,noheader,nounits'], text=True)
    parts = [part.strip() for part in raw.strip().split(',')]
    if len(parts) != 4 or int(parts[0]) != index or not parts[1].startswith('GPU-'):
        raise RuntimeError('Cannot verify selected physical GPU identity')
    if int(parts[3]) > 1024:
        raise RuntimeError('Selected GPU is occupied; choose another GPU')
    os.environ.update(CUDA_VISIBLE_DEVICES=parts[1], CUDA_DEVICE_ORDER='PCI_BUS_ID',
                      TOKENIZERS_PARALLELISM='false', USE_TF='0', USE_FLAX='0',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    return {'physical_index': index, 'uuid': parts[1], 'name': parts[2]}


def _git(*arguments):
    try:
        return subprocess.check_output(['git', *arguments], cwd=REPOSITORY_ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_hashes():
    from .upstream import VENDOR_ROOT
    files = list((BASELINE_ROOT / 'streamforest_videomem').glob('*.py'))
    files += [SHARED_ROOT / 'local_videomem' / 'data.py',
              VENDOR_ROOT / 'UPSTREAM.json', BASELINE_ROOT / 'requirements.txt']
    return {str(path.relative_to(REPOSITORY_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files)}


def load_inputs(args):
    """Validate reviewed query cutoffs before GPU allocation or output creation."""
    load_shared()
    from local_videomem.data import public_question, select_prefix
    manifest_bytes = Path(args.manifest).read_bytes()
    question_bytes = Path(args.questions).read_bytes()
    memory_bytes = Path(args.memory_config).read_bytes()
    manifest, raw_questions, memory_config = map(json.loads, (manifest_bytes, question_bytes, memory_bytes))
    if not isinstance(raw_questions, list) or not raw_questions:
        raise ValueError('questions must be a nonempty list of explicit timed questions')
    questions = [public_question(question) for question in raw_questions]
    if len({question['question_id'] for question in questions}) != len(questions):
        raise ValueError('Duplicate question IDs')
    for question in questions:
        select_prefix(manifest, question['query_time'])
    digests = {name + '_sha256': hashlib.sha256(content).hexdigest() for name, content in (
        ('manifest', manifest_bytes), ('questions', question_bytes), ('memory_config', memory_bytes))}
    return manifest, questions, memory_config, digests


def main(argv=None):
    args = parse_args(argv)
    manifest, questions, memory_config, digests = load_inputs(args)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        from .checkpoint import MODEL_ID, verify_checkpoint
        from .upstream import verify_source
        write(output / 'status.json', {'phase': 'verifying_source_and_weights'})
        upstream = verify_source()
        weights = verify_checkpoint(args.model_path, args.verification_file)
        gpu = bind_gpu(args.gpu)
        import torch
        import transformers
        from .memory import StreamForestMemory, validate_memory_config
        from .video import iter_frames, time_key
        from .bridge import load_model, encode_frame, generate_from_memory, PROMPT_CONTRACT
        from .session import run_session, PROMPT_CONTRACT as QUESTION_CONTRACT

        memory_config = validate_memory_config(memory_config)
        if memory_config['st_memory_tokens'][0] > 729:
            raise ValueError('now token count cannot exceed the native 729 SigLip tokens')
        torch.set_num_threads(4)
        torch.manual_seed(0)
        if torch.cuda.device_count() != 1:
            raise RuntimeError('GPU isolation failed; run in a fresh Python process')
        provenance = {
            'implementation': 'streamforest_native_online_persistent_v1',
            'repository_commit': _git('rev-parse', 'HEAD'),
            'repository_status': _git('status', '--porcelain'),
            'source_hashes': _source_hashes(), 'upstream': upstream,
            'settings': vars(args), 'memory_config': memory_config,
            'model_id': MODEL_ID, 'checkpoint_verification': weights,
            'torch': str(torch.__version__), 'transformers': str(transformers.__version__),
            'gpu': gpu, 'training': False, 'dtype': 'bfloat16', 'attention': 'sdpa',
            'decoding': 'greedy_fresh_kv_per_answer', 'prompt_contract': PROMPT_CONTRACT,
            'question_contract': QUESTION_CONTRACT, **digests,
            'memory_policy': 'official_memory_manager_update_then_get_per_frame',
            'vision': 'embedded_native_siglip_raw_729x1152',
            'projector': 'official_mlp_after_memory_no_batch_positional_additions',
            'query_lifecycle': 'explicit_questions_only', 'proactive': False,
            'time_mapping': 'original_pts_for_cutoffs; compact_ordinal_memory_clock; no_per_token_time_labels',
            'paper_evaluation_reproduction': False,
        }
        write(output / 'provenance.json', provenance)
        write(output / 'public_questions.json', questions)
        write(output / 'manifest.json', manifest)
        write(output / 'memory_config.json', memory_config)
        write(output / 'status.json', {'phase': 'loading_model', 'gpu': gpu})
        load_started = time.monotonic()
        model, tokenizer, processor = load_model(args.model_path, device='cuda:0', dtype=torch.bfloat16)
        load_seconds = time.monotonic() - load_started
        vision = model.get_vision_tower().config
        bank = StreamForestMemory(vision.hidden_size, vision.num_attention_heads, config=memory_config)
        decode_stats = {}
        frames = iter_frames(manifest, max(questions, key=time_key)['query_time'],
                             fps=args.fps, max_side=args.max_side, stats=decode_stats)
        counts = {'arrivals': 0, 'explicit': 0}

        def emit(event):
            with (output / 'events.jsonl').open('a') as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + '\n')
            counts['arrivals' if event['event'] == 'arrival' else event['kind']] += 1
            write(output / 'status.json', {'phase': 'streaming', **counts,
                  'bank': bank.stats(), 'wall_seconds': time.monotonic() - started})

        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            result = run_session(
                frames=frames, questions=questions, bank=bank,
                encode=lambda frame: encode_frame(model, processor, frame),
                answer=lambda snapshot, prompt: generate_from_memory(
                    model, tokenizer, snapshot, prompt, args.max_new_tokens),
                emit=emit, fps=args.fps)
        write(output / 'predictions.json', result['answers'])
        torch.save({'provenance': provenance, 'memory': bank.state_dict()}, output / 'memory.pt')
        summary = {key: value for key, value in result.items() if key != 'answers'}
        summary.update(
            phase='finished', training=False, model_id=MODEL_ID, decode=decode_stats,
            incomplete_answers=len(result['answers']) - result['complete_answers'],
            model_load_seconds=load_seconds, wall_seconds=time.monotonic() - started,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
        write(output / 'status.json', summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return 2 if summary['incomplete_answers'] else 0
    except Exception as exc:
        write(output / 'status.json', {'phase': 'execution_error', 'error_type': type(exc).__name__,
              'error': str(exc)[:2000], 'wall_seconds': time.monotonic() - started})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
