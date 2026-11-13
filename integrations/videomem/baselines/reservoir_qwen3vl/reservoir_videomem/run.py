"""Run uniform reservoir Qwen3-VL inference with an explicit memory budget and seed."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from .shared import BASELINE_ROOT, REPOSITORY_ROOT, SHARED_ROOT, load_shared


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='JSON defaults; CLI values override the same fields')
    for name in ('manifest', 'questions', 'output', 'model', 'verification'):
        parser.add_argument('--' + name)
    parser.add_argument('--gpu', type=int, help='Explicit physical GPU index; no device is chosen by default')
    parser.add_argument('--seed', type=int, help='Explicit nonnegative seed for the private reservoir RNG')
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument('--memory-frames', type=int,
                        help='Even number of sampled raw frames, at least 2 (two frames per grid)')
    budget.add_argument('--memory-tokens', type=int,
                        help='Maximum retained visual token rows; sample complete grids uniformly')
    parser.add_argument('--fps', type=float, default=1.0)
    parser.add_argument('--max-side', type=int, default=512)
    parser.add_argument('--max-pixels', type=int, default=256 * 32 * 32)
    parser.add_argument('--max-new-tokens', type=int, default=192)
    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.config:
        try:
            defaults = json.loads(Path(preliminary.config).expanduser().read_text())
        except (OSError, ValueError) as exc:
            parser.error(f'Cannot read config: {exc}')
        allowed = {action.dest for action in parser._actions} - {'help', 'config'}
        if not isinstance(defaults, dict) or set(defaults) - allowed:
            parser.error('Config must be an object with only supported settings')
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    for name in ('manifest', 'questions', 'output', 'model', 'verification'):
        value = getattr(args, name)
        if not isinstance(value, str) or not value.strip():
            parser.error(f'{name} must be provided through config or CLI')
        setattr(args, name, str(Path(value).expanduser()))
    if type(args.gpu) is not int or args.gpu < 0:
        parser.error('gpu must be an explicit nonnegative physical GPU index')
    if type(args.seed) is not int or args.seed < 0:
        parser.error('seed must be an explicit nonnegative integer')
    if (args.memory_frames is None) == (args.memory_tokens is None):
        parser.error('Specify exactly one of memory_frames or memory_tokens; there is no default budget')
    if args.memory_frames is not None:
        if type(args.memory_frames) is not int or args.memory_frames < 2 or args.memory_frames % 2:
            parser.error('memory_frames must be an even integer >= 2')
    if args.memory_tokens is not None:
        if type(args.memory_tokens) is not int or args.memory_tokens < 1:
            parser.error('memory_tokens must be a positive integer')
    for name in ('max_side', 'max_pixels', 'max_new_tokens'):
        if type(getattr(args, name)) is not int or getattr(args, name) < 1:
            parser.error(f'{name} must be a positive integer')
    if (isinstance(args.fps, bool) or not isinstance(args.fps, (int, float))
            or not math.isfinite(args.fps) or args.fps <= 0):
        parser.error('fps must be positive and finite')
    if args.max_pixels < 32 * 32:
        parser.error('max_pixels must fit at least one Qwen3-VL merged patch (1024 pixels)')
    return args


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def bind_gpu(index):
    """Select the explicitly requested idle GPU by UUID before importing Torch."""
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
    os.environ['CUDA_VISIBLE_DEVICES'] = parts[1]
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['USE_TF'] = '0'
    os.environ['USE_FLAX'] = '0'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    return {'physical_index': index, 'uuid': parts[1], 'name': parts[2]}


def _git(*arguments):
    try:
        return subprocess.check_output(['git', *arguments], cwd=REPOSITORY_ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_hashes():
    files = list((BASELINE_ROOT / 'reservoir_videomem').glob('*.py'))
    files += [SHARED_ROOT / name for name in (
        'local_videomem/data.py', 'streaming/__init__.py',
        'streaming/video.py', 'streaming/session.py', 'streaming/qwen3_bridge.py',
        'streaming/checkpoint.py')]
    return {str(path.relative_to(REPOSITORY_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files)}


def main(argv=None):
    args = parse_args(argv)
    load_shared()
    from local_videomem.data import public_question, select_prefix
    from streaming.checkpoint import MODEL_ID, verify_weights

    # Validate all cutoffs before allocating GPU/model memory or creating outputs.
    manifest_bytes = Path(args.manifest).read_bytes()
    question_bytes = Path(args.questions).read_bytes()
    manifest = json.loads(manifest_bytes)
    raw_questions = json.loads(question_bytes)
    if not isinstance(raw_questions, list) or not raw_questions:
        raise ValueError('questions must be a nonempty list of explicit timed questions')
    questions = [public_question(question) for question in raw_questions]
    if len({question['question_id'] for question in questions}) != len(questions):
        raise ValueError('Duplicate question IDs')
    for question in questions:
        select_prefix(manifest, question['query_time'])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        write(output / 'status.json', {'phase': 'verifying_weights'})
        weights = verify_weights(Path(args.model), args.verification)
        gpu = bind_gpu(args.gpu)
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from .memory import Qwen3ReservoirMemory
        from streaming.video import iter_pairs, time_key
        from streaming.qwen3_bridge import encode_pair, generate_from_memory
        from streaming.session import PROMPT_CONTRACT, run_session

        torch.set_num_threads(4)
        torch.manual_seed(0)
        if torch.cuda.device_count() != 1:
            raise RuntimeError('GPU isolation failed; run in a fresh Python process')
        provenance = {
            'implementation': 'reservoir_qwen3vl_algorithm_r_v1',
            'repository_commit': _git('rev-parse', 'HEAD'),
            'repository_status': _git('status', '--porcelain'),
            'source_hashes': _source_hashes(), 'settings': vars(args),
            'model_id': MODEL_ID, 'model_revision': weights['revision'],
            'model_module': Qwen3VLForConditionalGeneration.__module__,
            'native_model_source_sha256': hashlib.sha256(
                Path(inspect.getfile(Qwen3VLForConditionalGeneration)).read_bytes()).hexdigest(),
            'torch': str(torch.__version__), 'transformers': str(transformers.__version__),
            'gpu': gpu, 'training': False, 'dtype': 'bfloat16', 'attention': 'sdpa',
            'decoding': 'greedy_fresh_kv_per_answer', 'prompt_contract': PROMPT_CONTRACT,
            'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
            'questions_sha256': hashlib.sha256(question_bytes).hexdigest(),
            'memory_policy': 'uniform_algorithm_r_complete_grid_reservoir',
            'selection_timing': 'during_ingest_before_any_answer',
            'sampling_algorithm': 'Algorithm R',
            'sampling_unit': 'two_sampled_frames_per_temporal_grid',
            'rng': 'python_random_MT19937_private_instance', 'seed': args.seed,
            'python': sys.version,
            'capacity_unit': 'sampled_raw_frames' if args.memory_frames is not None else 'visual_token_rows',
            'partial_temporal_patch': 'exclude_unpaired_frame_no_future_padding',
            'deepstack': 'all_three_levels_retained_and_evicted_with_main_features',
            'query_lifecycle': 'explicit_questions_only', 'proactive': False,
            'time_mapping': 'Qwen3_native_per_group_timestamp_and_spatial_rope; compacted_sequence',
        }
        write(output / 'provenance.json', provenance)
        write(output / 'public_questions.json', questions)
        write(output / 'manifest.json', manifest)
        write(output / 'status.json', {'phase': 'loading_model', 'gpu': gpu})
        load_started = time.monotonic()
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model, local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation='sdpa', device_map={'': 'cuda:0'}).eval()
        model.requires_grad_(False)
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
        load_seconds = time.monotonic() - load_started
        bank = Qwen3ReservoirMemory(memory_frames=args.memory_frames, memory_tokens=args.memory_tokens, seed=args.seed)
        decode_stats = {}
        pairs = iter_pairs(manifest, max(questions, key=time_key)['query_time'],
                           fps=args.fps, max_side=args.max_side, stats=decode_stats)
        counts = {'arrivals': 0, 'explicit': 0, 'proactive': 0}

        def emit(event):
            with (output / 'events.jsonl').open('a') as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + '\n')
            if event['event'] == 'arrival':
                counts['arrivals'] += 1
            else:
                counts[event['kind']] += 1
            write(output / 'status.json', {'phase': 'streaming', **counts,
                  'bank': bank.stats(), 'wall_seconds': time.monotonic() - started})

        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            result = run_session(
                pairs=pairs, questions=questions, bank=bank,
                encode=lambda frames: encode_pair(model, processor, frames, args.max_pixels),
                answer=lambda snapshot, prompt: generate_from_memory(
                    model, processor, snapshot, prompt, args.max_new_tokens),
                emit=emit, fps=args.fps, proactive=False,
                temporal_tokens_per_second=2, time_id_mode='ordinal')
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
