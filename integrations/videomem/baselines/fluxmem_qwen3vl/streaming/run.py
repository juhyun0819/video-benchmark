"""FluxMem + Qwen3-VL-8B: training-free backbone variant on local video."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from local_videomem.run import bind_gpu, write
from streaming.checkpoint import MODEL_ID, verify_weights


def _git_metadata(*arguments):
    """Export metadata is optional for source ZIPs and an unborn checkout."""
    try:
        return subprocess.check_output(
            ['git', *arguments], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', help='JSON defaults; explicit CLI arguments take precedence')
    for name in ('manifest', 'questions', 'output'):
        p.add_argument('--'+name)
    p.add_argument('--model', default='/home/kai/models/Qwen3-VL-8B-Instruct')
    p.add_argument('--verification', default='/home/kai/models/.downloads/qwen3_vl_8b_instruct/verification.json')
    p.add_argument('--gpu', type=int, choices=(4, 5), default=4)
    p.add_argument('--fps', type=float, default=1)
    p.add_argument('--max-side', type=int, default=512)
    p.add_argument('--max-pixels', type=int, default=256*32*32)
    p.add_argument('--short-frames', type=int, default=8)
    p.add_argument('--medium-frames', type=int, default=64)
    p.add_argument('--long-frames', type=int, default=None)
    p.add_argument('--trigger-threshold', type=float,
                   help='Explicit experiment setting; the paper gives no recommended gamma')
    p.add_argument('--tas-policy', choices=('upstream', 'paper'), default='upstream')
    p.add_argument('--max-new-tokens', type=int, default=192)
    p.add_argument('--proactive', choices=('on', 'off'), default='on')
    preliminary, _ = p.parse_known_args()
    if preliminary.config:
        defaults = json.loads(Path(preliminary.config).read_text())
        allowed = {action.dest for action in p._actions} - {'help','config'}
        if not isinstance(defaults, dict) or set(defaults) - allowed:
            p.error('Config contains unknown settings')
        p.set_defaults(**defaults)
    args = p.parse_args()
    if any(not getattr(args, name) for name in ('manifest','questions','output')):
        p.error('manifest, questions and output are required through config or CLI')
    if args.trigger_threshold is None:
        p.error('trigger_threshold must be explicitly provided through config or CLI')
    if args.proactive not in ('on','off') or args.tas_policy not in ('upstream','paper'):
        p.error('Invalid proactive/tas_policy setting')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        gpu = bind_gpu(args.gpu)
        write(output/'status.json', {'phase': 'verifying_weights', 'gpu': gpu})
        weights = verify_weights(Path(args.model), args.verification)
        import torch
        import transformers
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        from local_videomem.data import public_question, select_prefix
        from streaming.qwen3_memory import Qwen3PersistentMemory
        from streaming.video import iter_pairs, time_key
        from streaming.qwen3_bridge import encode_pair, generate_from_memory
        from streaming.session import PROMPT_CONTRACT, run_session
        torch.set_num_threads(4)
        torch.manual_seed(0)
        if torch.cuda.device_count() != 1:
            raise RuntimeError('GPU isolation failed')
        manifest = json.loads(Path(args.manifest).read_text())
        private_questions = json.loads(Path(args.questions).read_text())
        questions = [public_question(q) for q in private_questions]
        if not questions:
            raise ValueError('No explicit timed questions')
        if len({q['question_id'] for q in questions}) != len(questions):
            raise ValueError('Duplicate question IDs')
        for q in questions:
            select_prefix(manifest, q['query_time'])
        source_hashes = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                         for base in ('models/qwen2-5-vl', 'models/qwen-vl-utils', 'local_videomem', 'streaming')
                         for f in (ROOT/base).rglob('*.py')}
        provenance = {'upstream_commit': '6fdbf65b6f2af1a94ca41d542be5be86a565d338', 'repository_commit': _git_metadata('rev-parse','HEAD'),
                      'branch': _git_metadata('branch','--show-current'),
                      'implementation': 'fluxmem_qwen3vl8b_training_free_backbone_variant',
                      'model_id': MODEL_ID,
                      'source_hashes': source_hashes, 'model_revision': weights['revision'],
                      'model_module': Qwen3VLForConditionalGeneration.__module__, 'gpu': gpu,
                      'native_model_source_sha256': hashlib.sha256(
                          Path(inspect.getfile(Qwen3VLForConditionalGeneration)).read_bytes()).hexdigest(),
                      'torch': str(torch.__version__), 'transformers': str(transformers.__version__),
                      'settings': vars(args), 'training': False, 'dtype': 'bfloat16',
                      'attention': 'sdpa', 'decoding': 'greedy_fresh_kv_per_answer',
                      'prompt_contract': PROMPT_CONTRACT,
                      'manifest_sha256': hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                      'questions_sha256': hashlib.sha256(Path(args.questions).read_bytes()).hexdigest(),
                      'query_lifecycle': 'last_successful_explicit_question_remains_active',
                      'partial_temporal_patch': 'exclude_unpaired_frame_no_future_padding',
                      'time_mapping': 'Qwen3_native_per_group_timestamp_and_spatial_rope; compacted_sequence',
                      'deepstack': 'all_three_levels; main_derived_TAS_masks_and_SDC_component_means'}
        write(output/'provenance.json', provenance)
        write(output/'public_questions.json', questions)
        write(output/'manifest.json', manifest)
        write(output/'status.json', {'phase': 'loading_model', 'gpu': gpu})
        load_start = time.monotonic()
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model, local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation='sdpa', device_map={'': 'cuda:0'}).eval()
        model.requires_grad_(False)
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
        load_seconds = time.monotonic() - load_start
        bank = Qwen3PersistentMemory(args.short_frames, args.medium_frames, args.long_frames,
                                args.trigger_threshold, args.tas_policy)
        decode_stats = {}
        pairs = iter_pairs(manifest, max(questions, key=time_key)['query_time'], fps=args.fps,
                           max_side=args.max_side, stats=decode_stats)
        counts = {'arrivals': 0, 'explicit': 0, 'proactive': 0}
        def emit(event):
            with (output/'events.jsonl').open('a') as handle:
                handle.write(json.dumps(event, ensure_ascii=False)+'\n')
                handle.flush()
            if event['event'] == 'arrival': counts['arrivals'] += 1
            else: counts[event['kind']] += 1
            write(output/'status.json', {'phase':'streaming', **counts,
                  'bank':bank.stats(), 'wall_seconds':time.monotonic()-started})
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            result = run_session(pairs=pairs, questions=questions, bank=bank,
                encode=lambda frames: encode_pair(model, processor, frames, args.max_pixels),
                answer=lambda snapshot, prompt: generate_from_memory(
                    model, processor, snapshot, prompt, args.max_new_tokens),
                emit=emit, fps=args.fps, proactive=args.proactive=='on',
                # IDs are bank identities only. Qwen3 timestamps and per-group
                # temporal coordinate 0 are assembled by qwen3_bridge.
                temporal_tokens_per_second=2, time_id_mode='ordinal')
        write(output/'predictions.json', result['answers'])
        # A serializable memory snapshot is reusable by the Python API; the CLI
        # deliberately has no implicit resume that could duplicate answers.
        torch.save({'provenance':provenance, 'memory':bank.state_dict()}, output/'memory.pt')
        summary = {k:v for k,v in result.items() if k != 'answers'}
        summary.update(phase='finished', training=False, model_id=MODEL_ID, decode=decode_stats,
            incomplete_answers=len(result['answers'])-result['complete_answers'],
            model_load_seconds=load_seconds, wall_seconds=time.monotonic()-started,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
        write(output/'status.json',summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        if summary['incomplete_answers']:
            raise SystemExit(2)
    except Exception as exc:
        write(output/'status.json', {'phase':'execution_error', 'error_type':type(exc).__name__,
              'error':str(exc)[:2000], 'wall_seconds':time.monotonic()-started})
        raise


if __name__ == '__main__':
    main()
