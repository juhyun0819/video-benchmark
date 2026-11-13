"""Run the isolated training-free persistent FluxMem extension on local video."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Prefer this clone's official packages, not the old editable installation.
for path in ('models/qwen2-5-vl/src', 'models/qwen-vl-utils/src'):
    sys.path.insert(0, str(ROOT/path))
from local_videomem.run import bind_gpu, verify_weights, write


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
    for name in ('manifest', 'questions', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--model', default='/home/kai/models/Qwen2.5-VL-7B-Instruct')
    p.add_argument('--verification', default='/home/kai/models/.downloads/qwen25vl_7b_fluxmem/verification.json')
    p.add_argument('--gpu', type=int, choices=(4, 5), default=4)
    p.add_argument('--fps', type=float, default=1)
    p.add_argument('--max-side', type=int, default=448)
    p.add_argument('--max-pixels', type=int, default=256*28*28)
    p.add_argument('--short-frames', type=int, default=8)
    p.add_argument('--medium-frames', type=int, default=64)
    p.add_argument('--long-frames', type=int, default=None)
    p.add_argument('--trigger-threshold', type=float, required=True,
                   help='Explicit experiment setting; the paper gives no recommended gamma')
    p.add_argument('--tas-policy', choices=('upstream', 'paper'), default='upstream')
    p.add_argument('--max-new-tokens', type=int, default=192)
    p.add_argument('--proactive', choices=('on', 'off'), default='on')
    args = p.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        gpu = bind_gpu(args.gpu)
        write(output/'status.json', {'phase': 'verifying_weights', 'gpu': gpu})
        weights = verify_weights(Path(args.model), args.verification)
        import torch
        import transformers
        import qwen2_5_vl_fluxmem
        from qwen2_5_vl_fluxmem import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
        from local_videomem.data import public_question, select_prefix
        from streaming.memory import PersistentMemory
        from streaming.video import iter_pairs, time_key
        from streaming.model_bridge import encode_pair, generate_from_memory
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
                      'implementation': 'local_paper_guided_extension_not_official_streaming_release',
                      'source_hashes': source_hashes, 'model_revision': weights['revision'],
                      'model_module': qwen2_5_vl_fluxmem.__file__, 'gpu': gpu,
                      'torch': torch.__version__, 'transformers': transformers.__version__,
                      'settings': vars(args), 'training': False, 'dtype': 'bfloat16',
                      'attention': 'sdpa', 'decoding': 'greedy_fresh_kv_per_answer',
                      'prompt_contract': PROMPT_CONTRACT,
                      'manifest_sha256': hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                      'questions_sha256': hashlib.sha256(Path(args.questions).read_bytes()).hexdigest(),
                      'query_lifecycle': 'last_successful_explicit_question_remains_active',
                      'partial_temporal_patch': 'exclude_unpaired_frame_no_future_padding',
                      'time_mapping': 'native_compacted_sequence; original_Day_and_file_PTS_in_audit_only'}
        write(output/'provenance.json', provenance)
        write(output/'public_questions.json', questions)
        write(output/'manifest.json', manifest)
        write(output/'status.json', {'phase': 'loading_model', 'gpu': gpu})
        load_start = time.monotonic()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model, local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation='sdpa', device_map={'': 'cuda:0'}).eval()
        model.requires_grad_(False)
        processor = Qwen2_5_VLProcessor.from_pretrained(args.model, local_files_only=True)
        load_seconds = time.monotonic() - load_start
        bank = PersistentMemory(args.short_frames, args.medium_frames, args.long_frames,
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
                temporal_tokens_per_second=model.config.vision_config.tokens_per_second)
        write(output/'predictions.json', result['answers'])
        # A serializable memory snapshot is reusable by the Python API; the CLI
        # deliberately has no implicit resume that could duplicate answers.
        torch.save({'provenance':provenance, 'memory':bank.state_dict()}, output/'memory.pt')
        summary = {k:v for k,v in result.items() if k != 'answers'}
        summary.update(phase='finished', training=False, decode=decode_stats,
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
