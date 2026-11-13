"""Executable native Cambrian-S evaluation; requires explicit local dependencies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from .runner import atomic_json, load_inputs, run_stream
from .stream_frames import sample_plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, help='Labeled canonical QA; omit for a CPU-only QA-pending preflight')
    parser.add_argument('--plan-only', action='store_true', help='Validate inputs and write a plan without loading a model or consuming the stream')
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True, help='Native Cambrian method JSON; no fallback model')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--upstream', type=Path)
    parser.add_argument('--budget-gib', type=float)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--days', type=int, nargs='+', help='Optional QA day labels; all earlier source observations are retained')
    parser.add_argument('--fps', type=float, default=1.0)
    parser.add_argument('--chunk-frames', type=int, default=8)
    parser.add_argument('--checkpoint-every', type=int, default=10)
    parser.add_argument('--max-questions', type=int)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    if config.get('method', 'cambrian_s') != 'cambrian_s':parser.error('Only native cambrian_s is supported')
    if args.model:config['model_path'] = str(args.model.resolve())
    if args.upstream:config['upstream_source'] = str(args.upstream.resolve())
    for key in ('model_path', 'upstream_source'):
        if not config.get(key):parser.error(f'{key} must name a local directory')
        config[key] = str(Path(config[key]).resolve())
    if args.budget_gib is not None:config['memory_budget_bytes'] = int(args.budget_gib * 2**30)
    if type(config.get('memory_budget_bytes')) is not int or config['memory_budget_bytes'] <= 0:
        parser.error('Supply positive memory_budget_bytes in config or --budget-gib')
    config.update(device='cuda:0', local_files_only=True)
    questions, sources, identity = load_inputs(args.root, args.manifest, args.sources, args.days)
    identity['config_sha256'] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if args.output.exists():parser.error('Output directory must not already exist')
    plan = sample_plan(sources, fps=args.fps)
    if args.manifest is None or args.plan_only:
        if args.resume is not None:parser.error('A resume checkpoint requires an unchanged QA manifest and an actual run')
        args.output.mkdir(parents=True, exist_ok=False)
        result = {'status': 'awaiting_qa' if args.manifest is None else 'preflight_complete',
            'target_dataset': 'ours', 'model_loaded': False, 'stream_observed': False,
            'inputs': identity, 'source_count': len(sources), 'planned_frames': len(plan),
            'first_frame_timestamp': plan[0]['requested_global_timestamp'],
            'last_frame_timestamp': plan[-1]['requested_global_timestamp'],
            'question_count': len(questions), 'sampling_fps': args.fps,
            'local_directories_present': {key: Path(config[key]).is_dir() for key in ('model_path', 'upstream_source')},
            'checkpoint_created': False,
            'next_step': 'Supply the final canonical QA manifest before starting the chronological run. Future memory cannot answer past QA.'}
        atomic_json(args.output/'status.json', result)
        atomic_json(args.output/'plan.json', {'config': config, **result})
        print(json.dumps(result, indent=2))
        return result
    for key in ('model_path', 'upstream_source'):
        if not Path(config[key]).is_dir():parser.error(f'{key} must be an existing local directory')
    used = int(subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
               '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
    if used >= 1024:raise RuntimeError(f'GPU {args.gpu} is occupied ({used} MiB); no other job was stopped')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      USE_TF='0', USE_FLAX='0', TOKENIZERS_PARALLELISM='false')
    import torch
    import transformers
    from baselines.streambench.adapters.cambrian_s import CambrianSMethod, SOURCE_REVISION, MODEL_REVISION
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if transformers.__version__ != '4.37.0':
        raise RuntimeError('Native Cambrian-S requires its isolated transformers==4.37.0 environment')
    config.setdefault('model_revision', MODEL_REVISION)
    identity['config_sha256'] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    provenance = {'method': 'Cambrian-S-7B-LFP', 'native_source_revision': SOURCE_REVISION,
        'config': config, 'inputs': identity, 'fps': args.fps, 'chunk_frames': args.chunk_frames,
        'gpu': args.gpu, 'torch': torch.__version__, 'transformers': transformers.__version__,
        'audio_input': False, 'training': False, 'day_boundary_reset': False,
        'qa_fork_restore': True, 'visual_feature_cache': False, 'generation': 'native greedy with repetition penalty',
        'weights_verification': 'Local model architecture checked by native adapter; external file provenance required.',
        'scorer': 'baselines.streambench.scoring.score_answer',
        'decoder_sha256': hashlib.sha256(Path(__file__).with_name('stream_frames.py').read_bytes()).hexdigest(),
        'runner_sha256': hashlib.sha256(Path(__file__).with_name('runner.py').read_bytes()).hexdigest(),
        'cli_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    method = CambrianSMethod(config)
    def stats():
        return {'peak_vram_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_vram_reserved_bytes': torch.cuda.max_memory_reserved()}
    try:
        result = run_stream(method, root=args.root, questions=questions, sources=sources,
            identity=identity, output=args.output, fps=args.fps, chunk_frames=args.chunk_frames,
            checkpoint_every=args.checkpoint_every, max_questions=args.max_questions, resume=args.resume,
            synchronize=torch.cuda.synchronize, memory_stats=stats)
    finally:
        if args.output.is_dir():atomic_json(args.output / 'provenance.json', provenance)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
