#!/usr/bin/env python3
"""Durable producer + validated smoke gate + overlapping CPU memory consumer."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[2]


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def feature_digest(directory, flux=False):
    # Compare values/provenance through typed loaders, independent of shard size.
    from baselines.feature_memory.byte_memory import SnapshotReader
    from baselines.feature_memory.fluxmem import FluxMemSnapshotReader
    import torch
    reader = FluxMemSnapshotReader(directory) if flux else SnapshotReader(directory)
    result = hashlib.sha256()
    try:
        for batch in reader.iter_batches(max_tokens=1024):
            result.update(batch['features'].contiguous().view(torch.uint8).numpy().tobytes())
            for name, value in sorted(batch['metadata'].items()):
                result.update(name.encode())
                result.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    finally:
        reader.close()
    return result.hexdigest()


def validate_smoke(smoke, golden):
    summary, reference = load(smoke/'summary.json'), load(golden/'summary.json')
    if (summary['status'] != 'completed' or summary['total_frames'] != 300 or summary['consumer_count'] != 16
            or summary['num_encoder_calls'] != 0 or not summary['all_hard_caps_verified_each_update']
            or not summary['snapshot_save_load_consistency_verified']
            or summary['cleanup']['worker_processes_alive_after_close'] != 0
            or not summary['cleanup']['shared_memory_unlinked']):
        raise ValueError('CPU process smoke did not satisfy the construction contract')
    if summary['ordered_shared_features_sha256'] != reference['ordered_shared_features_sha256']:
        raise ValueError('CPU stage did not consume the same real encoder features as the validated smoke')
    old_rows = {(r['method'],r['label']):r for r in reference['rows']}
    for row in summary['rows']:
        old = old_rows[(row['method'],row['label'])]
        if row['hard_budget_bytes'] != old['hard_budget_bytes'] or row['actual_memory_bytes'] > row['hard_budget_bytes']:
            raise ValueError('Fixed full-stream budget mismatch')
        for key in ('retained_visual_tokens','num_merged_tokens','num_removed_temporal_duplicates','fallback_eviction_events'):
            if key in old and row.get(key) != old[key]:
                raise ValueError(f'Native result mismatch: {row["method"]}/{row["label"]}/{key}')
    compared = {}
    for method in ('fifo','fluxmem'):
        suffix = Path('snapshots') / method / '5pct' / '300s'
        actual = feature_digest(smoke/suffix, flux=method=='fluxmem')
        expected = feature_digest(golden/suffix, flux=method=='fluxmem')
        if actual != expected:
            raise ValueError('Stored feature/provenance values changed: ' + method)
        compared[method] = actual
    result = {'passed': True, 'real_frames': 300, 'states':16,
        'encoder_calls_in_cpu_stage':0, 'shared_input_digest_matches_reference':True,
        'all_state_retained_counts_match_reference':True,
        'exact_feature_and_provenance_digest':compared,
        'snapshot_hashes_verified_by_workers':True,'cleanup':summary['cleanup'],
        'summary_sha256':hashlib.sha256((smoke/'summary.json').read_bytes()).hexdigest(),
        'validated_at_utc':datetime.now(timezone.utc).isoformat()}
    save(smoke/'validation.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('archive','output','smoke-output','job-status','seed-run','golden-smoke'):
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--project-root', type=Path, default=REPO.parent,
                        help='Parent workspace containing data/ and models/ (default: repository parent)')
    parser.add_argument('--python', type=Path, default=Path(sys.executable),
                        help='Python executable with the method dependencies (default: current interpreter)')
    parser.add_argument('--videos', type=Path, nargs='+', help='Video paths in chronological order')
    parser.add_argument('--model', type=Path, help='Local Qwen2.5-VL-7B-Instruct checkpoint')
    parser.add_argument('--dataset-status', type=Path, help='Verified source inventory status JSON')
    parser.add_argument('--timeline', type=Path, help='Optional source timeline JSON')
    parser.add_argument('--devices', nargs='+', default=list(map(str, range(8))),
                        help='Physical CUDA device identifiers for the producer (default: 0 through 7)')
    args = parser.parse_args()
    args.project_root = args.project_root.expanduser().resolve()
    # Resolving a venv interpreter symlink would select its base interpreter and
    # silently lose the environment's installed method dependencies.
    args.python = args.python.expanduser().absolute()
    for name in ('archive','output','smoke_output','job_status','seed_run','golden_smoke'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    videos = args.videos or [args.project_root/'data/videos'/f'day{day}.mp4' for day in range(1,8)]
    videos = [path.expanduser().resolve() for path in videos]
    model = (args.model or args.project_root/'models/Qwen2.5-VL-7B-Instruct').expanduser().resolve()
    dataset_status = (args.dataset_status or args.project_root/'dataset_download_status.json').expanduser().resolve()
    timeline = args.timeline
    if timeline is None:
        candidate = args.project_root/'data/daywise_annotations_qa_v2_20261108/timeline.json'
        timeline = candidate if candidate.is_file() else None
    if timeline is not None:
        timeline = timeline.expanduser().resolve()
    for path in (args.archive,args.output,args.smoke_output,args.job_status):
        if path.exists():
            raise FileExistsError(path)
    args.job_status.parent.mkdir(parents=True,exist_ok=True)
    args.archive.parent.mkdir(parents=True,exist_ok=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    golden = load(args.golden_smoke/'summary.json')
    receipt = load(args.golden_smoke/'validation.json')
    if not receipt.get('passed') or receipt['summary_sha256'] != hashlib.sha256((args.golden_smoke/'summary.json').read_bytes()).hexdigest():
        raise ValueError('Original real encoder smoke has no valid independent receipt')
    plan = golden['plan']
    required = (plan['sum_all_state_hard_budgets']*7 + plan['full_feature_bytes']) * 1.02
    if shutil.disk_usage(args.archive.parent).free < required:
        raise RuntimeError('Insufficient space for feature archive, live banks and all snapshots')
    producer = consumer = None
    logs = []
    phase = 'starting'

    def publish(state='running', **extra):
        info = {'state':state,'phase':phase,'updated_at_utc':datetime.now(timezone.utc).isoformat(),
            'launcher_pid':os.getpid(),'producer_pid':producer.pid if producer else None,
            'consumer_pid':consumer.pid if consumer else None,'archive':str(args.archive),
            'output':str(args.output),'smoke_output':str(args.smoke_output),
            'producer_log':str(args.archive.with_suffix('.log')),
            'consumer_log':str(args.output.with_suffix('.log')),
            'fixed_caps_bytes':[s['hard_budget_bytes'] for s in plan['settings']],
            'state_count':16,'feature_archive_authorized':True, **extra}
        save(args.job_status,info)
        print(json.dumps(info),flush=True)

    def spawn(command,path,gpu=False):
        log=path.open('x'); logs.append(log)
        return subprocess.Popen(command,cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES=','.join(args.devices) if gpu else ''))

    try:
        command=[str(args.python),'-m','baselines.feature_memory','archive',
            '--videos',*[str(path) for path in videos],
            '--dataset-status',str(dataset_status),
            '--model',str(model),
            '--mode','parallel','--devices',*map(str, range(len(args.devices))),
            '--archive-frames-per-shard','100','--chunk-frames','8','--max-inflight-chunks','8',
            '--seed-run',str(args.seed_run),'--output',str(args.archive)]
        if timeline is not None:
            command += ['--timeline', str(timeline)]
        producer=spawn(command,args.archive.with_suffix('.log'),gpu=True)
        phase='publishing_initial_shards'
        while not (args.archive/'ready_00000002.json').exists():
            if producer.poll() is not None:
                raise RuntimeError('Producer exited before three complete shards')
            publish(); time.sleep(5)
        consume=[str(args.python),'-m','baselines.feature_memory','consume','--archive',str(args.archive),
                 '--batch-frames','100','--shard-bytes','268435456']
        phase='cpu_smoke'
        consumer=spawn([*consume,'--max-frames','300','--snapshot-seconds','300',
                        '--output',str(args.smoke_output)],args.smoke_output.with_suffix('.log'))
        while consumer.poll() is None:
            publish(); time.sleep(10)
        if consumer.returncode:
            raise RuntimeError(f'CPU smoke exited {consumer.returncode}')
        phase='validating_cpu_smoke'; publish()
        validate_smoke(args.smoke_output,args.golden_smoke)
        phase='overlapped_producer_consumer'
        consumer=spawn([*consume,'--output',str(args.output)],args.output.with_suffix('.log'))
        while producer.poll() is None or consumer.poll() is None:
            if consumer.poll() not in (None,0):
                phase='consumer_failed_archive_continues'
            if producer.poll() not in (None,0):
                phase='producer_failed'
            if producer.poll()==0 and phase=='overlapped_producer_consumer':
                phase='cpu_processing_archive_complete'
            publish(producer_exit_code=producer.poll(),consumer_exit_code=consumer.poll())
            time.sleep(15)
        okay=producer.returncode==consumer.returncode==0
        phase='complete' if okay else 'failed'
        publish('completed' if okay else 'failed',producer_exit_code=producer.returncode,
                consumer_exit_code=consumer.returncode,summary=str(args.output/'summary.json'))
        return 0 if okay else 1
    except BaseException as exc:
        phase='failed'
        # Preserve all published shards and completed snapshots. End children
        # gracefully if the supervisor cannot carry out its validation contract.
        import signal
        for child in (consumer,producer):
            if child is not None and child.poll() is None:
                child.send_signal(signal.SIGINT)
        publish('failed',error_type=type(exc).__name__,error=str(exc)[:2000])
        raise
    finally:
        for log in logs: log.close()


if __name__=='__main__':
    sys.path.insert(0,str(REPO))
    raise SystemExit(main())
