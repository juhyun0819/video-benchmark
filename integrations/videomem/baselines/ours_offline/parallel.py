"""Supervise independent native-video QA shards and audit their merged coverage.

See PARALLEL.md. This module never downloads weights, restarts workers, or kills
processes. Only a fresh output directory can be launched.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from .backend import BACKBONES
from .inputs import load_inputs, sha256, finite_time
from .report import performance_breakdown
from baselines.streambench.scoring import score_answer


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def read_json(path):
    return json.loads(Path(path).read_text())


def _positive_int(value, label, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{label} must be an integer >= {minimum}')
    return value


def normalize_config(path):
    """Resolve relative paths against the config file, without loading models."""
    path = Path(path).resolve()
    config = read_json(path)
    allowed = {'root','canonical','sources','max_frames','sampling_fps','models','cpu_threads','decode_workers','poll_seconds','expected_questions'}
    if set(config)-allowed: raise ValueError(f'Unknown supervisor settings: {sorted(set(config)-allowed)}')
    for key in ('root','canonical','sources'):
        value = Path(config[key]).expanduser()
        config[key] = str((path.parent/value).resolve())
    config['max_frames'] = _positive_int(config.get('max_frames',32),'max_frames',2)
    config.setdefault('sampling_fps',None)
    if config['sampling_fps'] is not None and finite_time(config['sampling_fps'],'sampling_fps')<=0:
        raise ValueError('sampling_fps must be positive')
    config['cpu_threads'] = _positive_int(config.get('cpu_threads',4),'cpu_threads')
    if config['cpu_threads'] != 4:
        raise ValueError('Existing native worker uses four PyTorch CPU threads; cpu_threads must be 4')
    config['decode_workers'] = _positive_int(config.get('decode_workers',4),'decode_workers')
    interval = config.get('poll_seconds',5.)
    if isinstance(interval,bool) or not isinstance(interval,(float,int)) or not math.isfinite(interval) or not 0.1 <= interval <= 30:
        raise ValueError('poll_seconds must be between 0.1 and 30')
    config['poll_seconds'] = float(interval)
    if 'expected_questions' in config:
        _positive_int(config['expected_questions'],'expected_questions')
    models = config.get('models')
    if not isinstance(models,list) or not models: raise ValueError('At least one model allocation required')
    names, used_gpus = set(), set()
    options = {'name','backbone','model','model_revision','ledger','gpus','max_new_tokens','attention','dtype',
               'transformers_version','max_pixels','channels_last_3d','decode_workers'}
    for model in models:
        if not isinstance(model,dict) or set(model)-options: raise ValueError('Unknown model settings')
        name = model.get('name')
        if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',name) or name in names:
            raise ValueError('Model names must be unique safe directory names')
        names.add(name)
        if model.get('backbone') not in BACKBONES: raise ValueError('Unknown answer backbone')
        if not re.fullmatch(r'[0-9a-fA-F]{40}',model.get('model_revision','')):
            raise ValueError('Immutable 40-character model revision required')
        for key in ('model','ledger'):
            model[key] = str((path.parent/Path(model[key]).expanduser()).resolve())
        if not Path(model['model']).is_dir() or not Path(model['ledger']).is_file():
            raise ValueError(f'Missing local model directory or ledger for {name}')
        gpus = model.get('gpus')
        if not isinstance(gpus,list) or not gpus: raise ValueError('Explicit nonempty physical GPU allocation required')
        for gpu in gpus:
            _positive_int(gpu,'gpu',0)
            if gpu in used_gpus: raise ValueError('GPU allocations must be disjoint and contain no duplicates')
            used_gpus.add(gpu)
        model.setdefault('max_new_tokens',256)
        _positive_int(model['max_new_tokens'],'max_new_tokens')
        model.setdefault('attention','sdpa'); model.setdefault('dtype','bfloat16')
        model.setdefault('transformers_version','4.57.6'); model.setdefault('channels_last_3d',False)
        model.setdefault('max_pixels',None)
        model.setdefault('decode_workers',config['decode_workers'])
        _positive_int(model['decode_workers'],'model decode_workers')
        if model['attention'] not in ('sdpa','flash_attention_2','eager') or model['dtype'] not in ('bfloat16','float16','float32'):
            raise ValueError('Invalid attention or dtype')
        if type(model['channels_last_3d']) is not bool: raise ValueError('channels_last_3d must be boolean')
        if model['max_pixels'] is not None: _positive_int(model['max_pixels'],'max_pixels')
    return config


def expected_model(model):
    return {'backbone':model['backbone'],'model_path':model['model'],'model_revision':model['model_revision'],
            'device':'cuda:0','dtype':model['dtype'],'attn_implementation':model['attention'],
            'max_new_tokens':model['max_new_tokens'],'transformers_version':model['transformers_version'],
            'max_pixels':model['max_pixels'],'channels_last_3d':model['channels_last_3d']}


def build_plan(config, output):
    manifest, _, questions = load_inputs(config['root'],config['canonical'],config['sources'])
    if config.get('expected_questions',len(questions)) != len(questions):
        raise ValueError('Manifest count differs from explicit expected_questions')
    output = Path(output).resolve()
    plan = {'schema':'ours_offline_parallel_v1','config':config,'output':str(output),
            'canonical_sha256':sha256(config['canonical']),'sources_sha256':sha256(config['sources']),
            'scorer_sha256':manifest['scorer_sha256'],'qa_ids':[q['qa_id'] for q in questions],
            'code_sha256':{p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},
            'interpreter':sys.executable,'workers':[],'created_at':time.time()}
    for model in config['models']:
        if len(model['gpus']) > len(questions): raise ValueError('More QA shards than questions')
        model['ledger_sha256'] = sha256(model['ledger'])
        for shard,gpu in enumerate(model['gpus']):
            folder = output/model['name']/f'shard-{shard:03d}'
            argv = [sys.executable,'-m','baselines.ours_offline','run','--root',config['root'],
                '--canonical',config['canonical'],'--sources',config['sources'],'--output',str(folder),
                '--max-frames',str(config['max_frames']),'--shard',str(shard),'--shards',str(len(model['gpus'])),
                '--backbone',model['backbone'],'--model',model['model'],'--model-revision',model['model_revision'],
                '--ledger',model['ledger'],'--gpu',str(gpu),'--max-new-tokens',str(model['max_new_tokens']),
                '--decode-workers',str(model['decode_workers']),
                '--attention',model['attention'],'--dtype',model['dtype'],'--transformers-version',model['transformers_version']]
            if model['max_pixels'] is not None: argv.extend(['--max-pixels',str(model['max_pixels'])])
            if config.get('sampling_fps') is not None: argv.extend(['--sampling-fps',str(config['sampling_fps'])])
            if model['channels_last_3d']: argv.append('--channels-last-3d')
            plan['workers'].append({'model':model['name'],'shard':shard,'shards':len(model['gpus']),'gpu':gpu,
                'output':str(folder),'log':str(folder.with_suffix('.log')),'argv':argv,
                'qa_ids':plan['qa_ids'][shard::len(model['gpus'])]})
    return plan, questions


def probe_gpu(gpu):
    raw = subprocess.check_output(['nvidia-smi','-i',str(gpu),
        '--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    memory,utilization = [int(x.strip()) for x in raw.split(',')]
    if memory >= 1024 or utilization > 0:
        raise RuntimeError(f'GPU {gpu} occupied ({memory} MiB, {utilization}%); no jobs interrupted')
    return {'gpu':gpu,'memory_mib':memory,'utilization_percent':utilization,'checked_at':time.time()}


def sample_gpus(gpus):
    """One non-PyTorch sample; memory.used is process-external polling, not allocator peak."""
    raw = subprocess.check_output(['nvidia-smi','-i',','.join(str(g) for g in gpus),
        '--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15)
    result = {}
    for line in raw.strip().splitlines():
        gpu,memory,utilization = [int(x.strip()) for x in line.split(',')]
        result[str(gpu)] = {'memory_used_mib':memory,'utilization_percent':utilization,'sampled_at':time.time()}
    if set(result)!={str(g) for g in gpus}: raise ValueError('GPU metric sample omitted or added an index')
    return result


@contextmanager
def gpu_leases(gpus):
    """Exclude other supervisors of this user; worker occupancy checks still apply."""
    leases = []
    try:
        for gpu in sorted(gpus):
            path = Path('/tmp')/f'videomem-offline-{os.getuid()}-gpu-{gpu}.lock'
            descriptor = os.open(path,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
            leases.append(descriptor)
            try: fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: raise RuntimeError(f'GPU {gpu} reserved by another offline supervisor')
        yield
    finally:
        for descriptor in leases: os.close(descriptor)


def _worker_progress(worker):
    try: return read_json(Path(worker['output'])/'status.json')
    except (FileNotFoundError,ValueError): return {'status':'not_reported'}


def _audit_provenance(plan, worker, model, provenance):
    for key in ('canonical_sha256','sources_sha256','scorer_sha256','code_sha256'):
        if provenance.get(key) != plan[key]: raise ValueError(f'{key} differs from supervisor plan')
    for key in ('shard','shards','qa_ids'):
        if provenance.get(key) != worker[key]: raise ValueError(f'Worker {key} differs from allocation')
    if provenance.get('max_frames') != plan['config']['max_frames'] or provenance.get('model') != expected_model(model):
        raise ValueError('Worker model or frame budget differs from allocation')
    if provenance.get('decode_workers') != model['decode_workers']:
        raise ValueError('Worker decoder concurrency differs from allocation')
    if provenance.get('sampling_fps') != plan['config'].get('sampling_fps'):
        raise ValueError('Worker sampling rate differs from allocation')
    if provenance.get('verification',{}).get('ledger_sha256') != model['ledger_sha256']:
        raise ValueError('Worker model ledger differs from allocation')


def merge_model(plan, model, questions, worker_states):
    """Fail closed on duplicate/unknown IDs or mixed provenance; keep fixed denominator."""
    expected = {q['qa_id']:q for q in questions}
    rows, duplicates, errors, observed = {}, set(), [], []
    workers = [w for w in plan['workers'] if w['model']==model['name']]
    for worker in workers:
        folder = Path(worker['output'])
        predictions = folder/'predictions.jsonl'
        if not predictions.exists(): continue
        try:
            _audit_provenance(plan,worker,model,read_json(folder/'provenance.json'))
        except (OSError,ValueError,KeyError,TypeError) as exc:
            errors.append({'shard':worker['shard'],'error':f'Invalid provenance: {exc}'})
            continue
        with predictions.open() as stream:
            for line_number,line in enumerate(stream,1):
                try:
                    row = json.loads(line)
                    qa_id = row['qa_id']
                    if not isinstance(qa_id,str) or qa_id not in worker['qa_ids']:
                        raise ValueError('Unknown or wrong-shard QA ID')
                    observed.append(qa_id)
                    if qa_id in rows or qa_id in duplicates:
                        rows.pop(qa_id,None); duplicates.add(qa_id)
                        raise ValueError('Duplicate QA ID')
                    q = expected[qa_id]
                    if row.get('method') != 'Offline-'+model['backbone'] or row.get('query_time') != q['query_time'] or row.get('GT') != q['answer']:
                        raise ValueError('Method, query time or GT differs from canonical QA')
                    if row.get('status') not in ('ok','incomplete','error') or not isinstance(row.get('prediction'),str) or not isinstance(row.get('raw_output'),str):
                        raise ValueError('Malformed worker answer record')
                    stamps = [x['actual_global_timestamp'] for x in row['actual_frame_timestamps']]
                    if any(isinstance(t,bool) or not isinstance(t,(int,float)) or not math.isfinite(t) or t<0 or t>q['query_time']+1e-8 for t in stamps):
                        raise ValueError('Invalid or future frame timestamp')
                    if any(a>=b for a,b in zip(stamps,stamps[1:])):
                        raise ValueError('Nonchronological frame timestamps')
                    if row['status']=='ok' and not 2<=len(stamps)<=plan['config']['max_frames']:
                        raise ValueError('Invalid successful answer frame count')
                    score,kind = score_answer(row['prediction'],q)
                    row = dict(row,worker_reported_correct=row.get('correct'),
                               correct=bool(score and row['status']=='ok'),score_kind=kind,
                               worker_shard=worker['shard'],worker_gpu=worker['gpu'])
                    if row['status']!='ok' and not row.get('error'): row['error']='Worker answer did not complete successfully'
                    rows[qa_id] = row
                except (ValueError,KeyError,TypeError) as exc:
                    errors.append({'shard':worker['shard'],'line':line_number,'error':str(exc)})
    missing = [qa_id for qa_id in expected if qa_id not in rows]
    merged = []
    for qa_id,q in expected.items():
        row = rows.get(qa_id)
        if row is None:
            row = {'method':'Offline-'+model['backbone'],'qa_id':qa_id,'query_time':q['query_time'],
                'actual_frame_timestamps':[],'prediction':'','raw_output':'','GT':q['answer'],
                'correct':False,'answer_latency':None,'stream_update_time':0.,'status':'error',
                'error':'Supervisor coverage audit: duplicate QA ID' if qa_id in duplicates else 'Supervisor coverage audit: missing or rejected worker result',
                'supervisor_generated':True}
        merged.append(row)
    successful_workers = all(worker_states.get(f"{w['model']}:{w['shard']}",{}).get('returncode')==0 for w in workers)
    correct = sum(row['correct'] is True for row in merged)
    bad = sum(row['status']!='ok' for row in merged)
    coverage = not missing and not errors and len(observed)==len(expected) and not duplicates
    summary = {'model':model['name'],'backbone':model['backbone'],'status':'complete' if coverage and not bad and successful_workers else 'incomplete_or_failed',
        'expected':len(expected),'accepted_worker_rows':len(rows),'merged_rows':len(merged),
        'actual_successful_qa':sum(row['status']=='ok' for row in rows.values()),
        'synthesized_missing_or_rejected':len(missing),
        'missing_or_rejected_ids':missing,'duplicate_ids':sorted(duplicates),'audit_errors':errors,
        'exact_unique_coverage':coverage,'all_workers_exit_zero':successful_workers,
        'correct':correct,'errors_or_incomplete':bad,'accuracy_over_expected':correct/len(expected),
        'canonical_sha256':plan['canonical_sha256'],'sources_sha256':plan['sources_sha256'],
        'scorer_sha256':plan['scorer_sha256'],'max_frames':plan['config']['max_frames'],
        'decode_workers':model['decode_workers'],
        'model_settings':expected_model(model),'model_ledger_sha256':model['ledger_sha256']}
    summary['sampling_fps'] = plan['config'].get('sampling_fps')
    summary['performance'] = performance_breakdown(questions,merged)
    entries = [worker_states.get(f"{w['model']}:{w['shard']}",{}) for w in workers]
    if all('started_at' in e and 'finished_at' in e for e in entries):
        wall = max(e['finished_at'] for e in entries)-min(e['started_at'] for e in entries)
        summary['wall_seconds_including_verification_and_model_setup'] = wall
        summary['accepted_qa_per_second_including_setup'] = len(rows)/wall if wall>0 else None
    folder = Path(plan['output'])/model['name']; folder.mkdir(parents=True,exist_ok=True)
    destination = folder/'predictions.merged.jsonl'; temporary = destination.with_suffix('.jsonl.tmp')
    with temporary.open('w') as stream:
        for row in merged: stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(destination)
    summary['merged_sha256'] = sha256(destination)
    atomic_json(folder/'summary.json',summary)
    return summary


def merge_all(plan, states):
    config = plan['config']
    if sha256(config['canonical'])!=plan['canonical_sha256'] or sha256(config['sources'])!=plan['sources_sha256']:
        raise ValueError('Inputs changed since supervisor launch')
    _,_,questions = load_inputs(config['root'],config['canonical'],config['sources'])
    if [q['qa_id'] for q in questions]!=plan['qa_ids']: raise ValueError('QA order or coverage changed')
    return [merge_model(plan,model,questions,states) for model in config['models']]


def supervise(plan, *, popen=subprocess.Popen, occupancy=probe_gpu, metrics=sample_gpus, sleeper=time.sleep):
    output = Path(plan['output']); output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'plan.json',plan)
    state = {'status':'preflight','supervisor_pid':os.getpid(),'started_at':time.time(),'workers':{},'gpu_checks':[],
             'gpu_metrics':{},'memory_metric':'nvidia-smi observed memory.used; polling can miss peaks and includes all GPU processes'}
    atomic_json(output/'status.json',state)
    active = []
    try:
        with gpu_leases([w['gpu'] for w in plan['workers']]):
            # Check every requested GPU before launching any worker. Each worker
            # independently rechecks occupancy after verifying its local weights.
            state['gpu_checks'] = [occupancy(w['gpu']) for w in plan['workers']]
            env = os.environ.copy()
            env.pop('CUDA_VISIBLE_DEVICES',None)
            env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                USE_TF='0',USE_FLAX='0',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',
                OMP_NUM_THREADS=str(plan['config']['cpu_threads']),MKL_NUM_THREADS=str(plan['config']['cpu_threads']),
                OPENBLAS_NUM_THREADS=str(plan['config']['cpu_threads']),NUMEXPR_NUM_THREADS=str(plan['config']['cpu_threads']))
            state['status'] = 'launching'; atomic_json(output/'status.json',state)
            for worker in plan['workers']:
                Path(worker['log']).parent.mkdir(parents=True,exist_ok=True)
                with Path(worker['log']).open('xb') as log:
                    process = popen(worker['argv'],cwd=str(Path(__file__).resolve().parents[2]),
                                    env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                key = f"{worker['model']}:{worker['shard']}"
                active.append((worker,key,process))
                state['workers'][key] = {'pid':process.pid,'gpu':worker['gpu'],'returncode':None,
                    'started_at':time.time(),'expected':len(worker['qa_ids']),'output':worker['output'],'log':worker['log']}
                atomic_json(output/'status.json',state)
            while True:
                running = False
                for worker,key,process in active:
                    entry = state['workers'][key]
                    code = process.poll()
                    entry.update(returncode=code,progress=_worker_progress(worker))
                    if code is None: running=True
                    elif 'finished_at' not in entry: entry['finished_at']=time.time()
                try:
                    samples = metrics([w['gpu'] for w in plan['workers']])
                    for gpu,sample in samples.items():
                        old = state['gpu_metrics'].get(gpu,{})
                        state['gpu_metrics'][gpu] = dict(sample,samples=old.get('samples',0)+1,
                            peak_observed_memory_used_mib=max(sample['memory_used_mib'],old.get('peak_observed_memory_used_mib',0)))
                except (OSError,ValueError,subprocess.SubprocessError) as exc:
                    state['gpu_metrics_last_error'] = f'{type(exc).__name__}: {exc}'
                state.update(status='running' if running else 'merging',updated_at=time.time(),
                             elapsed_seconds=time.time()-state['started_at'])
                completed = sum(e.get('progress',{}).get('completed_qa',0) for e in state['workers'].values())
                state['worker_reported_completed_qa'] = completed
                state['expected_model_qa_pairs'] = len(plan['qa_ids'])*len(plan['config']['models'])
                state['worker_reported_qa_per_second_including_setup'] = completed/state['elapsed_seconds'] if state['elapsed_seconds']>0 else None
                atomic_json(output/'status.json',state)
                if not running: break
                sleeper(plan['config']['poll_seconds'])
            summaries = merge_all(plan,state['workers'])
            state.update(status='complete' if all(s['status']=='complete' for s in summaries) else 'incomplete_or_failed',
                         summaries=summaries,finished_at=time.time())
            atomic_json(output/'status.json',state)
            return 0 if state['status']=='complete' else 2
    except BaseException as exc:
        state.update(status='supervisor_failed',error=f'{type(exc).__name__}: {exc}',updated_at=time.time(),
                     worker_policy='No workers were killed or restarted; recorded workers may still be running')
        for worker,key,process in active:
            state['workers'][key].update(returncode=process.poll(),progress=_worker_progress(worker))
        atomic_json(output/'status.json',state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    for command in ('plan','run'):
        p = sub.add_parser(command); p.add_argument('--config',type=Path,required=True)
        p.add_argument('--output',type=Path,required=True)
    for command in ('status','merge'):
        p = sub.add_parser(command); p.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(argv)
    if args.command in ('plan','run'):
        plan,_ = build_plan(normalize_config(args.config),args.output)
        if args.command=='run': return supervise(plan)
        args.output.mkdir(parents=True,exist_ok=False)
        atomic_json(args.output/'plan.json',plan)
        atomic_json(args.output/'status.json',{'status':'planned','workers':len(plan['workers']),
            'expected_per_model':len(plan['qa_ids']),'gpu_checked':False,'models_loaded':False})
        return 0
    plan = read_json(args.output/'plan.json'); state = read_json(args.output/'status.json')
    if args.command=='status':
        state['live_worker_progress'] = {f"{w['model']}:{w['shard']}":_worker_progress(w) for w in plan['workers']}
        print(json.dumps(state,ensure_ascii=False,indent=2)); return 0
    if state.get('status') in ('launching','running','merging','preflight'):
        raise ValueError('Supervisor is active; use status until it finishes')
    summaries = merge_all(plan,state.get('workers',{}))
    print(json.dumps(summaries,ensure_ascii=False,indent=2))
    return 0 if all(s['status']=='complete' for s in summaries) else 2


if __name__=='__main__': raise SystemExit(main())
