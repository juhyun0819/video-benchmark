"""Plan or run offline Qwen3/LLaVA on the shared ours canonical QA manifest."""
from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path
import subprocess
import time

from .backend import BACKBONES, NativeVideoBackend, verify_model_ledger
from .inputs import decode_samples, load_inputs, uniform_samples, rate_uniform_samples, finite_time
from .report import performance_breakdown
from .inputs import sha256
from baselines.streambench.scoring import score_answer


def add_model_arguments(parser):
    parser.add_argument('--backbone', choices=sorted(BACKBONES), required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--model-revision', required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--attention', choices=['sdpa','flash_attention_2','eager'], default='sdpa')
    parser.add_argument('--dtype', choices=['bfloat16','float16','float32'], default='bfloat16')
    parser.add_argument('--transformers-version', default='4.57.6')
    parser.add_argument('--max-pixels', type=int, help='Qwen native whole-video T*H*W resize budget; not per-frame pixels')
    parser.add_argument('--channels-last-3d', action='store_true')


def backend_config(args):
    return {'backbone':args.backbone,'model_path':str(args.model),'model_revision':args.model_revision,
            'device':'cuda:0','dtype':args.dtype,'attn_implementation':args.attention,
            'max_new_tokens':args.max_new_tokens,'transformers_version':args.transformers_version,
            'max_pixels':args.max_pixels,'channels_last_3d':args.channels_last_3d}


def bind_gpu(gpu):
    if type(gpu) is not int or gpu < 0:
        raise ValueError('gpu must be a nonnegative physical GPU index')
    used = int(subprocess.check_output(['nvidia-smi','-i',str(gpu),
        '--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    if used >= 1024:
        raise RuntimeError(f'GPU {gpu} is occupied ({used} MiB); existing jobs are not interrupted')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                      USE_TF='0',USE_FLAX='0',TOKENIZERS_PARALLELISM='false')


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def select_questions(questions, *, shard=0, shards=1, qa_ids=None, limit=None):
    if type(shards) is not int or shards < 1 or type(shard) is not int or not 0 <= shard < shards:
        raise ValueError('Require 0 <= shard < shards')
    if qa_ids is not None:
        if not qa_ids or len(qa_ids) != len(set(qa_ids)):
            raise ValueError('qa_ids must be nonempty and unique')
        requested = set(qa_ids)
        if not requested <= {q['qa_id'] for q in questions}:
            raise ValueError('Unknown QA IDs in selected day subset')
        questions = [q for q in questions if q['qa_id'] in requested]
    selected = questions[shard::shards]
    if limit is not None:
        if type(limit) is not int or limit < 1:
            raise ValueError('limit must be positive')
        selected = selected[:limit]
    if not selected:
        raise ValueError('Empty QA shard')
    return selected


def evaluate_question(root, sources, question, backend, max_frames, *, decoder=decode_samples, selector=uniform_samples):
    """Labels remain on the evaluator side; only public strings reach backend."""
    record = {'method':f'Offline-{backend.backbone}','qa_id':question['qa_id'],
        'query_time':question['query_time'],'actual_frame_timestamps':[],
        'prediction':'','raw_output':'','GT':question['answer'],'correct':False,
        'answer_latency':None,'stream_update_time':0.0,'decode_seconds':0.0,
        'status':'error','error':None,'need_audio':question.get('need_audio'),
        'need_name':question.get('need_name')}
    try:
        requested = selector(sources, question['query_time'], max_frames)
        tick = time.perf_counter()
        frames, timestamps = decoder(root,requested,question['query_time'])
        record['decode_seconds'] = time.perf_counter()-tick
        record['actual_frame_timestamps'] = timestamps
        actual = [x['actual_global_timestamp'] for x in timestamps]
        if len(frames) != len(timestamps) or not actual or any(t > question['query_time']+1e-8 for t in actual):
            raise ValueError('Invalid or future decoded inputs')
        tick = time.perf_counter()
        result = backend.answer(frames,timestamps,question['question'],list(question['choices']))
        record['answer_latency'] = time.perf_counter()-tick
        record.update(prediction=result['prediction'],raw_output=result['raw_output'],
            status='ok' if result['complete'] else 'incomplete',
            error=None if result['complete'] else 'Generation did not finish with nonempty EOS-terminated output',
            generated_token_ids=result.get('generated_token_ids'),input_tokens=result.get('input_tokens'),
            model_inputs_metadata=result.get('model_inputs_metadata'))
        record['correct'],record['score_kind'] = score_answer(record['prediction'],question)
        if not result['complete']:
            record['correct'] = False
    except Exception as exc:
        record.update(status='error',error=f'{type(exc).__name__}: {exc}')
    return record


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    prepare=sub.add_parser('prepare',help='Probe media metadata and emit an empty pending-QA template')
    prepare.add_argument('--root',type=Path,required=True)
    prepare.add_argument('--source-list',type=Path,required=True)
    prepare.add_argument('--output',type=Path,required=True)
    validate=sub.add_parser('validate',help='Validate existing sources with optional QA; no model or GPU')
    validate.add_argument('--root',type=Path,required=True)
    validate.add_argument('--sources',type=Path,required=True)
    validate.add_argument('--canonical',type=Path)
    for command in ['plan','run']:
        p=sub.add_parser(command)
        p.add_argument('--root',type=Path,required=True)
        p.add_argument('--canonical',type=Path,required=True,help='Canonical QA JSON file (not directory)')
        p.add_argument('--sources',type=Path,required=True,help='Canonical source-index JSON file')
        p.add_argument('--output',type=Path,required=True)
        p.add_argument('--max-frames',type=int,required=True,help='Explicit uniform raw-frame read budget; >=2')
        p.add_argument('--decode-workers',type=int,default=1,help='Independent CPU raw-frame decoder threads per QA')
        p.add_argument('--sampling-fps',type=float,help='Optional source-local rate grid before uniform frame cap')
        p.add_argument('--days',type=int,nargs='+')
        p.add_argument('--qa-ids',nargs='+')
        p.add_argument('--shard',type=int,default=0)
        p.add_argument('--shards',type=int,default=1)
        p.add_argument('--limit',type=int)
        if command=='run':add_model_arguments(p)
    return parser


def main(argv=None):
    args=build_parser().parse_args(argv)
    if args.command=='prepare':
        from .prepare import write_prepared
        print(json.dumps(write_prepared(args.root,args.source_list,args.output),indent=2))
        return 0
    if args.command=='validate':
        from .inputs import validate_sources
        sources=validate_sources(args.root,json.loads(args.sources.read_text())['sources'])
        questions=[]
        if args.canonical is not None:
            manifest=json.loads(args.canonical.read_text())
            if manifest.get('questions'):
                _,sources,questions=load_inputs(args.root,args.canonical,args.sources)
            else:
                scorer=Path(__file__).parents[1]/'streambench/scoring.py'
                if manifest.get('scorer_sha256')!=sha256(scorer):raise ValueError('Pending-QA scorer hash mismatch')
                if manifest.get('sources_sha256') not in (None,sha256(args.sources)):raise ValueError('Pending-QA source index mismatch')
        print(json.dumps({'status':'ready' if questions else 'sources_ready_awaiting_QA','sources':len(sources),'questions':len(questions),'inference_performed':False},indent=2))
        return 0
    if args.output.exists():raise FileExistsError(args.output)
    if args.decode_workers < 1:raise ValueError('decode-workers must be positive')
    if args.sampling_fps is not None and finite_time(args.sampling_fps,'sampling-fps')<=0:
        raise ValueError('sampling-fps must be positive')
    selector = uniform_samples if args.sampling_fps is None else partial(rate_uniform_samples,sample_fps=args.sampling_fps)
    manifest,sources,questions=load_inputs(args.root,args.canonical,args.sources,args.days)
    questions=select_questions(questions,shard=args.shard,shards=args.shards,qa_ids=args.qa_ids,limit=args.limit)
    plans=[{'qa_id':q['qa_id'],'query_time':q['query_time'],
            'requested_frames':selector(sources,q['query_time'],args.max_frames)} for q in questions]
    args.output.mkdir(parents=True,exist_ok=False)
    provenance={'canonical_sha256':sha256(args.canonical),'sources_sha256':sha256(args.sources),
        'scorer_sha256':manifest['scorer_sha256'],
        'sampling':('uniform original CFR frames within real global-time causal prefix' if args.sampling_fps is None
                    else 'uniformly capped source-local rate grid within real global-time causal prefix'),
        'max_frames':args.max_frames,'decode_workers':args.decode_workers,
        'sampling_fps':args.sampling_fps,
        'sampling_grid':'all native CFR positions' if args.sampling_fps is None else 'source-local rate grid, floor to native CFR, uniformly capped over complete causal prefix',
        'days':args.days,'shard':args.shard,'shards':args.shards,
        'qa_ids':[q['qa_id'] for q in questions],'feature_cache':False,'persistent_kv':False,
        'audio_input':False,'gt_fields_passed_to_model':False,
        'implementation':'explicit read-budget adaptation with native HF video processor/model; not untouched default sampling',
        'code_sha256':{p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}}
    write_json(args.output/'plan.json',plans)
    write_json(args.output/'provenance.json',provenance)
    if args.command=='plan':
        write_json(args.output/'status.json',{'status':'planned','questions':len(questions),'inference_performed':False})
        return 0
    try:
        write_json(args.output/'status.json',{'status':'verifying_model','completed_qa':0})
        verification=verify_model_ledger(args.model,args.ledger,args.backbone,args.model_revision)
        bind_gpu(args.gpu)
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        config=backend_config(args)
        backend=NativeVideoBackend(config)
        provenance.update(model=config,verification=verification,gpu=args.gpu,
                          torch=torch.__version__,generation={'do_sample':False,'num_beams':1,'max_new_tokens':args.max_new_tokens})
        write_json(args.output/'provenance.json',provenance)
        write_json(args.output/'status.json',{'status':'loading','completed_qa':0})
        backend.initialize()
        completed=errors=incomplete=correct=0
        result_rows=[]
        with (args.output/'predictions.jsonl').open('w') as output:
            for q in questions:
                row=evaluate_question(args.root,sources,q,backend,args.max_frames,
                    decoder=partial(decode_samples,workers=args.decode_workers),selector=selector)
                output.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');output.flush()
                result_rows.append(row)
                completed+=1;errors+=int(row['status']=='error');incomplete+=int(row['status']=='incomplete');correct+=int(row['correct'] is True)
                write_json(args.output/'status.json',{'status':'running','completed_qa':completed,'errors':errors,'incomplete':incomplete,'correct':correct,
                    'performance':performance_breakdown(questions,result_rows)})
                if row['status']=='error' and 'OutOfMemoryError' in row['error']:
                    raise RuntimeError('GPU memory exceeded; reduce the explicitly declared frame budget and use a new output directory')
        status='complete' if errors==incomplete==0 else 'incomplete_or_failed'
        write_json(args.output/'status.json',{'status':status,'expected':len(questions),'completed_qa':completed,
            'errors':errors,'incomplete':incomplete,'correct':correct,'accuracy_over_expected':correct/len(questions),
            'performance':performance_breakdown(questions,result_rows)})
        return 0 if status=='complete' else 2
    except Exception as exc:
        status_path=args.output/'status.json'
        status=json.loads(status_path.read_text()) if status_path.exists() else {}
        status.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        write_json(status_path,status)
        raise


if __name__=='__main__':
    raise SystemExit(main())
