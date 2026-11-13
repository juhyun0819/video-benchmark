"""Plan or execute a privileged GT-source-clip reference; no automatic model run."""
import argparse
import json
import time
from pathlib import Path
from baselines.ours_offline.inputs import sha256
from baselines.ours_offline.inputs import load_inputs
from .inputs import load_evidence
from .evaluate import plan_question, answer_question


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('plan','run'):
        command=sub.add_parser(name)
        command.add_argument('--root',type=Path,required=True)
        command.add_argument('--manifest',type=Path,required=True)
        command.add_argument('--sources',type=Path,required=True)
        command.add_argument('--evidence',type=Path,help='Optional sidecar; otherwise use each QA evidence field')
        command.add_argument('--output',type=Path,required=True)
        command.add_argument('--max-frames',type=int,required=True)
        command.add_argument('--days',type=int,nargs='+',default=None)
        command.add_argument('--shard',type=int,default=0)
        command.add_argument('--shards',type=int,default=1)
        if name=='run':
            from baselines.ours_offline.run import add_model_arguments
            add_model_arguments(command)
    return p


def write_json(path,data):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def execute(args):
    if args.max_frames<2 or not 0<=args.shard<args.shards:
        raise ValueError('Need max_frames>=2 and 0<=shard<shards')
    manifest,sources,questions=load_inputs(args.root,args.manifest,args.sources,args.days)
    annotations=load_evidence(args.evidence,manifest,args.manifest)
    questions=questions[args.shard::args.shards]
    if not questions:raise ValueError('Empty QA shard')
    args.output.mkdir(parents=True,exist_ok=False)
    status={'status':'planning','expected':len(questions),'completed':0,'errors':0}
    write_json(args.output/'status.json',status)
    try:
        provenance={'method':'GT-Evidence-Clip','privileged_evidence_access':True,
            'scope':'Reference using privileged annotated points, source clips or intervals.',
            'manifest_sha256':sha256(args.manifest),'sources_sha256':sha256(args.sources),
            'evidence_sidecar_sha256':sha256(args.evidence) if args.evidence else None,'scorer_sha256':manifest['scorer_sha256'],
            'sampling':'uniform within declared evidence; at least two requested frames per clip; <=query_time',
            'max_frames':args.max_frames,'days':args.days,'shard':args.shard,'shards':args.shards,
            'audio_input':False,'target_time_passed_as_text_to_model':False,
            'missing_evidence_policy':'Record QA error; do not substitute neighboring clips or omit missing points.',
            'timecode':'global seconds on the explicitly supplied source timeline',
            'qa_ids':[q['qa_id'] for q in questions]}
        records=[plan_question(q,annotations.get(q['qa_id']),sources,args.max_frames) for q in questions]
        write_json(args.output/'plan.json',records)
        backend=None
        if args.command=='run':
            from baselines.ours_offline.run import backend_config,bind_gpu
            from baselines.ours_offline.backend import NativeVideoBackend,verify_model_ledger
            provenance['verification']=verify_model_ledger(args.model,args.ledger,args.backbone,args.model_revision)
            config=backend_config(args);provenance['backend']=config
            if any(r['status']=='planned' for r in records):
                bind_gpu(args.gpu)
                backend=NativeVideoBackend(config)
                started=time.perf_counter()
                backend.initialize()
                provenance['model_initialization_seconds']=time.perf_counter()-started
        write_json(args.output/'provenance.json',provenance)
        results=[]
        with (args.output/'predictions.jsonl').open('w') as stream:
            for question,record in zip(questions,records):
                if args.command=='run':record=answer_question(question,record,args.root,backend)
                results.append(record)
                stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
                status.update(completed=len(results),errors=sum(r['status'] in ('error','incomplete') for r in results))
                write_json(args.output/'status.json',status)
        status.update(status=('planned' if args.command=='plan' else 'complete') if not status['errors'] else 'incomplete',
            planned=sum(r['status']=='planned' for r in results),successful=sum(r['status']=='ok' for r in results),
            correct=sum(r['correct'] is True and r['status']=='ok' for r in results))
        if args.command=='run':status['accuracy_over_expected']=status['correct']/len(questions)
        write_json(args.output/'status.json',status)
        return 0 if not status['errors'] else 2
    except Exception as error:
        status.update(status='failed',error=f'{type(error).__name__}: {error}')
        write_json(args.output/'status.json',status)
        raise


def main():return execute(parser().parse_args())
