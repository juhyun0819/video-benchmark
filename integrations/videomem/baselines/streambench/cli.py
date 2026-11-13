"""Prepare once, observe without questions, evaluate cumulative checkpoints."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from .checkpoint import read_checkpoint, write_json
from .manifest import prepare_manifest, manifest_info, iter_chunks
from .method import UnsupportedBudget
from .registry import available_methods, create_method
from .runner import observe_stream, evaluate_checkpoint


def positive(value):
    result=int(value)
    if result<=0:raise argparse.ArgumentTypeError('must be a positive integer')
    return result


def parser():
    root=argparse.ArgumentParser(description=__doc__)
    sub=root.add_subparsers(dest='command',required=True)
    sub.add_parser('list-methods')
    prep=sub.add_parser('prepare')
    prep.add_argument('--sources',required=True)
    prep.add_argument('--output',required=True)
    prep.add_argument('--fps',type=float,default=1.)
    observe=sub.add_parser('observe')
    observe.add_argument('--config',required=True)
    observe.add_argument('--manifest',required=True)
    observe.add_argument('--output',required=True)
    observe.add_argument('--budget-mib',type=positive)
    observe.add_argument('--chunk-size',type=positive,default=8)
    observe.add_argument('--resume')
    observe.add_argument('--stop-after-day',type=positive)
    evaluate=sub.add_parser('evaluate')
    evaluate.add_argument('--config',required=True)
    evaluate.add_argument('--checkpoint',required=True)
    evaluate.add_argument('--questions',required=True)
    evaluate.add_argument('--output',required=True)
    sweep=sub.add_parser('sweep')
    sweep.add_argument('--config',required=True)
    sweep.add_argument('--manifest',required=True)
    sweep.add_argument('--output',required=True)
    sweep.add_argument('--budgets-mib',nargs='+',type=positive,default=[64,128,256,512,1024])
    sweep.add_argument('--chunk-size',type=positive,default=8)
    sweep.add_argument('--stop-after-day',type=positive)
    return root


def read_config(path, budget_mib=None):
    config=json.loads(Path(path).expanduser().read_text())
    if not isinstance(config,dict):raise ValueError('Config must be an object')
    if budget_mib is not None:config['memory_budget_bytes']=budget_mib*1024**2
    if type(config.get('memory_budget_bytes')) is not int or config['memory_budget_bytes']<=0:
        raise ValueError('Set positive memory_budget_bytes in config or --budget-mib')
    return config


def sweep(args):
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    reports=[]
    for budget in args.budgets_mib:
        target=output/f'{budget}MiB'
        command=[sys.executable,'-m','baselines.streambench','observe','--config',str(Path(args.config).resolve()),
                 '--manifest',str(Path(args.manifest).resolve()),'--output',str(target),
                 '--budget-mib',str(budget),'--chunk-size',str(args.chunk_size)]
        if args.stop_after_day:command+=['--stop-after-day',str(args.stop_after_day)]
        with (output/f'{budget}MiB.log').open('w') as log:
            process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        report_path=target/'summary.json'
        report=json.loads(report_path.read_text()) if report_path.exists() else {
            'status':'failed','reason':'Worker exited without summary; see log'}
        report.update(memory_budget_bytes=budget*1024**2,exit_code=process.returncode)
        reports.append(report)
        write_json(output/'sweep.json',reports)
    return {'budgets':reports,'output':str(output)}


def main(argv=None):
    args=parser().parse_args(argv)
    try:
        if args.command=='list-methods':result=available_methods()
        elif args.command=='prepare':
            result=prepare_manifest(json.loads(Path(args.sources).read_text()),args.output,fps=args.fps)
        elif args.command=='sweep':result=sweep(args)
        elif args.command=='observe':
            config=read_config(args.config,args.budget_mib)
            info=manifest_info(args.manifest)
            if config.get('method') == 'fluxmem_streaming' and info['fps'] > 2 * config.get('temporal_tokens_per_second', 2):
                raise ValueError('FluxMem native temporal coordinates require sampling FPS <= twice temporal_tokens_per_second (default max 4 FPS)')
            start=-1
            if args.resume:
                state=read_checkpoint(args.resume)
                start=state['metadata']['last_global_frame_id']
            method=create_method(config)
            chunks=iter_chunks(args.manifest,chunk_size=args.chunk_size,start_after_frame_id=start,
                               stop_after_day=args.stop_after_day)
            result=observe_stream(method,chunks,args.output,stream_id=info['stream_id'],
                                  manifest_digest=info['manifest_sha256'],method_config=config,
                                  resume_checkpoint=args.resume,stop_after_day=args.stop_after_day)
        else:
            target=Path(args.output)
            if target.exists():raise FileExistsError(target)
            target.parent.mkdir(parents=True,exist_ok=True)
            config=read_config(args.config)
            method=create_method(config)
            method.initialize()
            questions=json.loads(Path(args.questions).read_text())
            if not isinstance(questions,list) or not questions:raise ValueError('Questions must be a nonempty JSON list')
            rows=evaluate_checkpoint(method,args.checkpoint,questions,output=target,method_config=config)
            result={'status':'complete','questions':len(rows),'output':str(target)}
        print(json.dumps(result,ensure_ascii=False,allow_nan=False))
        return 0
    except UnsupportedBudget as exc:
        report=exc.as_dict()
        code=2
    except (ValueError,FileNotFoundError,FileExistsError,NotImplementedError,ImportError,RuntimeError) as exc:
        report={'status':'failed','error_type':type(exc).__name__,'reason':str(exc)}
        code=1
    if args.command=='observe':
        target=Path(args.output)
        if not target.exists():target.mkdir(parents=True)
        if not (target/'summary.json').exists() and not any(target.iterdir()):
            write_json(target/'summary.json',report)
    print(json.dumps(report,ensure_ascii=False),file=sys.stderr)
    return code
