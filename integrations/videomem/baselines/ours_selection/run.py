"""Offline query-adaptive selection; no pretrained inference runs on import."""
import argparse
import json
import os
from pathlib import Path
import time
from baselines.ours_offline.inputs import sha256
from .archive import prepare_archive,SelectionArchive
from .captions import CaptionManifest
from .selection import evaluate_question


def read_config(path):
    config=json.loads(Path(path).read_text())
    if config.get('method') not in ('aks','videotree'):
        raise ValueError('Config method must be aks or videotree')
    return config


def make_encoder(config):
    if config['method']=='aks':
        from baselines.streambench.adapters.aks import CLIPScorer
        return CLIPScorer(**config['candidate_encoder'])
    from baselines.streambench.adapters.videotree import EVAEncoder
    return EVAEncoder(**config['candidate_encoder'])


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    commands=p.add_subparsers(dest='command',required=True)
    prepare=commands.add_parser('prepare',help='Build query-independent CLIP/EVA selection features')
    evaluate=commands.add_parser('evaluate',help='Select causal raw frames and answer with target VLM')
    check=commands.add_parser('validate',help='Data-only readiness check; QA may still be pending')
    check.add_argument('--root',type=Path,required=True)
    check.add_argument('--sources',type=Path,required=True)
    check.add_argument('--archive',type=Path)
    check.add_argument('--canonical',type=Path)
    check.add_argument('--captions',type=Path)
    for command in (prepare,evaluate):
        command.add_argument('--config',type=Path,required=True)
        command.add_argument('--root',type=Path,required=True)
        command.add_argument('--sources',type=Path,required=True)
        command.add_argument('--output',type=Path,required=True)
    prepare.add_argument('--candidate-fps',type=float,default=1.)
    prepare.add_argument('--batch-size',type=int,default=8)
    evaluate.add_argument('--canonical',type=Path,required=True)
    evaluate.add_argument('--archive',type=Path,required=True)
    evaluate.add_argument('--captions',type=Path)
    evaluate.add_argument('--days',type=int,nargs='+',default=None)
    evaluate.add_argument('--shard',type=int,default=0)
    evaluate.add_argument('--shards',type=int,default=1)
    evaluate.add_argument('--limit',type=int)
    return p


def readiness(args):
    from baselines.ours_offline.inputs import validate_sources,load_inputs
    sources=validate_sources(args.root,json.loads(args.sources.read_text())['sources'])
    result={'status':'awaiting_QA','source_count':len(sources),'selection_features':'not_prepared',
            'qa_status':'awaiting_QA','qa_count':0,'model_loaded':False,
            'model_artifacts':'not_checked','ready_to_evaluate':False,'ready_for_model_preflight':False}
    archive=None
    if args.archive is not None:
        archive=SelectionArchive(args.archive,args.root,args.sources)
        result.update(selection_features='ready',feature_rows=len(archive.records),method=archive.metadata['method'])
        if archive.metadata['method']=='videotree':
            result['captions']='missing'
            if args.captions is not None:
                CaptionManifest(args.captions,archive);result['captions']='ready'
    if args.canonical is not None:
        data=json.loads(args.canonical.read_text())
        if not isinstance(data.get('questions'),list):raise ValueError('Canonical questions must be a list')
        if data['questions']:
            _,_,questions=load_inputs(args.root,args.canonical,args.sources)
            result.update(qa_status='ready',qa_count=len(questions))
            result['ready_for_model_preflight']=archive is not None and result.get('captions','ready')=='ready'
            result['status']=('missing_selection_features' if archive is None else
                              'missing_captions' if result.get('captions')=='missing' else 'ready_inputs')
    return result


def evaluate(args,config):
    from baselines.ours_offline.inputs import load_inputs
    from baselines.ours_offline.backend import NativeVideoBackend,verify_model_ledger
    if not 0<=args.shard<args.shards or (args.limit is not None and args.limit<1):
        raise ValueError('Invalid shard/limit')
    manifest,sources,questions=load_inputs(args.root,args.canonical,args.sources,days=args.days)
    questions=questions[args.shard::args.shards]
    if args.limit is not None:questions=questions[:args.limit]
    if not questions:raise ValueError('Empty QA shard')
    archive=SelectionArchive(args.archive,args.root,args.sources)
    if archive.metadata['method']!=config['method']:
        raise ValueError('Archive feature method does not match selector')
    candidate_encoder=make_encoder(config)
    if candidate_encoder.identity!=archive.metadata['encoder_identity']:
        raise ValueError('Configured candidate encoder does not match archived encoder identity')
    model_config=dict(config['answerer'])
    ledger=model_config.pop('ledger_path')
    verification=verify_model_ledger(model_config['model_path'],ledger,model_config['backbone'],model_config['model_revision'])
    backend=NativeVideoBackend(model_config)
    max_frames=int(config.get('max_answer_frames',64))
    if max_frames<2:raise ValueError('Native video answer frame budget must be at least two')
    captions=None
    if config['method']=='aks':
        scorer=candidate_encoder
        parameters=dict(max_num_frames=max_frames,ratio=1,t1=.8,t2=-100.,all_depth=5)
        parameters.update(config.get('selector_parameters',{}))
        if parameters['max_num_frames']!=max_frames:
            raise ValueError('AKS selection budget and answer budget must match')
    else:
        if args.captions is None:raise ValueError('VideoTree requires --captions')
        captions=CaptionManifest(args.captions,archive)
        from .relevance import LocalRelevanceScorer
        scorer=LocalRelevanceScorer(config['relevance'])
        parameters=dict(initial_clusters=8,maximum_clusters=32,high_relevance_threshold=4,branch_width=4,seed=0)
        parameters.update(config.get('selector_parameters',{}))
    # Model loading is a setup cost, not the first question's answer latency.
    backend.initialize()
    if config['method']=='videotree':scorer.initialize()
    else:scorer._load()  # Existing CLIP adapter exposes its lazy loader here.
    provenance={'method':config['method'],'backbone':model_config['backbone'],'config':config,
        'canonical_qa_sha256':sha256(args.canonical),'source_index_sha256':sha256(args.sources),
        'scorer_sha256':manifest['scorer_sha256'],'archive_sha256':archive.identity,
        'caption_identity':captions.identity if captions else None,'selection_model_identity':scorer.identity,
        'answer_model_verification':verification,'parameters':parameters,'max_answer_frames':max_frames,
        'answer_backend_settings':backend.config,
        'candidate_fps':archive.metadata['candidate_fps'],'qa_ids':[q['qa_id'] for q in questions],
        'sampling':'offline source-local candidate grid; strictly causal PTS cutoff',
        'variant':'CLIP AKS + selected raw frames to native VLM' if config['method']=='aks' else 'VideoTree + external causal captions + local relevance + selected raw frames to native VLM',
        'official_paper_reproduction':False,'audio_input':False,'persistent_memory_budget':None,
        'raw_source_reread_at_qa':True,'answer_visual_feature_cache':False,
        'runner_sha256':sha256(__file__),'selector_sha256':sha256(Path(__file__).with_name('selection.py'))}
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    start=time.perf_counter();rows=[]
    with (args.output/'predictions.jsonl').open('w') as stream:
        for q in questions:
            if hasattr(scorer,'calls'):scorer.calls=[]
            row=evaluate_question(args.root,archive,q,method=config['method'],scorer=scorer,
                parameters=parameters,backend=backend,max_frames=max_frames,captions=captions)
            if hasattr(scorer,'calls'):row['relevance_calls']=scorer.calls
            rows.append(row);stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
            status={'status':'running','completed':len(rows),'expected':len(questions),
                    'errors':sum(r['status']!='ok' for r in rows),'elapsed_seconds':time.perf_counter()-start}
            (args.output/'status.json').write_text(json.dumps(status,indent=2))
    status.update(status='complete' if status['errors']==0 else 'incomplete',
                  correct=sum(r['correct'] is True for r in rows),
                  accuracy_over_expected=sum(r['correct'] is True for r in rows)/len(questions))
    (args.output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    return 0 if status['status']=='complete' else 2


def main(argv=None):
    args=parser().parse_args(argv)
    if args.command=='validate':
        print(json.dumps(readiness(args),indent=2));return
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0',USE_FLAX='0',TOKENIZERS_PARALLELISM='false')
    config=read_config(args.config)
    if args.command=='prepare':
        encoder=make_encoder(config)
        prepare_archive(args.root,args.sources,args.output,config['method'],encoder,
                        fps=args.candidate_fps,batch_size=args.batch_size)
        return
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'status.json').write_text(json.dumps({'status':'initializing'}))
    try:
        code=evaluate(args,config)
    except Exception as exc:
        (args.output/'status.json').write_text(json.dumps({'status':'failed','error':f'{type(exc).__name__}: {exc}'}))
        raise
    if code:raise SystemExit(code)


if __name__=='__main__':
    main()
