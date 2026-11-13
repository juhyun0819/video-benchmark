"""python -m videomem_adapter.cli --help"""
import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from .common import read, write, public_question, digest
from .media import prepare_manifest
from .pipeline import Pipeline


def evaluate(predictions, labels):
    """The ONLY path that reads labels; errors remain in the denominator."""
    keys = [r['question']['question_id'] for r in predictions]
    if len(keys) != len(set(keys)): raise ValueError('Duplicate predictions')
    label_map = {str(r['question_id']):r['answer'] for r in labels}
    if len(label_map) != len(labels): raise ValueError('Duplicate labels')
    if set(keys) != set(label_map): raise ValueError('Prediction/label IDs must match exactly')
    rows = []
    for p in predictions:
        key = p['question']['question_id']
        answer = p.get('prediction')
        options = p['question'].get('choices',[])
        parsed = isinstance(answer,str) and bool(answer.strip()) and p.get('status')=='complete'
        if parsed and options:
            parsed = bool(re.fullmatch('[A-'+chr(64+len(options))+']',answer.strip()))
        correct = parsed and answer.strip().casefold() == str(label_map[key]).strip().casefold()
        rows.append({'question_id':key,'prediction':answer,'answer':label_map[key],
                     'status':p.get('status'),'parsed':parsed,'correct':bool(correct)})
    total = len(rows)
    return {'total':total,'correct':sum(r['correct'] for r in rows),
            'unparsed_or_failed':sum(not r['parsed'] for r in rows),
            'accuracy':sum(r['correct'] for r in rows)/total if total else None,
            'test_only':any(p.get('test_only') for p in predictions),'results':rows}


def run_rows(pipe,stage,rows):
    output=[]
    for row in rows:
        started=time.monotonic()
        try:
            result=pipe.query(row) if stage=='query' else pipe.answer(row)
        except Exception as exc:
            # No response body or credentials in error reports. Per-item errors
            # do not disappear from the evaluation denominator.
            raw = row if stage=='query' else row['question']
            result={'question':public_question(raw),'status':'execution_error',
                    'error_type':type(exc).__name__,'prediction':None,'retrieved':[],
                    'error_message':pipe.models._sanitize(str(exc))[:500],
                    'namespace':pipe.namespace,'manifest_hash':digest(pipe.manifest),
                    'test_only':pipe.config.get('test_only',False),'elapsed_seconds':time.monotonic()-started}
        output.append(result)
    return output


def smoke(work,index_engine='sqlite'):
    """Actual MP4/FFmpeg + official retrieval functions + pixel-reading doubles.

    Not an LLM/benchmark performance measurement. No API keys/network/GPU needed.
    """
    work=Path(work); work.mkdir(parents=True,exist_ok=True)
    video=work/'synthetic_red_then_blue.mp4'
    if not video.exists():
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=red:s=64x64:r=10:d=2',
            '-f','lavfi','-i','color=c=blue:s=64x64:r=10:d=2','-filter_complex',
            '[0:v][1:v]concat=n=2:v=1:a=0[v]','-map','[v]','-c:v','libx264','-pix_fmt','yuv420p','-y',str(video)],check=True)
    config=read(Path(__file__).resolve().parents[1]/'configs/videomem.json')
    config.update(test_only=True,index_engine=index_engine,clip_seconds=2,top_k=3)
    config['frames']={'fps':1,'max_frames':2,'max_side':64}
    config['embedding']={'backend':'fixture','test_only':True,'dimensions':32}
    for role in config['models']:
        config['models'][role]={'backend':'fixture','model':'pixel-contract-test-double-v1'}
    manifest=prepare_manifest([{'day_id':'Day1','day_order':1,'path':str(video.resolve()),
        'timeline_status':'direct_mp4'}],clip_seconds=2,fingerprint_cache_dir=work/'fingerprints')
    write(work/'config.json',config); write(work/'manifest.json',manifest)
    q={'question_id':'smoke-not-benchmark','question':'What color was the scene?',
       'choices':['red','blue'],'query_time':{'day_id':'Day1','day_order':1,'seconds':2},
       'answer':'FORBIDDEN_GOLD_MARKER','reason':'FORBIDDEN_REASON_MARKER',
       'keywords':'FORBIDDEN_KEYWORD_MARKER','target_time':'FORBIDDEN_TARGET_MARKER'}
    pipe=Pipeline(config,manifest,work/'run')
    create=pipe.create()
    if create['errors']: raise RuntimeError(f"Smoke preprocessing failed: {create['errors']}")
    # Materialize a full-video (future-containing) summary BEFORE the early query.
    future=pipe.summaries({'day_id':'Day1','day_order':1,'seconds':4})
    query=pipe.query(q)
    answer=pipe.answer(query)
    calls=pipe.models.calls
    replay=pipe.create()
    again=pipe.answer(pipe.query(q))
    assert pipe.models.calls==calls, 'Completed preprocessing/query should be reused'
    assert query['retrieved'] and all(r['clip']['day_end_seconds']<=2 for r in query['retrieved'])
    assert answer['used_video_segments'] and answer['prediction']=='A'
    assert all(f['file_seconds']<2 for r in answer['used_video_segments'] for f in r['frames'])
    prior_ids={r['summary_id'] for value in future.values() for r in value}
    assert not prior_ids.intersection(query['summary_ids']), 'Full future summary was reused'
    trace=(work/'run/model_calls.jsonl').read_text()
    assert 'FORBIDDEN_' not in trace
    trace_rows=[json.loads(s) for s in trace.splitlines()]
    assert any(r['role']=='answerer' and r['images'] for r in trace_rows)
    result={'status':'passed','test_only':True,'real_llm_inference':False,
            'created':create,'resume':replay,'new_model_calls_on_second_pass':pipe.models.calls-calls,
            'query':query,'answer':answer,'storage':pipe.storage(),
            'checks':['future_clip_blocked','full_day_summary_not_reused','gold_fields_hidden',
                      'actual_mp4_frame_decode','visual_answer_wire','resume_cache_hit']}
    write(work/'smoke_result.json',result)
    print(json.dumps({'status':'passed','test_only':True,'result':str(work/'smoke_result.json')}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='stage',required=True)
    p=sub.add_parser('prepare'); p.add_argument('--sources',required=True); p.add_argument('--output',required=True)
    p.add_argument('--clip-seconds',type=float,default=30); p.add_argument('--fingerprint-cache')
    p=sub.add_parser('sanitize'); p.add_argument('--questions',required=True); p.add_argument('--output',required=True)
    for stage in ['create','summarize','query','answer','storage']:
        p=sub.add_parser(stage)
        p.add_argument('--config',required=True); p.add_argument('--manifest',required=True); p.add_argument('--work',required=True)
        p.add_argument('--output')
        if stage in ['summarize','query']: p.add_argument('--questions',required=True)
        if stage=='answer': p.add_argument('--retrieval',required=True)
        if stage=='create':
            p.add_argument('--limit',type=int); p.add_argument('--allow-full',action='store_true')
    p=sub.add_parser('evaluate'); p.add_argument('--predictions',required=True); p.add_argument('--labels',required=True); p.add_argument('--output',required=True)
    p=sub.add_parser('smoke'); p.add_argument('--work',required=True)
    p.add_argument('--index-engine',choices=['sqlite','chroma'],default='sqlite')
    args=parser.parse_args()
    if args.stage=='prepare':
        write(args.output,prepare_manifest(args.sources,args.clip_seconds,fingerprint_cache_dir=args.fingerprint_cache)); return
    if args.stage=='sanitize':
        write(args.output,[public_question(q) for q in read(args.questions)]); return
    if args.stage=='evaluate':
        write(args.output,evaluate(read(args.predictions),read(args.labels))); return
    if args.stage=='smoke': smoke(args.work,args.index_engine); return
    manifest=read(args.manifest)
    if args.stage=='create' and args.limit is None and len(manifest['clips'])>12 and not args.allow_full:
        parser.error('Use --limit for a small run or explicitly --allow-full after validating the smoke run')
    pipe=Pipeline(read(args.config),manifest,args.work)
    if args.stage=='create': result=pipe.create(limit=args.limit)
    elif args.stage=='summarize':
        result=[{'question':pipe.checked_question(q),'summaries':pipe.summaries(pipe.checked_question(q)['query_time'])} for q in read(args.questions)]
    elif args.stage=='query': result=run_rows(pipe,'query',read(args.questions))
    elif args.stage=='answer': result=run_rows(pipe,'answer',read(args.retrieval))
    else: result=pipe.storage()
    dest=args.output or str(Path(args.work)/(args.stage+'.json'))
    write(dest,result); print(json.dumps({'stage':args.stage,'output':dest,'model_calls':pipe.models.calls,'cache_hits':pipe.hits}))


if __name__=='__main__': main()
