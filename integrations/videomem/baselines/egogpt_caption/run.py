"""Prepare and run resumable original-protocol EgoGPT caption workers."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .archive import atomic_json, object_hash, load_complete, save_record, is_complete, finish_reason
from .native import CAPTION_PROMPT
from .runtime import snapshot


def read(path): return json.loads(Path(path).read_text())


def reexec_clean_libraries(argv):
    """Let the pinned Torch wheel resolve its own CUDA/cuDNN libraries.

    The dynamic linker reads library overrides at process startup; changing
    os.environ just before importing Torch does not reliably clear that state.
    """
    if not any(os.environ.get(key) for key in ('LD_LIBRARY_PATH','LD_PRELOAD')):
        return False
    env=dict(os.environ)
    env.pop('LD_LIBRARY_PATH',None)
    env.pop('LD_PRELOAD',None)
    os.execve(sys.executable,[sys.executable,'-m','baselines.egogpt_caption',*argv],env)


def code_hashes():
    here=Path(__file__).parent
    files=list(here.glob('*.py'))+[here.parent/'qwen3_offline/video.py',
                                 here.parent/'fifo_qwen3vl/fifo_videomem/run.py',
                                 here.parent/'fifo_qwen3vl/fifo_videomem/shared.py']
    return {str(p.relative_to(here.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def prepare_job(inventory_path,runtime_path,output,workers):
    from .media import build_clips
    if workers<1: raise ValueError('workers must be positive')
    runtime=read(runtime_path)
    if runtime['status']!='ready': raise ValueError('Verified native runtime is not ready')
    inventory=read(inventory_path)
    clips=build_clips(inventory)
    spec={'inventory':inventory,'runtime':runtime,'runtime_snapshot':snapshot(runtime),
          'workers':workers,'code_hashes':code_hashes(),
          'protocol':{'seconds':30,'fps':1,'modalities':['video','audio'],'dtype':'float16',
                      'prompt':CAPTION_PROMPT,'conversation':'qwen_1_5 actual default system',
                      'do_sample':False,'temperature':0.5,'max_new_tokens':4096,
                      'generation_config':'official checkpoint inherited','vision_batch':'one full clip',
                      'memory_policy':runtime.get('memory_policy','native_gpu'),
                      'strict_causal':False,
                      'availability':'nominal clip end; conservative full-source end due to audio resampling',
                      'audio_policy':'user selected official full-source resampling over strict sample causality',
                      'query_information_used':False,
                      'adaptations':['source-local audio slicing fixes upstream HHMMSSFF bug',
                                     'full-source native librosa float32 audio cached losslessly',
                                     'explicit30s manifest includes partial last clip of each day']}}
    job_hash=object_hash(spec)
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    with (root/'.prepare.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (root/'job.json').exists():
            if read(root/'job.json')['job_hash']!=job_hash: raise ValueError('Existing archive has a different job')
        else:
            atomic_json(root/'job.json',{'job_hash':job_hash,**spec})
        atomic_json(root/'manifest.json',{'job_hash':job_hash,'clips':clips,'clip_count':len(clips),
                                        'sampled_frame_count':sum(len(c['frame_indices']) for c in clips)})
    return job_hash


def verify_job(root):
    job=read(root/'job.json')
    spec={k:v for k,v in job.items() if k!='job_hash'}
    if object_hash(spec)!=job['job_hash']: raise ValueError('Job identity hash mismatch')
    if job['code_hashes']!=code_hashes(): raise ValueError('Caption source code changed after job preparation')
    if job['runtime_snapshot']!=snapshot(job['runtime']): raise ValueError('Native runtime changed after preparation')
    from .media import build_clips
    manifest=read(root/'manifest.json')
    clips=build_clips(job['inventory'])
    if manifest['job_hash']!=job['job_hash'] or manifest['clips']!=clips:
        raise ValueError('Caption manifest changed')
    return job,clips


def load_native(runtime,gpu,port):
    from baselines.fifo_qwen3vl.fifo_videomem.run import bind_gpu
    gpu_info=bind_gpu(gpu)
    os.environ['MASTER_ADDR']='127.0.0.1';os.environ['MASTER_PORT']=str(port)
    sys.path.insert(0,runtime['upstream_path'])
    import torch
    import torch.distributed as dist
    torch.set_num_threads(4)
    torch.manual_seed(0)
    dist.init_process_group('gloo',rank=0,world_size=1)
    from egogpt.model.builder import load_pretrained_model
    from egogpt.model.language_model.egogpt_qwen import EgoGPTConfigQwen
    config=EgoGPTConfigQwen.from_pretrained(runtime['model_path'],local_files_only=True)
    config.mm_vision_tower=runtime['vision_path']
    config.speech_encoder=runtime['speech_path']
    tokenizer,model,_=load_pretrained_model(runtime['model_path'],device_map='cuda',config=config)
    model.eval()
    packages={name:importlib.metadata.version(name) for name in
              ['torch','torchvision','transformers','accelerate','decord','librosa','openai-whisper','deepspeed','numpy']}
    provenance={'gpu':gpu_info,'versions':packages,'cuda':torch.version.cuda,
                'cudnn_version':torch.backends.cudnn.version(),
                'library_overrides':{key:os.environ.get(key) for key in ('LD_LIBRARY_PATH','LD_PRELOAD')},
                'torch_num_threads':torch.get_num_threads(),'seed':0,
                'generation_config':model.generation_config.to_dict(),
                'attention':getattr(model.config,'_attn_implementation',None),
                'upstream_git_revision':subprocess.check_output(['git','-C',runtime['upstream_path'],'rev-parse','HEAD'],text=True).strip()}
    if provenance['upstream_git_revision']!=runtime['upstream_revision']:
        raise ValueError('Upstream source revision changed')
    return tokenizer,model,provenance


def worker(root,index,gpu,audio_cache,port,clip_ids=None,limit=None):
    from .media import decode_clip
    from .memory import offload_idle_encoders
    from .native import caption
    root=Path(root)
    job,clips=verify_job(root)
    if index<0 or index>=job['workers']: raise ValueError('Worker outside configured range')
    assigned=[c for i,c in enumerate(clips) if i%job['workers']==index]
    shard_total=len(assigned)
    if clip_ids:
        wanted=set(clip_ids)
        if not wanted <= {c['clip_id'] for c in assigned}: raise ValueError('Requested clips not assigned to this worker')
        assigned=[c for c in assigned if c['clip_id'] in wanted]
    if limit is not None:
        if limit<1: raise ValueError('limit must be positive')
        assigned=assigned[:limit]
    completion_status='complete' if len(assigned)==shard_total else 'selection_complete'
    status=root/'status'/f'worker_{index:02d}.json'
    def write_status(data):
        atomic_json(status,{'job_hash':job['job_hash'],'shard_total':shard_total,
                            'selected':len(assigned),'updated_unix':time.time(),**data})
    root.mkdir(parents=True,exist_ok=True)
    with (root/f'.worker_{index:02d}.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pending=[]
        for clip in assigned:
            if load_complete(root,clip['clip_id'],job['job_hash']) is None: pending.append(clip)
        if not pending:
            write_status({'status':completion_status,'assigned':len(assigned),'completed':len(assigned)})
            return 0
        completed=len(assigned)-len(pending);errors=0;consecutive_errors=0
        write_status({'status':'loading','assigned':len(assigned),'completed':completed})
        try:
            tokenizer,model,provenance=load_native(job['runtime'],gpu,port)
            memory_policy=job['runtime'].get('memory_policy','native_gpu')
            provenance['memory_policy']=memory_policy
            provenance['memory_validation']=job['runtime'].get('memory_validation')
            provenance_hash=object_hash(provenance)
            atomic_json(root/'provenance'/f'{provenance_hash}.json',provenance)
            for clip in pending:
                write_status({'status':'running','assigned':len(assigned),'completed':completed,
                                    'errors':errors,'clip_id':clip['clip_id'],'updated_unix':time.time()})
                started=time.monotonic()
                try:
                    video,audio,media=decode_clip(job['inventory'],clip,audio_cache)
                    decoded_at=time.monotonic()
                    context=(offload_idle_encoders(model) if memory_policy=='offload_idle_encoders'
                             else nullcontext())
                    with context:
                        output=caption(tokenizer,model,video,audio)
                    generated_at=time.monotonic()
                    termination={'max_new_tokens':job['protocol']['max_new_tokens'],
                                 'generation_returned_normally':True}
                    reason=finish_reason(output['generated_token_ids'],output['eos_token_id'],**termination)
                    done=is_complete(output['caption'],output['generated_token_ids'],output['eos_token_id'],
                                     **termination)
                    record={'clip_id':clip['clip_id'],'clip':clip,'media':media,'job_hash':job['job_hash'],
                            'status':'complete' if done else 'incomplete','worker':index,
                            **termination,'finish_reason':reason,'truncated':reason=='length',
                            'provenance_hash':provenance_hash,'strict_causal':False,
                            'nominal_clip_end_global_seconds_fraction':clip['global_end_fraction'],
                            'conservative_available_at_global_seconds_fraction':
                                media['audio']['full_source_global_end_fraction'],
                            'decode_seconds':decoded_at-started,
                            'generation_seconds':generated_at-decoded_at,
                            'elapsed_seconds':generated_at-started,**output}
                    save_record(root,record)
                    completed+=int(done);errors+=int(not done);consecutive_errors=0 if done else consecutive_errors+1
                    del video,audio,output
                except Exception as exc:
                    errors+=1;consecutive_errors+=1
                    atomic_json(root/'errors'/f"{clip['clip_id']}.json",
                                {'clip_id':clip['clip_id'],'job_hash':job['job_hash'],'error':str(exc),
                                 'traceback':traceback.format_exc()})
                    import torch
                    torch.cuda.empty_cache()
                if consecutive_errors>=3: raise RuntimeError('Three consecutive failed/incomplete clips; inspect errors')
            state=completion_status if errors==0 else 'incomplete'
            write_status({'status':state,'assigned':len(assigned),'completed':completed,'errors':errors})
            return 0 if errors==0 else 2
        except Exception as exc:
            write_status({'status':'error','assigned':len(assigned),'completed':completed,'errors':errors,
                                'error':str(exc),'traceback':traceback.format_exc()})
            raise
        finally:
            if 'torch.distributed' in sys.modules:
                import torch.distributed as dist
                if dist.is_initialized(): dist.destroy_process_group()


def main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare')
    for name in ['inventory','runtime','output']: p.add_argument('--'+name,required=True)
    p.add_argument('--workers',type=int,default=1)
    p=sub.add_parser('audio-cache')
    p.add_argument('--inventory',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('worker')
    p.add_argument('--output',required=True);p.add_argument('--index',type=int,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--audio-cache',required=True)
    p.add_argument('--port',type=int,default=29580);p.add_argument('--limit',type=int)
    p.add_argument('--clip-ids',nargs='+')
    args=parser.parse_args(argv)
    if args.command=='prepare':
        print(prepare_job(args.inventory,args.runtime,args.output,args.workers));return 0
    if args.command=='audio-cache':
        from .media import prepare_audio_cache
        receipts=prepare_audio_cache(read(args.inventory),args.output)
        print(json.dumps({'audio_sources_ready':len(receipts)}));return 0
    reexec_clean_libraries(argv)
    return worker(args.output,args.index,args.gpu,args.audio_cache,args.port,args.clip_ids,args.limit)
