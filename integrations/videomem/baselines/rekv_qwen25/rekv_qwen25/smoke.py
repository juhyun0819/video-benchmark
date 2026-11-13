"""Real GPU0 query-free backbone/cache fidelity and KV footprint preflight."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import psutil
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .archive import KVArchiveWriter, load_block
from .inputs import read_feature_frames, validate_checkpoint

MODEL=Path('/convei_nas2/zzuhyun/hyunjun/models/Qwen2.5-VL-7B-Instruct')
FEATURES=Path('/convei_nas2/zzuhyun/hyunjun/feature_archives/qwen25_day1_7_v1')
LEDGER=MODEL.parent/'.downloads/Qwen2.5-VL-7B-Instruct/verification.json'
ROOT=Path('/convei_nas2/zzuhyun/hyunjun/rekv_qwen25vl7b')
PREFIX='<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nVisual observation.\n'


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def compare(a,b):
    if a.shape!=b.shape or a.dtype!=b.dtype:
        return dict(exact=False,shape_match=a.shape==b.shape,dtype_match=a.dtype==b.dtype)
    return dict(exact=torch.equal(a,b),max_abs=float((a.float()-b.float()).abs().max()),
                shape=list(a.shape),dtype=str(a.dtype))


def past_layers(cache):
    return tuple(cache.to_legacy_cache())


class Capture:
    """Passive hooks: observe native projection outputs without replacing them."""
    def __init__(self,model):
        self.model=model;self.handles=[];self.kv={};self.inputs={};self.mask=None

    def __enter__(self):
        def inputs(module,args,kwargs):
            for key in ['inputs_embeds','position_ids','attention_mask']:
                value=kwargs.get(key)
                self.inputs[key]=value.detach().cpu().clone() if torch.is_tensor(value) else value
        self.handles.append(self.model.model.language_model.register_forward_pre_hook(inputs,with_kwargs=True))
        def mask(module,args,kwargs):
            value=kwargs.get('attention_mask')
            self.mask=None if value is None else dict(shape=list(value.shape),dtype=str(value.dtype),sha256=tensor_hash(value))
        self.handles.append(self.model.model.language_model.layers[0].self_attn.register_forward_pre_hook(mask,with_kwargs=True))
        for i,layer in enumerate(self.model.model.language_model.layers):
            for kind in ['key','value']:
                projection=layer.self_attn.k_proj if kind=='key' else layer.self_attn.v_proj
                def record(module,args,output,i=i,kind=kind):
                    self.kv[f'layer_{i:02d}.{kind}']=output.detach().reshape(1,-1,4,128).transpose(1,2).contiguous().cpu()
                self.handles.append(projection.register_forward_hook(record))
        return self

    def __exit__(self,*args):
        for handle in self.handles:handle.remove()


def observation_inputs(model,processor,features,start_frame=0):
    if start_frame != 0:
        raise ValueError('This native full-prefix helper starts at frame zero; use StreamingInputs for temporal shards')
    n=features.shape[0]
    prefix=processor.tokenizer(PREFIX,add_special_tokens=False).input_ids
    block=[model.config.vision_start_token_id]+[model.config.image_token_id]*676+[model.config.vision_end_token_id]
    ids=torch.tensor([prefix+block*n],device=model.device)
    grid=torch.tensor([[1,52,52]]*n,device=model.device)
    mask=torch.ones_like(ids)
    positions,delta=model.model.get_rope_index(ids,image_grid_thw=grid,attention_mask=mask)
    embeddings=model.get_input_embeddings()(ids)
    embeddings=embeddings.masked_scatter((ids==model.config.image_token_id).unsqueeze(-1),features.to(model.device).reshape(-1))
    return dict(input_ids=ids,image_grid_thw=grid,attention_mask=mask,position_ids=positions),embeddings,len(prefix),delta


def golden_images(start,count):
    baseline=Path(__file__).resolve().parents[2]/'streamforest'
    sys.path.insert(0,str(baseline))
    from streamforest_videomem.native_archive import frames_for_range
    inventory=json.loads(Path('/home/zzuhyun/현준/streamforest_work/inventory.json').read_text())
    images=[]
    for image,row in frames_for_range(inventory,start,start+count):images.append(image)
    return images


def forward_cached(model,inputs,embeddings):
    return model(inputs_embeds=embeddings,position_ids=inputs['position_ids'],
                 attention_mask=inputs['attention_mask'],use_cache=True,logits_to_keep=1,return_dict=True)


def metadata(rows):
    return dict(frame_ids=[r.get('global_frame_id',r.get('frame_id')) for r in rows],
                timestamps=[r.get('timestamp_seconds',r.get('timestamp')) for r in rows],frames=rows,
                sampling_rule='1fps first PTS at or after source-local integer second',
                scope='query-free GPU0 preflight; fixed open user prefix + independent image blocks')


def monitor(stop,path,gpu):
    process=psutil.Process()
    while not stop.is_set():
        try:
            out=subprocess.check_output(['nvidia-smi',f'--id={gpu}',
                '--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True,timeout=10)
            with path.open('a') as f:f.write(json.dumps(dict(time=time.time(),gpu_csv=out.strip(),cpu_rss=process.memory_info().rss))+'\n')
        except Exception:pass
        stop.wait(1)


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--output',default=str(ROOT/'benchmarks/gpu0_fidelity_v1'))
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_device(args.gpu)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    stop=threading.Event();monitor_thread=threading.Thread(target=monitor,args=(stop,out/'telemetry.jsonl',args.gpu),daemon=True);monitor_thread.start()
    try:
        identity=validate_checkpoint(MODEL,LEDGER)
        features,rows=read_feature_frames(FEATURES,0,16)
        print('Loading exact frozen Qwen2.5 checkpoint',flush=True)
        model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(MODEL),local_files_only=True,
                torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':f'cuda:{args.gpu}'},use_safetensors=True)
        model.eval().requires_grad_(False)
        processor=AutoProcessor.from_pretrained(str(MODEL),local_files_only=True,use_fast=False)
        config=dict(backbone=identity,sampling_fps=1,frame_feature_shape=[676,3584],dtype='bfloat16',
                    input_protocol='open fixed user prefix, independent image blocks; no query or closing footer',
                    official_rekv_commit='5a17285d5e1a7488228eb0098b5e586d72a73e3a',
                    key_representation='pre_rope',n_local=15000,
                    temporal_policy='causal context preserved; no unvalidated independent reset')
        results=[]
        for n in [1,4,8,16]:
            inputs,embeddings,n_init,delta=observation_inputs(model,processor,features[:n])
            torch.cuda.reset_peak_memory_stats()
            # A and B use the identical native forward and identical full input.
            t=time.perf_counter();a=forward_cached(model,inputs,embeddings);torch.cuda.synchronize()
            native_seconds=time.perf_counter()-t
            with Capture(model) as cap:
                t=time.perf_counter();b=forward_cached(model,inputs,embeddings);torch.cuda.synchronize()
                instrumented_seconds=time.perf_counter()-t
            kv_exact=all(torch.equal(x,y) for l,r in zip(past_layers(a.past_key_values),past_layers(b.past_key_values)) for x,y in zip(l,r))
            evidence=dict(frames=n,sequence_tokens=inputs['input_ids'].shape[1],n_init=n_init,
                layers=len(past_layers(b.past_key_values)),key_shape=list(past_layers(b.past_key_values)[0][0].shape),
                raw_key_shape=list(cap.kv['layer_00.key'].shape),dtype='bfloat16',
                instrumentation_kv_exact=kv_exact,instrumentation_logits=compare(a.logits,b.logits),
                input_embeddings_exact=torch.equal(embeddings.cpu(),cap.inputs['inputs_embeds']),
                position_ids_exact=torch.equal(inputs['position_ids'].cpu(),cap.inputs['position_ids']),
                attention_mask=cap.mask,rope_deltas=delta.tolist(),native_forward_seconds=native_seconds,
                instrumented_forward_seconds=instrumented_seconds,processing_fps=n/instrumented_seconds,
                peak_vram_bytes=torch.cuda.max_memory_allocated(),cpu_rss=psutil.Process().memory_info().rss)
            if not (kv_exact and evidence['instrumentation_logits']['exact'] and evidence['input_embeddings_exact'] and evidence['position_ids_exact']):
                raise RuntimeError('Passive instrumentation changed native model values')
            # Raw image path comparison uses actual original images, including native vision batching.
            images=golden_images(0,n)
            pixel=processor.image_processor(images=images,return_tensors='pt')
            for im in images:im.close()
            with Capture(model) as raw_cap:
                raw=model(**inputs,pixel_values=pixel['pixel_values'].to(model.device),
                    use_cache=True,logits_to_keep=1,return_dict=True)
            evidence['raw_vs_cached_embeddings']=compare(raw_cap.inputs['inputs_embeds'],cap.inputs['inputs_embeds'])
            evidence['raw_vs_cached_positions_exact']=torch.equal(raw_cap.inputs['position_ids'],cap.inputs['position_ids'])
            evidence['raw_vs_cached_logits']=compare(raw.logits,b.logits)
            evidence['raw_vs_cached_postrope_kv_exact']=all(torch.equal(x,y) for l,r in zip(past_layers(raw.past_key_values),past_layers(b.past_key_values)) for x,y in zip(l,r))
            evidence['raw_vs_cached_prerope_kv_exact']=all(torch.equal(raw_cap.kv[k],cap.kv[k]) for k in cap.kv)
            if n==1 and not(evidence['raw_vs_cached_embeddings']['exact'] and evidence['raw_vs_cached_postrope_kv_exact']):
                write_json(out/'fidelity-failure.json',evidence)
                raise RuntimeError('Single-frame raw/cache equality failed; cache reuse is blocked')
            del a,raw,raw_cap
            writer=KVArchiveWriter(out/f'kv_{n:02d}_frames',config)
            payload=dict(cap.kv,position_ids=inputs['position_ids'].cpu(),input_ids=inputs['input_ids'].cpu())
            t=time.perf_counter();receipt=writer.commit(0,payload,metadata(rows[:n]));elapsed=time.perf_counter()-t
            restored,_=load_block(writer.root,0,writer.config_id)
            evidence['reload_exact']=all(torch.equal(v,restored[k]) for k,v in payload.items())
            evidence['kv_payload_bytes']=sum(v.numel()*v.element_size() for v in cap.kv.values())
            evidence['bytes_per_observation_frame']=28*2*4*128*2*678
            evidence['serialized_bytes']=receipt['serialized_bytes']
            evidence['atomic_commit_seconds']=elapsed
            evidence['atomic_commit_MB_per_second']=receipt['serialized_bytes']/elapsed/1e6
            evidence['estimated_full_payload_bytes']=28*2*4*128*2*(678*221728+n_init)
            results.append(evidence);write_json(out/'measurements.json',results)
            print('GPU0 fidelity sample',n,'frames:',json.dumps(evidence),flush=True)
            # Test native post-RoPE cache serialize/reload continuation with one fixed non-query token.
            from transformers import DynamicCache
            token=torch.tensor([[processor.tokenizer.encode('\n',add_special_tokens=False)[0]]],device=model.device)
            nextpos=torch.full((3,1,1),int(inputs['position_ids'].max())+1,device=model.device,dtype=torch.long)
            copied=DynamicCache.from_legacy_cache(tuple((k.clone(),v.clone()) for k,v in past_layers(b.past_key_values)))
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
            cos,sin=model.model.language_model.rotary_emb(embeddings,restored['position_ids'].to(model.device))
            recovered=[]
            for layer_index in range(28):
                key=restored[f'layer_{layer_index:02d}.key'].to(model.device)
                value=restored[f'layer_{layer_index:02d}.value'].to(model.device)
                _,key=apply_multimodal_rotary_pos_emb(key,key,cos,sin,[16,24,24])
                recovered.append((key,value))
            evidence['disk_prerope_to_native_postrope_exact']=all(torch.equal(x,y) for l,r in zip(recovered,past_layers(b.past_key_values)) for x,y in zip(l,r))
            reload_cache=DynamicCache.from_legacy_cache(tuple(recovered))
            c=model(input_ids=token,position_ids=nextpos,past_key_values=copied,use_cache=True,logits_to_keep=1)
            d=model(input_ids=token,position_ids=nextpos,past_key_values=reload_cache,use_cache=True,logits_to_keep=1)
            evidence['native_cache_reload_continuation_exact']=torch.equal(c.logits,d.logits)
            if not all(evidence[k] for k in ['reload_exact','disk_prerope_to_native_postrope_exact','native_cache_reload_continuation_exact']):
                write_json(out/'fidelity-failure.json',evidence)
                raise RuntimeError('Exact disk reload or native continuation fidelity failed')
            write_json(out/'measurements.json',results)
            del b,c,d,copied,reload_cache,recovered,payload,restored,cap,embeddings
            torch.cuda.empty_cache()
        report=dict(status='gpu0_backbone_fidelity_complete',samples=results,
            checkpoint=identity,visual_cache_reuse_single_image_exact=results[0]['raw_vs_cached_prerope_kv_exact'],
            full_run_started=False,qa_used=False,retrieval_implemented=False,
            next_stage='actual ReKV sliding-window adaptation, context boundary tests, then scaling')
        write_json(out/'report.json',report)
    finally:
        stop.set();monitor_thread.join(timeout=15)


if __name__=='__main__':main()
