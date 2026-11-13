"""Preflight only. Never starts the full archive or constructs a forest."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from . import native_archive as na


def compare_archives(reference, candidate):
    ref, actual = na.Archive(reference), na.Archive(candidate)
    if ref.config_id != actual.config_id:
        raise ValueError('Smoke checkpoint/config mismatch')
    ids=[]
    max_difference=0.0
    with torch.inference_mode():
        for a,b in zip(ref,actual,strict=True):
            if a.global_frame_id!=b.global_frame_id or a.timestamp!=b.timestamp:
                raise ValueError('Smoke frame ordering/timestamps changed')
            # Device labels are expected to differ. All input and position metadata must agree.
            keys=set(a.metadata)-{'encoder_gpu','feature_sha256','decode_seconds'}
            if any(a.metadata[k]!=b.metadata[k] for k in keys):
                raise ValueError('Smoke preprocessing/input metadata changed')
            difference=(a.visual_tokens.float()-b.visual_tokens.float()).abs().max().item()
            max_difference=max(max_difference,difference)
            if not torch.equal(a.visual_tokens,b.visual_tokens):
                return dict(bitwise_equal=False,first_different_frame=a.global_frame_id,max_absolute_difference=max_difference)
            ids.append(a.global_frame_id)
    if ids!=list(range(300)) or any(c!=1 for c in Counter(ids).values()):
        raise ValueError('Smoke frame coverage is not exactly 300 unique frames')
    return dict(bitwise_equal=True,unique_frames=len(ids),missing_frames=[],duplicate_frames=[],
                ordering_equal=True,timestamps_equal=True,preprocessing_equal=True,
                reload_bitwise_equal=True,max_absolute_difference=max_difference)


def tune_and_native_boundary(model_path, inventory, reference):
    """Tune only against exact native sequential output; no tolerance relaxation."""
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    tower=na.load_tower(model_path,0)
    reference_rows, reference_raw=na.Archive(reference).shard(0)
    expected=np.asarray(reference_raw[:32]).copy()
    tuning=[]
    selected=1
    best_fps=0.0
    with torch.inference_mode():
        for batch in [1,4,8,16,32]:
            raw,rows,metrics=na.encode_range(tower,inventory,0,32,batch)
            exact=np.array_equal(raw,expected)
            difference=(torch.from_numpy(raw.copy()).view(torch.float16).float()-
                        torch.from_numpy(expected.copy()).view(torch.float16).float()).abs().max().item()
            tuning.append(dict(batch_size=batch,bitwise_equal_to_sequential=exact,
                               max_absolute_difference=difference,**metrics))
            if exact and metrics['frames_per_second']>best_fps:
                selected,best_fps=batch,metrics['frames_per_second']
            print('Batch tuning:',batch,'exact:',exact,'fps:',metrics['frames_per_second'],flush=True)
        # Execute the actual official online entrypoint and stop at its first memory
        # input. Replacing the manager with a capture prevents forest construction.
        from .upstream import load_native
        native=load_native()
        class Captured(Exception):
            pass
        captured=[]
        class CaptureOnly:
            def __init__(self,*args,**kwargs):pass
            def update(self,feature):
                captured.append(feature.detach().cpu().clone())
                raise Captured()
        stub=SimpleNamespace(get_model=lambda:SimpleNamespace(get_vision_tower=lambda:tower))
        frame=next(na.frames_for_range(inventory,0,1))[0]
        pixels=tower.image_processor.preprocess(frame,return_tensors='pt')['pixel_values'].half().to(tower.device)
        frame.close()
        with patch('llava.model.multimodal_projector.memory_manager.MemoryManager',CaptureOnly):
            try:
                native.LlavaQwenForCausalLM.generate_online(stub,images=[pixels])
            except Captured:
                pass
        expected_first=torch.from_numpy(expected[0].copy()).view(torch.float16)
        if len(captured)!=1 or not torch.equal(captured[0],expected_first):
            raise ValueError('Cache differs from the official raw-frame online memory input')
    del tower
    torch.cuda.empty_cache()
    return selected,tuning,dict(official_generate_online_input_bitwise_equal=True,
        memory_update_intercepted_before_execution=True,qa_inference_performed=False,
        official_dtype='float16',checkpoint_storage_dtype='bfloat16')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',required=True)
    parser.add_argument('--model-path',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    root=Path(args.output)
    root.mkdir(parents=True,exist_ok=True)
    inventory=json.loads(Path(args.inventory).read_text())
    config=na.make_config(inventory,args.model_path,100)
    reference=root/'reference_300'
    parallel=root/'parallel_300'
    print('Starting 300-frame sequential native reference',flush=True)
    na.extract(config,args.model_path,reference,frames=300,gpus=[0],batch_size=1)
    batch,tuning,boundary=tune_and_native_boundary(args.model_path,inventory,reference)
    print('Starting 300-frame 8-GPU smoke; candidate batch:',batch,flush=True)
    na.extract(config,args.model_path,parallel,frames=300,gpus=list(range(8)),batch_size=batch)
    equality=compare_archives(reference,parallel)
    if not equality['bitwise_equal']:
        if batch==1:
            raise ValueError(f'8 GPU native equality failed: {equality}')
        parallel=root/'parallel_300_batch1'
        batch=1
        na.extract(config,args.model_path,parallel,frames=300,gpus=list(range(8)),batch_size=1)
        equality=compare_archives(reference,parallel)
        if not equality['bitwise_equal']:
            raise ValueError(f'8 GPU native equality failed: {equality}')
    summary=json.loads((parallel/'logs/extraction-summary.json').read_text())
    counts=Counter()
    for task in summary['worker_tasks']:counts[task['gpu']]+=task['frames']
    if set(counts)!=set(range(8)) or sum(counts.values())!=300:
        raise ValueError('Smoke did not use all 8 GPUs exactly once per production frame')
    packages={p:importlib.metadata.version(p) for p in ['torch','transformers','torchvision','av','numpy','pillow','safetensors','accelerate']}
    report=dict(status='preflight_passed_awaiting_user_review',checkpoint=config['checkpoint'],
        checkpoint_revision=config['checkpoint_revision'],vision_encoder=config['vision_encoder'],
        projector_connector=config['projector'],cache_boundary=config['cache_boundary'],
        per_frame_tensor_shape=config['shape'],dtype=config['dtype'],visual_tokens_per_frame=729,
        bytes_per_frame=na.FRAME_BYTES,total_inventory_frames=inventory['total_frames'],
        total_inventory_duration_seconds=inventory['total_duration_seconds'],
        total_inventory_hours=inventory['total_duration_seconds']/3600,
        estimated_archive_payload_bytes=inventory['total_frames']*na.FRAME_BYTES,
        smoke_300_throughput_fps=summary['steady_frames_per_second'],
        smoke_300_steady_wall_seconds=summary['steady_wall_seconds'],
        smoke_300_total_wall_seconds=summary['total_wall_seconds'],
        estimated_8gpu_full_run_seconds=summary['projected_full_run_seconds'],
        full_run_estimate_basis='300-frame measured wall throughput, excluding worker load; short smoke uncertainty remains',
        selected_batch_size=batch,gpu_production_frame_counts=dict(counts),
        equality=equality,native_boundary=boundary,batch_tuning=tuning,
        summary_path=str(parallel/'logs/extraction-summary.json'),packages=packages,
        reference_encoding_frames=300,production_smoke_encoding_frames=300,
        tuning_encodings_are_validation_only=True,full_run_started=False,
        qa_read=False,forest_constructed=False)
    na.write_json(root/'preflight-report.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
