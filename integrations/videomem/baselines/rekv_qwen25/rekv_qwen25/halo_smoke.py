"""Real-model finite-history test against one uninterrupted observation stream.

Two sessions share one frozen model on GPU0 and retain independent GPU state.
The replay session starts with deliberately incorrect recent KV, true fixed
initial sinks, and the exact absolute positions/chunk boundaries. No archive
generation or parallel scaling is authorized by this script's mere existence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import threading
import time

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .attention import ReKVSession
from .archive import KVArchiveWriter
from .inputs import read_feature_frames, validate_checkpoint
from .smoke import MODEL, FEATURES, LEDGER, ROOT, compare, monitor, write_json
from .state_io import flatten_state
from .stream import StreamingInputs, observe


def zero_boundary(session, stream, frame, sinks):
    """A deliberately wrong local cache; only a verified halo can repair it."""
    total = stream.n_init + frame * 678
    if total <= session.n_local:
        raise ValueError('Artificial boundary must be beyond the local window')
    count = math.ceil(session.n_local / 678)
    positions = torch.cat([stream.frame_positions + j * stream.stride
                          for j in range(frame - count, frame)], dim=-1)
    positions = positions[:, :, -session.n_local:].cpu().contiguous()
    state = session.export_state()
    state['total_seen'] = total
    state['native_post_rope_cache'] = None
    state['states'] = []
    for layer, sink in zip(session.attentions, sinks):
        shape = (1, layer.num_key_value_heads, session.n_local, layer.head_dim)
        state['states'].append(dict(key=torch.zeros(shape,dtype=layer.k_proj.weight.dtype),
            value=torch.zeros(shape,dtype=layer.k_proj.weight.dtype), positions=positions,
            init_key=sink['init_key'], init_value=sink['init_value']))
    session.import_state(state)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--boundary-frame',type=int,default=24)
    parser.add_argument('--n-local',type=int,default=15000)
    parser.add_argument('--verify-frames',type=int,default=4)
    parser.add_argument('--output',default=str(ROOT/'benchmarks/gpu0_halo_v1'))
    parser.add_argument('--initial-state-root',default=str(ROOT/'initial_state'))
    args = parser.parse_args()
    if args.verify_frames < 1:
        parser.error('--verify-frames must be positive')
    torch.set_num_threads(4);torch.cuda.set_device(args.gpu)
    out = Path(args.output);out.mkdir(parents=True,exist_ok=True)
    stop = threading.Event()
    thread = threading.Thread(target=monitor,args=(stop,out/'telemetry.jsonl',args.gpu),daemon=True)
    thread.start()
    identity = validate_checkpoint(MODEL,LEDGER)
    print('Loading checkpoint for continuous versus finite-halo replay',flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(MODEL),local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':f'cuda:{args.gpu}'})
    model.eval().requires_grad_(False)
    processor = AutoProcessor.from_pretrained(str(MODEL),local_files_only=True,use_fast=False)
    stream = StreamingInputs(model,processor)
    continuous = ReKVSession(model,stream.n_init,args.n_local);continuous.restore()
    replay = ReKVSession(model,stream.n_init,args.n_local);replay.restore()
    layers = len(continuous.attentions)
    halo = math.ceil(layers * args.n_local / 678)
    end = args.boundary_frame + halo + args.verify_frames
    checks = []
    started = time.perf_counter()
    replay_seconds = continuous_seconds = 0.0
    sinks = None
    torch.cuda.reset_peak_memory_stats()
    try:
        continuous.install()
        observe(model,continuous,stream.prefix_inputs())
        continuous.restore()
        saved = continuous.export_state()
        sinks = [{k:s[k] for k in ('init_key','init_value')} for s in saved['states']]
        initial_tensors, initial_descriptor = flatten_state(saved)
        initial_writer = KVArchiveWriter(args.initial_state_root, dict(key_representation='pre_rope',
            backbone=identity,policy=continuous.policy,upstream=continuous.upstream,
            input_protocol='separate_fixed_prefix_then_one_frame',prefix=processor.tokenizer.decode(stream.prefix),
            n_init=stream.n_init,n_local=args.n_local,dtype='bfloat16',
            attention_sha256=hashlib.sha256(Path(__file__).with_name('attention.py').read_bytes()).hexdigest()))
        initial_receipt = initial_writer.commit(0,initial_tensors,dict(frame_ids=[0],timestamps=[0.0],
            scope='Immutable query-free text-prefix state; contains no video frames',resume_state=initial_descriptor))
        del saved
        for batch_start in range(0,end,100):
            features,_ = read_feature_frames(FEATURES,batch_start,min(100,end-batch_start))
            for index, feature in enumerate(features):
                frame = batch_start + index
                inputs = stream.build(feature.unsqueeze(0),frame)
                continuous.install()
                t = time.perf_counter()
                reference = observe(model,continuous,inputs)
                torch.cuda.synchronize();continuous_seconds += time.perf_counter()-t
                continuous.restore()
                if frame == args.boundary_frame:
                    zero_boundary(replay,stream,frame,sinks)
                if frame >= args.boundary_frame:
                    replay.install()
                    t = time.perf_counter()
                    actual = observe(model,replay,inputs)
                    torch.cuda.synchronize();replay_seconds += time.perf_counter()-t
                    replay.restore()
                    if (frame-args.boundary_frame)%23 == 0 or frame>=args.boundary_frame+halo:
                        keys = [torch.equal(continuous.latest_kv[f'layer_{i:02d}.key'], replay.latest_kv[f'layer_{i:02d}.key'])
                                and torch.equal(continuous.latest_kv[f'layer_{i:02d}.value'], replay.latest_kv[f'layer_{i:02d}.value']) for i in range(layers)]
                        row = dict(frame_id=frame,replayed_frames=frame-args.boundary_frame+1,
                            exact_layers=keys,all_kv_exact=all(keys),logits=compare(reference,actual))
                        checks.append(row)
                        if frame == args.boundary_frame and all(keys) and row['logits']['exact']:
                            raise RuntimeError('Artificial boundary did not perturb the reference; test is vacuous')
                        if frame == args.boundary_frame + halo:
                            row['bounded_state_exact'] = all(torch.equal(a[k],b[k])
                                for a,b in zip(continuous.layer_states,replay.layer_states) for k in a)
                            if not row['bounded_state_exact']:
                                write_json(out/'failure.json',row)
                                raise RuntimeError('Halo did not reproduce the full bounded continuation state')
                        print(json.dumps(row),flush=True)
                        if frame>=args.boundary_frame+halo and not(all(keys) and row['logits']['exact']):
                            write_json(out/'failure.json',row)
                            raise RuntimeError('Full-model halo did not reproduce continuous KV/logits exactly')
                if frame%10 == 0 or frame==end-1:
                    write_json(out/'status.json',dict(state='running',frame_id=frame,total_frames=end,
                        elapsed_seconds=time.perf_counter()-started,continuous_seconds=continuous_seconds,
                        replay_seconds=replay_seconds,halo_frames=halo,latest_check=checks[-1] if checks else None))
        verified = [r for r in checks if r['frame_id']>=args.boundary_frame+halo]
        report = dict(status='passed',checkpoint=identity,halo_frames=halo,halo_tokens=halo*678,
            initial_state_binding=dict(config_id=initial_writer.config_id,
                payload_sha256=initial_receipt['payload_sha256'],index_sha256=initial_receipt['index_sha256'],
                recorded_by='halo smoke from the canonical prefix state used in this run'),
            n_local=args.n_local,n_init=stream.n_init,policy=continuous.policy,upstream=continuous.upstream,
            torch_version=torch.__version__,cuda_version=torch.version.cuda,short_backend='sdpa',
            adaptation_sha256=hashlib.sha256(Path(__file__).with_name('attention.py').read_bytes()).hexdigest(),
            conservative_receptive_bound_tokens=layers*args.n_local,boundary_frame=args.boundary_frame,
            fixed_original_per_layer_sinks=True,absolute_positions=True,chunk_frames=1,
            verification_frames=len(verified),checks=checks,elapsed_seconds=time.perf_counter()-started,
            continuous_seconds=continuous_seconds,replay_seconds=replay_seconds,
            peak_vram_bytes=torch.cuda.max_memory_allocated(),full_run_started=False,
            limitation='One real-video boundary; verifies the fixed protocol and this backend, not arbitrary kernels or chunking')
        write_json(out/'report.json',report)
        write_json(out/'status.json',dict(state='complete',frame_id=end-1,total_frames=end))
    except Exception as exc:
        write_json(out/'status.json',dict(state='failed',error=str(exc)))
        raise
    finally:
        continuous.restore();replay.restore();stop.set();thread.join(timeout=15)


if __name__=='__main__':
    main()
