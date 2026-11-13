"""Single continuous StreamTTT Day1 state, with causal QA forks and resume.

No temporal shards, training, feature cache, or state reset at a file/day boundary.
The seven-day extension must append to the same sample timeline and checkpoint.
"""
import argparse
import bisect
import json
import os
from pathlib import Path
import subprocess
import time

from .inputs import sha256
from .stream_frames import sample_plan, decode_window
from baselines.streambench.scoring import score_answer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--canonical', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--ledger', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--max-questions', type=int)
    p.add_argument('--audit-state', action='store_true')
    p.add_argument('--window-frames', type=int, default=192)
    p.add_argument('--resume', type=Path)
    p.add_argument('--days', type=int, nargs='+', default=[1])
    p.add_argument('--source-index', default='day1_sources_v1.json')
    p.add_argument('--resume-prefix-canonical', type=Path)
    p.add_argument('--resume-window-frames-from', type=int)
    p.add_argument('--secondary-gpu', type=int)
    p.add_argument('--prefetch-raw', action='store_true')
    args = p.parse_args()
    assert args.resume or args.resume_window_frames_from is None
    assert args.window_frames >= 2 and args.window_frames % 2 == 0
    used = int(subprocess.check_output(['nvidia-smi','-i',str(args.gpu),
        '--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    assert used < 1024, f'GPU occupied: {used} MiB'
    if args.secondary_gpu is not None:
        assert args.secondary_gpu != args.gpu
        used_secondary=int(subprocess.check_output(['nvidia-smi','-i',str(args.secondary_gpu),
            '--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        assert used_secondary < 1024, f'Second GPU occupied: {used_secondary} MiB'
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'status.json').write_text(json.dumps({'status':'initializing','pid':os.getpid()}))
    visible=str(args.gpu) if args.secondary_gpu is None else f'{args.gpu},{args.secondary_gpu}'
    os.environ.update(CUDA_VISIBLE_DEVICES=visible, HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', USE_TF='0', USE_FLAX='0', TOKENIZERS_PARALLELISM='false')
    import numpy as np
    import torch
    import transformers
    import flash_attn
    import causal_conv1d
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3_vl
    assert torch.__version__.split('+')[0] == '2.8.0'
    assert transformers.__version__ == '4.57.1'
    assert flash_attn.__version__.startswith('2.8.3')
    assert causal_conv1d.__version__ == '1.5.4'
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    apply_liger_kernel_to_qwen3_vl(rope=False, fused_linear_cross_entropy=False)
    from .ttt_native import NativeStreamTTT, state_digest

    canonical = json.loads((args.canonical/'jake_qa_v1.json').read_text())
    source_file = args.canonical/args.source_index
    sources = json.loads(source_file.read_text())['sources']
    scorer = Path(__file__).parents[1]/'streambench/scoring.py'
    assert sha256(scorer) == canonical['scorer_sha256']
    ledger = json.loads(args.ledger.read_text())
    assert ledger['status'] == 'verified' and Path(ledger['model_path']).resolve() == args.model.resolve()
    for source in sources:
        s = (args.root/source['path']).stat()
        assert source['file_state'] == [s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
    plan = sample_plan(sources, fps=2)
    times = [r['requested_global_timestamp'] for r in plan]
    questions = [q for q in canonical['questions'] if q['day'] in args.days]
    assert questions and all(q['query_time'] is not None for q in questions)
    assert set(args.days) <= {s['day'] for s in sources}
    config = {'memory_budget_bytes':16*2**30, 'model_path':str(args.model),
        'model_revision':ledger['revision'], 'upstream_source':str(args.upstream),
        'recent_tokens':4096, 'window_frames':args.window_frames, 'device':'cuda:0',
        'attn_implementation':'flash_attention_2', 'local_files_only':True,
        'processor_kwargs':{'do_resize':False}}
    method = NativeStreamTTT(config)
    identity = {'canonical_qa_sha256':sha256(args.canonical/'jake_qa_v1.json'),
        'source_index_sha256':sha256(source_file), 'scorer_sha256':canonical['scorer_sha256']}
    provenance = dict(identity, method='StreamTTT-4B', config=config, fps=2,
        max_window_duration_seconds=args.window_frames/2, native_pixel_budget=262144,
        patch_conv_memory_format='channels_last_3d', training=False,
        temporal_shards=False, native_raw_video=True, day_boundary_reset=False,
        qa_state_fork_restore=True, torch=torch.__version__, transformers=transformers.__version__,
        gpu=args.gpu, python_pid=os.getpid(), source_revision=method.identity['source_revision'],
        multi_gpu='Official evaluator sample-parallel; no audited single-stream model-parallel recipe.',
        runner_sha256=sha256(__file__), native_adapter_sha256=sha256(Path(__file__).with_name('ttt_native.py')),
        stream_decoder_sha256=sha256(Path(__file__).with_name('stream_frames.py')),
        overlapping_clip_policy='Next source owns its start timestamp; preceding tail is trimmed.',
        audio_input=False, generation='official multiple-choice option logits; no free-form decoding',
        secondary_gpu=args.secondary_gpu, prefetch_raw=args.prefetch_raw,
        placement='native decoder layers 18-35 on second GPU' if args.secondary_gpu is not None else 'single GPU',
        input_scope={'days':args.days,'source_index':args.source_index},
        temporal_window_note='At QA, commit available even temporal pairs; odd frame padded only on private QA branch.')
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2))
    status = {'status':'loading','pid':os.getpid(),'completed_qa':0,'stream_frames':0,'errors':0}
    def publish():
        temp=args.output/'status.tmp';temp.write_text(json.dumps(status,indent=2));temp.replace(args.output/'status.json')
    publish()
    method.initialize()
    completed, timestamps, cursor = [], [], 0
    previous_rows = []
    if args.resume:
        if args.resume_window_frames_from is None:
            method.load_checkpoint(args.resume)
        else:
            from baselines.streambench.checkpoint import read_checkpoint
            payload = read_checkpoint(args.resume)
            assert payload['adapter'] == type(method).__module__ + '.' + type(method).__qualname__
            assert payload['memory_budget_bytes'] == method.memory_budget_bytes
            saved = payload['snapshot']['method']
            old_identity = dict(saved['identity'])
            assert old_identity['window_frames'] == args.resume_window_frames_from
            new_identity = dict(old_identity, window_frames=args.window_frames)
            assert new_identity == method.identity, 'Only forward window size may change on resume'
            saved['identity'] = new_identity
            expected_digest = state_digest(saved)
            method.restore(payload['snapshot'])
            assert expected_digest == state_digest(method.state_dict()), 'Continuation tensors changed on restore'
            method.enforce_budget()
            (args.output/'resume-window-audit.json').write_text(json.dumps({
                'status':'passed','checkpoint':str(args.resume),'old_window_frames':args.resume_window_frames_from,
                'new_window_frames':args.window_frames,'restored_state_sha256':expected_digest,
                'all_continuation_tensors_preserved':True,
                'note':'Only subsequent forward batching changes; no claim of bitwise-identical later outputs.'},indent=2))
            del payload, saved
        proto=method.protocol_state
        cursor=proto['cursor'];timestamps=list(proto['actual_timestamps']);completed=list(proto['completed_qa_ids'])
        if proto['input_identity'] != identity:
            assert args.resume_prefix_canonical, 'Changed manifest requires a verified prefix extension'
            old=args.resume_prefix_canonical
            assert sha256(old/'jake_qa_v1.json')==proto['input_identity']['canonical_qa_sha256']
            assert sha256(old/'day1_sources_v1.json')==proto['input_identity']['source_index_sha256']
            assert identity['scorer_sha256']==proto['input_identity']['scorer_sha256']
            old_sources=json.loads((old/'day1_sources_v1.json').read_text())['sources']
            assert sample_plan(old_sources,fps=2)[:cursor]==plan[:cursor]
            old_q={q['qa_id']:q for q in json.loads((old/'jake_qa_v1.json').read_text())['questions']}
            new_q={q['qa_id']:q for q in questions}
            assert all(old_q[i]==new_q[i] for i in completed)
            (args.output/'resume-prefix-audit.json').write_text(json.dumps({'status':'passed','frames':cursor,'qa_ids':completed,'old_identity':proto['input_identity'],'new_identity':identity},indent=2))
        previous = args.resume.parent/'predictions.jsonl'
        previous_rows = [json.loads(l) for l in previous.read_text().splitlines() if l.strip()]
        previous_rows = [r for r in previous_rows if r['qa_id'] in completed]
        assert {r['qa_id'] for r in previous_rows} == set(completed)
        assert method.frames_seen == cursor == len(timestamps)
        status.update(completed_qa=len(completed),stream_frames=cursor,checkpoint=str(args.resume))
        publish()
    if args.secondary_gpu is not None:
        from .placement import install_two_gpu_layers, placement_neutral_state
        before=state_digest(placement_neutral_state(method.state_dict()))
        placement_handles=install_two_gpu_layers(method)
        assert before==state_digest(placement_neutral_state(method.state_dict()))
        (args.output/'placement-audit.json').write_text(json.dumps({'status':'passed',
            'all_state_tensor_bytes_preserved':True,'state_sha256':before,'gpus':[args.gpu,args.secondary_gpu]},indent=2))
    frames, metadata = decode_window(args.root,plan[:4])
    tensors=[torch.from_numpy(np.asarray(im).copy()).permute(2,0,1) for im in frames]
    token_audit=method.audit_native_tokens(tensors,metadata)
    (args.output/'native-token-audit.json').write_text(json.dumps(token_audit,indent=2))
    del frames,metadata,tensors
    for i in range(torch.cuda.device_count()):torch.cuda.reset_peak_memory_stats(i)
    def peaks():return [torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())]
    raw_buffer_bound=3*(args.window_frames+1)*max(s['width']*s['height']*3 for s in sources)
    total_decode=total_update=0.0
    if args.resume:
        # The parent status can include work after this checkpoint, including
        # failed work. Restore counters at the actual saved QA boundary.
        total_decode=method.protocol_state.get('decode_seconds', previous_rows[-1]['stream_decode_time'])
        total_update=method.protocol_state.get('update_seconds', previous_rows[-1]['stream_update_time'])
    start=time.perf_counter()
    updates=(args.output/'updates.jsonl').open('w')
    output=(args.output/'predictions.jsonl').open('w')
    for row in previous_rows:output.write(json.dumps(row,ensure_ascii=False)+'\n')
    output.flush()

    def advance(end):
        nonlocal cursor,total_decode,total_update
        from .prefetch import decoded_windows
        for begin,stop,frames,meta,dt in decoded_windows(args.root,plan,times,cursor,end,args.window_frames,args.prefetch_raw):
            assert begin==cursor
            assert all(r['global_timestamp']<=times[stop-1]+1e-8 for r in meta)
            torch.cuda.synchronize();tick=time.perf_counter()
            method.observe(frames,meta)
            method.finalize_chunk()
            torch.cuda.synchronize();ut=time.perf_counter()-tick
            memory=method.enforce_budget()
            assert memory['persistent_total_bytes']+raw_buffer_bound <= method.memory_budget_bytes
            timestamps.extend(r['global_timestamp'] for r in meta)
            total_decode+=dt;total_update+=ut
            event={'start_frame':cursor,'end_frame_exclusive':stop,'actual_frame_timestamps':[r['global_timestamp'] for r in meta],
                'decode_seconds':dt,'update_seconds':ut,'update_fps':len(frames)/ut,
                'persistent_memory_bytes':memory['persistent_total_bytes'],
                'raw_buffer_upper_bound_bytes':raw_buffer_bound,'peak_vram_allocated_by_gpu':peaks(),
                'peak_vram_allocated_bytes':torch.cuda.max_memory_allocated(),
                'peak_vram_reserved_bytes':torch.cuda.max_memory_reserved(),'token_offset':method._token_offset}
            updates.write(json.dumps(event)+'\n');updates.flush()
            cursor=stop
            status.update(status='streaming',stream_frames=cursor,completed_qa=len(completed),
                update_seconds=total_update,decode_seconds=total_decode,elapsed_seconds=time.perf_counter()-start,
                peak_vram_allocated_by_gpu=peaks(),
                peak_vram_allocated_bytes=torch.cuda.max_memory_allocated())
            publish();print('UPDATE',cursor,round(ut,3),'seconds',flush=True)

    def checkpoint(name):
        method.protocol_state={'input_identity':identity,'cursor':cursor,
            'actual_timestamps':timestamps,'completed_qa_ids':completed,'day_boundary_reset':False,
            'decode_seconds':total_decode,'update_seconds':total_update}
        path=args.output/name
        method.save_checkpoint(path)
        status['checkpoint']=str(path);publish()
        return path

    processed=0
    try:
        for q in questions:
            if q['qa_id'] in completed:continue
            end=bisect.bisect_right(times,q['query_time']+1e-8)
            advance(end)
            assert timestamps[-1] <= q['query_time']+1e-8
            before=state_digest(method.state_dict()) if args.audit_state else None
            torch.cuda.synchronize();tick=time.perf_counter()
            response=method.answer(q['question'],q['choices'])
            torch.cuda.synchronize();latency=time.perf_counter()-tick
            if before is not None:assert before==state_digest(method.state_dict()),'QA changed continuous state'
            correct,kind=score_answer(response['answer'],q)
            row={'method':'StreamTTT-4B','qa_id':q['qa_id'],'query_time':q['query_time'],
                'original_query_time':q['original_query_time'], 'need_audio':q['need_audio'], 'need_name':q['need_name'],
                'actual_frame_timestamps':list(timestamps),'prediction':response['answer'],
                'raw_output':response['raw_option_logits'],'GT':q['answer'],'correct':correct,
                'answer_latency':latency,'stream_update_time':total_update,'stream_decode_time':total_decode,
                'status':'ok','error':None,'score_kind':kind,'qa_state_digest_preserved':before is not None,
                'state_digest':before,'peak_vram_allocated_bytes':torch.cuda.max_memory_allocated(),
                'peak_vram_reserved_bytes':torch.cuda.max_memory_reserved(),
                'persistent_memory_bytes':method.memory_bytes()['persistent_total_bytes'],
                'temporal_padding_frames':response['qa_temporal_padding_frames']}
            output.write(json.dumps(row,ensure_ascii=False)+'\n');output.flush()
            completed.append(q['qa_id']);processed+=1
            status.update(completed_qa=len(completed),last_qa_id=q['qa_id'],last_answer_latency=latency)
            publish();print('QA',q['qa_id'],response['answer'],round(latency,3),flush=True)
            if args.max_questions and processed>=args.max_questions:break
            if len(completed)%10==0:checkpoint(f'checkpoint_qa{len(completed):03d}')
        if len(completed)==len(questions):
            # Ingest the tail after the final QA, so Day2 can continue full Day1 state.
            advance(len(plan))
        checkpoint('checkpoint')
        status.update(status='stream_complete_state_preserved' if len(completed)==len(questions) else 'smoke_complete',
            completed_qa=len(completed),stream_frames=cursor,elapsed_seconds=time.perf_counter()-start,
            update_seconds=total_update,decode_seconds=total_decode,
            peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_vram_reserved_bytes=torch.cuda.max_memory_reserved())
        publish()
    except Exception as error:
        if 'q' in locals() and q['qa_id'] not in completed:
            output.write(json.dumps({'method':'StreamTTT-4B','qa_id':q['qa_id'],'query_time':q['query_time'],
                'actual_frame_timestamps':timestamps,'prediction':'','raw_output':None,'GT':q['answer'],
                'correct':False,'answer_latency':None,'stream_update_time':total_update,
                'status':'error','error':f'{type(error).__name__}: {error}'})+'\n');output.flush()
        status.update(status='failed',errors=1,error=f'{type(error).__name__}: {error}');publish();raise
    finally:
        output.close();updates.close()


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        import sys
        if '--output' in sys.argv:
            path=Path(sys.argv[sys.argv.index('--output')+1])/'status.json'
            if path.exists():
                current=json.loads(path.read_text())
                if current.get('pid')==os.getpid():
                    current.update(status='failed',error=f'{type(error).__name__}: {error}')
                    path.write_text(json.dumps(current,indent=2))
        raise
