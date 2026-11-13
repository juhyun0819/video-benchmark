"""Causal JAKE replay with one shared memory and independent GPU QA replicas.

No new feature archive/checkpoint is written. The readout is question-independent
uniform sampling of retained original tokens, with their native sparse image
mRoPE coordinates. This is an explicit sparse-token QA adaptation, not a native
full-video Qwen evaluation or a learned text-to-visual retriever.
"""
import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import time
import traceback

import torch

from .config import RoutineConfig
from .memory import RoutineAwareMemory


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def public_question(row):
    return {k: row[k] for k in ('qa_id', 'day', 'query_time', 'question', 'choices')}


def absolute_time(row):
    # canonical_v2 has already added the DAY offset. Never add it a second time.
    timestamp = float(row['query_time'])
    if not (int(row['day'])-1)*86400 <= timestamp < int(row['day'])*86400:
        raise ValueError('Expected canonical absolute query_time, not day-local time')
    return timestamp


def select_evidence(memory, budget, cutoff):
    """Bounded, question-independent readout; snapshots own selected tokens only."""
    rows = sorted(memory.retained_evidence(), key=lambda r: (r['timestamp'], r['frame_id'], r['kind']))
    if any(r.get('end_timestamp', r['timestamp']) > cutoff for r in rows):
        raise ValueError('Future evidence in QA readout')
    total = sum(len(r['features']) for r in rows)
    count = min(budget, total)
    if not count:
        return []
    selected = torch.div(torch.arange(count) * total, count, rounding_mode='floor')
    output, offset = [], 0
    for row in rows:
        local = selected[(selected >= offset) & (selected < offset + len(row['features']))] - offset
        offset += len(row['features'])
        if len(local):
            output.append({'features': row['features'].index_select(0, local).clone(),
                           'token_indices': row['token_indices'].index_select(0, local).clone(),
                           'frame_id': row['frame_id'], 'timestamp': row['timestamp'], 'kind': row['kind']})
    return output


def sparse_prompt(tokenizer, config, question, evidence, grid_width=50):
    """Text is sequential; selected image rows retain full-grid t/h/w positions."""
    ids, positions, replacements = [], [], []
    next_pos = 0
    def text(value):
        nonlocal next_pos
        tokens = tokenizer.encode(value, add_special_tokens=False)
        ids.extend(tokens)
        positions.append(torch.arange(next_pos, next_pos + len(tokens)).expand(3, -1))
        next_pos += len(tokens)
    text('<|im_start|>system\nYou are a helpful assistant. Answer using only the observed visual evidence.\n<|im_end|>\n<|im_start|>user\n')
    text('These are sparse visual observations from my past. Times are DAY plus seconds from midnight.\n')
    for record in evidence:
        day = int(record['timestamp'] // 86400) + 1
        clock = record['timestamp'] % 86400
        text(f'DAY{day} time={clock:.3f}s frame={record["frame_id"]}\n<|vision_start|>')
        indices = record['token_indices']
        count = len(indices)
        replacements.append((len(ids), record['features']))
        ids.extend([config.image_token_id] * count)
        native = torch.stack((torch.zeros_like(indices), indices // grid_width, indices % grid_width))
        positions.append(native + next_pos)
        # Removing image rows must not shift following text/image coordinates.
        # The original square merged grid still occupies grid_width positions.
        next_pos += grid_width
        text('<|vision_end|>\n')
    text(f'Current query time: DAY{question["day"]} {question["query_time"] % 86400:.3f}s\nQuestion: {question["question"]}\nOptions:\n')
    text('\n'.join(f'{chr(65+i)}. {choice}' for i, choice in enumerate(question['choices'])))
    text('\nAnswer with only the letter of the best option.<|im_end|>\n<|im_start|>assistant\n')
    return torch.tensor([ids]), torch.cat(positions, 1).unsqueeze(1), replacements, next_pos


def worker(gpu, model_path, jobs, results, max_new_tokens):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    torch.set_num_threads(1)
    try:
        from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation='sdpa', device_map={'': 'cuda:0'}).eval()
        eos = model.generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos)
        results.put({'event': 'ready', 'gpu': gpu})
        while True:
            task = jobs.get()
            if task is None:
                return
            started = time.perf_counter()
            try:
                with torch.inference_mode():
                    torch.cuda.reset_peak_memory_stats()
                    ids, positions, replacements, next_pos = sparse_prompt(tokenizer, model.config, task['question'], task['evidence'])
                    ids, positions = ids.cuda(), positions.cuda()
                    embeds = model.get_input_embeddings()(ids)
                    for begin, features in replacements:
                        embeds[0, begin:begin+len(features)] = features.to('cuda:0', dtype=embeds.dtype)
                    if ids.shape[1] + max_new_tokens > model.config.max_position_embeddings:
                        raise ValueError('QA readout exceeds model context')
                    model.model.rope_deltas = None
                    output = model(inputs_embeds=embeds, position_ids=positions, use_cache=True, logits_to_keep=1)
                    generated, completed = [], False
                    for step in range(max_new_tokens):
                        token = int(output.logits[0, -1].argmax())
                        generated.append(token)
                        if token in eos:
                            completed = True
                            break
                        output = model(input_ids=torch.tensor([[token]], device='cuda:0'),
                                       position_ids=torch.full((3, 1, 1), next_pos + step, device='cuda:0', dtype=torch.long),
                                       past_key_values=output.past_key_values, use_cache=True, logits_to_keep=1)
                    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
                    result = {'event': 'prediction', 'qa_id': task['question']['qa_id'], 'gpu': gpu,
                              'answer': answer, 'status': 'ok' if completed else 'generation_truncated',
                              'completed_eos': completed, 'input_tokens': ids.shape[1],
                              'visual_tokens_read': sum(len(r['features']) for r in task['evidence']),
                              'generated_tokens': len(generated), 'latency_seconds': time.perf_counter()-started,
                              'peak_gpu_bytes': torch.cuda.max_memory_allocated()}
                    del output, embeds, ids, positions
                    torch.cuda.empty_cache()
            except Exception as exc:
                result = {'event': 'prediction', 'qa_id': task['question']['qa_id'], 'gpu': gpu,
                          'status': 'error', 'answer': '', 'error': str(exc), 'traceback': traceback.format_exc()}
            results.put(result)
            del task
    except Exception as exc:
        results.put({'event': 'worker_error', 'gpu': gpu, 'error': str(exc), 'traceback': traceback.format_exc()})


def run(args):
    from baselines.feature_memory.archive import iter_archive
    from baselines.streambench.scoring import score_answer
    torch.set_num_threads(args.cpu_threads)
    archive_readers=getattr(args,'archive_readers',1)
    archive_buffer_shards=getattr(args,'archive_buffer_shards',max(archive_readers,6))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    qa_raw = Path(args.qa).read_bytes()
    questions = json.loads(qa_raw)['questions']
    selected = sorted([q for q in questions if not q.get('need_audio', False)], key=lambda q:(absolute_time(q), str(q['qa_id'])))
    if args.max_questions:
        selected = selected[:args.max_questions]
    if not selected:
        raise ValueError('No visual-only questions')
    labels = {q['qa_id']: q for q in selected}
    header_path = Path(args.archive) / 'archive.json'
    header = json.loads(header_path.read_text())
    identity = header.get('identity', {})
    if (header.get('frame_shape') != [2500,3584] or header.get('dtype') != 'bfloat16'
            or identity.get('participant') != 'A1_JAKE' or not identity.get('independent_image_encoding')
            or identity.get('checkpoint_repo_id') != 'Qwen/Qwen2.5-VL-7B-Instruct'):
        raise ValueError('Expected native JAKE independent-image Qwen archive')
    for filename, field in [('config.json','model_config_sha256'), ('preprocessor_config.json','processor_config_sha256')]:
        if hashlib.sha256((Path(args.model)/filename).read_bytes()).hexdigest() != identity.get(field):
            raise ValueError('Model/archive identity differs')
    memory = RoutineAwareMemory(RoutineConfig(budget_bytes=args.budget_bytes))
    memory.set_source_context({'archive_id': header['archive_id'], 'header_sha256':hashlib.sha256(header_path.read_bytes()).hexdigest(),
                               'participant':'A1_JAKE', 'representation':'independent_image_post_merger'})
    write_json(output/'run_config.json', {**vars(args), 'selected_qa_count':len(selected),
               'total_qa_count':len(questions), 'excluded_audio_qa_count':sum(bool(q.get('need_audio',False)) for q in questions),
               'archive_id':header['archive_id'], 'cache_identity':identity,
               'source_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in ('online_qa.py','memory.py','components.py','budget.py','config.py','parallel_archive.py')},
               'input_readers':archive_readers,'input_buffer_shards':archive_buffer_shards,
               'read_timing_basis':'consumer_wait_wall' if archive_readers>1 else 'serial_read_and_hash_wall',
               'time_basis':'canonical_v2 query_time already contains DAY offset; archive timestamps use the same absolute clock',
               'qa_sha256':hashlib.sha256(qa_raw).hexdigest(), 'ground_truth_in_model':False,
               'audio_used':False, 'memory_checkpoints_written':False, 'new_archive_written':False,
               'readout':'question-independent uniform retained-token rows; native sparse mRoPE adaptation',
               'online_policy':'single chronological memory; query-cutoff frozen readout; no day reset',
               'replication':'six QA model replicas, one shared 16GB memory; replicas have no persistent QA KV'})
    ctx = mp.get_context('spawn')
    jobs, results = ctx.Queue(maxsize=len(args.gpus)), ctx.Queue()
    workers = [ctx.Process(target=worker, args=(g,args.model,jobs,results,args.max_new_tokens)) for g in args.gpus]
    for process in workers:
        process.start()
    started, cursor, submitted, completed, correct, failures, ready = time.perf_counter(), 0, {}, 0, 0, 0, set()
    read_seconds, update_seconds, reader_worker_seconds, delivered_bytes = 0.0, 0.0, 0.0, 0
    status = {'phase':'loading_models', 'gpus':args.gpus, 'qa_total':len(selected), 'pid':os.getpid()}
    def save_status(phase):
        status.update(phase=phase, qa_submitted=len(submitted), qa_completed=completed, correct=correct,
                      errors=failures, accuracy=correct/completed if completed else None,
                      elapsed_seconds=time.perf_counter()-started, memory=memory.stats(), ready_gpus=sorted(ready),
                      archive_read_seconds=read_seconds, memory_update_seconds=update_seconds,
                      archive_reader_worker_seconds=reader_worker_seconds, archive_delivered_bytes=delivered_bytes,
                      archive_read_timing_basis='consumer_wait_wall' if archive_readers>1 else 'serial_read_and_hash_wall',
                      archive_readers=archive_readers,cpu_threads=args.cpu_threads,
                      new_encoder_calls=0, raw_video_decoding_calls=0)
        write_json(output/'status.json',status)
    with (output/'predictions.jsonl').open('w',buffering=1) as predictions, (output/'metrics.jsonl').open('w',buffering=1) as metrics, (output/'operations.jsonl').open('w',buffering=1) as operations, (output/'predictions.csv').open('w',newline='',buffering=1) as csv_file:
        csv_columns=['qa_id','day','query_time','gpu','answer','ground_truth','correct','status',
                     'routines','events','candidates','event_to_routine_promotions',
                     'short_term_bytes','routine_memory_bytes','event_memory_bytes','metadata_bytes','actual_memory_bytes']
        csv_writer=csv.DictWriter(csv_file,fieldnames=csv_columns)
        csv_writer.writeheader()
        def receive(block=False):
            nonlocal completed, correct, failures
            try:
                result = results.get(timeout=5 if block else .001)
            except queue.Empty:
                if any(p.exitcode is not None and p.exitcode != 0 for p in workers):
                    raise RuntimeError('QA worker exited unexpectedly')
                return False
            if result['event']=='worker_error':
                raise RuntimeError(str(result))
            if result['event']=='ready':
                ready.add(result['gpu'])
                save_status('loading_models')
                return True
            row = labels[result['qa_id']]
            scored, scoring = score_answer(result.get('answer',''),row)
            passed = bool(scored) and result['status']=='ok'
            completed += 1
            correct += int(passed)
            failures += int(result['status']!='ok')
            result.update(question=public_question(row), ground_truth=row['answer'], correct=passed,
                          scoring=scoring, memory_at_query=submitted[result['qa_id']],
                          causal_audit_passed=True, audio_used=False)
            predictions.write(json.dumps(result,allow_nan=False)+'\n')
            csv_writer.writerow({key:result.get(key,public_question(row).get(key,submitted[row['qa_id']].get(key,''))) for key in csv_columns})
            save_status('running')
            return True
        try:
            save_status('loading_models')
            while len(ready)<len(workers):
                receive(True)
                save_status('loading_models')
                if time.perf_counter()-started > 900:
                    raise RuntimeError('GPU model loading timed out')
            save_status('streaming')
            if archive_readers>1:
                from .parallel_archive import iter_parallel_archive
                batches=iter_parallel_archive(args.archive,archive_readers,archive_buffer_shards)
            else:
                batches=iter_archive(args.archive,wait=False)
            for batch in batches:
                read_seconds += batch['read_seconds']
                reader_worker_seconds += batch.get('reader_worker_seconds',batch['read_seconds'])
                delivered_bytes += batch.get('file_bytes_read',0)
                for features, frame in zip(batch['features'],batch['frames']):
                    stamp=float(frame['timestamp'])
                    while cursor<len(selected) and absolute_time(selected[cursor])<stamp:
                        row=selected[cursor]
                        cutoff=absolute_time(row)
                        stats=memory.stats()
                        if stats['last_timestamp'] is not None and stats['last_timestamp']>cutoff:
                            raise ValueError('Future frame entered query memory')
                        evidence=select_evidence(memory,args.qa_read_tokens,cutoff)
                        task={'question':public_question(row),'evidence':evidence}
                        while True:
                            try:
                                jobs.put(task,timeout=.1)
                                break
                            except queue.Full:
                                receive(True)
                        submitted[row['qa_id']]={**stats,'query_absolute_timestamp':cutoff,
                                                'readout_timestamps':[r['timestamp'] for r in evidence],
                                                'readout_frame_ids':[r['frame_id'] for r in evidence]}
                        cursor+=1
                        metrics.write(json.dumps({'event':'query_boundary','qa_id':row['qa_id'],**submitted[row['qa_id']]})+'\n')
                        while receive():
                            pass
                    if cursor==len(selected):
                        break
                    decision=memory.update(features,frame['frame_id'],stamp,
                        metadata={'source_id':frame['source_id'],'day':frame['day'],'feature_sha256':frame['shared_feature_sha256']})
                    update_seconds += decision['processing_seconds']
                    operations.write(json.dumps(decision,allow_nan=False)+'\n')
                    if frame['frame_id']%30==0:
                        save_status('streaming')
                        metrics.write(json.dumps({'event':'stream_progress',**status})+'\n')
                        while receive():
                            pass
                if cursor==len(selected):
                    break
            if cursor!=len(selected):
                raise RuntimeError('Archive ended before all requested query times')
            save_status('waiting_for_answers')
            while completed<len(submitted):
                receive(True)
            save_status('complete')
            write_json(output/'summary.json',status)
        except BaseException as exc:
            status['error']=str(exc)
            save_status('failed')
            raise
        finally:
            for process in workers:
                if process.is_alive():
                    process.terminate()
            for process in workers:
                process.join(timeout=15)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',required=True)
    parser.add_argument('--qa',required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--gpus',type=int,nargs='+',default=[0])
    parser.add_argument('--budget-bytes',type=int,default=16_000_000_000)
    parser.add_argument('--qa-read-tokens',type=int,default=8192)
    parser.add_argument('--max-new-tokens',type=int,default=32)
    parser.add_argument('--max-questions',type=int,default=0)
    parser.add_argument('--cpu-threads',type=int,default=2)
    parser.add_argument('--archive-readers',type=int,default=1)
    parser.add_argument('--archive-buffer-shards',type=int,default=6)
    args=parser.parse_args()
    if len(set(args.gpus))!=len(args.gpus) or args.qa_read_tokens<1 or args.max_new_tokens<1:
        parser.error('GPUs must be unique and token limits positive')
    if args.cpu_threads<1 or args.archive_readers<1 or args.archive_buffer_shards<args.archive_readers:
        parser.error('CPU/readers must be positive; buffer_shards must cover readers')
    run(args)


if __name__=='__main__':
    main()
