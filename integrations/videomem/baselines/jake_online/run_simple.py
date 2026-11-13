"""QA-parallel SimpleStream recency policy with native Qwen3 raw-image input.

This preserves DeepStack and native mRoPE rather than using the upstream cached
embedding builder. It is explicitly recorded as a native-input adaptation.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from .inputs import decode_sample, recent_samples, sha256
from baselines.streambench.scoring import score_answer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--canonical', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--ledger', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--limit', type=int)
    p.add_argument('--budget-gib', type=int, default=16)
    p.add_argument('--days', type=int, nargs='+', default=[1])
    p.add_argument('--source-index', default='day1_sources_v1.json')
    args = p.parse_args()
    assert 0 <= args.shard < args.shards and args.budget_gib > 0
    used = int(subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
        '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
    assert used < 1024, f'GPU occupied: {used} MiB'
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', USE_TF='0', USE_FLAX='0', TOKENIZERS_PARALLELISM='false')
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.enabled = True

    manifest = json.loads((args.canonical / 'jake_qa_v1.json').read_text())
    sources = json.loads((args.canonical / args.source_index).read_text())['sources']
    scorer_path = Path(__file__).parents[1] / 'streambench/scoring.py'
    assert sha256(scorer_path) == manifest['scorer_sha256']
    ledger = json.loads(args.ledger.read_text())
    assert args.model.resolve() == Path(ledger['model_path']).resolve()
    assert ledger['status'] == 'verified'
    questions = [q for q in manifest['questions'] if q['day'] in args.days]
    assert questions and all(q['query_time'] is not None for q in questions)
    assert set(args.days) <= {s['day'] for s in sources}
    selected = questions[args.shard::args.shards]
    if args.limit is not None:
        selected = selected[:args.limit]
    for source in sources:
        s = (args.root / source['path']).stat()
        assert [s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns] == source['file_state']
    provenance = {'method': 'SimpleStream-' + ('8B' if '8B' in args.model.name else '4B'),
        'implementation': 'native_raw_multi_image_adaptation_not_upstream_cached_builder',
        'canonical_qa_sha256': sha256(args.canonical / 'jake_qa_v1.json'),
        'sources_sha256': sha256(args.canonical / args.source_index), 'days': args.days,
        'scorer_sha256': manifest['scorer_sha256'], 'model_revision': ledger['revision'],
        'official_source_revision': 'be227ee8ce90a258b286a34161688c2a59eeeadb',
        'budget_bytes': args.budget_gib * 2**30, 'budget_scope': 'retained observation state; excludes model, temporary prefill and QA KV',
        'recent_frames': 4, 'fps': 1, 'sampling': 'source-local grid; inclusive causal cutoff',
        'max_new_tokens': 256, 'do_sample': False, 'dtype': 'bfloat16',
        'attention': 'flash_attention_2', 'gpu': args.gpu,
        'cudnn_enabled': True, 'cudnn_version': torch.backends.cudnn.version(),
        'patch_conv_memory_format': 'channels_last_3d',
        'runtime_note': 'Native Conv3D and unchanged values/weights; channels-last storage avoids RTX3090 BF16 biased Conv3D stall.',
        'shard': args.shard, 'shards': args.shards, 'qa_ids': [q['qa_id'] for q in selected],
        'audio_input': False, 'name_identity_supervision': False,
        'raw_gt_fields_never_passed_to_model': True,
        'runner_sha256': sha256(__file__), 'inputs_sha256': sha256(Path(__file__).with_name('inputs.py'))}
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2', device_map={'': 'cuda:0'}).eval()
    model.model.visual.patch_embed.proj.to(memory_format=torch.channels_last_3d)
    model.model.visual.patch_embed.proj.register_forward_pre_hook(
        lambda module, values: (values[0].contiguous(memory_format=torch.channels_last_3d),))
    started = time.perf_counter()
    results = []
    with (args.output / 'predictions.jsonl').open('w') as output:
        for question in selected:
            record = {'method': provenance['method'], 'qa_id': question['qa_id'],
                'query_time': question['query_time'], 'original_query_time': question['original_query_time'],
                'GT': question['answer'], 'prediction': '', 'raw_output': '', 'correct': False,
                'status': 'error', 'error': None, 'actual_frame_timestamps': [],
                'answer_latency': None, 'stream_update_time': 0.0,
                'need_audio': question['need_audio'], 'need_name': question['need_name']}
            try:
                print('decode', question['qa_id'], flush=True)
                decode_start = time.perf_counter()
                samples = recent_samples(sources, question['query_time'], n=4)
                pairs = [decode_sample(args.root, s, question['query_time']) for s in samples]
                images, actual = zip(*pairs)
                assert len(images) == 4
                assert len({(s['source'], s['source_pts']) for s in actual}) == 4
                assert all(s['actual_global_timestamp'] <= question['query_time'] + 1e-8 for s in actual)
                record['actual_frame_timestamps'] = actual
                record['decode_seconds'] = time.perf_counter() - decode_start
                record['retained_observation_bytes'] = sum(im.width * im.height * len(im.getbands()) for im in images)
                assert record['retained_observation_bytes'] <= provenance['budget_bytes']
                options = '; '.join(f'{chr(65+i)}. {choice}' for i, choice in enumerate(question['choices'])) + ';'
                prompt = question['question'] + '\nOptions: ' + options + "\nOnly give the best option's letter directly."
                content = [{'type': 'image', 'image': im} for im in images] + [{'type': 'text', 'text': prompt}]
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                answer_start = time.perf_counter()
                inputs = processor.apply_chat_template([{'role': 'user', 'content': content}],
                    tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt').to('cuda')
                print('generate', question['qa_id'], inputs['input_ids'].shape[-1], flush=True)
                record['input_tokens'] = inputs['input_ids'].shape[-1]
                record['image_grid_thw'] = inputs['image_grid_thw'].tolist()
                # New generation call and native raw pixels preserve every DeepStack feature.
                model.model.rope_deltas = None
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=256, do_sample=False, use_cache=True)
                torch.cuda.synchronize()
                record['answer_latency'] = time.perf_counter() - answer_start
                tail = generated[:, inputs['input_ids'].shape[-1]:]
                raw = processor.batch_decode(tail, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                record.update(prediction=raw.strip(), raw_output=raw, status='ok',
                    generated_token_ids=tail[0].tolist(),
                    peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_vram_reserved_bytes=torch.cuda.max_memory_reserved())
                eos = model.generation_config.eos_token_id
                eos = eos if isinstance(eos, list) else [eos]
                record['stopped_on_eos'] = bool(tail.shape[-1] and int(tail[0, -1]) in eos)
                if not record['stopped_on_eos']:
                    record.update(status='incomplete_generation', error='Generation ended without EOS')
                record['correct'], record['score_kind'] = score_answer(raw, question)
                del inputs, generated, tail
            except Exception as error:
                record['error'] = f'{type(error).__name__}: {error}'
            results.append(record)
            output.write(json.dumps(record, ensure_ascii=False) + '\n'); output.flush()
            elapsed = time.perf_counter() - started
            status = {'status': 'running', 'completed': len(results), 'total': len(selected),
                'errors': sum(r['status'] != 'ok' for r in results), 'elapsed_seconds': elapsed,
                'qa_per_second': len(results) / elapsed,
                'peak_vram_allocated_bytes': max((r.get('peak_vram_allocated_bytes', 0) for r in results), default=0)}
            (args.output / 'status.json').write_text(json.dumps(status, indent=2))
            print(question['qa_id'], record['status'], record['prediction'], flush=True)
            if record['status'] != 'ok':
                status['status'] = 'failed'
                (args.output / 'status.json').write_text(json.dumps(status, indent=2))
                raise RuntimeError(record['error'])
    status['status'] = 'complete'
    (args.output / 'status.json').write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
