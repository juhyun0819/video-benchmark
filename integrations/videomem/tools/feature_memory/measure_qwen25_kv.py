#!/usr/bin/env python3
"""Short, question-free Qwen2.5-VL cache probe using committed image features.

No video decode, vision re-encoding, generation, or memory policy runs here.
Each requested frame count is an independent batch-one prefill. The native
image-placeholder and MRoPE operations are reused; archived features replace
only the already-computed post-merger image embeddings.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def cache_tensors(cache):
    from transformers.cache_utils import DynamicCache
    if not isinstance(cache, DynamicCache):
        raise TypeError('Expected native DynamicCache, got ' + type(cache).__name__)
    result = {}
    for index, layer in enumerate(cache.layers):
        result[f'layer_{index:02d}.key'] = layer.keys
        result[f'layer_{index:02d}.value'] = layer.values
    if not result:
        raise ValueError('Native forward returned an empty cache')
    return result


def describe_cache(cache, expected_layers, expected_tokens, expected_heads, expected_head_dim):
    tensors = cache_tensors(cache)
    if len(tensors) != expected_layers * 2:
        raise ValueError('Cache layer count differs from model config')
    records, storages = [], {}
    for name, tensor in tensors.items():
        expected = (1, expected_heads, expected_tokens, expected_head_dim)
        if tuple(tensor.shape) != expected:
            raise ValueError(f'{name}: expected {expected}, got {tuple(tensor.shape)}')
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        storages[key] = storage.nbytes()
        records.append({'name': name, 'shape': list(tensor.shape),
                        'dtype': str(tensor.dtype).removeprefix('torch.'),
                        'device': str(tensor.device), 'stride': list(tensor.stride()),
                        'payload_bytes': tensor.numel() * tensor.element_size(),
                        'backing_storage_bytes': storage.nbytes()})
    return {'cache_class': type(cache).__name__, 'tensors': records,
            'kv_tensor_payload_bytes': sum(x['payload_bytes'] for x in records),
            'kv_unique_backing_storage_bytes': sum(storages.values()),
            'sequence_length': cache.get_seq_length()}


def serialize_cache(cache, path):
    """Save K/V tensors only, then reload every tensor and check exact equality."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    started = time.perf_counter()
    cpu = {name: value.detach().to(device='cpu').contiguous()
           for name, value in cache_tensors(cache).items()}
    copied = time.perf_counter() - started
    started = time.perf_counter()
    save_file(cpu, str(path), metadata={'kind': 'qwen2_5_vl_prefill_key_value_tensors_only',
                                      'not_a_complete_generation_checkpoint': 'true'})
    saved = time.perf_counter() - started
    started = time.perf_counter()
    with safe_open(str(path), framework='pt', device='cpu') as handle:
        if set(handle.keys()) != set(cpu):
            raise ValueError('Serialized cache tensor names differ')
        for name, reference in cpu.items():
            actual = handle.get_tensor(name)
            if actual.dtype != reference.dtype or not torch.equal(actual, reference):
                raise ValueError('Serialized cache reload differs: ' + name)
    verified = time.perf_counter() - started
    payload = sum(x.numel() * x.element_size() for x in cpu.values())
    return {'path': str(path), 'serialized_file_bytes': path.stat().st_size,
            'serialized_header_bytes': path.stat().st_size - payload,
            'file_sha256': sha256(path), 'reload_exact_equal': True,
            'serialization_scope': 'K/V tensors only; JSON metadata stored separately',
            'cache_copy_to_cpu_seconds': copied, 'serialize_seconds': saved,
            'reload_validation_seconds': verified}


def cpu_native_equivalence_check():
    """Small native model: ordinary placeholder insertion vs reused embeddings.

    The synthetic vision output is identical on both branches. This verifies
    insertion/MRoPE/cache semantics, not pretrained model accuracy.
    """
    import torch
    import tempfile
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    torch.manual_seed(19)
    config = Qwen2_5_VLConfig(
        text_config={'vocab_size': 32, 'hidden_size': 32, 'intermediate_size': 64,
                     'num_hidden_layers': 2, 'num_attention_heads': 4,
                     'num_key_value_heads': 2, 'rope_scaling': {'type': 'mrope', 'mrope_section': [1, 1, 2]}},
        vision_config={'depth': 1, 'hidden_size': 32, 'intermediate_size': 64,
                       'num_heads': 4, 'out_hidden_size': 32, 'patch_size': 2,
                       'spatial_merge_size': 2, 'temporal_patch_size': 2,
                       'fullatt_block_indexes': [0]},
        image_token_id=4, video_token_id=5, vision_start_token_id=2, vision_end_token_id=3)
    config._attn_implementation = 'sdpa'
    model = Qwen2_5_VLForConditionalGeneration(config).eval()
    ids = torch.tensor([[1, 2, 4, 4, 4, 4, 3, 2, 4, 4, 4, 4, 3, 6]])
    grid = torch.tensor([[1, 4, 4], [1, 4, 4]])
    features = torch.randn(8, 32)
    model.model.get_image_features = lambda *args, **kwargs: (features[:4], features[4:])
    mask = torch.ones_like(ids)
    with torch.inference_mode():
        native = model(input_ids=ids, pixel_values=torch.zeros(1), image_grid_thw=grid,
                       attention_mask=mask, use_cache=True, logits_to_keep=1)
        embeddings = model.get_input_embeddings()(ids)
        image_mask, _ = model.model.get_placeholder_mask(ids, embeddings, image_features=features)
        embeddings = embeddings.masked_scatter(image_mask, features)
        positions, deltas = model.model.get_rope_index(ids, grid, attention_mask=mask)
        reused = model(inputs_embeds=embeddings, position_ids=positions, attention_mask=mask,
                       use_cache=True, logits_to_keep=1)
    native_cache, reused_cache = cache_tensors(native.past_key_values), cache_tensors(reused.past_key_values)
    if not torch.equal(native.logits, reused.logits):
        raise AssertionError('Native and reused-feature logits differ')
    for name in native_cache:
        if not torch.equal(native_cache[name], reused_cache[name]):
            raise AssertionError('Native and reused-feature KV differ: ' + name)
    with tempfile.TemporaryDirectory(prefix='qwen25-kv-cpu-check-') as scratch:
        serialized = serialize_cache(reused.past_key_values, Path(scratch) / 'tiny_kv.safetensors')
    return {'passed': True, 'native_kv_exact_equal': True, 'native_logits_exact_equal': True,
            'serialization_reload_exact_equal': serialized['reload_exact_equal'],
            'layers': 2, 'frames': 2, 'tokens_per_frame': 4, 'sequence_length': ids.shape[1],
            'position_shape': list(positions.shape), 'rope_deltas': deltas.tolist(),
            'scope': 'Tiny CPU native architecture; synthetic post-merger features, no pretrained inference'}


def build_input(tokenizer, config, frame_count, tokens_per_frame, grid_hw):
    import torch
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
    image = '<|vision_start|>' + '<|image_pad|>' * tokens_per_frame + '<|vision_end|>'
    prompt = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': 'Visual observation.\n' + image * frame_count}],
        tokenize=False, add_generation_prompt=False)
    batch = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
    ids, mask = batch['input_ids'], batch['attention_mask']
    merge = config.vision_config.spatial_merge_size
    grid = torch.tensor([[1, grid_hw[0] * merge, grid_hw[1] * merge]] * frame_count)
    positions, deltas = Qwen2_5_VLModel.get_rope_index(
        SimpleNamespace(config=config), ids, grid, attention_mask=mask)
    visual = int((ids == config.image_token_id).sum())
    starts = int((ids == config.vision_start_token_id).sum())
    ends = int((ids == config.vision_end_token_id).sum())
    if visual != frame_count * tokens_per_frame or starts != frame_count or ends != frame_count:
        raise ValueError('Tokenizer image placeholders do not match archived feature geometry')
    if int((ids == config.video_token_id).sum()):
        raise ValueError('Video tokens unexpectedly appeared in independent-image prompt')
    metadata = {'frames': frame_count, 'visual_tokens': visual, 'sequence_tokens': ids.numel(),
                'nonvisual_tokens': ids.numel() - visual, 'image_boundary_tokens': starts + ends,
                'fixed_text_tokens': ids.numel() - visual - starts - ends,
                'image_grid_thw': grid.tolist(), 'position_ids_shape': list(positions.shape),
                'rope_deltas': deltas.tolist(), 'prompt': prompt.replace('<|image_pad|>' * tokens_per_frame,
                                                                       f'[IMAGE_PAD x {tokens_per_frame}]')}
    return ids, mask, positions, deltas, metadata


def prepare(args):
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from baselines.feature_memory.archive import iter_archive
    from baselines.feature_memory.encoder import _checkpoint_info
    archive = Path(args.archive).resolve(strict=True)
    model_path = Path(args.model).resolve(strict=True)
    header = json.loads((archive / 'archive.json').read_text())
    if header['dtype'] != 'bfloat16' or header['frame_shape'] != [676, 3584]:
        raise ValueError('Expected the existing Qwen2.5-VL [676,3584] bf16 archive')
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    if config.model_type != 'qwen2_5_vl' or config.text_config.hidden_size != 3584:
        raise ValueError('Wrong pretrained model architecture')
    if config.text_config.use_sliding_window:
        raise ValueError('This probe requires the checkpoint full-attention cache')
    index = json.loads((model_path / 'model.safetensors.index.json').read_text())
    required = ['config.json', 'preprocessor_config.json', 'model.safetensors.index.json',
                *sorted(set(index['weight_map'].values()))]
    checkpoint = _checkpoint_info(model_path, required)
    archive_identity = header.get('model_identity', {})
    for field, filename in [('config_sha256', 'config.json'),
                            ('preprocessor_sha256', 'preprocessor_config.json')]:
        if archive_identity.get(field) != sha256(model_path / filename):
            raise ValueError('Archive features and current model differ: ' + field)
    for field in ('repo_id', 'revision'):
        if archive_identity.get(field) and archive_identity[field] != checkpoint.get(field):
            raise ValueError('Archive features and verified checkpoint differ: ' + field)
    started = time.perf_counter()
    batches = list(iter_archive(archive, wait=False, max_frames=max(args.frames)))
    if not batches or sum(len(x['frames']) for x in batches) != max(args.frames):
        raise ValueError('Insufficient committed archive frames for requested probe')
    features = torch.cat([x['features'] for x in batches], dim=0)
    frames = [f for batch in batches for f in batch['frames']]
    archive_read_seconds = time.perf_counter() - started
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    inputs = [build_input(tokenizer, config, n, 676, (26, 26)) for n in args.frames]
    if len({x[-1]['fixed_text_tokens'] for x in inputs}) != 1:
        raise AssertionError('Prompt fixed text overhead changed with frame count')
    if any(x[-1]['sequence_tokens'] >= config.text_config.max_position_embeddings for x in inputs):
        raise ValueError('Probe sequence reaches or exceeds the checkpoint context limit')
    return config, features, inputs, {
        'checkpoint': checkpoint, 'archive': str(archive), 'archive_id': header['archive_id'],
        'archive_model_identity': archive_identity, 'archive_checkpoint_identity_matches': True,
        'model_geometry': {'num_hidden_layers': config.text_config.num_hidden_layers,
                           'num_attention_heads': config.text_config.num_attention_heads,
                           'num_key_value_heads': config.text_config.num_key_value_heads,
                           'hidden_size': config.text_config.hidden_size,
                           'head_dim': config.text_config.hidden_size // config.text_config.num_attention_heads,
                           'max_position_embeddings': config.text_config.max_position_embeddings,
                           'use_sliding_window': config.text_config.use_sliding_window},
        'archive_header_sha256': sha256(archive / 'archive.json'), 'archive_read_seconds': archive_read_seconds,
        'archive_read_validation': 'committed shard and individual frame SHA256 verified by iter_archive',
        'frames': [{'frame_id': f['frame_id'], 'timestamp': f['timestamp'],
                    'feature_sha256': f['shared_feature_sha256']} for f in frames],
        'feature_shape_per_frame': [676, 3584], 'feature_dtype': 'bfloat16',
        'feature_bytes_per_frame': 676 * 3584 * 2,
        'planned_full_frames': header['inventory']['total_1fps_frames'],
        'input_mode': 'independent 1fps image features, one image block per frame',
        'native_video_equivalent': False,
        'native_video_difference': 'Archive encodes each image independently with same-image temporal padding; native video pairs two source frames. Image MRoPE is used, not video timestamp MRoPE.',
        'qa_inference': False, 'generated_tokens': 0, 'vision_forward_calls': 0,
        'policies_or_compression': False, 'timestamps_enter_model_prompt': False,
        'cache_scope': 'All causal prefill K/V positions across all decoder layers; frozen model weights excluded',
        'inputs': [x[-1] for x in inputs]}


def run(args):
    import torch
    import transformers
    from transformers import Qwen2_5_VLForConditionalGeneration
    torch.set_num_threads(args.cpu_threads)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    status = {'pid': os.getpid(), 'model': args.model, 'device': args.device,
              'frames_processed': 0, 'total_frames': max(args.frames),
              'summary_path': str(output / 'summary.json')}
    def update_status(phase, state='running', **fields):
        status.update(status=state, phase=phase, **fields)
        write_json(output / 'status.json', status)
    try:
        update_status('cpu_checks')
        check = cpu_native_equivalence_check()
        config, features, inputs, metadata = prepare(args)
        metadata.update(cpu_native_equivalence=check, torch_version=torch.__version__,
                        transformers_version=transformers.__version__, script_sha256=sha256(__file__))
        write_json(output / 'input_plan.json', metadata)
        if args.cpu_check:
            update_status('cpu_checks_passed', state='cpu_checks_passed', gpu_started=False, checks=check)
            print(json.dumps({'status': 'cpu_checks_passed', 'output': str(output),
                              'inputs': metadata['inputs']}, indent=2), flush=True)
            return
        device = torch.device(args.device)
        if device.type != 'cuda' or not torch.cuda.is_available():
            raise ValueError('Actual pretrained probe requires an explicitly selected CUDA device')
        torch.cuda.set_device(device)
        update_status('model_loading', device=str(device))
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model, local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation='sdpa', device_map={'': str(device)}, low_cpu_mem_usage=True)
        model.eval().requires_grad_(False)
        torch.cuda.synchronize(device)
        metadata.update(model_load_seconds=time.perf_counter() - started,
                        model_load_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                        model_load_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                        model_parameter_bytes=sum(x.numel() * x.element_size() for x in model.parameters()),
                        gpu_name=torch.cuda.get_device_name(device), device=str(device),
                        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                        attention_implementation='sdpa', logits_to_keep=1)
        # A guard makes accidental vision re-encoding an explicit failure.
        def forbid_vision(*unused):
            raise AssertionError('The archived-feature probe must not run the vision encoder')
        hook = model.model.visual.register_forward_pre_hook(forbid_vision)
        text_config = config.text_config
        per_token = 2 * text_config.num_hidden_layers * text_config.num_key_value_heads * (
            text_config.hidden_size // text_config.num_attention_heads) * 2
        rows = []
        for ids_cpu, mask_cpu, positions_cpu, deltas, input_metadata in inputs:
            count = input_metadata['frames']
            update_status('prefill', current_probe_frames=count)
            gc.collect()
            torch.cuda.empty_cache()
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            with torch.inference_mode():
                ids, mask = ids_cpu.to(device), mask_cpu.to(device)
                positions = positions_cpu.to(device)
                visual = features[:count].reshape(-1, 3584).to(device)
                embeddings = model.get_input_embeddings()(ids)
                image_mask, _ = model.model.get_placeholder_mask(ids, embeddings, image_features=visual)
                embeddings = embeddings.masked_scatter(image_mask, visual)
                torch.cuda.synchronize(device)
                preparation_seconds = time.perf_counter() - started
                started = time.perf_counter()
                result = model(inputs_embeds=embeddings, position_ids=positions, attention_mask=mask,
                               use_cache=True, logits_to_keep=1, return_dict=True)
                torch.cuda.synchronize(device)
                prefill_seconds = time.perf_counter() - started
            report = describe_cache(result.past_key_values, text_config.num_hidden_layers, ids.numel(),
                                    text_config.num_key_value_heads,
                                    text_config.hidden_size // text_config.num_attention_heads)
            if report['kv_tensor_payload_bytes'] != ids.numel() * per_token:
                raise AssertionError('Actual KV bytes disagree with native model geometry')
            if any(x['dtype'] != 'bfloat16' for x in report['tensors']):
                raise AssertionError('Unexpected actual cache dtype')
            if result.logits.shape[1] != 1:
                raise AssertionError('logits_to_keep=1 was not honored')
            report.update(input_metadata, input_prepare_h2d_seconds=preparation_seconds,
                          prefill_seconds=prefill_seconds, gpu_baseline_allocated_bytes=baseline,
                          gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                          gpu_peak_increment_allocated_bytes=torch.cuda.max_memory_allocated(device) - baseline,
                          bytes_per_kv_position=per_token,
                          visual_position_kv_bytes=input_metadata['visual_tokens'] * per_token,
                          nonvisual_position_kv_bytes=input_metadata['nonvisual_tokens'] * per_token,
                          visual_position_kv_bytes_per_frame=676 * per_token)
            report.update(serialize_cache(result.past_key_values, output / f'kv_{count:02d}_frames.safetensors'))
            write_json(output / f'kv_{count:02d}_frames.json', report)
            rows.append(report)
            update_status('sample_complete', frames_processed=count)
            print(json.dumps({key: report[key] for key in ('frames', 'sequence_tokens', 'kv_tensor_payload_bytes',
                                                          'serialized_file_bytes', 'prefill_seconds',
                                                          'gpu_peak_allocated_bytes')}), flush=True)
            del result, embeddings, visual, image_mask, ids, mask, positions
        hook.remove()
        total_frames = metadata['planned_full_frames']
        metadata.update(status='completed', measurements=rows, projection={
            'is_projection_not_full_history_llm_run': True, 'frames': total_frames,
            'kv_bytes_per_position': per_token, 'visual_only_kv_bytes': total_frames * 676 * per_token,
            'with_per_image_boundary_and_fixed_prompt_bytes': (
                total_frames * (676 + 2) + inputs[0][-1]['fixed_text_tokens']) * per_token,
            'does_not_imply_full_history_fits_model_context_or_gpu': True,
            'visual_kv_to_archived_feature_payload_ratio': (676 * per_token) / metadata['feature_bytes_per_frame']})
        write_json(output / 'summary.json', metadata)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        update_status('completed', state='completed', summary=str(output / 'summary.json'))
    except BaseException as exc:
        update_status('failed', state='failed', error_type=type(exc).__name__, error=str(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, help='Committed visual-feature archive directory')
    parser.add_argument('--model', required=True, help='Local Qwen2.5-VL-7B-Instruct checkpoint matching the archive')
    parser.add_argument('--frames', type=int, nargs='+', default=[1, 4, 8])
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--cpu-check', action='store_true', help='Validate archive, native MRoPE and tiny CPU model only')
    args = parser.parse_args()
    if args.frames != sorted(set(args.frames)) or not args.frames or min(args.frames) < 1 or max(args.frames) > 8:
        parser.error('--frames must be increasing unique counts in 1..8 for this short probe')
    if args.cpu_threads < 1:
        parser.error('--cpu-threads must be positive')
    run(args)


if __name__ == '__main__':
    main()
