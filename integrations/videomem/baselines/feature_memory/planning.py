"""Convert retention ratios to immutable absolute persistent-byte budgets."""
from decimal import Decimal, ROUND_FLOOR
from fractions import Fraction
import json
import math
from pathlib import Path


RATIOS = (0.05, 0.10, 0.25, 0.50)
METHODS = ('fifo', 'uniform', 'reservoir', 'fluxmem')


def make_plan(inventory, model_path, *, max_seconds=None, ratios=RATIOS):
    from transformers import Qwen2VLImageProcessor
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    if inventory.get('total_duration_seconds') is None:
        raise ValueError('All source durations must be known')
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError('max_seconds must be positive and finite')
    full_stop = Fraction(inventory['total_duration_fraction'])
    stop = min(full_stop, Fraction(str(max_seconds)) if max_seconds is not None else full_stop)
    config = json.loads((Path(model_path) / 'config.json').read_text())
    if config.get('model_type') != 'qwen2_5_vl':
        raise ValueError('The shared representation must use Qwen2.5-VL')
    vision = config['vision_config']
    hidden = int(vision['out_hidden_size'])
    processor = Qwen2VLImageProcessor.from_pretrained(str(model_path), local_files_only=True)
    sources = []
    frames = tokens = run_frames = run_tokens = 0
    grids = set()
    for source in inventory['sources']:
        start = Fraction(source['global_start_fraction'])
        if not source['verified'] or source['width'] is None or source['height'] is None:
            raise ValueError('Requested source must be verified with known dimensions: ' + source['path'])
        duration = Fraction(source['duration_fraction'])
        count = math.ceil(duration)
        run_count = math.ceil(min(duration, max(Fraction(0), stop - start)))
        height, width = smart_resize(source['height'], source['width'],
                                     factor=processor.patch_size * processor.merge_size,
                                     min_pixels=processor.min_pixels, max_pixels=processor.max_pixels)
        gh, gw = height // processor.patch_size, width // processor.patch_size
        per_frame = gh * gw // processor.merge_size ** 2
        grids.add((gh // processor.merge_size, gw // processor.merge_size))
        sources.append({'path': source['path'], 'frames': count, 'run_frames': run_count,
                        'tokens_per_frame': per_frame, 'post_merger_grid_hw': [gh // processor.merge_size, gw // processor.merge_size],
                        'feature_bytes_per_frame': per_frame * hidden * 2})
        frames += count
        tokens += count * per_frame
        run_frames += run_count
        run_tokens += run_count * per_frame
    # Native FluxMem streaming grids have fixed spatial geometry. Reject mixed
    # grids explicitly rather than silently changing the shared representation.
    if len(grids) != 1:
        raise ValueError('All four methods currently require a fixed shared post-merger spatial grid')
    total = tokens * hidden * 2
    settings = []
    used = set()
    for ratio in ratios:
        value = Decimal(str(ratio))
        if not value.is_finite() or not 0 < value <= 1 or value in used:
            raise ValueError('Ratios must be unique, finite, and in (0,1]')
        used.add(value)
        cap = int((Decimal(total) * value).to_integral_value(rounding=ROUND_FLOOR))
        nominal_tokens = cap // (hidden * 2)
        label = format(value * 100, 'f').rstrip('0').rstrip('.') if '.' in format(value * 100, 'f') else format(value * 100, 'f')
        settings.append({'retention_ratio': float(value), 'label': label + 'pct',
                         'hard_budget_bytes': cap, 'hard_budget_GB': cap / 1e9,
                         'target_visual_tokens': nominal_tokens,
                         'approximate_frame_equivalent': nominal_tokens / (tokens / frames),
                         'target_tokens_are_payload_only_upper_bound': True})
    return {'schema': 'feature_memory_byte_budget_v1', 'methods': list(METHODS),
            'dtype': 'bfloat16', 'hidden_dim': hidden, 'element_size': 2,
            'post_merger_grid_hw': list(next(iter(grids))),
            'tokens_per_frame': tokens / frames, 'bytes_per_token': hidden * 2,
            'planned_frames': run_frames, 'planned_visual_tokens': run_tokens,
            'reference_frames': frames, 'reference_visual_tokens': tokens,
            'planned_run_feature_bytes': run_tokens * hidden * 2,
            'full_feature_bytes': total, 'full_feature_GB': total / 1e9,
            'full_feature_TB': total / 1e12, 'duration_seconds': float(stop),
            'reference_duration_seconds': float(full_stop),
            'budget_basis': 'fixed uncompressed bf16 post-merger footprint of the FULL source inventory; independent of smoke/run cutoff or snapshot time',
            'full_feature_bytes_are_planned_not_yet_observed': True,
            'settings': settings, 'sources': sources,
            'sum_all_state_hard_budgets': sum(s['hard_budget_bytes'] for s in settings) * len(METHODS)}


def plan_table(plan):
    rows = ['Method | Ratio | Hard Budget GB | Actual GB | Frames Seen | Retained Tokens | Frame-equivalent | Compression Ratio | Build Time',
            '--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:']
    for method in plan['methods']:
        for setting in plan['settings']:
            rows.append(f"{method} | {setting['retention_ratio']:.0%} | {setting['hard_budget_GB']:.6f} | pending | 0 | pending | pending | pending | pending")
    return '\n'.join(rows) + '\n'
