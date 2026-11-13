"""Inspectable repeated visual patterns; no learned model or QA labels."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import time


def synthetic_stream():
    """Three separated visits, then routine, local residual, and new activity.

    A visit contains several similar frames. Its timestamp gap and intervening
    orthogonal patterns make recurrence distinct from contiguous persistence.
    """
    import torch
    base = torch.zeros(16, 8, dtype=torch.float32)
    base[:, 0] = 1.0
    for group in range(4):
        base[group * 4:(group + 1) * 4, group + 2] = 0.05
    separator = torch.zeros_like(base)
    separator[:, 7] = 1.0
    residual = base.clone()
    residual[12:] = 0
    residual[12:, 1] = 1.0
    novel = torch.zeros_like(base)
    novel[:, 6] = 1.0
    phases = [('novel_A_visit_1', 0, base), ('separator_1', 8, separator),
              ('A_visit_2', 40, base), ('separator_2', 48, separator),
              ('A_visit_3_promotion', 80, base), ('explained_A', 88, base),
              ('partial_B_local_residual', 96, residual), ('unexplained_C', 104, novel)]
    frame_id = 0
    for label, start, pattern in phases:
        for offset in range(8):
            yield {'features': pattern.to(torch.bfloat16).clone(), 'frame_id': frame_id,
                   'timestamp': float(start + offset), 'phase': label}
            frame_id += 1


def run_synthetic(args):
    import torch
    from .config import RoutineConfig
    from .memory import RoutineAwareMemory
    from .checkpoint import save_checkpoint
    from .runner import _write, _progress, make_config
    torch.set_num_threads(args.cpu_threads)
    if args.config:
        config = make_config(args.config, args.budget_bytes)
    else:
        config = RoutineConfig(short_term_frames=4, spatial_bins=4, temporal_bins=4,
                               evidence_tokens=16, stability_frames=3,
                               occurrence_gap_seconds=20.0,
                               **({'budget_bytes': args.budget_bytes} if args.budget_bytes is not None else {}))
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    memory = RoutineAwareMemory(config=config)
    memory.set_source_context({'kind': 'deterministic_synthetic_v1', 'frame_shape': [16, 8],
                               'dtype': 'bfloat16', 'no_pretrained_encoder': True})
    _write(output / 'run_config.json', {'config': asdict(config), 'source': 'deterministic_synthetic_v1',
                                       'feature_shape': [16, 8], 'feature_dtype': 'bfloat16',
                                       'qa_inference': False, 'new_encoder_calls': 0})
    started, update_seconds = time.perf_counter(), 0.0
    counts, phase_counts = {'A': 0, 'B': 0, 'C': 0}, {}
    promotion_rows = []
    try:
        with (output / 'decisions.jsonl').open('w') as decisions, (output / 'metrics.jsonl').open('w') as metrics:
            previous_promotions = 0
            for row in synthetic_stream():
                before = time.perf_counter()
                decision = memory.update(row['features'], frame_id=row['frame_id'], timestamp=row['timestamp'],
                                         metadata={'source_id': 'synthetic_v1'})
                seconds = time.perf_counter() - before
                update_seconds += seconds
                case = decision['case']
                counts[case] += 1
                phase_counts.setdefault(row['phase'], {'A': 0, 'B': 0, 'C': 0})[case] += 1
                audit = {'frame_id': row['frame_id'], 'timestamp': row['timestamp'],
                         'synthetic_phase': row['phase'], 'decision': decision, 'update_seconds': seconds}
                decisions.write(json.dumps(audit, sort_keys=True, allow_nan=False) + '\n')
                report = _progress(memory, started, row['frame_id'] + 1, update_seconds, 0.0)
                metrics.write(json.dumps(report, sort_keys=True, allow_nan=False) + '\n')
                promotions = report.get('promotions', report.get('event_to_routine_promotions', 0))
                if promotions > previous_promotions:
                    promotion_rows.append(audit)
                    previous_promotions = promotions
        manifest = save_checkpoint(memory, output / 'checkpoint_final')
        summary = {'status': 'completed', 'config': asdict(config),
                   **_progress(memory, started, row['frame_id'] + 1, update_seconds, 0.0),
                   'case_counts': counts, 'case_counts_by_phase': phase_counts,
                   'promotion_examples': promotion_rows, 'checkpoint': str(output / 'checkpoint_final'),
                   'checkpoint_manifest': manifest, 'qa_inference': False,
                   'synthetic_semantic_labels_are_for_audit_only': True,
                   'new_encoder_calls': 0, 'full_history_output_created': False}
        _write(output / 'summary.json', summary)
        _write(output / 'status.json', summary)
        print(json.dumps({'event': 'synthetic_complete', 'output': str(output),
                          'case_counts': counts, 'case_counts_by_phase': phase_counts,
                          'promotion_examples': promotion_rows}, sort_keys=True), flush=True)
        return summary
    except BaseException as exc:
        _write(output / 'status.json', {'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)})
        raise
