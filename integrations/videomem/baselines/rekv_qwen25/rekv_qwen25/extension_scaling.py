"""One explicitly requested 8-GPU extension of completed 1/2/4-GPU scaling.

Only inputs.py's recorded source hash may differ from the prior configuration.
The frozen runner, weights, attention, prefix, frame protocol and all semantic
configuration fields remain identical. No full archive run is launched here.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import signal
import sys

from . import coordinator
from .archive import _normalize_config, committed_blocks, load_block


_BASE_WORKER_VALIDATOR = coordinator.validate_worker_report


def extension_plan(root):
    return dict(gpu_count=8, archive_root=str(Path(root) / 'benchmarks/scaling_n8'),
        workers=[dict(worker_id=f'gpu{gpu}', gpu=gpu, start_frame=(gpu+1)*100,
                      end_frame=(gpu+2)*100) for gpu in range(8)])


def validate_lineage(prior_config, current_config):
    """Permit exactly the documented reader repair, with no semantic changes."""
    before, after = copy.deepcopy(_normalize_config(prior_config)), copy.deepcopy(_normalize_config(current_config))
    old_sources, new_sources = before.get('extraction_source_sha256'), after.get('extraction_source_sha256')
    if not isinstance(old_sources, dict) or not isinstance(new_sources, dict) or set(old_sources) != set(new_sources):
        raise ValueError('Both configurations must bind the same extraction source files')
    changed = sorted(name for name in old_sources if old_sources[name] != new_sources[name])
    if changed != ['inputs.py']:
        raise ValueError('Only the inputs.py reader-repair source hash may change')
    new_sources['inputs.py'] = old_sources['inputs.py']
    if before != after:
        raise ValueError('Scaling extension configuration changes algorithm or other semantic fields')
    for filename, expected in current_config['extraction_source_sha256'].items():
        if Path(filename).name != filename or not filename.endswith('.py'):
            raise ValueError('Invalid extraction source filename')
        if coordinator.sha256_file(Path(__file__).with_name(filename)) != expected:
            raise ValueError(f'Current extraction source does not match config: {filename}')
    return dict(changed_source_files=changed, semantic_config_equal=True,
                prior_inputs_sha256=old_sources['inputs.py'],
                current_inputs_sha256=current_config['extraction_source_sha256']['inputs.py'])


def validate_extension_worker_report(report, worker, expected_config_id, *, benchmark=False):
    _BASE_WORKER_VALIDATOR(report, worker, expected_config_id, benchmark=False)
    if benchmark:
        start = worker['start_frame']
        replay_start = max(0, start-620)
        if 17 + replay_start*678 <= 15000:
            replay_start = 0
        if (report.get('resumed_frames') != 0 or report.get('committed_frames') != 100
                or worker['end_frame']-start != 100 or report.get('warmup_frames') != start-replay_start):
            raise ValueError('8-GPU benchmark requires fresh 100-frame output and the validated replay length')
    return report


@contextmanager
def _extension_worker_validation():
    """Inject the extended warmup contract only in this coordinator process."""
    previous = coordinator.validate_worker_report
    coordinator.validate_worker_report = validate_extension_worker_report
    try:
        yield
    finally:
        coordinator.validate_worker_report = previous


def _checked_indices(stage):
    root = Path(stage['archive_root'])
    entries = committed_blocks(root, expected_config_id=stage['config_id'])
    expected = {frame//100 for worker in stage['workers']
                for frame in range(worker['start_frame'], worker['end_frame'], 100)}
    if {entry['chunk_id'] for entry in entries} != expected:
        raise ValueError('Scaling archive does not match its assigned chunk coverage')
    return {entry['chunk_id']: json.loads((root/'indices'/f"chunk_{entry['chunk_id']:08d}.json").read_text())
            for entry in entries}


def compare_transition_chunks(reference, candidate):
    """Compare receipt-verified tensor specs/hashes across reader-only config IDs."""
    left, right = _checked_indices(reference), _checked_indices(candidate)
    common = sorted(set(left) & set(right))
    if not common:
        raise ValueError('No overlapping chunks for configuration transition fidelity')
    result = []
    for chunk in common:
        first, second = left[chunk], right[chunk]
        if first['tensors'] != second['tensors']:
            raise ValueError(f'Configuration transition changed tensor values/geometry in chunk {chunk}')
        a, b = copy.deepcopy(first['metadata']), copy.deepcopy(second['metadata'])
        if a.pop('config_id', None) != reference['config_id'] or b.pop('config_id', None) != candidate['config_id']:
            raise ValueError('Chunk metadata does not bind its actual archive configuration')
        if a != b:
            raise ValueError(f'Configuration transition changed frame/source/position metadata in chunk {chunk}')
        result.append(dict(status='passed', chunk_id=chunk,
            reference_archive_root=reference['archive_root'], candidate_archive_root=candidate['archive_root'],
            reference_config_id=reference['config_id'], candidate_config_id=candidate['config_id'],
            reference_gpu_count=reference['gpu_count'], candidate_gpu_count=candidate['gpu_count'],
            frame_count=first['metadata']['frame_count'], tensor_count=len(first['tensors']),
            all_tensor_sha256_exact=True, all_tensor_geometry_exact=True,
            metadata_equivalent_except_config=True, ignored_metadata_fields=['config_id']))
    return result


def prepare_extension(root, config_path, prior_config_path, prior_report_path=None, initial_state_root=None):
    root, config_path, prior_config_path = Path(root).resolve(), Path(config_path).resolve(), Path(prior_config_path).resolve()
    prior_report_path = Path(prior_report_path or root/'benchmarks/scaling_report.json').resolve()
    initial = Path(initial_state_root or root/'initial_state').resolve()
    current, validation = coordinator.validate_config(config_path, initial)
    prior_config = json.loads(prior_config_path.read_text())
    lineage = validate_lineage(prior_config, current)
    prior = json.loads(prior_report_path.read_text())
    prior_id, current_id = coordinator.config_id(prior_config), validation['config_id']
    if (prior.get('status') != 'complete' or prior.get('cross_gpu_fidelity', {}).get('status') != 'passed'
            or prior.get('validation', {}).get('config_id') != prior_id
            or [stage.get('gpu_count') for stage in prior.get('stages', [])] != [1, 2, 4]
            or any(stage.get('status') != 'complete' or stage.get('config_id') != prior_id for stage in prior['stages'])):
        raise ValueError('Prior report is not completed, verified 1/2/4-GPU evidence for the baseline config')
    if current['total_frames'] < 900:
        raise ValueError('8-GPU trial requires visual frames through 899')
    # Execute the runner's full proof/runtime/canonical-artifact checks before
    # launching any subprocess. Loading this small CPU artifact does not use CUDA.
    from .runner import _validate_config
    from .state_io import unflatten_state
    tensors, metadata = load_block(initial, 0, current['initial_state_config_id'])
    canonical = unflatten_state(tensors, metadata['resume_state'])
    receipt = json.loads((initial/'receipts/chunk_00000000.ready.json').read_text())
    binding = {key: receipt[key] for key in ('config_id', 'payload_sha256', 'index_sha256')}
    _validate_config(current, canonical, binding)
    lineage.update(prior_config_path=str(prior_config_path), prior_config_id=prior_id,
        prior_config_file_sha256=coordinator.sha256_file(prior_config_path),
        current_config_path=str(config_path), current_config_id=current_id,
        current_config_file_sha256=coordinator.sha256_file(config_path),
        prior_scaling_report_path=str(prior_report_path),
        prior_scaling_report_sha256=coordinator.sha256_file(prior_report_path))
    return dict(validation=validation, config_lineage=lineage, prior=prior,
                stage=extension_plan(root), initial_state_root=str(initial))


def run_extension(root, config_path, prior_config_path, prior_report_path=None,
                  initial_state_root=None, *, python_executable=None):
    prepared = prepare_extension(root, config_path, prior_config_path, prior_report_path, initial_state_root)
    output = Path(root).resolve()/'benchmarks/extended_scaling_report.json'
    if output.exists():
        raise ValueError(f'Refusing to overwrite prior extension evidence: {output}')
    prior = prepared['prior']
    summary = dict(schema='rekv_scaling_extension_v1', status='running',
        validation=prepared['validation'], config_lineage=prepared['config_lineage'],
        original_scaling_report=copy.deepcopy(prior), stages=copy.deepcopy(prior['stages']),
        cross_gpu_fidelity=copy.deepcopy(prior['cross_gpu_fidelity']),
        config_transition_fidelity=dict(status='pending', comparisons=[]),
        user_authorization=dict(requested_gpu_count=8, frames_per_gpu=100,
            source='explicit_user_request', scope='one_additional_scaling_trial'),
        started_at=coordinator.utc_now(), full_run_started=False)
    coordinator.write_json(output, summary)
    try:
        with _extension_worker_validation():
            stage = coordinator.run_stage(prepared['stage'], config_path,
                prepared['validation']['config_id'], prepared['initial_state_root'],
                python_executable=python_executable)
        for previous in prior['stages']:
            summary['config_transition_fidelity']['comparisons'].extend(compare_transition_chunks(previous, stage))
        lineage = prepared['config_lineage']
        for path_field, hash_field in [('prior_config_path', 'prior_config_file_sha256'),
                ('current_config_path', 'current_config_file_sha256'),
                ('prior_scaling_report_path', 'prior_scaling_report_sha256')]:
            if coordinator.sha256_file(lineage[path_field]) != lineage[hash_field]:
                raise ValueError('Configuration or prior evidence changed during the extension trial')
        stage['speedup_over_one_gpu'] = stage['steady_fps']/prior['stages'][0]['steady_fps']
        stage['parallel_efficiency'] = stage['speedup_over_one_gpu']/8
        stage['speedup_over_four_gpu'] = stage['steady_fps']/prior['stages'][2]['steady_fps']
        summary['stages'].append(stage)
        summary['config_transition_fidelity']['status'] = 'passed'
        summary.update(status='complete', completed_at=coordinator.utc_now(),
            fastest_stage_gpu_count=max(summary['stages'], key=lambda item: item['steady_fps'])['gpu_count'],
            total_serialized_bytes=sum(item['serialized_bytes'] for item in summary['stages']))
        coordinator.write_json(output, summary)
        return summary
    except BaseException as exc:
        summary.update(status='failed', error=f'{type(exc).__name__}: {exc}', updated_at=coordinator.utc_now())
        coordinator.write_json(output, summary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root', required=True)
    parser.add_argument('--config', required=True, help='Reader-repair configuration v2')
    parser.add_argument('--baseline-config', required=True, help='Preserved configuration v1')
    parser.add_argument('--baseline-report')
    parser.add_argument('--initial-state-root')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f'Extension coordinator received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    function = prepare_extension if args.plan_only else run_extension
    kwargs = {} if args.plan_only else {'python_executable': args.python}
    result = function(args.archive_root, args.config, args.baseline_config,
                      args.baseline_report, args.initial_state_root, **kwargs)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
