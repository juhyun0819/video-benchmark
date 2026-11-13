"""CPU-only reader-fix lineage and explicitly authorized 8-GPU orchestration."""
import copy
import json
from pathlib import Path

import pytest
import torch

from rekv_qwen25 import coordinator
from rekv_qwen25 import extension_scaling as extension
from rekv_qwen25.archive import KVArchiveWriter


@pytest.fixture
def configs():
    sources = {name: coordinator.sha256_file(Path(extension.__file__).with_name(name))
               for name in ('inputs.py', 'attention.py', 'runner.py')}
    current = dict(n_local=15000, prefix_protocol=coordinator.PREFIX_PROTOCOL,
        backbone={'repo_id': coordinator.CHECKPOINT, 'revision': coordinator.REVISION},
        extraction_source_sha256=sources)
    prior = copy.deepcopy(current)
    prior['extraction_source_sha256']['inputs.py'] = '0' * 64
    return prior, current


def test_plan_has_exactly_eight_disjoint_100_frame_workers(tmp_path):
    stage = extension.extension_plan(tmp_path)
    assert stage['gpu_count'] == 8
    assert [worker['gpu'] for worker in stage['workers']] == list(range(8))
    assert [(worker['start_frame'], worker['end_frame']) for worker in stage['workers']] == [
        ((i+1)*100, (i+2)*100) for i in range(8)]
    assert stage['archive_root'] == str(tmp_path/'benchmarks/scaling_n8')


def test_lineage_accepts_only_reader_hash_repair_without_mutating_configs(configs):
    prior, current = configs
    snapshots = copy.deepcopy(configs)
    result = extension.validate_lineage(prior, current)
    assert result['changed_source_files'] == ['inputs.py']
    assert result['semantic_config_equal'] is True
    assert configs == snapshots


@pytest.mark.parametrize('mutation', ['window', 'weights', 'prefix', 'other_source',
                                    'new_field', 'missing_source', 'no_change', 'stale_current_source'])
def test_lineage_rejects_any_unapproved_difference(configs, mutation):
    prior, current = configs
    if mutation == 'window': current['n_local'] = 14000
    elif mutation == 'weights': current['backbone']['revision'] = 'other'
    elif mutation == 'prefix': current['prefix_protocol'] = 'other'
    elif mutation == 'other_source': current['extraction_source_sha256']['attention.py'] = '1' * 64
    elif mutation == 'new_field': current['new_runtime_metadata'] = 'not-authorized'
    elif mutation == 'missing_source': del current['extraction_source_sha256']['runner.py']
    elif mutation == 'no_change': current['extraction_source_sha256']['inputs.py'] = prior['extraction_source_sha256']['inputs.py']
    else: current['extraction_source_sha256']['inputs.py'] = 'f' * 64
    with pytest.raises(ValueError):
        extension.validate_lineage(prior, current)


def worker_report(worker, config_id='current'):
    start = worker['start_frame']
    return dict(status='complete', config_id=config_id, **worker,
        committed_frames=100, resumed_frames=0, warmup_frames=min(start, 620),
        measured_seconds=100.0, read_seconds=1.0, infer_seconds=90.0,
        atomic_nas_seconds=9.0, serialized_bytes=1000, peak_vram_bytes=2000)


@pytest.mark.parametrize('gpu', range(8))
def test_extended_validator_accepts_proved_halo_and_rejects_wrong_replay(tmp_path, gpu):
    worker = extension.extension_plan(tmp_path)['workers'][gpu]
    report = worker_report(worker)
    assert extension.validate_extension_worker_report(report, worker, 'current', benchmark=True) is report
    report['warmup_frames'] += 1
    with pytest.raises(ValueError, match='replay length'):
        extension.validate_extension_worker_report(report, worker, 'current', benchmark=True)


def test_validator_override_is_scoped_even_on_failure():
    original = coordinator.validate_worker_report
    with pytest.raises(RuntimeError):
        with extension._extension_worker_validation():
            assert coordinator.validate_worker_report is extension.validate_extension_worker_report
            raise RuntimeError('simulated interrupted coordinator')
    assert coordinator.validate_worker_report is original


def archived_stage(root, config, *, gpu_count, changed_tensor=False, changed_metadata=False):
    writer = KVArchiveWriter(root, config)
    tensors = {'layer_00.key': torch.tensor([1, 2 + int(changed_tensor)], dtype=torch.bfloat16),
               'position_ids': torch.tensor([1, 2], dtype=torch.int64)}
    metadata = dict(frame_ids=list(range(100, 200)), timestamps=list(map(float, range(100, 200))),
        config_id=writer.config_id, position_convention='wrong' if changed_metadata else 'absolute-three-axis')
    writer.commit(1, tensors, metadata)
    return dict(gpu_count=gpu_count, archive_root=str(root), config_id=writer.config_id,
                workers=[dict(worker_id='gpu0', gpu=0, start_frame=100, end_frame=200)])


def test_transition_compares_receipt_verified_tensors_ignoring_only_config_id(tmp_path, configs):
    prior, current = configs
    reference = archived_stage(tmp_path/'old', prior, gpu_count=1)
    candidate = archived_stage(tmp_path/'new', current, gpu_count=8)
    comparisons = extension.compare_transition_chunks(reference, candidate)
    assert len(comparisons) == 1
    assert comparisons[0]['all_tensor_sha256_exact'] is True
    assert comparisons[0]['all_tensor_geometry_exact'] is True
    assert comparisons[0]['metadata_equivalent_except_config'] is True
    assert comparisons[0]['reference_config_id'] != comparisons[0]['candidate_config_id']


@pytest.mark.parametrize('difference', ['tensor', 'metadata', 'tampered_index'])
def test_transition_rejects_changed_values_metadata_or_unverified_index(tmp_path, configs, difference):
    prior, current = configs
    reference = archived_stage(tmp_path/'old', prior, gpu_count=1)
    candidate = archived_stage(tmp_path/'new', current, gpu_count=8,
        changed_tensor=difference == 'tensor', changed_metadata=difference == 'metadata')
    if difference == 'tampered_index':
        path = tmp_path/'new/indices/chunk_00000001.json'
        path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError):
        extension.compare_transition_chunks(reference, candidate)


def prepared_run(tmp_path, configs):
    prior_config, current_config = configs
    prior_path, current_path = tmp_path/'v1.json', tmp_path/'v2.json'
    prior_path.write_text(json.dumps(prior_config)); current_path.write_text(json.dumps(current_config))
    prior = dict(status='complete', validation={'config_id': 'prior'},
        cross_gpu_fidelity={'status': 'passed', 'comparisons': [{'original': True}]},
        stages=[dict(status='complete', gpu_count=n, config_id='prior', steady_fps=rate,
                     serialized_bytes=n*1000, archive_root=str(tmp_path/f'old{n}'))
                for n, rate in [(1, 0.67), (2, 1.08), (4, 1.46)]])
    prior_report = tmp_path/'prior_scaling.json'; prior_report.write_text(json.dumps(prior))
    lineage = dict(prior_config_path=str(prior_path), current_config_path=str(current_path),
        prior_config_file_sha256=coordinator.sha256_file(prior_path),
        current_config_file_sha256=coordinator.sha256_file(current_path),
        prior_scaling_report_path=str(prior_report), prior_scaling_report_sha256=coordinator.sha256_file(prior_report),
        prior_config_id='prior', current_config_id='current', changed_source_files=['inputs.py'],
        semantic_config_equal=True)
    return dict(prior=prior, validation={'config_id': 'current'}, config_lineage=lineage,
                stage=extension.extension_plan(tmp_path), initial_state_root=str(tmp_path/'initial'))


def test_extension_preserves_original_evidence_and_records_user_authorized_scope(tmp_path, configs, monkeypatch):
    prepared = prepared_run(tmp_path, configs)
    original = copy.deepcopy(prepared['prior'])
    monkeypatch.setattr(extension, 'prepare_extension', lambda *a: prepared)

    def stage_run(stage, *args, **kwargs):
        assert coordinator.validate_worker_report is extension.validate_extension_worker_report
        for worker in stage['workers']:
            coordinator.validate_worker_report(worker_report(worker), worker, 'current', benchmark=True)
        return dict(status='complete', gpu_count=8, steady_fps=1.8, serialized_bytes=8000,
                    config_id='current', archive_root=stage['archive_root'], workers=stage['workers'])

    monkeypatch.setattr(coordinator, 'run_stage', stage_run)
    monkeypatch.setattr(extension, 'compare_transition_chunks', lambda before, after: [
        dict(status='passed', chunk_id=i+1) for i in range(before['gpu_count'])])
    result = extension.run_extension(tmp_path, tmp_path/'v2.json', tmp_path/'v1.json')
    assert result['stages'][:3] == original['stages']
    assert result['original_scaling_report'] == original
    assert result['cross_gpu_fidelity'] == original['cross_gpu_fidelity']
    assert len(result['config_transition_fidelity']['comparisons']) == 7
    assert result['config_transition_fidelity']['status'] == 'passed'
    assert result['user_authorization']['requested_gpu_count'] == 8
    assert result['user_authorization']['frames_per_gpu'] == 100
    assert result['full_run_started'] is False
    assert result['stages'][-1]['speedup_over_four_gpu'] == pytest.approx(1.8/1.46)
    assert json.loads((tmp_path/'prior_scaling.json').read_text()) == original
    with pytest.raises(ValueError, match='overwrite'):
        extension.run_extension(tmp_path, tmp_path/'v2.json', tmp_path/'v1.json')


def test_failed_trial_records_failure_and_restores_original_validator(tmp_path, configs, monkeypatch):
    prepared = prepared_run(tmp_path, configs)
    monkeypatch.setattr(extension, 'prepare_extension', lambda *a: prepared)
    def failed(*a, **kw):
        raise RuntimeError('mock worker failure')
    monkeypatch.setattr(coordinator, 'run_stage', failed)
    original_validator = coordinator.validate_worker_report
    with pytest.raises(RuntimeError, match='mock worker failure'):
        extension.run_extension(tmp_path, tmp_path/'v2.json', tmp_path/'v1.json')
    report = json.loads((tmp_path/'benchmarks/extended_scaling_report.json').read_text())
    assert report['status'] == 'failed'
    assert len(report['stages']) == 3
    assert coordinator.validate_worker_report is original_validator
