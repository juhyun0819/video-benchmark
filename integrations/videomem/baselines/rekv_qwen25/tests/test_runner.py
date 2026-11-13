"""CPU contracts for production gating, ready-marker waiting and exact ranges.

These tests do not load weights, initialize CUDA or approve a real halo run.
Synthetic proof/artifact metadata exercises fail-closed validation; the real
model proof and archive reconstruction integration have separate tests.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rekv_qwen25 import runner


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Runner contract tests must not load models or initialize CUDA')

    monkeypatch.setattr(torch.cuda, 'set_device', forbidden)
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    monkeypatch.setattr(runner.Qwen2_5_VLForConditionalGeneration, 'from_pretrained', forbidden)
    monkeypatch.setattr(runner.AutoProcessor, 'from_pretrained', forbidden)


@pytest.fixture
def proof_case(tmp_path):
    attention_hash = sha256(Path(runner.__file__).with_name('attention.py'))
    binding = dict(config_id='canonical-config', payload_sha256='a' * 64,
                   index_sha256='b' * 64)
    initial = dict(n_local=15000, n_init=17, layers=28, total_seen=17,
        native_post_rope_cache=[('synthetic-key', 'synthetic-value')],
        policy={'synthetic_test_policy': 'fixed-sinks'}, upstream={'revision': 'test-upstream'})
    checks = [dict(frame_id=24, all_kv_exact=False, logits={'exact': False})]
    checks.extend(dict(frame_id=frame, all_kv_exact=True, logits={'exact': True},
                       exact_layers=[True] * 28) for frame in range(644, 648))
    checks[1]['bounded_state_exact'] = True
    report = dict(status='passed', halo_frames=620, adaptation_sha256=attention_hash,
        verification_frames=4, checkpoint={'repo_id': runner.REPO_ID, 'revision': runner.REVISION},
        n_local=15000, n_init=17, policy=copy.deepcopy(initial['policy']),
        upstream=copy.deepcopy(initial['upstream']), short_backend='sdpa', chunk_frames=1,
        fixed_original_per_layer_sinks=True, absolute_positions=True,
        initial_state_binding=copy.deepcopy(binding), torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda, halo_tokens=420360,
        conservative_receptive_bound_tokens=420000, boundary_frame=24, checks=checks)
    report_path = tmp_path / 'synthetic_halo_report.json'
    config = dict(n_local=15000, n_init=17, frames_per_block=100, total_frames=221728,
        transformers_version='4.57.3', torch_version=str(torch.__version__),
        initial_state_config_id=binding['config_id'], initial_state_payload_sha256=binding['payload_sha256'],
        initial_state_index_sha256=binding['index_sha256'], prefix_protocol=runner.PREFIX_PROTOCOL,
        backbone={'repo_id': runner.REPO_ID, 'revision': runner.REVISION},
        policy=copy.deepcopy(initial['policy']), halo_validation=dict(status='passed', halo_frames=620,
            attention_sha256=attention_hash, report_path=str(report_path)))

    def publish_report():
        report_path.write_text(json.dumps(report))
        config['halo_validation']['report_sha256'] = sha256(report_path)

    publish_report()
    return SimpleNamespace(config=config, initial=initial, binding=binding, report=report,
                           report_path=report_path, publish=publish_report, attention_hash=attention_hash)


def validate(case):
    return runner._validate_config(case.config, case.initial, case.binding)


def test_proof_binds_actual_protocol_runtime_and_initial_artifact(proof_case):
    assert validate(proof_case) == (620, proof_case.attention_hash)
    assert proof_case.report_path.read_bytes()


@pytest.mark.parametrize('field,value', [
    ('n_local', 14999), ('n_init', 16), ('frames_per_block', 64),
    ('prefix_protocol', 'combined_prefix_and_frame'), ('total_frames', True),
    ('transformers_version', '4.57.2'), ('torch_version', 'different-runtime'),
    ('initial_state_config_id', 'other-config'), ('initial_state_payload_sha256', 'c' * 64),
    ('initial_state_index_sha256', 'd' * 64),
])
def test_changed_controlled_configuration_is_rejected(proof_case, field, value):
    proof_case.config[field] = value
    with pytest.raises(ValueError):
        validate(proof_case)


def test_actual_transformers_runtime_is_checked(proof_case, monkeypatch):
    monkeypatch.setattr(runner.transformers, '__version__', '4.58.0')
    with pytest.raises(ValueError, match='runtime'):
        validate(proof_case)


@pytest.mark.parametrize('field,value', [('total_seen', 695), ('native_post_rope_cache', None), ('layers', 27)])
def test_initial_state_must_be_exact_separate_prefix(proof_case, field, value):
    proof_case.initial[field] = value
    with pytest.raises(ValueError):
        validate(proof_case)


def test_report_file_hash_is_checked_before_semantic_acceptance(proof_case):
    proof_case.report_path.write_text(proof_case.report_path.read_text() + '\n')
    with pytest.raises(ValueError, match='content hash'):
        validate(proof_case)


@pytest.mark.parametrize('mutation', [
    'missing_window', 'missing_sink_length', 'wrong_policy', 'wrong_upstream',
    'wrong_backend', 'wrong_chunking', 'different_sinks', 'reset_positions',
    'wrong_torch', 'wrong_cuda', 'wrong_bound', 'wrong_halo_tokens',
    'wrong_checkpoint', 'wrong_initial_payload', 'wrong_initial_index',
    'missing_comparisons', 'missing_bounded_state', 'inexact_layer',
    'inexact_logits', 'wrong_verification_count', 'vacuous_perturbation',
])
def test_hash_valid_but_insufficient_scientific_proof_is_rejected(proof_case, mutation):
    report = proof_case.report
    replacements = {
        'wrong_policy': ('policy', {'different': True}),
        'wrong_upstream': ('upstream', {'revision': 'different'}),
        'wrong_backend': ('short_backend', 'eager'),
        'wrong_chunking': ('chunk_frames', 4),
        'different_sinks': ('fixed_original_per_layer_sinks', False),
        'reset_positions': ('absolute_positions', False),
        'wrong_torch': ('torch_version', 'other-torch'),
        'wrong_cuda': ('cuda_version', 'other-cuda'),
        'wrong_bound': ('conservative_receptive_bound_tokens', 15000),
        'wrong_halo_tokens': ('halo_tokens', 420000),
        'wrong_checkpoint': ('checkpoint', {'repo_id': runner.REPO_ID, 'revision': 'other'}),
        'missing_comparisons': ('checks', []),
        'wrong_verification_count': ('verification_frames', 3),
    }
    if mutation in replacements:
        field, value = replacements[mutation]
        report[field] = value
    elif mutation == 'missing_window':
        del report['n_local']
    elif mutation == 'missing_sink_length':
        del report['n_init']
    elif mutation == 'wrong_initial_payload':
        report['initial_state_binding']['payload_sha256'] = 'c' * 64
    elif mutation == 'wrong_initial_index':
        report['initial_state_binding']['index_sha256'] = 'c' * 64
    elif mutation == 'missing_bounded_state':
        del report['checks'][1]['bounded_state_exact']
    elif mutation == 'inexact_layer':
        report['checks'][1]['exact_layers'][27] = False
    elif mutation == 'inexact_logits':
        report['checks'][2]['logits']['exact'] = False
    elif mutation == 'vacuous_perturbation':
        report['checks'][0].update(all_kv_exact=True, logits={'exact': True})
    proof_case.publish()  # A matching file hash alone must not license the run.
    with pytest.raises(ValueError):
        validate(proof_case)


def test_source_hashes_are_checked_without_modifying_source(proof_case):
    source = Path(runner.__file__)
    proof_case.config['extraction_source_sha256'] = {'runner.py': sha256(source)}
    assert validate(proof_case)[0] == 620
    proof_case.config['extraction_source_sha256']['runner.py'] = '0' * 64
    with pytest.raises(ValueError, match='source changed'):
        validate(proof_case)
    proof_case.config['extraction_source_sha256'] = {'../runner.py': sha256(source)}
    with pytest.raises(ValueError, match='filename'):
        validate(proof_case)


@pytest.fixture
def feature_markers(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    header = dict(archive_id='visual-archive', frames_per_shard=100,
                  inventory={'total_1fps_frames': 325})
    (root / 'archive.json').write_text(json.dumps(header))
    receipts = {}
    for shard in range(4):
        receipt = dict(archive_id=header['archive_id'], shard_id=shard,
            feature_file=f'features_{shard:08d}.bin', index_file=f'index_{shard:08d}.json')
        receipts[shard] = receipt
        for field in ('feature_file', 'index_file'):
            (root / receipt[field]).touch()
        (root / f'ready_{shard:08d}.json').write_text(json.dumps(receipt))
    return SimpleNamespace(root=root, receipts=receipts,
                           config={'total_frames': 325, 'feature_archive_id': header['archive_id']})


def test_feature_range_waits_for_marker_then_continues(feature_markers, monkeypatch):
    case = feature_markers
    marker = case.root / 'ready_00000002.json'
    marker.unlink()
    states, sleeps = [], []

    def producer_finishes(seconds):
        sleeps.append(seconds)
        marker.write_text(json.dumps(case.receipts[2]))

    monkeypatch.setattr(runner.time, 'sleep', producer_finishes)
    archive_id, seconds = runner._feature_range(case.root, 100, 300, case.config,
        wait=True, timeout=60, status=lambda state, **kw: states.append((state, kw)))
    assert archive_id == 'visual-archive' and seconds >= 0
    assert sleeps == [5]
    assert states[0][0] == 'waiting_for_features'
    assert states[0][1]['missing_receipt'] == marker.name


@pytest.mark.parametrize('mutation', ['missing_payload', 'missing_index', 'wrong_archive',
                                    'wrong_filename', 'malformed_marker', 'missing_header'])
def test_committed_source_corruption_never_waits(feature_markers, monkeypatch, mutation):
    case = feature_markers
    receipt = case.receipts[1]
    marker = case.root / 'ready_00000001.json'
    if mutation == 'missing_payload':
        (case.root / receipt['feature_file']).unlink()
    elif mutation == 'missing_index':
        (case.root / receipt['index_file']).unlink()
    elif mutation == 'wrong_archive':
        receipt['archive_id'] = 'other-archive'
        marker.write_text(json.dumps(receipt))
    elif mutation == 'wrong_filename':
        receipt['feature_file'] = 'uncommitted.partial'
        marker.write_text(json.dumps(receipt))
    elif mutation == 'malformed_marker':
        marker.write_text('{broken json')
    else:
        (case.root / 'archive.json').unlink()

    def must_not_sleep(*args):
        pytest.fail('Committed source corruption must fail immediately')

    monkeypatch.setattr(runner.time, 'sleep', must_not_sleep)
    with pytest.raises((ValueError, FileNotFoundError)):
        runner._feature_range(case.root, 100, 200, case.config, wait=True, timeout=60,
                              status=lambda *a, **k: None)


def test_scaling_mode_does_not_wait_for_uncommitted_source(feature_markers, monkeypatch):
    case = feature_markers
    (case.root / 'ready_00000001.json').unlink()
    monkeypatch.setattr(runner.time, 'sleep', lambda *args: pytest.fail('Waiting is disabled'))
    with pytest.raises(FileNotFoundError):
        runner._feature_range(case.root, 100, 200, case.config, wait=False, timeout=60,
                              status=lambda *a, **k: None)


@pytest.fixture
def fake_clock(monkeypatch):
    clock = SimpleNamespace(now=0.0, sleeps=[])

    def sleep(seconds):
        assert seconds >= 0
        clock.sleeps.append(seconds)
        clock.now += seconds

    monkeypatch.setattr(runner.time, 'monotonic', lambda: clock.now)
    monkeypatch.setattr(runner.time, 'sleep', sleep)
    return clock


def test_source_wait_has_bounded_timeout(feature_markers, fake_clock):
    case = feature_markers
    (case.root / 'ready_00000001.json').unlink()
    with pytest.raises(TimeoutError, match='did not become ready'):
        runner._feature_range(case.root, 100, 200, case.config, wait=True, timeout=6,
                              status=lambda *a, **k: None)
    assert fake_clock.sleeps == [5, 1]


@pytest.mark.parametrize('start,expected', [(0, 0), (100, 0), (620, 0), (642, 0), (643, 23), (1000, 380)])
def test_replay_starts_from_prefix_or_sufficient_long_halo(start, expected):
    assert runner._warmup_start(start, 620, 17, 15000) == expected
    if expected:
        assert start - expected == 620
        assert 17 + expected * 678 > 15000


@pytest.mark.parametrize('start,end', [(0, 100), (100, 300), (221700, 221728)])
def test_worker_range_accepts_global_blocks_and_exact_tail(start, end):
    runner._check_range(start, end, 221728)


@pytest.mark.parametrize('start,end', [(-100, 100), (1, 100), (100, 151), (100, 100), (221700, 221729)])
def test_worker_range_rejects_partial_or_invalid_blocks(start, end):
    with pytest.raises(ValueError):
        runner._check_range(start, end, 221728)


def entry(start, end):
    return dict(chunk_id=start // 100, first_frame_id=start, last_frame_id=end - 1, frame_count=end - start)


def test_resume_uses_only_owned_contiguous_prefix():
    entries = [entry(200, 300), entry(0, 100), entry(400, 500), entry(100, 200)]
    cursor, owned = runner._owned_prefix(entries, 100, 400)
    assert cursor == 300
    assert {row['chunk_id'] for row in owned} == {1, 2}
    assert runner._owned_prefix([], 100, 400) == (100, [])
    assert runner._owned_prefix([entry(221700, 221728)], 221700, 221728)[0] == 221728


@pytest.mark.parametrize('entries', [
    [entry(200, 300)], [entry(100, 200), entry(300, 400)],
    [entry(100, 200), entry(100, 200)], [entry(100, 151)],
    [dict(entry(100, 200), chunk_id=0)], [entry(99, 199)],
])
def test_resume_rejects_gaps_duplicates_and_foreign_partial_blocks(entries):
    with pytest.raises(ValueError, match='coverage'):
        runner._owned_prefix(entries, 100, 400)


@pytest.fixture
def frame_payload():
    session = SimpleNamespace(attentions=[SimpleNamespace(num_key_value_heads=1, head_dim=4)],
        latest_kv={f'layer_00.{kind}': torch.full((1, 1, 678, 4), index + 1, dtype=torch.bfloat16)
                   for index, kind in enumerate(('key', 'value'))}, total_seen=17 + 102 * 678)
    inputs = dict(position_ids=torch.arange(678).reshape(1, 1, -1).expand(3, 1, -1),
                  input_ids=torch.arange(678).reshape(1, -1))
    return session, inputs, runner._allocate_block(session, 2)


def test_frame_capture_preserves_exact_offsets_and_owns_copied_data(frame_payload):
    session, inputs, payload = frame_payload
    runner._append_frame(payload, session, inputs, 0)
    first = {name: value.clone() for name, value in session.latest_kv.items()}
    for tensor in session.latest_kv.values():
        tensor.add_(10)
    next_inputs = {name: value + 1000 for name, value in inputs.items()}
    runner._append_frame(payload, session, next_inputs, 1)
    for name in first:
        assert torch.equal(payload[name][:, :, :678], first[name])
        assert torch.equal(payload[name][:, :, 678:], session.latest_kv[name])
    assert torch.equal(payload['position_ids'][:, :, :678], inputs['position_ids'])
    assert torch.equal(payload['position_ids'][:, :, 678:], next_inputs['position_ids'])
    assert torch.equal(payload['input_ids'][:, 678:], next_inputs['input_ids'])
    runner._require_finite_payload(payload)


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'dtype', 'shape'])
def test_capture_rejects_incomplete_or_incompatible_kv(frame_payload, mutation):
    session, inputs, payload = frame_payload
    if mutation == 'missing':
        del session.latest_kv['layer_00.value']
    elif mutation == 'extra':
        session.latest_kv['unexpected'] = torch.tensor(1)
    elif mutation == 'dtype':
        session.latest_kv['layer_00.value'] = session.latest_kv['layer_00.value'].float()
    else:
        session.latest_kv['layer_00.value'] = session.latest_kv['layer_00.value'][:, :, :1]
    with pytest.raises(ValueError):
        runner._append_frame(payload, session, inputs, 0)


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_raw_kv_cannot_be_committed(invalid):
    payload = {'layer_00.key': torch.tensor([1, invalid], dtype=torch.bfloat16),
               'position_ids': torch.tensor([1, 2], dtype=torch.int64)}
    with pytest.raises(ValueError, match='nonfinite raw KV'):
        runner._require_finite_payload(payload)


def test_frame_metadata_retains_timeline_and_block_relative_token_offsets(frame_payload):
    session, _, _ = frame_payload
    rows = [dict(frame_id=100 + i, timestamp_seconds=100.25 + i, day_id='day2') for i in range(2)]
    metadata = runner._block_metadata(rows, session, 100, 'config', 'initial', 'visual')
    assert metadata['frame_ids'] == [100, 101]
    assert metadata['timestamps'] == [100.25, 101.25]
    assert metadata['next_frame_id'] == 102 and metadata['total_seen'] == 17 + 102 * 678
    assert [(row['token_start'], row['token_end']) for row in metadata['frames']] == [(0, 678), (678, 1356)]
    assert 'token_start' not in rows[0]  # The source metadata stays immutable.
    with pytest.raises(ValueError, match='causal frame sequence'):
        runner._block_metadata(rows[::-1], session, 100, 'config', 'initial', 'visual')


def test_barrier_publishes_ready_before_waiting_for_coordinator(tmp_path, monkeypatch):
    directory = tmp_path / 'barrier'
    ready = dict(status='ready', worker_id='gpu0', start_frame=100, end_frame=200, config_id='controlled')
    states = []

    def release(seconds):
        assert json.loads((directory / 'gpu0.ready.json').read_text()) == ready
        (directory / 'start.json').write_text(json.dumps({'workers': ['gpu0', 'gpu1']}))

    monkeypatch.setattr(runner.time, 'sleep', release)
    runner._wait_barrier(directory, 'gpu0', ready, 60, lambda state: states.append(state))
    assert states == ['waiting_for_barrier']


def test_barrier_rejects_release_for_other_workers(tmp_path):
    (tmp_path / 'start.json').write_text(json.dumps({'workers': ['gpu1']}))
    with pytest.raises(ValueError, match='authorize this worker'):
        runner._wait_barrier(tmp_path, 'gpu0', {'worker_id': 'gpu0'}, 60, lambda *a: None)


def test_barrier_timeout_is_bounded(tmp_path, fake_clock):
    with pytest.raises(TimeoutError, match='did not release'):
        runner._wait_barrier(tmp_path, 'gpu0', {'worker_id': 'gpu0'}, 0.5, lambda *a: None)
    assert fake_clock.sleeps == [0.25, 0.25]
