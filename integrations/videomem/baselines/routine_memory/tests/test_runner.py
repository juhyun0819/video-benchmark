import json
from pathlib import Path

import pytest
import torch

from baselines.feature_memory.archive import ArchiveWriter
from baselines.routine_memory.runner import inspect_archive, main, make_config


@pytest.fixture
def archive(tmp_path):
    directory = tmp_path / 'archive'
    frames = 6
    payload = frames * 676 * 3584 * 2
    header = {'frame_shape': [676, 3584],
              'model_identity': {'repo_id': 'Qwen/Qwen2.5-VL-7B-Instruct', 'revision': 'c' * 40,
                                 'config_sha256': 'a' * 64, 'preprocessor_sha256': 'b' * 64},
              'plan': {'hidden_dim': 3584, 'tokens_per_frame': 676,
                       'dtype': 'bfloat16', 'full_feature_bytes': payload},
              'inventory': {'fps': 1.0, 'total_1fps_frames': frames, 'total_duration_seconds': float(frames)}}
    writer = ArchiveWriter(directory, header, frames_per_shard=3)
    feature = torch.zeros(676, 3584, dtype=torch.bfloat16)
    feature[:, 0] = 1
    for fid in range(frames):
        writer.append(feature, {'frame_id': fid, 'timestamp': float(fid), 'source_id': 'synthetic_archive'})
    writer.complete(frames)
    writer.close()
    return directory


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'budget_bytes': 16_000_000, 'short_term_frames': 2,
                               'spatial_bins': 4, 'temporal_bins': 2, 'evidence_tokens': 4,
                               'stability_frames': 1, 'occurrence_gap_seconds': 2.0}))
    return path


def assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype and torch.equal(left, right)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_state_equal(a, b)
    else:
        assert left == right


def test_archive_build_cutoff_resume_and_query(archive, config_path, tmp_path):
    from baselines.routine_memory.checkpoint import load_checkpoint
    prefix = tmp_path / 'prefix'
    result = main(['build', '--archive', str(archive), '--config', str(config_path),
                   '--max-frames', '3', '--snapshot-seconds', '2', '--output', str(prefix)])
    assert result['total_frames_seen'] == 3
    assert result['new_encoder_calls'] == 0
    assert not (prefix / 'history').exists()
    snapshot = load_checkpoint(prefix / 'snapshots/t_2')
    assert snapshot.stats()['total_frames_seen'] == 2
    assert snapshot.stats()['last_timestamp'] == 1.0
    assert snapshot.control['runner_source']['archive_id'] == _load(archive / 'archive.json')['archive_id']
    resumed = tmp_path / 'resumed'
    result = main(['build', '--archive', str(archive), '--max-frames', '6',
                   '--resume-checkpoint', str(prefix / 'checkpoint_final'), '--output', str(resumed)])
    assert result['resume_frames'] == 3 and result['frames_processed_this_run'] == 3
    fresh = tmp_path / 'fresh'
    main(['build', '--archive', str(archive), '--config', str(config_path),
          '--max-frames', '6', '--output', str(fresh)])
    a = load_checkpoint(resumed / 'checkpoint_final').state_dict()
    b = load_checkpoint(fresh / 'checkpoint_final').state_dict()
    assert_state_equal(a, b)
    rows = [json.loads(line) for line in (resumed / 'decisions.jsonl').read_text().splitlines()]
    assert [row['frame_id'] for row in rows] == [3, 4, 5]
    retrieval = main(['query', '--checkpoint', str(resumed / 'checkpoint_final'),
                      '--archive', str(archive), '--frame-id', '0', '--top-k', '2',
                      '--output', str(tmp_path / 'query')])
    assert retrieval['qa_inference'] is False
    assert len(retrieval['matches']) <= 2
    assert (tmp_path / 'query/retrieval.json').is_file()


def _load(path):
    return json.loads(Path(path).read_text())


def test_header_mismatch_and_ratio_reference(archive, tmp_path):
    header, source = inspect_archive(archive)
    config = make_config(budget_ratio='0.05', full_reference_bytes=source['full_reference_feature_bytes'])
    assert config.budget_bytes == source['full_reference_feature_bytes'] * 5 // 100
    header['model_identity']['config_sha256'] = 'invalid'
    (archive / 'archive.json').write_text(json.dumps(header))
    with pytest.raises(ValueError, match='model hash'):
        inspect_archive(archive)


def test_synthetic_promotes_after_independent_visits_and_routes_all_cases(tmp_path):
    output = tmp_path / 'synthetic'
    result = main(['synthetic', '--output', str(output)])
    assert result['config']['budget_bytes'] == 16_000_000_000
    assert result['total_frames_seen'] == 64
    assert all(result['case_counts'][case] > 0 for case in ('A', 'B', 'C'))
    assert result['promotion_examples']
    assert all(row['timestamp'] >= 80 for row in result['promotion_examples'])
    assert result['case_counts_by_phase']['explained_A']['A'] > 0
    assert result['case_counts_by_phase']['partial_B_local_residual']['B'] > 0
    assert result['case_counts_by_phase']['unexplained_C']['C'] > 0


def test_default_is_bounded_prefix_and_full_requires_explicit_flag(monkeypatch, tmp_path):
    import baselines.routine_memory.runner as runner
    observed = []
    monkeypatch.setattr(runner, 'build', lambda args: observed.append(args))
    main(['build', '--archive', str(tmp_path), '--output', str(tmp_path / 'out')])
    assert observed[0].full is False and observed[0].max_frames is None
    with pytest.raises(SystemExit):
        main(['build', '--archive', str(tmp_path), '--full', '--max-frames', '10', '--output', str(tmp_path / 'out')])
