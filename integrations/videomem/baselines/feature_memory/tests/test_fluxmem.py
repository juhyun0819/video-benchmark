"""Official-kernel equivalence, causal provenance and byte-cap checks."""
import importlib

import pytest
import torch

from baselines.feature_memory.fluxmem import (
    FluxMemByteMemory, FluxMemSnapshotReader, _provenance_class, kernel_provenance,
    load_snapshot, official_classes,
)


def sequence(count=12, dtype=torch.float32):
    rng = torch.Generator().manual_seed(419)
    result = []
    for index in range(count):
        frame = torch.randn(9, 8, generator=rng)
        frame[1] = frame[0] * .97 + frame[1] * .03
        if index % 3 == 1:
            frame[:3] = result[-1][:3].float()
        result.append(frame.to(dtype))
    return result


def upstream(sequence, short=4, medium=4):
    FluxMem, _ = official_classes()
    tokens = 9
    hidden = torch.cat((torch.zeros(1, 8, dtype=sequence[0].dtype), *sequence,
                        torch.zeros(1, 8, dtype=sequence[0].dtype))).unsqueeze(0)
    t = torch.arange(len(sequence)).repeat_interleave(tokens)
    h = torch.arange(3).repeat_interleave(3).repeat(len(sequence))
    w = torch.arange(3).repeat(3).repeat(len(sequence))
    positions = torch.nn.functional.pad(torch.stack((t, h, w)), (1, 1)).unsqueeze(1)
    input_ids = torch.tensor([[90] + [5] * (len(sequence) * tokens) + [91]])
    rotary = torch.zeros((3, 1, hidden.shape[1], 1))
    result = FluxMem(90, 91, short_term_frames=short, mid_term_frames=medium).process_memory_streaming(
        hidden, positions, (rotary, rotary), input_ids,
        video_grid_thw=torch.tensor([[len(sequence), 6, 6]]))
    return result[0][0, 1:-1], result[2][:, 0, 1:-1]


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_instrumented_stream_exactly_matches_released_kernel_each_prefix(dtype):
    bank = _provenance_class()(short_frames=4, medium_frames=4)
    values = sequence(dtype=dtype)
    for index, features in enumerate(values):
        bank.ingest(features, 3, 3, index, float(index), {'frame_id': index})
        bank.advance()
        actual = bank.snapshot()
        expected_features, expected_positions = upstream(values[:index + 1])
        torch.testing.assert_close(actual['embeddings'], expected_features, rtol=0, atol=0)
        torch.testing.assert_close(actual['positions'], expected_positions, rtol=0, atol=0)
        for grid in bank._long:
            counts = grid.metadata['merge_counts']
            assert counts.shape == (grid.tokens,)
            assert bool((counts >= 1).all())
    assert bank.num_removed_temporal_duplicates > 0
    assert bank.num_merged_tokens > 0
    assert sum(int((g.metadata['merge_counts'] - 1).sum()) for g in bank._long) == bank.num_merged_tokens


def test_video_import_does_not_mix_official_kernel_namespaces():
    importlib.import_module('baselines.feature_memory.video')
    kernel, persistent = official_classes()
    assert kernel.__module__ == '_feature_memory_official_fluxmem.fluxmem'
    assert persistent.__module__ == '_feature_memory_official_fluxmem.persistent'
    assert kernel_provenance()['kernel_math_modified'] is False


def _make(tmp_path, name, cap=100000):
    return FluxMemByteMemory(tmp_path / name, cap, hidden_dim=8, dtype='float32',
                            tokens_per_frame=9, height=3, width=3,
                            native_short_grids=2, native_medium_grids=2,
                            retention_ratio=.05, shard_bytes=4096)


def _records(bank):
    batches = list(bank.iter_batches(max_tokens=7))
    features = torch.cat([r['features'] for r in batches])
    metadata = {k: torch.cat([r['metadata'][k] for r in batches]) for k in batches[0]['metadata']}
    order = torch.argsort(metadata['global_token_id'])
    return features[order], {k: v[order] for k, v in metadata.items()}


def test_byte_cap_provenance_snapshot_restore_and_continuation(tmp_path):
    bank = _make(tmp_path, 'work')
    values = sequence(30)
    for i, features in enumerate(values[:20]):
        bank.update(features, i, float(i))
        assert bank.stats()['actual_memory_bytes'] <= bank.hard_budget_bytes
    before = _records(bank)
    receipt = bank.snapshot(tmp_path / 'snapshot', timestamp=20)
    assert receipt['snapshot_serialized_bytes'] <= bank.hard_budget_bytes
    files_before = {p.name: (p.stat().st_size, p.stat().st_mtime_ns)
                    for p in (tmp_path / 'snapshot').iterdir()}
    reader = FluxMemSnapshotReader(tmp_path / 'snapshot')
    verified = reader.verify()
    assert verified['retained_visual_tokens'] == len(before[0]) == reader.count
    assert reader.max_timestamp == 19.
    read_only = _records(reader)
    torch.testing.assert_close(before[0], read_only[0], rtol=0, atol=0)
    for name in before[1]:
        torch.testing.assert_close(before[1][name], read_only[1][name], rtol=0, atol=0)
    reader.close()
    assert files_before == {p.name: (p.stat().st_size, p.stat().st_mtime_ns)
                            for p in (tmp_path / 'snapshot').iterdir()}
    restored = load_snapshot(tmp_path / 'snapshot', tmp_path / 'restored')
    after = _records(restored)
    torch.testing.assert_close(before[0], after[0], rtol=0, atol=0)
    for name in before[1]:
        torch.testing.assert_close(before[1][name], after[1][name], rtol=0, atol=0)
    assert bool((after[1]['start_timestamp'] == after[1]['end_timestamp']).all())
    assert bool((after[1]['start_frame'] == after[1]['end_frame']).all())
    assert bool((after[1]['timestamp'] < 20).all())
    for i in range(20, 30):
        bank.update(values[i], i, float(i))
        restored.update(values[i], i, float(i))
    final, resumed = _records(bank), _records(restored)
    torch.testing.assert_close(final[0], resumed[0], rtol=0, atol=0)
    for name in final[1]:
        torch.testing.assert_close(final[1][name], resumed[1][name], rtol=0, atol=0)
    with pytest.raises(ValueError, match='strictly increase'):
        restored.update(values[-1], 29, 29.)
    bank.close()
    restored.close()


def test_insufficient_cap_rejected_before_store_creation(tmp_path):
    with pytest.raises(ValueError, match='cannot fit'):
        _make(tmp_path, 'small', cap=100)
    assert not (tmp_path / 'small').exists()


def test_native_fixed_point_precedes_any_byte_cap_fifo_fallback(tmp_path):
    bank = _make(tmp_path, 'bounded', cap=80000)
    for frame_id, features in enumerate(sequence(60)):
        bank.update(features, frame_id, float(frame_id))
        assert bank.stats()['actual_memory_bytes'] <= 80000
        if bank.stats()['fallback_eviction_events']:
            assert bank.stats()['bank_sdc_fixedpoint']
            assert bank.stats()['native_bank_compactions'] == 1
            assert bank.stats()['last_fallback_cause'].startswith('native_sdc_fixed_point')
    assert bank.stats()['evicted_compressed_tokens'] > 0
    assert bank.stats()['fallback_eviction_bytes'] == bank.stats()['evicted_compressed_tokens'] * bank.store.record_bytes
    assert bank.stats()['native_recompression_events'] > 0
    assert bank.stats()['native_recompression_bytes_saved'] > 0
    # Every retained finalized frame group is now a released-SDC fixed point.
    rows = bank.store.read_slots(torch.arange(bank.store.count))
    for frame in torch.unique(rows['metadata']['frame_id']):
        keep = rows['metadata']['frame_id'] == frame
        features = rows['features'][keep]
        metadata = {key: value[keep] for key, value in rows['metadata'].items()}
        recompressed, out_meta, _, _ = bank._native_sdc_once(features, metadata)
        assert len(recompressed) == len(features)
        assert int(out_meta['merge_count'].sum()) == int(metadata['merge_count'].sum())
    assert bank.stats()['effective_retention_ratio'] == bank.stats()['actual_memory_bytes'] / bank.stats()['full_feature_bytes_seen']
    bank.close()
