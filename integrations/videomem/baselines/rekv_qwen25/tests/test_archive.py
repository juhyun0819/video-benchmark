import json
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest
import torch

from rekv_qwen25.archive import KVArchiveWriter, committed_blocks, load_block


def tensors():
    # BF16 and scalar integer resume state exercise actual storage bit patterns.
    return {'layer_00.key': torch.arange(24).reshape(1, 2, 3, 4).to(torch.bfloat16),
            'layer_00.value': -torch.arange(24).reshape(1, 2, 3, 4).to(torch.bfloat16),
            'resume_state.position': torch.tensor(3, dtype=torch.int64)}


def writer(root):
    return KVArchiveWriter(root, {'checkpoint': 'Qwen/Qwen2.5-VL-7B-Instruct',
                                  'revision': 'fixed-test-revision', 'dtype': 'bfloat16'})


def meta(frame_ids=(0, 1), timestamps=(0.0, 1.0)):
    return dict(frame_ids=list(frame_ids), timestamps=list(timestamps),
                day='Day1', source_video='/source/day1.mp4', sampling_rule='1fps')


def test_exact_bf16_resume_state_reload_and_reuse(tmp_path):
    archive = writer(tmp_path)
    original = tensors()
    receipt = archive.commit(0, original, meta())
    assert not receipt['reused']
    loaded, metadata = load_block(tmp_path, 0, archive.config_id)
    for key in original:
        assert loaded[key].dtype == original[key].dtype
        assert loaded[key].shape == original[key].shape
        assert torch.equal(loaded[key], original[key])
    assert metadata['availability_timestamp'] == 1
    again = writer(tmp_path).commit(0, original, meta())
    assert again['reused']
    assert again['payload_sha256'] == receipt['payload_sha256']
    assert len(committed_blocks(tmp_path)) == 1


def test_causal_filter_and_fractional_day_boundary(tmp_path):
    archive = writer(tmp_path)
    archive.commit(0, tensors(), meta())
    archive.commit(1, tensors(), meta((2, 3), (1.4, 2.4)))
    assert [d['chunk_id'] for d in committed_blocks(tmp_path, 1.0)] == [0]
    assert [d['chunk_id'] for d in committed_blocks(tmp_path, 2.4)] == [0, 1]
    with pytest.raises(ValueError, match='future'):
        load_block(tmp_path, 1, cutoff_timestamp=2.0)
    assert committed_blocks(tmp_path, -1.0) == []


@pytest.mark.parametrize('folder,suffix', [('shards', '.safetensors'), ('indices', '.json')])
def test_corruption_rejected(tmp_path, folder, suffix):
    writer(tmp_path).commit(0, tensors(), meta())
    path = tmp_path / folder / ('chunk_00000000' + suffix)
    with path.open('r+b') as stream:
        value = stream.read(1)
        stream.seek(0)
        stream.write(bytes([value[0] ^ 1]))
    with pytest.raises(ValueError, match='checksum mismatch'):
        load_block(tmp_path, 0)


def test_uncommitted_partial_and_prepared_index_not_exposed(tmp_path):
    archive = writer(tmp_path)
    (tmp_path / 'shards' / 'orphan.partial').write_bytes(b'incomplete')
    archive.commit(0, tensors(), meta())
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    manifest['chunks']['0']['state'] = 'prepared'
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    (tmp_path / 'receipts' / 'chunk_00000000.ready.json').unlink()
    assert committed_blocks(tmp_path) == []
    with pytest.raises(FileNotFoundError):
        load_block(tmp_path, 0)
    receipt = archive.commit(0, tensors(), meta())
    assert not receipt['reused']
    assert len(committed_blocks(tmp_path)) == 1


def test_ready_missing_receipt_is_corruption(tmp_path):
    archive = writer(tmp_path)
    archive.commit(0, tensors(), meta())
    (tmp_path / 'receipts' / 'chunk_00000000.ready.json').unlink()
    with pytest.raises(ValueError, match='missing its ready receipt'):
        committed_blocks(tmp_path)
    with pytest.raises(ValueError, match='missing its ready receipt'):
        archive.commit(0, tensors(), meta())


def test_model_config_and_recommit_changes_rejected(tmp_path):
    archive = writer(tmp_path)
    archive.commit(0, tensors(), meta())
    with pytest.raises(ValueError, match='different model/config'):
        KVArchiveWriter(tmp_path, {'checkpoint': 'wrong'})
    with pytest.raises(ValueError, match='identifier'):
        load_block(tmp_path, 0, expected_config_id='wrong')
    changed = tensors()
    changed['layer_00.key'][0, 0, 0, 0] = 42
    with pytest.raises(ValueError, match='already committed'):
        archive.commit(0, changed, meta())


def test_out_of_order_commit_declares_gap_until_filled(tmp_path):
    archive = writer(tmp_path)
    archive.commit(1, tensors(), meta((2, 3), (2.0, 3.0)))
    assert [d['chunk_id'] for d in committed_blocks(tmp_path)] == [1]
    with pytest.raises(ValueError, match='missing chunk'):
        committed_blocks(tmp_path, require_contiguous=True)
    archive.commit(0, tensors(), meta())
    assert [d['chunk_id'] for d in committed_blocks(tmp_path, require_contiguous=True)] == [0, 1]
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert manifest['continuity']['contiguous_prefix_frames'] == 4
    assert not manifest['continuity']['has_chunk_gaps']


def test_duplicate_missing_frames_and_forged_availability_rejected(tmp_path):
    archive = writer(tmp_path)
    for metadata in [meta((0, 0)), meta((0, 2)), dict(meta(), availability_timestamp=0.0)]:
        with pytest.raises(ValueError):
            archive.commit(0, tensors(), metadata)
    archive.commit(0, tensors(), meta())
    with pytest.raises(ValueError, match='overlap'):
        archive.commit(1, tensors(), meta((1, 2), (2.0, 3.0)))
    with pytest.raises(ValueError, match='missing frame'):
        archive.commit(1, tensors(), meta((3, 4), (3.0, 4.0)))


def test_ready_commit_before_manifest_update_can_be_recovered(tmp_path):
    archive = writer(tmp_path)
    archive.commit(0, tensors(), meta())
    path = tmp_path / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['chunks']['0']['state'] = 'prepared'
    path.write_text(json.dumps(manifest))
    assert len(committed_blocks(tmp_path)) == 1
    assert archive.commit(0, tensors(), meta())['reused']
    assert json.loads(path.read_text())['chunks']['0']['state'] == 'ready'


def test_interrupted_empty_bootstrap_recovers_but_lost_committed_manifest_fails(tmp_path):
    archive = writer(tmp_path)
    (tmp_path / 'manifest.json').unlink()
    recovered = writer(tmp_path)
    assert recovered.config_id == archive.config_id
    recovered.commit(0, tensors(), meta())
    (tmp_path / 'manifest.json').unlink()
    with pytest.raises(ValueError, match='Manifest is missing'):
        writer(tmp_path)


def test_distinct_chunk_payload_writes_overlap_and_recheck_manifest(tmp_path, monkeypatch):
    import rekv_qwen25.archive as module
    archive = writer(tmp_path)
    original = module.save_file
    barrier = threading.Barrier(2)

    def concurrent_save(*args, **kwargs):
        # A global lock held during save_file would make this barrier fail.
        barrier.wait(timeout=5)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, 'save_file', concurrent_save)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(archive.commit, i, tensors(),
                           meta((i*2, i*2+1), (float(i*2), float(i*2+1))))
                for i in range(2)]
        receipts = [job.result(timeout=10) for job in jobs]
    assert all(not receipt['reused'] for receipt in receipts)
    assert len(committed_blocks(tmp_path, require_contiguous=True)) == 2
    assert json.loads((tmp_path / 'manifest.json').read_text())['continuity']['committed_frame_count'] == 4


def test_same_chunk_concurrent_commit_writes_once(tmp_path, monkeypatch):
    import rekv_qwen25.archive as module
    archives = [writer(tmp_path), writer(tmp_path)]
    original = module.save_file
    start = threading.Barrier(2)
    calls = []

    def counted_save(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    def commit(archive):
        start.wait(timeout=5)
        return archive.commit(0, tensors(), meta())

    monkeypatch.setattr(module, 'save_file', counted_save)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(commit, archive) for archive in archives]
        receipts = [job.result(timeout=10) for job in jobs]
    assert sorted(receipt['reused'] for receipt in receipts) == [False, True]
    assert len(calls) == 1
    assert receipts[0]['payload_sha256'] == receipts[1]['payload_sha256']
    assert len(committed_blocks(tmp_path)) == 1


def test_payload_reader_does_not_block_another_chunk_commit(tmp_path, monkeypatch):
    import rekv_qwen25.archive as module
    archive = writer(tmp_path)
    archive.commit(0, tensors(), meta())
    original_hash = module.file_sha256
    hashing = threading.Event()
    other_committed = threading.Event()

    def delayed_hash(path):
        if str(path).endswith('chunk_00000000.safetensors'):
            hashing.set()
            assert other_committed.wait(5), 'Reader held global lock during payload hashing'
        return original_hash(path)

    def commit_other():
        assert hashing.wait(5)
        result = archive.commit(1, tensors(), meta((2,3),(2.0,3.0)))
        other_committed.set()
        return result

    monkeypatch.setattr(module, 'file_sha256', delayed_hash)
    with ThreadPoolExecutor(max_workers=2) as pool:
        reader = pool.submit(load_block, tmp_path, 0)
        writer_job = pool.submit(commit_other)
        assert not writer_job.result(timeout=10)['reused']
        assert torch.equal(reader.result(timeout=10)[0]['layer_00.key'], tensors()['layer_00.key'])


def test_concurrent_overlapping_chunks_only_one_can_become_ready(tmp_path, monkeypatch):
    import rekv_qwen25.archive as module
    archive = writer(tmp_path)
    original = module.save_file
    barrier = threading.Barrier(2)

    def concurrent_save(*args, **kwargs):
        barrier.wait(timeout=5)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, 'save_file', concurrent_save)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(archive.commit, i, tensors(), meta()) for i in range(2)]
        outcomes = []
        for job in jobs:
            try:
                outcomes.append(job.result(timeout=10))
            except ValueError as error:
                outcomes.append(str(error))
    assert sum(isinstance(outcome, dict) for outcome in outcomes) == 1
    assert len(committed_blocks(tmp_path)) == 1
