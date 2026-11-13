"""Byte caps, overwrite/compaction, exact persistence and streaming reads."""
import json
import errno
import hashlib
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from baselines.feature_memory.byte_memory import ByteBudgetMemory, DiskTokenStore, SnapshotReader
from baselines.feature_memory.memory import TokenMemory
from baselines.feature_memory import byte_memory


def frame(frame_id, tokens=7, hidden=3, dtype=torch.bfloat16):
    return (torch.arange(tokens).view(-1,1) + frame_id*tokens + torch.arange(hidden).view(1,-1)/16).to(dtype)


def flattened(memory):
    batches=list(memory.iter_batches(max_tokens=4))
    if not batches:
        return None,{}
    features=torch.cat([b['features'] for b in batches])
    metadata={k:torch.cat([b['metadata'][k] for b in batches]) for k in batches[0]['metadata']}
    order=torch.argsort(metadata['global_token_id'])
    return features[order],{k:v[order] for k,v in metadata.items()}


def assert_real_bytes(memory):
    stats=memory.stats()
    actual_files=sum(p.stat().st_size for p in memory.store.directory.iterdir() if p.is_file())
    assert stats['persistent_disk_bytes']==actual_files
    assert stats['actual_memory_bytes']>=actual_files
    assert stats['actual_memory_bytes']<=stats['hard_budget_bytes']
    assert not any(p.suffix=='.npy' for p in memory.store.directory.iterdir())


@pytest.mark.parametrize('method',['fifo','uniform','reservoir'])
@pytest.mark.parametrize('capacity',[1,5,17])
def test_disk_selection_matches_in_memory_reference_and_hard_cap(tmp_path,method,capacity):
    budget=4096+capacity*(3*2+32)
    memory=ByteBudgetMemory(method,budget,.1,hidden_dim=3,tokens_per_frame=7,seed=123,
                            directory=tmp_path/'live',shard_bytes=3*38)
    reference=TokenMemory(method,capacity,seed=123)
    try:
        for i in range(18):
            value=frame(i)
            memory.update(value,i,float(i))
            reference.update(value,i,float(i))
            actual,meta=flattened(memory)
            expected=reference.snapshot()['tensors']
            assert torch.equal(actual,expected['features'])
            for ours,theirs in [('global_token_id','global_token_ids'),('frame_id','frame_ids'),
                                ('timestamp','frame_timestamps'),('token_index','token_indices')]:
                assert torch.equal(meta[ours],expected[theirs])
            assert memory.stats()['total_frames_seen']==i+1
            assert memory.stats()['total_visual_tokens_seen']==(i+1)*7
            assert_real_bytes(memory)
        memory.flush()
        assert_real_bytes(memory)
        assert len(list(memory.store.directory.glob('records_*.bin')))<=6
    finally:
        memory.close()


@pytest.mark.parametrize('method',['fifo','uniform','reservoir'])
def test_snapshot_read_only_roundtrip_and_resume_reproducibility(tmp_path,method):
    capacity=17
    kwargs=dict(hard_budget_bytes=4096+capacity*38,retention_ratio=.25,hidden_dim=3,
                tokens_per_frame=7,seed=456,shard_bytes=4*38)
    live=ByteBudgetMemory(method,directory=tmp_path/'live',**kwargs)
    restored=None
    try:
        for i in range(8): live.update(frame(i),i,float(i))
        before=flattened(live)
        snapshot=live.snapshot(tmp_path/'snapshot',8.0)
        actual_snapshot=sum(p.stat().st_size for p in (tmp_path/'snapshot').iterdir())
        assert snapshot['snapshot_serialized_bytes']==actual_snapshot
        original_listing={p.name:p.stat().st_size for p in (tmp_path/'snapshot').iterdir()}
        with SnapshotReader(tmp_path/'snapshot') as reader:
            receipt=reader.verify()
            assert receipt['passed']
            assert receipt['retained_visual_tokens']==live.store.count
            actual=flattened(reader)
            assert torch.equal(actual[0],before[0])
            for key in before[1]: assert torch.equal(actual[1][key],before[1][key])
            with pytest.raises(PermissionError): reader.truncate(0)
        assert {p.name:p.stat().st_size for p in (tmp_path/'snapshot').iterdir()}==original_listing
        restored=ByteBudgetMemory.load(tmp_path/'snapshot',tmp_path/'restored')
        for i in range(8,15):
            live.update(frame(i),i,float(i))
            restored.update(frame(i),i,float(i))
            left,right=flattened(live),flattened(restored)
            assert torch.equal(left[0],right[0])
            for key in left[1]: assert torch.equal(left[1][key],right[1][key])
            assert restored.store.control==live.store.control
            assert_real_bytes(restored)
    finally:
        live.close()
        if restored is not None: restored.close()


def test_generalized_provenance_roundtrip_canonical_order(tmp_path):
    extras={'z_merge_count':'int64','a_start_timestamp':'float64','m_end_frame':'int64'}
    store=DiskTokenStore(tmp_path/'live',20000,hidden_dim=3,extra_fields=extras,shard_bytes=200)
    metadata={'global_token_id':np.arange(7),'frame_id':np.arange(7),
              'timestamp':np.arange(7,dtype=np.float64),'token_index':np.arange(7),
              'z_merge_count':np.arange(7)+20,'a_start_timestamp':np.arange(7)/2,
              'm_end_frame':np.arange(7)+50}
    try:
        store.append(frame(0),metadata)
        store.snapshot(tmp_path/'snapshot',{'method':'test_flux_provenance'})
        with SnapshotReader(tmp_path/'snapshot') as reader:
            assert reader.verify()['passed']
            observed=reader.read_slots(np.arange(7))
            assert torch.equal(observed['features'],frame(0))
            for key,value in metadata.items():
                assert torch.equal(observed['metadata'][key],torch.from_numpy(value))
        restored=DiskTokenStore.load(tmp_path/'snapshot',tmp_path/'restore')
        try:
            observed=restored.read_slots(np.arange(7))
            for key,value in metadata.items():
                assert torch.equal(observed['metadata'][key],torch.from_numpy(value))
        finally:
            restored.close()
    finally:
        store.close()


def test_checksum_detects_mutated_retained_feature(tmp_path):
    memory=ByteBudgetMemory('fifo',6000,.1,hidden_dim=3,tokens_per_frame=7,directory=tmp_path/'live')
    try:
        memory.update(frame(0),0,0.0)
        memory.snapshot(tmp_path/'snapshot',1.0)
        shard=next((tmp_path/'snapshot').glob('records_*.bin'))
        with shard.open('r+b') as handle:
            original=handle.read(1)
            handle.seek(0)
            handle.write(bytes([original[0]^1]))
        with SnapshotReader(tmp_path/'snapshot') as reader:
            with pytest.raises(ValueError,match='checksum'): reader.verify()
        with pytest.raises(ValueError,match='checksum'):
            ByteBudgetMemory.load(tmp_path/'snapshot',tmp_path/'bad_restore')
    finally:
        memory.close()


def test_invalid_budget_and_oversized_control_fail_without_hidden_storage(tmp_path):
    with pytest.raises(ValueError,match='cannot fit'):
        DiskTokenStore(tmp_path/'invalid',4096,hidden_dim=3584)
    assert not (tmp_path/'invalid').exists()
    store=DiskTokenStore(tmp_path/'live',10000,hidden_dim=3)
    before=store.stats()
    try:
        with pytest.raises(ValueError,match='reserve'):
            store.set_control({'unbounded_provenance':'x'*10000})
        assert store.stats()==before
        with pytest.raises(ValueError,match='existing retained slots'):
            store.write_slots([0,0],torch.ones(2,3,dtype=torch.bfloat16),{})
    finally:
        store.close()


def test_stats_and_updates_do_not_scan_directory_or_allocate_capacity(tmp_path,monkeypatch):
    # A huge logical cap must not cause capacity-sized RAM/file allocation.
    memory=ByteBudgetMemory('reservoir',500_000_000_000,.5,hidden_dim=3,tokens_per_frame=7,
                            directory=tmp_path/'live')
    try:
        def forbid_scan(*a,**kw):
            raise AssertionError('per-frame stats must not scan shard directories')
        monkeypatch.setattr(Path,'iterdir',forbid_scan)
        for i in range(3):
            memory.update(frame(i),i,float(i))
            stats=memory.stats()
            assert stats['persistent_disk_bytes']<5000
            assert stats['retained_visual_tokens']==7*(i+1)
        assert not any(isinstance(v,torch.Tensor) for v in vars(memory.store).values())
    finally:
        memory.close()


@pytest.mark.parametrize('factor',[2,3,2048])
def test_uniform_compaction_reads_contiguous_bounded_blocks(tmp_path,monkeypatch,factor):
    count,hidden=6000,3
    record_bytes=hidden*2+32
    store=DiskTokenStore(tmp_path/'live',4096+count*record_bytes,
                         hidden_dim=hidden,shard_bytes=1700*record_bytes)
    values=torch.arange(count*hidden,dtype=torch.float32).reshape(count,hidden).to(torch.bfloat16)
    metadata={'global_token_id':np.arange(count),'frame_id':np.arange(count),
              'timestamp':np.arange(count,dtype=np.float64),'token_index':np.zeros(count,dtype=np.int64)}
    reads=[]
    original=os.pread
    try:
        store.append(values,metadata)
        with monkeypatch.context() as patch:
            def counted_read(fd,size,offset):
                reads.append(size)
                return original(fd,size,offset)
            patch.setattr(os,'pread',counted_read)
            store.compact_stride(factor)
        expected=np.arange(0,count,factor)
        assert store.count==len(expected)
        # Block/shard boundaries add a few reads; calls must not scale with
        # retained token count (the previous path issued ~3000 reads here).
        assert len(reads)<=math.ceil(count/1024)+math.ceil(count/1700)
        assert max(reads)<=1024*record_bytes
        batches=list(store.iter_batches())
        actual=torch.cat([b['features'] for b in batches])
        actual_ids=torch.cat([b['metadata']['global_token_id'] for b in batches])
        assert torch.equal(actual,values[torch.from_numpy(expected)])
        assert torch.equal(actual_ids,torch.from_numpy(expected))
        disk_records=sum(p.stat().st_size for p in store.directory.glob('records_*.bin'))
        assert disk_records==len(expected)*record_bytes
        assert store.stats()['actual_memory_bytes']<=store.hard_budget_bytes
    finally:
        store.close()


def test_snapshot_kernel_copy_handles_partial_calls_and_hashes_source(tmp_path, monkeypatch):
    data = bytes(range(128))
    source, destination = tmp_path/'source', tmp_path/'destination'
    source.write_bytes(data)
    calls = []
    def partial(src, dst, count):
        calls.append(count)
        chunk = os.read(src, min(count, 7))
        return os.write(dst, chunk)
    monkeypatch.setattr(byte_memory, '_copy_file_range', partial)
    size, digest = DiskTokenStore._copy_hashed(source, destination)
    assert size == len(data) and digest == hashlib.sha256(data).hexdigest()
    assert destination.read_bytes() == data
    assert len(calls) == math.ceil(len(data)/7)


@pytest.mark.parametrize('first_partial', [False, True])
def test_snapshot_unsupported_copy_restarts_streaming_without_duplicate_prefix(tmp_path, monkeypatch, first_partial):
    data = bytes(range(128))
    source, destination = tmp_path/'source', tmp_path/'destination'
    source.write_bytes(data)
    calls = []
    def unsupported(src, dst, count):
        calls.append(count)
        if first_partial and len(calls) == 1:
            return os.write(dst, os.read(src, 7))
        raise OSError(errno.EXDEV, 'cross-device server copy')
    monkeypatch.setattr(byte_memory, '_copy_file_range', unsupported)
    size, digest = DiskTokenStore._copy_hashed(source, destination)
    assert size == len(data) and digest == hashlib.sha256(data).hexdigest()
    assert destination.read_bytes() == data
    assert len(calls) == 1 + int(first_partial)


def test_snapshot_copy_rejects_source_truncation_and_cleans_partial_file(tmp_path, monkeypatch):
    source, destination = tmp_path/'source', tmp_path/'destination'
    source.write_bytes(bytes(range(128)))
    def truncating(src, dst, count):
        copied = os.write(dst, os.read(src, count))
        source.write_bytes(b'x')
        return copied
    monkeypatch.setattr(byte_memory, '_copy_file_range', truncating)
    with pytest.raises(IOError, match='source changed'):
        DiskTokenStore._copy_hashed(source, destination)
    assert not destination.exists()


@pytest.mark.parametrize('failure', ['zero', 'enospc'])
def test_snapshot_copy_hard_errors_are_not_silently_retried(tmp_path, monkeypatch, failure):
    source, destination = tmp_path/'source', tmp_path/'destination'
    source.write_bytes(b'nonempty')
    calls = []
    def failed(*args):
        calls.append(True)
        if failure == 'zero':
            return 0
        raise OSError(errno.ENOSPC, 'disk full')
    monkeypatch.setattr(byte_memory, '_copy_file_range', failed)
    with pytest.raises(OSError):
        DiskTokenStore._copy_hashed(source, destination)
    assert calls == [True] and not destination.exists()


def test_snapshot_copy_preserves_existing_destination_and_supports_empty_source(tmp_path):
    source, destination = tmp_path/'source', tmp_path/'destination'
    source.write_bytes(b'')
    destination.write_bytes(b'existing')
    with pytest.raises(FileExistsError):
        DiskTokenStore._copy_hashed(source, destination)
    assert destination.read_bytes() == b'existing'
    size, digest = DiskTokenStore._copy_hashed(source, tmp_path/'empty')
    assert size == 0 and digest == hashlib.sha256(b'').hexdigest()


def test_snapshot_copy_closes_nfs_destination_before_checking_cached_size(tmp_path, monkeypatch):
    source, destination = tmp_path/'source', tmp_path/'destination'
    source.write_bytes(b'features')
    original_copy, original_stat, original_sync = byte_memory._copy_file_range, os.fstat, os.fsync
    original_path_stat = Path.stat
    state = {'destination_fd': None, 'synced': False, 'checked_closed': False}
    def copying(src, dst, count):
        state['destination_fd'] = dst
        return original_copy(src, dst, count)
    def cached_stat(fd):
        if fd == state['destination_fd']:
            return SimpleNamespace(st_size=0)
        return original_stat(fd)
    def path_stat(path, *args, **kwargs):
        if path == destination:
            with pytest.raises(OSError) as closed:
                original_stat(state['destination_fd'])
            assert closed.value.errno == errno.EBADF
            state['checked_closed'] = True
        return original_path_stat(path, *args, **kwargs)
    def syncing(fd):
        if fd == state['destination_fd']:
            state['synced'] = True
        return original_sync(fd)
    monkeypatch.setattr(byte_memory, '_copy_file_range', copying)
    monkeypatch.setattr(os, 'fstat', cached_stat)
    monkeypatch.setattr(os, 'fsync', syncing)
    monkeypatch.setattr(Path, 'stat', path_stat)
    size, digest = DiskTokenStore._copy_hashed(source, destination)
    assert state['synced'] and state['checked_closed'] and size == 8
    assert digest == hashlib.sha256(b'features').hexdigest()
