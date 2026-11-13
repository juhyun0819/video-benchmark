import hashlib
import json
import time

import pytest
import torch

from baselines.feature_memory.archive import ArchiveError,ArchiveWriter,iter_archive
from baselines.routine_memory import RoutineAwareMemory
from baselines.routine_memory import parallel_archive
from baselines.routine_memory.tests.test_integration_contract import fingerprint


def make_archive(tmp_path,frames=7):
    path=tmp_path/'archive'
    writer=ArchiveWriter(path,{'frame_shape':[4,8]},frames_per_shard=2)
    for i in range(frames):
        writer.append(torch.full((4,8),float(i+1),dtype=torch.bfloat16),
                      {'frame_id':i,'timestamp':float(i)})
    writer.complete(frames)
    return path


def test_parallel_delivery_matches_serial_memory_despite_out_of_order_reads(tmp_path,monkeypatch):
    path=make_archive(tmp_path)
    original=parallel_archive._read_shard
    calls=[]
    def delayed(directory,header,shard):
        calls.append(shard)
        if shard==0:time.sleep(.04)
        return original(directory,header,shard)
    monkeypatch.setattr(parallel_archive,'_read_shard',delayed)
    stream=parallel_archive.iter_parallel_archive(path,readers=2,buffer_shards=3)
    first=next(stream)
    assert max(calls)<=2
    batches=[first]+list(stream)
    assert [r['frame_id'] for batch in batches for r in batch['frames']]==list(range(7))
    serial,parallel=RoutineAwareMemory(),RoutineAwareMemory()
    for memory,source in [(serial,iter_archive(path,wait=False)),(parallel,batches)]:
        for batch in source:
            for feature,row in zip(batch['features'],batch['frames']):
                memory.update(feature,row['frame_id'],row['timestamp'])
    assert fingerprint(serial.state_dict())==fingerprint(parallel.state_dict())


def test_parallel_prefix_trims_future_frames(tmp_path):
    path=make_archive(tmp_path)
    batches=list(parallel_archive.iter_parallel_archive(path,2,3,max_frames=3))
    assert [r['frame_id'] for b in batches for r in b['frames']]==[0,1,2]
    assert batches[-1]['features'].shape[0]==1


@pytest.mark.parametrize('name',['features_00000000.bin','index_00000000.json'])
def test_parallel_corruption_refused(tmp_path,name):
    path=make_archive(tmp_path)
    target=path/name
    data=bytearray(target.read_bytes());data[len(data)//2]^=1;target.write_bytes(data)
    with pytest.raises(ArchiveError,match='SHA256'):
        list(parallel_archive.iter_parallel_archive(path,2,2))


def test_parallel_cross_shard_future_order_rejected_even_with_valid_index_hash(tmp_path):
    path=make_archive(tmp_path)
    target=path/'index_00000001.json'
    index=json.loads(target.read_text())
    for row in index['frames']:row['timestamp']-=100
    raw=json.dumps(index).encode();target.write_bytes(raw)
    target=path/'ready_00000001.json'
    receipt=json.loads(target.read_text());receipt['index_sha256']=hashlib.sha256(raw).hexdigest()
    target.write_text(json.dumps(receipt))
    with pytest.raises(ArchiveError,match='cross-shard'):
        list(parallel_archive.iter_parallel_archive(path,2,2))
