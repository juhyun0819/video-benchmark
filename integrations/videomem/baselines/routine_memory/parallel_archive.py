"""Bounded parallel I/O/SHA verification with strictly ordered delivery.

Completed archives only. Input read-ahead is transient workspace, never visible
to the memory policy before ordered delivery. Existing baseline readers are not
modified. The same committed shard/index/frame hashes are checked.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from baselines.feature_memory.archive import ArchiveError, _state


def _read_shard(directory, header, shard_id):
    started=time.perf_counter()
    receipt=json.loads((directory/f'ready_{shard_id:08d}.json').read_text())
    feature_path=directory/f'features_{shard_id:08d}.bin'
    index_path=directory/f'index_{shard_id:08d}.json'
    if (receipt.get('archive_id')!=header['archive_id'] or receipt.get('shard_id')!=shard_id
            or receipt.get('feature_file')!=feature_path.name or receipt.get('index_file')!=index_path.name):
        raise ArchiveError('Parallel reader receipt identity mismatch')
    raw_index=index_path.read_bytes()
    if hashlib.sha256(raw_index).hexdigest()!=receipt['index_sha256']:
        raise ArchiveError('Parallel reader index SHA256 mismatch')
    index=json.loads(raw_index)
    shape=header['frame_shape'];frame_bytes=math.prod(shape)*2
    frames=index['frames']
    if (index.get('archive_id')!=header['archive_id'] or index.get('shard_id')!=shard_id
            or index.get('frame_shape')!=shape or index.get('dtype')!='bfloat16'
            or len(frames)!=receipt['frame_count'] or not 0<len(frames)<=header['frames_per_shard']):
        raise ArchiveError('Parallel reader index structure mismatch')
    previous=-float('inf')
    for number,frame in enumerate(frames):
        if (frame.get('frame_id')!=receipt['first_frame_id']+number
                or frame.get('offset')!=number*frame_bytes or frame.get('feature_bytes')!=frame_bytes
                or frame.get('shape')!=shape or frame.get('dtype')!='bfloat16'
                or not math.isfinite(frame['timestamp']) or frame['timestamp']<=previous):
            raise ArchiveError('Parallel reader invalid frame geometry/order')
        previous=frame['timestamp']
    before=_state(feature_path)
    expected=len(frames)*frame_bytes
    if before[2]!=expected or receipt.get('feature_bytes')!=expected:
        raise ArchiveError('Parallel reader payload size mismatch')
    raw=torch.empty(expected,dtype=torch.uint8)
    view=memoryview(raw.numpy()).cast('B')
    digest=hashlib.sha256()
    with feature_path.open('rb',buffering=0) as handle:
        offset=0
        while offset<len(view):
            chunk=view[offset:offset+16*1024**2]
            count=handle.readinto(chunk)
            if not count:
                raise ArchiveError('Parallel reader truncated payload')
            digest.update(chunk[:count]);offset+=count
        if handle.read(1):
            raise ArchiveError('Parallel reader trailing payload')
    if _state(feature_path)!=before or digest.hexdigest()!=receipt['feature_sha256']:
        raise ArchiveError('Parallel reader feature SHA256/identity mismatch')
    for number,frame in enumerate(frames):
        if hashlib.sha256(view[number*frame_bytes:(number+1)*frame_bytes]).hexdigest()!=frame['shared_feature_sha256']:
            raise ArchiveError('Parallel reader per-frame SHA256 mismatch')
    return {'features':raw.view(torch.bfloat16).reshape(len(frames),*shape),
            'frames':frames,'shard_id':shard_id,'reader_worker_seconds':time.perf_counter()-started,
            'file_bytes_read':expected,'payload_bytes_yielded':expected}


def iter_parallel_archive(archive_dir,readers=4,buffer_shards=6,max_frames=None):
    if type(readers) is not int or readers<1 or type(buffer_shards) is not int or buffer_shards<readers:
        raise ValueError('Positive readers and buffer_shards>=readers are required')
    if max_frames is not None and (type(max_frames) is not int or max_frames<0):
        raise ValueError('max_frames must be a nonnegative integer')
    directory=Path(archive_dir)
    header_raw=(directory/'archive.json').read_bytes()
    header=json.loads(header_raw)
    complete=json.loads((directory/'complete.json').read_text())
    if (header.get('schema')!='feature_memory_archive_v1' or header.get('dtype')!='bfloat16'
            or header.get('byte_order')!='little' or complete.get('archive_id')!=header['archive_id']
            or complete.get('archive_header_sha256')!=hashlib.sha256(header_raw).hexdigest()):
        raise ArchiveError('Parallel reader requires a verified completed archive')
    total=complete['total_frames'];shards=complete['shards']
    target=total if max_frames is None else min(total,max_frames)
    if not target:
        return
    count,next_shard,previous=0,0,-float('inf')
    pending=deque()
    executor=ThreadPoolExecutor(max_workers=readers,thread_name_prefix='routine-archive')
    try:
        while next_shard<min(shards,buffer_shards):
            pending.append(executor.submit(_read_shard,directory,header,next_shard));next_shard+=1
        while pending and count<target:
            started=time.perf_counter()
            batch=pending.popleft().result()
            frames=batch['frames']
            if frames[0]['frame_id']!=count or frames[0]['timestamp']<=previous:
                raise ArchiveError('Parallel reader cross-shard stream order mismatch')
            take=min(len(frames),target-count)
            batch['features']=batch['features'][:take]
            batch['frames']=frames[:take]
            batch['payload_bytes_yielded']=take*math.prod(header['frame_shape'])*2
            batch['read_seconds']=time.perf_counter()-started
            batch['read_timing_basis']='consumer_wait_wall; worker I/O overlaps memory processing'
            yield batch
            count+=take;previous=frames[-1]['timestamp']
            if next_shard<shards and count<target:
                pending.append(executor.submit(_read_shard,directory,header,next_shard));next_shard+=1
        if count!=target:
            raise ArchiveError('Parallel reader archive ended before declared target')
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True,cancel_futures=True)
