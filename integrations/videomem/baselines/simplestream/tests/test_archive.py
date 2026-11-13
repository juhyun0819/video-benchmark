from fractions import Fraction
import hashlib
import json
import math

import numpy as np
import pytest
import torch

from baselines.simplestream import archive as module
from baselines.rekv_qwen25.rekv_qwen25 import inputs as frozen_inputs


def write_json(path,value):
    path.write_text(json.dumps(value))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive_fixture(root,durations=(Fraction(7,5),Fraction(11,5))):
    sources=[];base=Fraction(0)
    for day,duration in enumerate(durations,start=1):
        sources.append(dict(duration_fraction=str(duration),global_start_fraction=str(base),
            expected_1fps_frames=math.ceil(duration),avg_frame_rate='25/1',day_id=f'Day{day}',
            day_order=day,source_id=f'source-{day}',path=f'/videos/day{day}.mp4',source_fingerprint=str(day)*64))
        base+=duration
    total=sum(source['expected_1fps_frames'] for source in sources)
    header=dict(schema='feature_memory_archive_v1',archive_id='test-qwen',byte_order='little',dtype='bfloat16',
        frame_shape=list(frozen_inputs.FRAME_SHAPE),frames_per_shard=2,feature_compression='none',
        model_identity=dict(repo_id=frozen_inputs.REPO_ID,revision=frozen_inputs.REVISION),
        inventory=dict(fps=1.0,interval_convention='[start,end)',sources=sources,total_1fps_frames=total))
    write_json(root/'archive.json',header)
    for shard in range((total+1)//2):
        first,end=shard*2,min(total,shard*2+2);rows=[];blocks=[]
        for frame_id in range(first,end):
            envelope=frozen_inputs._frame_envelope(header['inventory'],frame_id)
            # Exact BF16 encodings of the small nonnegative integer frame IDs.
            bits=np.float32(frame_id).view(np.uint32)>>16
            raw=np.full(frozen_inputs.FRAME_SHAPE,bits,dtype='<u2').tobytes()
            rows.append(dict(frame_id=frame_id,timestamp=envelope['timestamp'],
                source_timestamp=envelope['source_timestamp'],day=envelope['day'],
                offset=(frame_id-first)*frozen_inputs.FRAME_BYTES,feature_bytes=frozen_inputs.FRAME_BYTES,
                shape=list(frozen_inputs.FRAME_SHAPE),dtype='bfloat16',
                shared_feature_sha256=hashlib.sha256(raw).hexdigest(),
                encoder=dict(grid_thw=[1,52,52],tensor_shape=list(frozen_inputs.FRAME_SHAPE)),
                metadata=dict(day_id=envelope['day_id'],source_id=envelope['source_id'],
                    file_seconds=envelope['source_timestamp'],source_fingerprint=envelope['source_sha256'])))
            blocks.append(raw)
        payload=root/f'features_{shard:08d}.bin';payload.write_bytes(b''.join(blocks))
        index=root/f'index_{shard:08d}.json'
        write_json(index,dict(schema='feature_memory_archive_index_v1',archive_id='test-qwen',
            shard_id=shard,dtype='bfloat16',frame_shape=list(frozen_inputs.FRAME_SHAPE),frames=rows))
        write_json(root/f'ready_{shard:08d}.json',dict(schema='feature_memory_archive_ready_v1',archive_id='test-qwen',
            shard_id=shard,feature_file=payload.name,index_file=index.name,feature_bytes=payload.stat().st_size,
            feature_sha256=sha(payload),index_sha256=sha(index),first_frame_id=first,frame_count=end-first,
            last_timestamp=rows[-1]['timestamp']))


def test_recent_crosses_fractional_day_boundary_and_reads_only_selected_range(tmp_path,monkeypatch):
    archive_fixture(tmp_path)
    source=module.QwenFeatureArchive(tmp_path)
    calls=[];read=module.read_feature_frames
    def requested(root,start,count):
        calls.append((start,count));return read(root,start,count)
    monkeypatch.setattr(module,'read_feature_frames',requested)
    frames=source.recent(1.4,2)
    assert calls==[(1,2)]
    assert [frame.frame_id for frame in frames]==[1,2]
    assert [frame.timestamp for frame in frames]==[1.0,1.4]
    assert [frame.metadata['day_id'] for frame in frames]==['Day1','Day2']
    assert [frame.metadata['source_timestamp'] for frame in frames]==[1.0,0.0]
    assert frames[1].metadata['source_video']=='/videos/day2.mp4'
    assert frames[1].metadata['model_revision']==frozen_inputs.REVISION
    assert frames[1].grid_thw.tolist()==[1,52,52]
    assert frames[1].features.dtype==torch.bfloat16 and frames[1].features.shape==(676,3584)
    assert frames[1].features[0,0].item()==2


def test_strict_stored_float_cutoff_handles_nondecimal_fraction(tmp_path):
    archive_fixture(tmp_path,(Fraction(1,3),Fraction(4,3)))
    source=module.QwenFeatureArchive(tmp_path);boundary=float(Fraction(1,3))
    assert [frame.frame_id for frame in source.recent(boundary,4)]==[0,1]
    assert [frame.frame_id for frame in source.recent(math.nextafter(boundary,-math.inf),4)]==[0]
    assert source.recent(-0.01,4)==[]


def test_future_missing_shard_is_excluded_but_missing_eligible_frame_fails(tmp_path):
    archive_fixture(tmp_path)
    (tmp_path/'ready_00000001.json').unlink()
    (tmp_path/'features_00000001.bin').write_bytes(b'incomplete-future')
    source=module.QwenFeatureArchive(tmp_path)
    assert [frame.frame_id for frame in source.recent(1.0,2)]==[0,1]
    with pytest.raises(FileNotFoundError):source.recent(1.4,2)


def test_header_and_returned_frames_are_owned_and_repeated_selection_is_exact(tmp_path):
    archive_fixture(tmp_path);source=module.QwenFeatureArchive(tmp_path)
    header=source.header;header['inventory']['sources'][0]['path']='changed'
    first=source.recent(0,1)[0];first.features.fill_(999);first.metadata['day_id']='changed'
    again=source.recent(0,1)[0]
    assert again.features[0,0].item()==0 and again.metadata['day_id']=='Day1'
    assert source.header['inventory']['sources'][0]['path']=='/videos/day1.mp4'


@pytest.mark.parametrize('mutation',['revision','header_source','index_source','payload'])
def test_model_source_identity_and_corruption_are_rejected(tmp_path,mutation):
    archive_fixture(tmp_path);source=module.QwenFeatureArchive(tmp_path)
    if mutation in ('revision','header_source'):
        path=tmp_path/'archive.json';header=json.loads(path.read_text())
        if mutation=='revision':header['model_identity']['revision']='different'
        else:header['inventory']['sources'][0]['source_fingerprint']='x'*64
        write_json(path,header)
    elif mutation=='index_source':
        path=tmp_path/'index_00000000.json';index=json.loads(path.read_text())
        index['frames'][0]['metadata']['source_id']='different'
        write_json(path,index)
        ready_path=tmp_path/'ready_00000000.json';ready=json.loads(ready_path.read_text())
        ready['index_sha256']=sha(path);write_json(ready_path,ready)
    else:
        with (tmp_path/'features_00000000.bin').open('r+b') as stream:stream.write(b'xx')
    with pytest.raises(ValueError):source.recent(0,1)


@pytest.mark.parametrize('cutoff,window',[(float('nan'),2),(float('inf'),2),(True,2),(1,0),(1,True)])
def test_invalid_cutoff_or_window_rejected(tmp_path,cutoff,window):
    archive_fixture(tmp_path)
    with pytest.raises(ValueError):module.QwenFeatureArchive(tmp_path).recent(cutoff,window)
