import hashlib
import json

import numpy as np
import pytest
import torch

from streamforest_videomem import native_archive as na


def fixture_archive(root):
    inventory={'sources':[dict(expected_1fps_frames=2,global_start_fraction='0',
                             duration_fraction='7/5',source_id='one',day_id='Day1'),
                          dict(expected_1fps_frames=1,global_start_fraction='7/5',
                             duration_fraction='1/2',source_id='two',day_id='Day2')],'total_frames':3}
    for source in inventory['sources']:
        path=root/source['source_id']
        path.write_bytes(b'verified source placeholder; completed resume must not decode')
        source['path']=str(path)
        source['verified_state']=na.source_state(path)
    config=dict(schema=na.SCHEMA,dtype='float16',shape=list(na.SHAPE),
                frames_per_shard=2,inventory=inventory,checkpoint=na.REPO,checkpoint_revision=na.REVISION)
    cid=hashlib.sha256(na.canonical(config)).hexdigest()
    na.write_json(root/'config.json',config)
    na.write_json(root/'manifest.json',dict(config_id=cid,requested_frames=3,status='complete',
                                         batch_size=1,task_frames=1,gpus=[0]))
    values=np.arange(3,dtype=np.float16)[:,None,None]*np.ones((3,*na.SHAPE),dtype=np.float16)
    raw=values.view(np.int16)
    for sid,(start,end) in enumerate([(0,2),(2,3)]):
        rows=[]
        for frame_id in range(start,end):
            rows.append(dict(global_frame_id=frame_id,**na.frame_envelope(inventory,frame_id),
                tensor_shape=list(na.SHAPE),dtype='float16',
                feature_sha256=hashlib.sha256(raw[frame_id].tobytes()).hexdigest()))
        na.commit_shard(root,cid,sid,rows,[raw[start:end]])
    return cid,values


def test_lossless_reload_order_and_fractional_day_boundary(tmp_path):
    cid,values=fixture_archive(tmp_path)
    frames=list(na.Archive(tmp_path,expected_config_id=cid))
    assert [f.global_frame_id for f in frames]==[0,1,2]
    assert [f.timestamp for f in frames]==[0,1,1.4]
    for frame,value in zip(frames,values):
        assert frame.visual_tokens.dtype==torch.float16
        assert torch.equal(frame.visual_tokens,torch.from_numpy(value))


@pytest.mark.parametrize('filename',['tokens.bin','index.json'])
def test_payload_or_index_corruption_rejected(tmp_path,filename):
    fixture_archive(tmp_path)
    p=tmp_path/'shards'/'shard_000000'/filename
    with p.open('r+b') as stream:
        original=stream.read(1)
        stream.seek(0)
        stream.write(bytes([original[0]^1]))
    with pytest.raises(ValueError,match='hash mismatch'):
        list(na.Archive(tmp_path))


def test_missing_receipt_rejects_uncommitted_payload(tmp_path):
    fixture_archive(tmp_path)
    (tmp_path/'receipts'/'000000.ready.json').unlink()
    with pytest.raises(FileNotFoundError):
        list(na.Archive(tmp_path))


def test_changed_checkpoint_config_rejected(tmp_path):
    fixture_archive(tmp_path)
    config=json.loads((tmp_path/'config.json').read_text())
    config['checkpoint_revision']='different'
    na.write_json(tmp_path/'config.json',config)
    with pytest.raises(ValueError,match='config identifier mismatch'):
        na.Archive(tmp_path)


def test_rehashed_index_with_wrong_time_still_rejected(tmp_path):
    fixture_archive(tmp_path)
    folder=tmp_path/'shards'/'shard_000000'
    index=json.loads((folder/'index.json').read_text())
    index['frames'][1]['timestamp_seconds']=99
    na.write_json(folder/'index.json',index)
    receipt_path=tmp_path/'receipts'/'000000.ready.json'
    receipt=json.loads(receipt_path.read_text())
    receipt['index_sha256']=na.digest(folder/'index.json')
    na.write_json(receipt_path,receipt)
    with pytest.raises(ValueError,match='timestamp/source mismatch'):
        list(na.Archive(tmp_path))


def test_incomplete_manifest_rejected_by_default(tmp_path):
    fixture_archive(tmp_path)
    manifest=json.loads((tmp_path/'manifest.json').read_text())
    manifest['status']='running'
    na.write_json(tmp_path/'manifest.json',manifest)
    with pytest.raises(ValueError,match='incomplete'):
        na.Archive(tmp_path)
    assert na.Archive(tmp_path,require_complete=False)


def test_decoder_seek_chunks_match_sequential_across_day_boundary(tmp_path):
    import av
    sources=[]
    for order in [1,2]:
        path=tmp_path/f'day{order}.mp4'
        with av.open(str(path),'w') as output:
            stream=output.add_stream('mpeg4',rate=25)
            stream.width=stream.height=64
            stream.pix_fmt='yuv420p'
            for i in range(80):
                image=np.full((64,64,3),(i+order*40)%255,dtype=np.uint8)
                frame=av.VideoFrame.from_ndarray(image,format='rgb24')
                for packet in stream.encode(frame):output.mux(packet)
            for packet in stream.encode():output.mux(packet)
        sources.append(dict(path=str(path),expected_1fps_frames=4,duration_fraction='16/5',
            global_start_fraction='0' if order==1 else '16/5',day_id=f'Day{order}',
            day_order=order,source_id=f'source{order}',source_fingerprint=na.digest(path),
            verified_state=na.source_state(path)))
    inventory=dict(sources=sources,total_frames=8)
    na.validate_inventory(inventory)
    def read(start,end):
        rows=[]
        for image,row in na.frames_for_range(inventory,start,end):
            image.close()
            rows.append({k:v for k,v in row.items() if k!='decode_seconds'})
        return rows
    sequential=read(0,8)
    chunked=read(0,3)+read(3,6)+read(6,8)
    assert chunked==sequential
    assert [r['timestamp_seconds'] for r in sequential]==[0,1,2,3,3.2,4.2,5.2,6.2]
    assert [r['global_frame_id'] for r in sequential]==list(range(8))


def test_completed_resume_validates_without_reencoding(tmp_path):
    fixture_archive(tmp_path)
    before={p.name:na.digest(p) for p in (tmp_path/'receipts').iterdir()}
    config=json.loads((tmp_path/'config.json').read_text())
    result=na.extract(config,tmp_path/'model-does-not-exist',tmp_path,
                      frames=3,gpus=[0],batch_size=1,task_frames=1)
    assert result['status']=='complete'
    assert before=={p.name:na.digest(p) for p in (tmp_path/'receipts').iterdir()}
