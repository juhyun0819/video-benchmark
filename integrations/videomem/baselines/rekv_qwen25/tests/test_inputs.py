import hashlib
from fractions import Fraction
import json
import os

import pytest
import torch

from rekv_qwen25 import inputs


def write_json(path, value):
    path.write_text(json.dumps(value))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive_fixture(root):
    sources = [dict(duration_fraction='7/5', global_start_fraction='0', expected_1fps_frames=2,
                    avg_frame_rate='25/1', day_id='Day1', day_order=1, source_id='one',
                    path='/videos/day1.mp4', source_fingerprint='1'*64),
               dict(duration_fraction='11/5', global_start_fraction='7/5', expected_1fps_frames=3,
                    avg_frame_rate='25/1', day_id='Day2', day_order=2, source_id='two',
                    path='/videos/day2.mp4', source_fingerprint='2'*64)]
    header = dict(schema='feature_memory_archive_v1', archive_id='test-archive',
        byte_order='little', dtype='bfloat16', frame_shape=list(inputs.FRAME_SHAPE),
        frames_per_shard=2, feature_compression='none',
        model_identity=dict(repo_id=inputs.REPO_ID, revision=inputs.REVISION),
        inventory=dict(fps=1.0, interval_convention='[start,end)', sources=sources, total_1fps_frames=5))
    write_json(root/'archive.json', header)
    for shard_id in range(3):
        first, last = shard_id*2, min(shard_id*2+2, 5)
        rows, blocks = [], []
        for frame_id in range(first, last):
            values = torch.full(inputs.FRAME_SHAPE, frame_id, dtype=torch.bfloat16)
            raw = values.view(torch.int16).numpy().tobytes()
            timestamp = frame_id if frame_id < 2 else float(Fraction(7,5)+frame_id-2)
            rows.append(dict(frame_id=frame_id, timestamp=timestamp,
                offset=(frame_id-first)*inputs.FRAME_BYTES, bytes=inputs.FRAME_BYTES,
                feature_bytes=inputs.FRAME_BYTES, shape=list(inputs.FRAME_SHAPE), dtype='bfloat16',
                shared_feature_sha256=hashlib.sha256(raw).hexdigest(),
                encoder=dict(grid_thw=[1,52,52], tensor_shape=list(inputs.FRAME_SHAPE))))
            blocks.append(raw)
        payload = root/f'features_{shard_id:08d}.bin'
        payload.write_bytes(b''.join(blocks))
        index = root/f'index_{shard_id:08d}.json'
        write_json(index, dict(schema='feature_memory_archive_index_v1', archive_id='test-archive',
            shard_id=shard_id, dtype='bfloat16', frame_shape=list(inputs.FRAME_SHAPE), frames=rows))
        write_json(root/f'ready_{shard_id:08d}.json', dict(schema='feature_memory_archive_ready_v1',
            archive_id='test-archive', shard_id=shard_id, feature_file=payload.name,
            index_file=index.name, feature_bytes=payload.stat().st_size, feature_sha256=sha(payload),
            index_sha256=sha(index), first_frame_id=first, frame_count=last-first,
            last_timestamp=rows[-1]['timestamp']))


def test_requested_frames_and_seed_source_metadata_are_exact(tmp_path):
    archive_fixture(tmp_path)
    values, rows = inputs.read_feature_frames(tmp_path, 1, 3)
    assert values.dtype == torch.bfloat16
    assert tuple(values.shape) == (3, 676, 3584)
    assert [r['global_frame_id'] for r in rows] == [1,2,3]
    assert [r['timestamp'] for r in rows] == [1,1.4,2.4]
    assert [r['source_timestamp'] for r in rows] == [1,0,1]
    assert [r['day_id'] for r in rows] == ['Day1','Day2','Day2']
    for i, value in enumerate(values, start=1):
        assert bool((value == i).all())
    again, _ = inputs.read_feature_frames(tmp_path, 1, 3)
    assert torch.equal(values, again)


def test_growing_archive_does_not_require_future_shards(tmp_path):
    archive_fixture(tmp_path)
    (tmp_path/'ready_00000002.json').unlink()
    (tmp_path/'features_00000002.bin').write_bytes(b'uncommitted')
    assert inputs.read_feature_frames(tmp_path, 0, 1)[0].shape[0] == 1
    with pytest.raises(FileNotFoundError, match='not committed'):
        inputs.read_feature_frames(tmp_path, 4, 1)


def test_payload_mutation_invalidates_cached_hash(tmp_path):
    archive_fixture(tmp_path)
    inputs.read_feature_frames(tmp_path, 0, 1)
    with (tmp_path/'features_00000000.bin').open('r+b') as stream:
        stream.seek(10)
        stream.write(b'xx')
    with pytest.raises(ValueError, match='payload checksum'):
        inputs.read_feature_frames(tmp_path, 0, 1)


def test_rehashed_invalid_frame_offsets_rejected(tmp_path):
    archive_fixture(tmp_path)
    path = tmp_path/'index_00000000.json'
    value = json.loads(path.read_text())
    value['frames'][0]['offset'] = inputs.FRAME_BYTES
    write_json(path, value)
    ready_path = tmp_path/'ready_00000000.json'
    ready = json.loads(ready_path.read_text())
    ready['index_sha256'] = sha(path)
    write_json(ready_path, ready)
    with pytest.raises(ValueError, match='offset'):
        inputs.read_feature_frames(tmp_path, 0, 1)


def rewrite_index(root, shard_id, mutate):
    path = root/f'index_{shard_id:08d}.json'
    value = json.loads(path.read_text())
    mutate(value['frames'])
    write_json(path, value)
    ready_path = root/f'ready_{shard_id:08d}.json'
    ready = json.loads(ready_path.read_text())
    ready['index_sha256'] = sha(path)
    write_json(ready_path, ready)


def test_mixed_seed_and_native_byte_count_schemas_preserve_exact_values(tmp_path):
    archive_fixture(tmp_path)
    expected, expected_rows = inputs.read_feature_frames(tmp_path, 0, 4)
    # The producer can transition inside a shard after the reused seed prefix.
    rewrite_index(tmp_path, 0, lambda rows: rows[1].pop('bytes'))
    rewrite_index(tmp_path, 1, lambda rows: rows[0].pop('feature_bytes'))
    actual, rows = inputs.read_feature_frames(tmp_path, 0, 4)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert [r['shared_feature_sha256'] for r in rows] == [r['shared_feature_sha256'] for r in expected_rows]
    assert [r['global_frame_id'] for r in rows] == [0,1,2,3]
    assert [r['timestamp'] for r in rows] == [0,1,1.4,2.4]
    assert 'bytes' not in rows[1] and rows[1]['feature_bytes'] == inputs.FRAME_BYTES


@pytest.mark.parametrize('mutation', ['missing_both', 'bad_bytes', 'bad_feature_bytes',
                                     'float_size', 'bad_native_offset', 'bad_native_shape', 'bad_native_dtype'])
def test_native_size_alias_support_does_not_bypass_contract_validation(tmp_path, mutation):
    archive_fixture(tmp_path)
    def mutate(rows):
        row = rows[0]
        if mutation == 'missing_both':
            del row['bytes'];del row['feature_bytes']
        elif mutation == 'bad_bytes': row['bytes'] -= 2
        elif mutation == 'bad_feature_bytes': row['feature_bytes'] -= 2
        else:
            del row['bytes']
            if mutation == 'float_size': row['feature_bytes'] = float(inputs.FRAME_BYTES)
            elif mutation == 'bad_native_offset': row['offset'] = inputs.FRAME_BYTES
            elif mutation == 'bad_native_shape': row['shape'] = [675,3584]
            elif mutation == 'bad_native_dtype': row['dtype'] = 'float16'
    rewrite_index(tmp_path, 0, mutate)
    with pytest.raises(ValueError, match='byte count|offset/size/shape/dtype'):
        inputs.read_feature_frames(tmp_path, 0, 1)


def test_native_schema_still_verifies_each_returned_frame_hash(tmp_path):
    archive_fixture(tmp_path)
    def mutate(rows):
        del rows[0]['bytes']
        rows[0]['shared_feature_sha256'] = '0'*64
    rewrite_index(tmp_path, 0, mutate)
    # The index and whole feature shard both have valid receipt hashes, but the
    # individual frame digest is false and must not be accepted.
    with pytest.raises(ValueError, match='frame checksum mismatch'):
        inputs.read_feature_frames(tmp_path, 0, 1)


def test_wrong_archive_revision_rejected(tmp_path):
    archive_fixture(tmp_path)
    path = tmp_path/'archive.json'
    value = json.loads(path.read_text())
    value['model_identity']['revision'] = 'different'
    write_json(path, value)
    with pytest.raises(ValueError, match='identity mismatch'):
        inputs.read_feature_frames(tmp_path, 0, 1)


def checkpoint_fixture(root):
    model = root/'model'
    model.mkdir()
    data = {'config.json': json.dumps(dict(model_type='qwen2_5_vl',hidden_size=3584,
                num_hidden_layers=28,num_key_value_heads=4)).encode(),
            'model.safetensors.index.json': json.dumps({'weight_map':{
                f'tensor_{i}':name for i,name in enumerate(sorted(inputs.WEIGHT_SHARDS))}}).encode(),
            'preprocessor_config.json': b'{}', 'tokenizer.json':b'{}', 'tokenizer_config.json':b'{}',
            'merges.txt':b'merges', 'vocab.json':b'{}'}
    data.update({name: name.encode() for name in inputs.WEIGHT_SHARDS})
    records, states = [], {}
    for name, raw in data.items():
        path = model/name
        path.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest() if name in inputs.WEIGHT_SHARDS else hashlib.sha1(
            f'blob {len(raw)}\0'.encode()+raw).hexdigest()
        records.append(dict(path=name,bytes=len(raw),hash=digest))
        s=path.stat()
        states[name]=[s.st_dev+100,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
    records.append(dict(path='.gitattributes',bytes=1,hash='0'*40))
    ledger = root/'verification.json'
    write_json(ledger, dict(schema_version='baseline_remote_verification_v1', repo_id=inputs.REPO_ID,
        revision=inputs.REVISION, remote_revision_verified=True,
        verified_files=records, materialized_file_states=states))
    return model, ledger


def test_checkpoint_reuses_portable_states_and_ignores_missing_nonmodel_file(tmp_path):
    model, ledger = checkpoint_fixture(tmp_path)
    result = inputs.validate_checkpoint(model, ledger)
    assert not result['rehashed_files']
    assert result['omitted_non_model_files'] == ['.gitattributes']
    assert set(result['verified_weight_shards']) == inputs.WEIGHT_SHARDS


def test_checkpoint_rehashes_only_changed_files_with_correct_algorithms(tmp_path):
    model, ledger = checkpoint_fixture(tmp_path)
    selected = ['config.json',sorted(inputs.WEIGHT_SHARDS)[0]]
    for name in selected:
        path = model/name
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns,stat.st_mtime_ns+1_000_000_000))
    result = inputs.validate_checkpoint(model, ledger)
    assert {v['path']:v['algorithm'] for v in result['rehashed_files']} == {
        selected[0]:'git_blob_sha1',selected[1]:'sha256'}


def test_changed_checkpoint_content_and_missing_weight_fail(tmp_path):
    model, ledger = checkpoint_fixture(tmp_path)
    path = model/sorted(inputs.WEIGHT_SHARDS)[0]
    original = path.read_bytes()
    path.write_bytes(bytes([original[0]^1])+original[1:])
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns,stat.st_mtime_ns+1_000_000_000))
    with pytest.raises(ValueError, match='hash mismatch'):
        inputs.validate_checkpoint(model, ledger)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        inputs.validate_checkpoint(model, ledger)
