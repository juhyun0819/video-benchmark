import json

import pytest

from baselines.egogpt_caption import media, native, run
from baselines.egogpt_caption.archive import atomic_json, load_complete


@pytest.fixture
def job(tmp_path, monkeypatch):
    clips = [{'clip_id': f'day1_{i:06d}', 'frame_indices': [i],
              'global_end_fraction': str((i + 1) * 30)} for i in range(4)]
    monkeypatch.setattr(media, 'build_clips', lambda inventory: clips)
    monkeypatch.setattr(run, 'snapshot', lambda runtime: {})
    inventory = tmp_path / 'inventory.json'
    runtime = tmp_path / 'runtime.json'
    root = tmp_path / 'output'
    atomic_json(inventory, {'sources': []})
    atomic_json(runtime, {'status': 'ready'})
    run.prepare_job(inventory, runtime, root, 2)
    return root, clips


def test_manifest_tampering_rejected_before_model_load(job):
    root, _ = job
    manifest = run.read(root / 'manifest.json')
    manifest['clips'][0]['global_end_fraction'] = '999'
    atomic_json(root / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='manifest changed'):
        run.verify_job(root)


def test_worker_retries_truncation_and_skips_verified_complete(job, monkeypatch):
    root, clips = job
    visited = []
    monkeypatch.setattr(run, 'load_native', lambda *args: (None, None, {}))
    monkeypatch.setattr(media, 'decode_clip', lambda inv, clip, cache:
                        (clip, None, {'audio': {'full_source_global_end_fraction': '120'}}))

    def generate(tokenizer, model, video, audio):
        visited.append(video['clip_id'])
        # The second assigned clip initially hits its output cap without EOS.
        truncated = len(visited) == 2
        return {'caption': 'I look around.', 'generated_token_ids': [3 if truncated else 9],
                'eos_token_id': 9}

    monkeypatch.setattr(native, 'caption', generate)
    assert run.worker(root, 0, 0, '/unused', 29580) == 2
    assert visited == [clips[0]['clip_id'], clips[2]['clip_id']]
    job_hash = run.read(root / 'job.json')['job_hash']
    assert load_complete(root, clips[2]['clip_id'], job_hash) is None
    first = (root / 'captions' / f"{clips[0]['clip_id']}.json").read_bytes()
    assert run.worker(root, 0, 0, '/unused', 29580) == 0
    assert visited == [clips[0]['clip_id'], clips[2]['clip_id'], clips[2]['clip_id']]
    assert (root / 'captions' / f"{clips[0]['clip_id']}.json").read_bytes() == first
    result = load_complete(root, clips[2]['clip_id'], job_hash)
    assert result['nominal_clip_end_global_seconds_fraction'] == clips[2]['global_end_fraction']
    assert result['conservative_available_at_global_seconds_fraction'] == '120'
    assert result['strict_causal'] is False
    assert run.read(root / 'status/worker_00.json')['completed'] == 2


def test_worker_does_not_load_model_when_all_assigned_records_complete(job, monkeypatch):
    root, clips = job
    from baselines.egogpt_caption.archive import save_record
    job_hash = run.read(root / 'job.json')['job_hash']
    for clip in (clips[1], clips[3]):
        save_record(root, {'clip_id': clip['clip_id'], 'job_hash': job_hash,
                          'status': 'complete', 'caption': 'I walk.',
                          'generated_token_ids': [9], 'eos_token_id': 9})
    def fail(*args):
        raise AssertionError('Must resume without loading a model')
    monkeypatch.setattr(run, 'load_native', fail)
    assert run.worker(root, 1, 1, '/unused', 29581) == 0


def test_worker_cannot_process_another_workers_clip(job):
    root, clips = job
    with pytest.raises(ValueError, match='not assigned'):
        run.worker(root, 0, 0, '/unused', 29580, clip_ids=[clips[1]['clip_id']])


def test_smoke_subset_cannot_claim_entire_shard_complete(job, monkeypatch):
    root, clips = job
    from baselines.egogpt_caption.archive import save_record
    job_hash = run.read(root / 'job.json')['job_hash']
    save_record(root, {'clip_id': clips[0]['clip_id'], 'job_hash': job_hash,
                      'status': 'complete', 'caption': 'I walk.',
                      'generated_token_ids': [9], 'eos_token_id': 9})
    assert run.worker(root, 0, 0, '/unused', 29580, limit=1) == 0
    status = run.read(root / 'status/worker_00.json')
    assert status['status'] == 'selection_complete'
    assert status['shard_total'] == 2 and status['selected'] == 1


def test_worker_uses_offload_only_around_native_call(job,monkeypatch):
    from contextlib import contextmanager
    from baselines.egogpt_caption import memory
    root,clips=job
    verify=run.verify_job
    def verified(path):
        spec,selection=verify(path)
        spec['runtime']['memory_policy']='offload_idle_encoders'
        return spec,selection
    monkeypatch.setattr(run,'verify_job',verified)
    monkeypatch.setattr(run,'load_native',lambda *args:(None,object(),{}))
    monkeypatch.setattr(media,'decode_clip',lambda *args:
                        (None,None,{'audio':{'full_source_global_end_fraction':'120'}}))
    events=[]
    @contextmanager
    def offload(model):
        events.append('enter')
        try:yield
        finally:events.append('exit')
    def caption(*args):
        assert events[-1]=='enter'
        events.append('caption')
        return {'caption':'I walk.','generated_token_ids':[9],'eos_token_id':9}
    monkeypatch.setattr(memory,'offload_idle_encoders',offload)
    monkeypatch.setattr(native,'caption',caption)
    assert run.worker(root,0,0,'/unused',29580)==0
    assert events==['enter','caption','exit']*2
    record=run.read(root/'captions'/f"{clips[0]['clip_id']}.json")
    assert record['finish_reason']=='eos' and record['truncated'] is False


def test_worker_preserves_native_length_stop_and_resumes_without_regeneration(job,monkeypatch):
    root,clips=job
    monkeypatch.setattr(run,'load_native',lambda *args:(None,None,{}))
    monkeypatch.setattr(media,'decode_clip',lambda *args:
                        (None,None,{'audio':{'full_source_global_end_fraction':'120'}}))
    calls=[]
    def caption(*args):
        calls.append(1)
        return {'caption':'I keep describing.','generated_token_ids':[3]*4096,'eos_token_id':9}
    monkeypatch.setattr(native,'caption',caption)
    assert run.worker(root,0,0,'/unused',29580,limit=1)==0
    assert run.worker(root,0,0,'/unused',29580,limit=1)==0
    assert len(calls)==1
    record=run.read(root/'captions'/f"{clips[0]['clip_id']}.json")
    assert record['finish_reason']=='length' and record['truncated'] is True
    assert record['generation_returned_normally'] is True
    assert record['max_new_tokens']==4096
