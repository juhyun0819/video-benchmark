"""CPU fake-model integration for bounded submit/ack ordering and exact restart."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

import pytest
import torch

from rekv_qwen25 import async_runner as module
from rekv_qwen25 import runner as base


def metrics():
    return dict(enqueue_seconds=0.0, writer_wait_seconds=0.0, finite_validation_seconds=0.0,
        atomic_nas_seconds=0.0, serialized_bytes=0, committed_frames=0,
        compute_write_overlap_seconds=0.0, overlapped_observation_calls=0)


def metadata(start, count=100):
    return dict(frame_ids=list(range(start, start+count)), next_frame_id=start+count,
        total_seen=17+(start+count)*678, config_id='controlled')


@pytest.mark.parametrize('mode,pending,sources', [
    ('sync', 1, {}), (module.WRITER_MODE, 2, {}), (module.WRITER_MODE, True, {}),
    (module.WRITER_MODE, 1, {'async_runner.py': 'only-one-bound-module'}),
])
def test_async_mode_and_both_source_bindings_required_before_loading_artifacts(tmp_path, monkeypatch, mode, pending, sources):
    path = tmp_path/'config.json'
    path.write_text(json.dumps(dict(writer_mode=mode, max_pending_blocks=pending,
                                    extraction_source_sha256=sources)))
    monkeypatch.setattr(base, 'load_block', lambda *a, **k: pytest.fail('Invalid async config must fail before artifact loading'))
    with pytest.raises(ValueError):
        module._load_canonical(SimpleNamespace(config=path))


class EnvelopeWriter:
    pid = 999999

    def __init__(self):
        self.pending = None
        self.mutate = lambda envelope: envelope
        self.events = []

    def submit(self, chunk_id, tensors, info):
        assert self.pending is None
        self.events.append(('submit', chunk_id))
        self.pending = (chunk_id, tensors, info)

    def wait(self):
        chunk_id, tensors, info = self.pending
        self.pending = None
        self.events.append(('wait', chunk_id))
        receipt = dict(config_id=info['config_id'], chunk_id=chunk_id,
            first_frame_id=info['frame_ids'][0], last_frame_id=info['frame_ids'][-1],
            frame_count=len(info['frame_ids']), serialized_bytes=100)
        return self.mutate(dict(receipt=receipt, writer_pid=self.pid, writer_rss_bytes=1000,
            cuda_initialized=False, finite_validation_seconds=1.0, atomic_nas_seconds=10.0,
            started_time=10.0, finished_time=20.0, seconds=10.0))


def test_pipeline_orders_ack_before_second_submit_and_snapshots_boundary():
    writer, counters, intervals = EnvelopeWriter(), metrics(), []
    pipeline = module.CommitPipeline(writer, 'controlled', 0, counters, intervals, lambda *a, **k: None)
    original = metadata(0)
    payload_a = {'raw': torch.tensor([1], dtype=torch.bfloat16)}
    pipeline.submit(payload_a, original)
    original['total_seen'] = -1
    assert writer.pending[2]['total_seen'] == 17+100*678
    assert pipeline.committed_next_frame == 0 and pipeline.submitted_next_frame == 100
    payload_b = {'raw': torch.tensor([2], dtype=torch.bfloat16)}
    pipeline.submit(payload_b, metadata(100))
    assert writer.events == [('submit', 0), ('wait', 0), ('submit', 1)]
    assert pipeline.committed_next_frame == 100
    pipeline.drain()
    assert pipeline.committed_next_frame == 200
    assert counters['committed_frames'] == 200 and len(intervals) == 2
    assert torch.equal(payload_a['raw'], torch.tensor([1], dtype=torch.bfloat16))


def test_overlap_is_actual_interval_intersection_and_buffer_is_bounded():
    writer, counters, intervals = EnvelopeWriter(), metrics(), []
    writer.mutate = lambda envelope: dict(envelope, writer_peak_rss_bytes=5000)
    pipeline = module.CommitPipeline(writer, 'controlled', 0, counters, intervals, lambda *a, **k: None)
    pipeline.submit({}, metadata(0))
    for frame, start, end in [(100, 9.0, 12.0), (101, 14.0, 18.0), (102, 19.0, 21.0)]:
        pipeline.record_observation(frame, start, end)
    pipeline.wait()
    assert counters['compute_write_overlap_seconds'] == 7.0
    assert counters['overlapped_observation_calls'] == 3
    assert pipeline.max_inflight_blocks == 2 and pipeline.max_pending_observed == 1
    assert pipeline.max_tracked_observation_intervals == 3
    assert pipeline.peak_writer_rss_bytes == 5000
    assert pipeline._observation_intervals == []
    pipeline.submit({}, metadata(100))
    for frame in range(200, 300):
        pipeline.record_observation(frame, 21.0, 22.0)
    with pytest.raises((ValueError, RuntimeError)):
        pipeline.record_observation(300, 22.0, 23.0)
    assert len(pipeline._observation_intervals) == 100


@pytest.mark.parametrize('failure', ['wrong_chunk', 'wrong_count', 'wrong_config', 'same_process',
                                   'cuda_child', 'nan_timing', 'child_error'])
def test_failed_ack_never_advances_durable_cursor(failure):
    writer = EnvelopeWriter()
    def corrupt(envelope):
        if failure == 'wrong_chunk': envelope['receipt']['chunk_id'] = 9
        elif failure == 'wrong_count': envelope['receipt']['frame_count'] = 99
        elif failure == 'wrong_config': envelope['receipt']['config_id'] = 'other'
        elif failure == 'same_process': envelope['writer_pid'] = os.getpid()
        elif failure == 'cuda_child': envelope['cuda_initialized'] = True
        elif failure == 'nan_timing': envelope['atomic_nas_seconds'] = float('nan')
        else: raise RuntimeError('writer died')
        return envelope
    writer.mutate = corrupt
    counters = metrics()
    pipeline = module.CommitPipeline(writer, 'controlled', 0, counters, [], lambda *a, **k: None)
    pipeline.submit({}, metadata(0))
    with pytest.raises((ValueError, RuntimeError)):
        pipeline.wait()
    assert pipeline.committed_next_frame == 0
    assert counters['committed_frames'] == 0 and counters['serialized_bytes'] == 0


class DeferredCPUWriter:
    """A deterministic fake child: actual atomic storage happens on acknowledgement."""

    def __init__(self, root, config, events, fail_chunk=None):
        self.archive = base.KVArchiveWriter(root, config)
        self.config_id, self.pid = self.archive.config_id, os.getpid()+100000
        self.events, self.fail_chunk = events, fail_chunk
        self.pending, self.exitcode, self.aborted = None, None, False

    def submit(self, chunk_id, payload, info):
        assert self.pending is None
        self.pending = (chunk_id, payload, copy.deepcopy(info))
        self.events.append(('submit', chunk_id))

    def wait(self):
        chunk_id, payload, info = self.pending
        self.events.append(('wait', chunk_id))
        if chunk_id == self.fail_chunk:
            raise RuntimeError('simulated child write failure')
        base._require_finite_payload(payload)
        began = time.time()
        receipt = self.archive.commit(chunk_id, payload, info)
        finished = time.time()
        self.pending = None
        return dict(receipt=receipt, writer_pid=self.pid, writer_rss_bytes=100000,
            cuda_initialized=False, finite_validation_seconds=0.0,
            atomic_nas_seconds=finished-began, seconds=finished-began,
            started_time=began, finished_time=finished)

    def close(self):
        assert self.pending is None
        self.events.append(('close', None))
        self.exitcode = 0

    def abort(self):
        self.aborted = True
        self.pending = None
        self.exitcode = -15
        self.events.append(('abort', None))


@pytest.fixture
def fake_runtime(tmp_path, monkeypatch):
    events, captured, imported, writers = [], {}, [], []
    prefix = torch.zeros((1, 1, 17, 2), dtype=torch.bfloat16)
    initial = dict(schema=1, n_init=17, n_local=15000, layers=1,
        policy={'cpu_fixture': True}, upstream={'revision': 'cpu-fixture'},
        geometry=[dict(kv_heads=1, head_dim=2, dtype='torch.bfloat16')], total_seen=17,
        states=[dict(key=prefix.clone(), value=prefix.clone(), positions=torch.zeros(3, 1, 17, dtype=torch.int64),
                     init_key=prefix.clone(), init_value=prefix.clone())],
        native_post_rope_cache=[(prefix.clone(), prefix.clone())])
    config = dict(writer_mode=module.WRITER_MODE, max_pending_blocks=1,
                  total_frames=300, n_init=17, n_local=15000)
    binding = dict(config_id='canonical', payload_sha256='payload', index_sha256='index')
    monkeypatch.setattr(module, '_load_canonical', lambda args: (config, initial, binding, 620, 'attention'))

    class FakeModel:
        model = SimpleNamespace(language_model=SimpleNamespace(layers=[None]*28))
        def eval(self): return self
        def requires_grad_(self, enabled):
            assert enabled is False
            return self

    class FakeSession:
        def __init__(self, model, n_init, n_local):
            self.attentions = [SimpleNamespace(num_key_value_heads=1, head_dim=2)]
            self.total_seen, self.latest_kv = 0, {}
        def import_state(self, state):
            self.total_seen = state['total_seen']
            imported.append(copy.deepcopy(state))
        def restore(self): pass

    class FakeStream:
        n_init, stride = 17, 28
        def __init__(self, *args): pass
        def build(self, features, frame):
            assert int(features.item()) == frame
            return dict(frame=frame, input_ids=torch.full((1, 678), frame, dtype=torch.int64),
                position_ids=(torch.arange(678, dtype=torch.int64).reshape(1, 1, -1).expand(3, 1, -1)+frame*1000))

    def observe(model, session, inputs):
        frame = inputs['frame']
        assert session.total_seen == 17+frame*678
        values = (torch.arange(678).reshape(1, 1, 678, 1).expand(1, 1, 678, 2)+frame).to(torch.bfloat16)
        session.latest_kv = {'layer_00.key': values.clone(), 'layer_00.value': values+1}
        captured[frame] = {key: value.clone() for key, value in session.latest_kv.items()}
        session.total_seen += 678
        events.append(('observe', frame))

    def features(root, start, count):
        return torch.arange(start, start+count).reshape(count, 1), [
            dict(frame_id=frame, timestamp_seconds=float(frame)) for frame in range(start, start+count)]

    monkeypatch.setattr(base.Qwen2_5_VLForConditionalGeneration, 'from_pretrained', lambda *a, **k: FakeModel())
    monkeypatch.setattr(base.AutoProcessor, 'from_pretrained', lambda *a, **k: object())
    monkeypatch.setattr(base, 'ReKVSession', FakeSession)
    monkeypatch.setattr(base, 'StreamingInputs', FakeStream)
    monkeypatch.setattr(base, 'observe', observe)
    monkeypatch.setattr(base, 'read_feature_frames', features)
    monkeypatch.setattr(base, '_feature_range', lambda *a, **k: ('visual', 0.0))
    monkeypatch.setattr(base, 'validate_checkpoint', lambda *a: {'revision': 'cpu-fixture'})
    monkeypatch.setattr(base, 'monitor', lambda stop, *a: stop.wait())
    monkeypatch.setattr(torch.cuda, '_lazy_init', lambda: pytest.fail('CUDA must not initialize in CPU integration'))
    for name in ('set_device', 'synchronize', 'reset_peak_memory_stats'):
        monkeypatch.setattr(torch.cuda, name, lambda *a: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    runtime = SimpleNamespace(events=events, captured=captured, imported=imported, writers=writers,
                              fail_chunk=None, config=config)

    def create_writer(root, config, **kwargs):
        writer = DeferredCPUWriter(root, config, events, runtime.fail_chunk)
        writers.append(writer)
        return writer

    monkeypatch.setattr(module, 'create_writer', create_writer)
    runtime.args = module._parser().parse_args([
        '--archive-root', str(tmp_path/'archive'), '--config', str(tmp_path/'config.json'),
        '--initial-state-root', str(tmp_path/'initial'), '--start-frame', '0', '--end-frame', '200',
        '--gpu', '0', '--worker-id', 'cpu_fixture'])
    return runtime


def test_fake_model_computes_next_block_before_ack_and_flushes_before_completion(fake_runtime):
    runtime = fake_runtime
    report = module._run(runtime.args)
    events = runtime.events
    assert events.index(('submit', 0)) < events.index(('observe', 100))
    assert events.index(('observe', 199)) < events.index(('wait', 0)) < events.index(('submit', 1))
    assert events[-2:] == [('wait', 1), ('close', None)]
    assert report['writer_drained'] is True and report['writer_exitcode'] == 0
    assert report['writer_pid'] != report['parent_pid']
    assert report['committed_frames'] == 200 and len(report['commit_intervals']) == 2
    assert report['observed_next_frame'] == report['committed_next_frame'] == 200
    assert report['max_inflight_blocks'] == 2 and report['max_pending_observed'] == 1
    assert report['max_tracked_observation_intervals'] == 100
    for chunk in (0, 1):
        tensors, info = base.load_block(runtime.args.archive_root, chunk, report['config_id'])
        assert info['total_seen'] == 17+(chunk+1)*100*678
        for frame in range(chunk*100, (chunk+1)*100):
            lo = (frame % 100)*678
            for name in ('layer_00.key', 'layer_00.value'):
                assert torch.equal(tensors[name][:, :, lo:lo+678], runtime.captured[frame][name])


def test_child_failure_preserves_durable_cursor_and_resume_recomputes_uncommitted_frames(fake_runtime):
    runtime = fake_runtime
    runtime.args.end_frame = 300
    runtime.fail_chunk = 1
    with pytest.raises(RuntimeError, match='child write failure'):
        module._run(runtime.args)
    status_path = runtime.args.archive_root/'logs/worker_cpu_fixture.status.json'
    failure = json.loads(status_path.read_text())
    assert failure['observed_next_frame'] == 300
    assert failure['committed_next_frame'] == failure['next_frame_id'] == 100
    assert runtime.writers[0].aborted is True
    assert [entry['chunk_id'] for entry in base.committed_blocks(runtime.args.archive_root)] == [0]
    original_frames = copy.deepcopy(runtime.captured)
    runtime.fail_chunk = None
    runtime.events.clear()
    report = module._run(runtime.args)
    assert report['resumed_frames'] == 100 and report['committed_frames'] == 200
    assert ('observe', 99) not in runtime.events and ('observe', 100) in runtime.events
    restored = next(state for state in runtime.imported if state['total_seen'] == 17+100*678)
    assert restored['states'][0]['key'].shape[-2] == 15000
    for chunk in (1, 2):
        tensors, _ = base.load_block(runtime.args.archive_root, chunk, report['config_id'])
        for frame in range(chunk*100, (chunk+1)*100):
            lo = (frame % 100)*678
            for name in ('layer_00.key', 'layer_00.value'):
                assert torch.equal(tensors[name][:, :, lo:lo+678], original_frames[frame][name])
    assert report['writer_drained'] is True and report['committed_next_frame'] == 300


def test_last_block_write_failure_cannot_report_complete(fake_runtime):
    runtime = fake_runtime
    runtime.fail_chunk = 1
    with pytest.raises(RuntimeError, match='child write failure'):
        module._run(runtime.args)
    assert not (runtime.args.archive_root/'logs/worker_cpu_fixture.report.json').exists()
    status = json.loads((runtime.args.archive_root/'logs/worker_cpu_fixture.status.json').read_text())
    assert status['state'] == 'failed' and status['committed_next_frame'] == 100
    assert runtime.writers[0].aborted


def test_completed_resume_does_not_start_new_writer_or_recompute(fake_runtime):
    runtime = fake_runtime
    module._run(runtime.args)
    runtime.events.clear()
    count = len(runtime.writers)
    report = module._run(runtime.args)
    assert report['already_complete'] is True and report['resumed_frames'] == 200
    assert report['committed_frames'] == 0 and report['writer_pid'] is None
    assert len(runtime.writers) == count and runtime.events == []
    assert report['committed_next_frame'] == report['observed_next_frame'] == 200


def test_cpu_fake_model_with_real_spawned_writer_flushes_two_blocks(fake_runtime, monkeypatch):
    from rekv_qwen25.bounded_writer import BoundedKVWriter
    runtime = fake_runtime
    monkeypatch.setattr(module, 'create_writer', lambda root, config, **kw: BoundedKVWriter(
        root, config, startup_timeout=60, commit_timeout=60, cpu_threads=1))
    report = module._run(runtime.args)
    assert report['writer_pid'] != os.getpid()
    assert report['writer_drained'] is True and report['writer_exitcode'] == 0
    assert report['writer_cuda_initialized'] is False
    assert len(report['commit_intervals']) == 2 and report['committed_frames'] == 200
    assert len(base.committed_blocks(runtime.args.archive_root)) == 2
