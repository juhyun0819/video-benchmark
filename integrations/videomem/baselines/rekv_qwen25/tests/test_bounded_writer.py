from functools import partial
import os
from pathlib import Path
import signal
import time
import weakref

import pytest
import torch
import torch.multiprocessing as multiprocessing

from rekv_qwen25.archive import KVArchiveWriter, load_block
from rekv_qwen25.bounded_writer import BoundedKVWriter, WriterProcessError


def tensors():
    values=torch.arange(24).reshape(1,2,3,4).to(torch.bfloat16)
    return {'layer_00.key':values,'layer_00.value':-values,
            'position_ids':torch.arange(9,dtype=torch.int64).reshape(3,1,3)}


def metadata(chunk=0):
    return dict(frame_ids=[chunk*2,chunk*2+1],timestamps=[float(chunk*2),float(chunk*2+1)])


class GatedWriter(KVArchiveWriter):
    def __init__(self,root,config,*,entered,released):
        super().__init__(root,config)
        self.entered,self.released=entered,released

    def commit(self,*args,**kwargs):
        self.entered.set()
        if not self.released.wait(30):
            raise TimeoutError('Test gate was not released')
        return super().commit(*args,**kwargs)


class FailedWriter(KVArchiveWriter):
    def commit(self,*args,**kwargs):
        raise OSError('injected storage failure')


class DeadWriter(KVArchiveWriter):
    def commit(self,*args,**kwargs):
        os._exit(23)


class FailedStartup:
    def __init__(self,*args,**kwargs):
        raise ValueError('injected startup failure')


def test_spawn_cpu_only_exact_bf16_roundtrip_and_clean_close(tmp_path):
    original=tensors()
    expected={name:value.clone() for name,value in original.items()}
    with BoundedKVWriter(tmp_path,{'checkpoint':'test'},cpu_threads=2) as writer:
        assert writer.pid != os.getpid()
        assert writer.startup_info['start_method']=='spawn'
        assert writer.startup_info['cuda_initialized'] is False
        assert writer.startup_info['cpu_threads']==2
        writer.submit(0,original,metadata())
        assert writer.is_pending and all(value.is_shared() for value in original.values())
        reply=writer.wait(timeout=20)
        assert not writer.is_pending
        assert reply['writer_pid']==writer.pid and reply['cuda_initialized'] is False
        assert reply['finite_validation_seconds']>=0
        assert reply['atomic_nas_seconds']==reply['seconds']>0
        assert reply['finished_time']>=reply['started_time']
        assert reply['writer_rss_bytes']>0 and reply['writer_peak_rss_bytes']>=reply['writer_rss_bytes']
        assert reply['receipt']['config_id']==writer.config_id
        loaded,_=load_block(tmp_path,0,writer.config_id)
        for name in expected:
            assert torch.equal(loaded[name].view(torch.uint8),expected[name].view(torch.uint8))
        # Ownership returns after the ACK; mutation now cannot change the file.
        original['layer_00.key'].fill_(99)
        assert torch.equal(load_block(tmp_path,0)[0]['layer_00.key'],expected['layer_00.key'])
    assert writer.exitcode==0
    writer.close()  # Successful close is idempotent.


def test_parent_can_work_while_writer_is_blocked_and_second_submit_is_rejected(tmp_path):
    context=multiprocessing.get_context('spawn')
    entered,released=context.Event(),context.Event()
    factory=partial(GatedWriter,entered=entered,released=released)
    writer=BoundedKVWriter(tmp_path,{'checkpoint':'test'},_writer_factory=factory,cpu_threads=1)
    try:
        payload=tensors();reference=weakref.ref(payload['layer_00.key'])
        meta=metadata()
        writer.submit(0,payload,meta)
        meta['timestamps'][0]=999  # Metadata was copied at submission.
        del payload
        assert entered.wait(10)
        assert reference() is not None  # Strong immutable ownership until ACK.
        assert writer.is_pending and not list((tmp_path/'receipts').glob('*.ready.json'))
        independent_result=sum(value*value for value in range(10000))
        assert independent_result==333283335000
        with pytest.raises(RuntimeError,match='already pending'):
            writer.submit(1,tensors(),metadata(1))
        released.set()
        writer.wait(timeout=20)
        assert load_block(tmp_path,0)[1]['timestamps']==[0.0,1.0]
        writer.submit(1,tensors(),metadata(1))
        final=writer.close()  # close drains this second block.
        assert final['chunk_id']==1 and writer.exitcode==0
    finally:
        if writer.exitcode is None:writer.abort()


@pytest.mark.parametrize('failure',['nonfinite','commit_error','process_death'])
def test_failure_propagates_without_deadlock_and_close_cannot_claim_success(tmp_path,failure):
    factory=FailedWriter if failure=='commit_error' else DeadWriter if failure=='process_death' else KVArchiveWriter
    writer=BoundedKVWriter(tmp_path,{'checkpoint':'test'},_writer_factory=factory,cpu_threads=1)
    payload=tensors()
    if failure=='nonfinite':payload['layer_00.value'][0,0,0,0]=float('nan')
    writer.submit(0,payload,metadata())
    with pytest.raises(WriterProcessError):writer.wait(timeout=20)
    assert writer.exitcode is not None and not writer.is_pending
    assert not list((tmp_path/'receipts').glob('*.ready.json'))
    with pytest.raises(WriterProcessError,match='failure'):writer.close()
    writer.abort()


def test_timeout_stops_child_and_uncommitted_payload_is_not_exposed(tmp_path):
    context=multiprocessing.get_context('spawn')
    entered,released=context.Event(),context.Event()
    writer=BoundedKVWriter(tmp_path,{'checkpoint':'test'},cpu_threads=1,
                          _writer_factory=partial(GatedWriter,entered=entered,released=released))
    writer.submit(0,tensors(),metadata())
    assert entered.wait(10)
    began=time.monotonic()
    with pytest.raises(TimeoutError):writer.wait(timeout=0.1)
    assert time.monotonic()-began<5 and writer.exitcode is not None
    assert not list((tmp_path/'receipts').glob('*.ready.json'))


def test_parent_exception_aborts_instead_of_draining_blocked_commit(tmp_path):
    context=multiprocessing.get_context('spawn')
    entered,released=context.Event(),context.Event()
    with pytest.raises(RuntimeError,match='parent failed'):
        with BoundedKVWriter(tmp_path,{'checkpoint':'test'},cpu_threads=1,
                _writer_factory=partial(GatedWriter,entered=entered,released=released)) as writer:
            writer.submit(0,tensors(),metadata())
            assert entered.wait(10)
            raise RuntimeError('parent failed')
    assert writer.exitcode is not None and not writer.is_pending
    assert not list((tmp_path/'receipts').glob('*.ready.json'))


def test_startup_exception_is_reported_and_child_is_joined(tmp_path):
    before={process.pid for process in multiprocessing.active_children()}
    with pytest.raises(WriterProcessError,match='injected startup failure'):
        BoundedKVWriter(tmp_path,{'checkpoint':'test'},cpu_threads=1,_writer_factory=FailedStartup)
    assert {process.pid for process in multiprocessing.active_children()}==before


def orphan_owner(root,connection):
    writer=BoundedKVWriter(root,{'checkpoint':'test'},cpu_threads=1)
    connection.send(writer.pid)
    time.sleep(30)


def process_running(pid):
    path=Path(f'/proc/{pid}/stat')
    try:
        # An adopted zombie is already stopped and cannot keep writing.
        return path.read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:
        return False


@pytest.mark.skipif(not Path('/proc').is_dir(),reason='Linux parent-death signal test')
def test_abrupt_parent_death_does_not_leave_active_writer(tmp_path):
    context=multiprocessing.get_context('spawn')
    receiver,sender=context.Pipe(duplex=False)
    owner=context.Process(target=orphan_owner,args=(str(tmp_path),sender))
    owner.start();sender.close();writer_pid=None
    try:
        assert receiver.poll(30),'Owner/writer startup timed out'
        writer_pid=receiver.recv()
        assert process_running(writer_pid)
        os.kill(owner.pid,signal.SIGKILL);owner.join(10)
        deadline=time.monotonic()+5
        while process_running(writer_pid) and time.monotonic()<deadline:time.sleep(0.05)
        assert not process_running(writer_pid)
    finally:
        receiver.close()
        if owner.is_alive():owner.kill()
        owner.join(10)
        if writer_pid is not None and process_running(writer_pid):os.kill(writer_pid,signal.SIGKILL)
