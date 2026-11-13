"""Threaded FluxMem fanout equals serial updates and drains on every failure."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import threading
import time

import pytest
import torch

from baselines.feature_memory.byte_memory import ByteBudgetMemory
from baselines.feature_memory.fanout import update_managers
from baselines.feature_memory.fluxmem import FluxMemByteMemory


def _banks(directory):
    entries = []
    for method in ('fifo', 'uniform', 'reservoir', 'fluxmem'):
        for ratio in (.05, .10, .25, .50):
            cap = int(2_000_000 * ratio)
            common = dict(directory=directory / f'{method}_{ratio}', hard_budget_bytes=cap,
                          retention_ratio=ratio, hidden_dim=8, tokens_per_frame=9,
                          dtype='bfloat16', shard_bytes=4096, seed=127)
            if method == 'fluxmem':
                manager = FluxMemByteMemory(height=3, width=3, native_short_grids=2,
                                            native_medium_grids=2, **common)
            else:
                manager = ByteBudgetMemory(method=method, **common)
            entries.append(((method, ratio), manager, {'hard_budget_bytes': cap}))
    return entries


def _digest(bank):
    result = hashlib.sha256()
    for batch in bank.iter_batches(max_tokens=7):
        result.update(batch['features'].contiguous().view(torch.uint8).numpy().tobytes())
        for name, tensor in sorted(batch['metadata'].items()):
            result.update(name.encode())
            result.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def test_native_sixteen_state_threaded_fanout_matches_serial_exactly(tmp_path):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    serial, threaded = _banks(tmp_path / 'serial'), _banks(tmp_path / 'threaded')
    rng = torch.Generator().manual_seed(419)
    previous = None
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            for frame_id in range(20):
                features = torch.randn(9, 8, generator=rng)
                features[1] = features[0] * .97 + features[1] * .03
                if frame_id % 3 == 1:
                    features[:3] = previous[:3].float()
                features = features.bfloat16()
                original = features.clone()
                timings = update_managers(serial, features, frame_id, float(frame_id))
                parallel_timings = update_managers(threaded, features, frame_id, float(frame_id), executor)
                assert set(timings) == set(parallel_timings) == {entry[0] for entry in serial}
                assert all(value >= 0 for value in parallel_timings.values())
                torch.testing.assert_close(features, original, rtol=0, atol=0)
                for (key, reference, _), (other_key, candidate, setting) in zip(serial, threaded):
                    assert key == other_key
                    assert candidate.stats()['actual_memory_bytes'] <= setting['hard_budget_bytes']
                    assert candidate.stats()['total_frames_seen'] == frame_id + 1
                    assert _digest(candidate) == _digest(reference)
                previous = original
        for _, manager, _ in threaded:
            if manager.method == 'fluxmem':
                assert manager.stats()['num_merged_tokens'] > 0
    finally:
        for _, manager, _ in serial + threaded:
            manager.close()
        torch.set_num_threads(old_threads)


class _Manager:
    def __init__(self, method, delay=0, fail=False, overshoot=False):
        self.method, self.delay, self.fail, self.overshoot = method, delay, fail, overshoot
        self.done = threading.Event()
        self.seen = []
        self.thread_ids = []
        self.input_ids = []
        self.running = False

    def update(self, features, frame_id, timestamp):
        assert not self.running
        assert frame_id == len(self.seen)
        self.running = True
        self.thread_ids.append(threading.get_ident())
        self.input_ids.append(id(features))
        try:
            time.sleep(self.delay)
            self.seen.append((frame_id, timestamp))
            if self.fail:
                raise RuntimeError('intentional state failure')
        finally:
            self.running = False
            self.done.set()

    def stats(self):
        return {'actual_memory_bytes': 101 if self.overshoot else 99}


def test_single_current_frame_barrier_and_thread_assignment():
    light = _Manager('fifo')
    flux = [_Manager('fluxmem', delay=.01) for _ in range(4)]
    entries = [('light', light, {'hard_budget_bytes': 100})]
    entries += [(f'flux{i}', manager, {'hard_budget_bytes': 100}) for i, manager in enumerate(flux)]
    shared = object()
    caller = threading.get_ident()
    with ThreadPoolExecutor(max_workers=4) as executor:
        for frame in range(3):
            result = update_managers(entries, shared, frame, float(frame), executor)
            assert len(result) == 5
            assert all(not manager.running and len(manager.seen) == frame + 1 for manager in [light, *flux])
    assert light.thread_ids == [caller] * 3
    assert all(all(tid != caller for tid in manager.thread_ids) for manager in flux)
    assert all(manager.input_ids == [id(shared)] * 3 for manager in [light, *flux])


@pytest.mark.parametrize('failure', ['light', 'flux', 'cap'])
def test_failure_waits_for_every_submitted_state_before_raising(failure):
    light = _Manager('fifo', fail=failure == 'light')
    flux = [_Manager('fluxmem', delay=.03, fail=failure == 'flux' and index == 0,
                     overshoot=failure == 'cap' and index == 0) for index in range(4)]
    entries = [('light', light, {'hard_budget_bytes': 100})]
    entries += [(f'flux{i}', manager, {'hard_budget_bytes': 100}) for i, manager in enumerate(flux)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        with pytest.raises(RuntimeError):
            update_managers(entries, object(), 0, 0.0, executor)
        assert all(manager.done.is_set() and not manager.running for manager in flux)


def test_partial_submission_failure_drains_already_submitted_work():
    managers = [_Manager('fluxmem', delay=.03) for _ in range(2)]
    entries = [(index, manager, {'hard_budget_bytes': 100}) for index, manager in enumerate(managers)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        class RejectedExecutor:
            calls = 0

            def submit(self, *args, **kwargs):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError('submission rejected')
                return pool.submit(*args, **kwargs)

        with pytest.raises(RuntimeError, match='submission rejected'):
            update_managers(entries, object(), 0, 0.0, RejectedExecutor())
        assert managers[0].done.is_set() and not managers[0].running
        assert not managers[1].seen


def test_reject_aliasing_state_objects_before_updates():
    manager = _Manager('fluxmem')
    entries = [('a', manager, {'hard_budget_bytes': 100}), ('b', manager, {'hard_budget_bytes': 100})]
    with pytest.raises(ValueError, match='distinct manager'):
        update_managers(entries, object(), 0, 0.0)
    assert manager.seen == []
