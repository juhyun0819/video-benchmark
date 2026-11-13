"""Share one frame's immutable features, with an explicit per-frame barrier.

The caller owns the executor. Only FluxMem updates are submitted to it; the
lightweight policies run in the caller. At most one future per FluxMem state is
live (four for the current experiment), and every future finishes before this
function returns or raises. No encoder or next frame is accessed here.
"""
from concurrent.futures import wait
import json
import time


def _update_one(key, manager, setting, features, frame_id, timestamp):
    started = time.perf_counter()
    manager.update(features, frame_id, timestamp)
    state = manager.stats()
    actual = state['actual_memory_bytes']
    cap = setting['hard_budget_bytes']
    if actual > cap:
        raise RuntimeError(json.dumps({
            'method': getattr(manager, 'method', str(key)),
            'budget': cap, 'timestamp': timestamp,
            'overshoot_bytes': actual - cap,
            'cause': 'persistent bytes exceeded the frozen absolute budget during fanout',
        }))
    return time.perf_counter() - started


def update_managers(managers, features, frame_id, timestamp, executor=None):
    """Update ``[(key, manager, setting), ...]`` and return per-key seconds.

    Entries must refer to distinct state objects and unique hashable keys.
    Futures share the exact incoming tensor; managers must own their retained
    storage and may not mutate that input. The cross-method contract tests
    enforce this for all four implementations. The runner can additionally
    verify its shared-feature digest around this call.
    """
    entries = list(managers)
    keys = [key for key, _, _ in entries]
    if len(keys) != len(set(keys)):
        raise ValueError('Memory fanout keys must be unique')
    identities = [id(manager) for _, manager, _ in entries]
    if len(identities) != len(set(identities)):
        raise ValueError('Memory fanout requires distinct manager objects')
    for _, _, setting in entries:
        cap = setting.get('hard_budget_bytes')
        if not isinstance(cap, int) or isinstance(cap, bool) or cap <= 0:
            raise ValueError('Every manager needs a positive integer hard_budget_bytes')
    elapsed = {}
    if executor is None:
        for key, manager, setting in entries:
            elapsed[key] = _update_one(key, manager, setting, features, frame_id, timestamp)
        return elapsed

    futures = {}
    try:
        for key, manager, setting in entries:
            if getattr(manager, 'method', None) == 'fluxmem':
                futures[key] = executor.submit(_update_one, key, manager, setting,
                                               features, frame_id, timestamp)
        for key, manager, setting in entries:
            if key not in futures:
                elapsed[key] = _update_one(key, manager, setting, features, frame_id, timestamp)
        for key, future in futures.items():
            elapsed[key] = future.result()
        return elapsed
    finally:
        # A lightweight-state error, worker error, or submit failure must not
        # leave another current-frame update racing snapshot/close/error cleanup.
        if futures:
            wait(tuple(futures.values()))
