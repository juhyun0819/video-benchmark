"""The same encoder output is immutable input to all sixteen memory states."""
import hashlib

import torch

from baselines.feature_memory.byte_memory import ByteBudgetMemory
from baselines.feature_memory.fluxmem import FluxMemByteMemory


def feature_digest(manager):
    digest = hashlib.sha256()
    for batch in manager.iter_batches(max_tokens=7):
        digest.update(batch['features'].contiguous().view(torch.uint8).numpy().tobytes())
        for name, value in sorted(batch['metadata'].items()):
            digest.update(name.encode())
            digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def test_all_sixteen_consumers_preserve_shared_input_and_own_retained_values(tmp_path):
    managers = []
    for method in ('fifo', 'uniform', 'reservoir', 'fluxmem'):
        for ratio in (.05, .10, .25, .50):
            cap = int(2_000_000 * ratio)
            common = dict(directory=tmp_path / f'{method}_{ratio}', hard_budget_bytes=cap,
                          retention_ratio=ratio, hidden_dim=8, tokens_per_frame=9,
                          dtype='bfloat16', shard_bytes=4096, seed=127)
            if method == 'fluxmem':
                bank = FluxMemByteMemory(height=3, width=3, native_short_grids=2,
                                         native_medium_grids=2, **common)
            else:
                bank = ByteBudgetMemory(method=method, **common)
            managers.append((bank, cap))
    assert len(managers) == 16
    rng = torch.Generator().manual_seed(419)
    shared = torch.empty(9, 8, dtype=torch.bfloat16)
    previous = None
    encoder_calls = 0
    try:
        for frame_id in range(20):
            generated = torch.randn(9, 8, generator=rng)
            generated[1] = generated[0] * .97 + generated[1] * .03
            if frame_id % 3 == 1:
                generated[:3] = previous[:3].float()
            # One producer call/output buffer, deliberately reused next frame.
            shared.copy_(generated)
            encoder_calls += 1
            original = shared.clone()
            address = shared.data_ptr()
            for bank, cap in managers:
                bank.update(shared, frame_id, float(frame_id))
                assert shared.data_ptr() == address
                torch.testing.assert_close(shared, original, rtol=0, atol=0)
                assert bank.stats()['actual_memory_bytes'] <= cap
            previous = original
        assert encoder_calls == 20
        assert all(bank.stats()['total_frames_seen'] == encoder_calls for bank, _ in managers)
        assert sum(bank.stats().get('num_merged_tokens', 0) for bank, _ in managers) > 0
        before = [feature_digest(bank) for bank, _ in managers]
        shared.fill_(float('nan'))
        after = [feature_digest(bank) for bank, _ in managers]
        assert before == after  # No retained live tier aliases encoder output.
    finally:
        for bank, _ in managers:
            bank.close()
