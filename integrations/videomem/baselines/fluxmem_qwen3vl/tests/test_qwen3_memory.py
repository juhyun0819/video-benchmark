"""CPU-only alignment tests against the unchanged released FluxMem kernel."""
import io

import pytest
import torch
import torch.nn.functional as F

from streaming.memory import PersistentMemory
from streaming.qwen3_memory import Qwen3PersistentMemory
from streaming.upstream_kernel import FluxMem


torch.set_num_threads(2)


def sequence(count=10, dtype=torch.float32):
    generator = torch.Generator().manual_seed(419)
    result = []
    for index in range(count):
        frame = torch.randn(9, 8, generator=generator)
        frame[1] = frame[0] * 0.97 + 0.03 * frame[1]
        if index % 3 == 1:
            frame[:3] = result[-1][:3].float()
        result.append(frame.to(dtype))
    return result


def auxiliary(frame, index=0):
    # Deliberately unlike the main features: cannot drive selection/anchors.
    generator = torch.Generator().manual_seed(775 + index)
    return [(torch.randn(frame.shape, generator=generator) * (level + 1) * 100).to(frame.dtype)
            for level in range(3)]


def arrive(bank, frame, index, *, height=3, width=3, deepstack=None):
    return bank.ingest(frame, height, width, index * 4 + 2, index * 0.5,
                       {"index": index}, auxiliary(frame, index) if deepstack is None else deepstack)


def official_snapshot(frames, short, medium):
    hidden = torch.cat((frames[0].new_zeros(1, 8), *frames,
                        frames[0].new_zeros(1, 8))).unsqueeze(0)
    time = torch.tensor([index * 4 + 2 for index in range(len(frames))]).repeat_interleave(9)
    height = torch.arange(3).repeat_interleave(3).repeat(len(frames))
    width = torch.arange(3).repeat(3 * len(frames))
    positions = F.pad(torch.stack((time, height, width)), (1, 1)).unsqueeze(1)
    ids = torch.tensor([[90] + [5] * (len(frames) * 9) + [91]])
    rotary = torch.zeros((3, 1, hidden.shape[1], 1))
    result = FluxMem(90, 91, short_term_frames=short, mid_term_frames=medium).process_memory_streaming(
        hidden, positions, (rotary, rotary), ids,
        video_grid_thw=torch.tensor([[len(frames), 6, 6]]),
    )
    return result[0][0, 1:-1], result[2][:, 0, 1:-1]


@pytest.mark.parametrize("short,medium", [(2, 0), (4, 4), (3, 1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_main_features_and_anchors_match_official_for_every_prefix(short, medium, dtype):
    frames = sequence(dtype=dtype)
    bank = Qwen3PersistentMemory(short_frames=short, medium_frames=medium)
    for index, frame in enumerate(frames):
        arrive(bank, frame, index)
        bank.advance()
        snapshot = bank.snapshot()
        expected_main, expected_positions = official_snapshot(frames[:index + 1], short, medium)
        torch.testing.assert_close(snapshot["embeddings"], expected_main, rtol=0, atol=0)
        torch.testing.assert_close(snapshot["positions"], expected_positions, rtol=0, atol=0)
        assert len(snapshot["deepstack"]) == 3
        assert all(level.shape == expected_main.shape for level in snapshot["deepstack"])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_sdc_averages_auxiliary_over_main_components_at_official_main_anchors(dtype):
    main = torch.tensor([[1, 0], [2, 0], [3, 0], [0, 1], [0, 2], [0, 3]], dtype=dtype)
    levels = [torch.tensor([[0, 100], [1000, 0], [1, -100],
                           [2, 30], [-400, 0], [4, -30]], dtype=dtype) * (level + 1)
              for level in range(3)]
    bank = Qwen3PersistentMemory()
    arrive(bank, main, 0, height=1, width=6, deepstack=levels)
    merged = bank._merge(bank._short[0])
    # Main components are {0,1,2}, {3,4,5}; anchors are nearest main means.
    assert merged.positions[2].tolist() == [1, 4]
    torch.testing.assert_close(merged.embeddings, torch.tensor([[2, 0], [0, 2]], dtype=dtype),
                               rtol=0, atol=0)
    for original, retained in zip(levels, merged.deepstack):
        expected = torch.stack((original[:3].float().mean(0), original[3:].float().mean(0))).to(dtype)
        torch.testing.assert_close(retained, expected, rtol=0, atol=0)


def reference_components(grid, kernel):
    """Independent small Python graph traversal for sparse retained grids."""
    normalized = grid.normalized if grid.normalized.dtype == torch.float32 else F.normalize(
        grid.embeddings.float(), p=2, dim=1, eps=1e-8)
    edges, distances = [], []
    coordinates = grid.positions[1:].T.tolist()
    for left in range(grid.tokens):
        for right in range(left + 1, grid.tokens):
            if max(abs(a - b) for a, b in zip(coordinates[left], coordinates[right])) == 1:
                edges.append((left, right))
                distances.append((1 - (normalized[left] * normalized[right]).sum()).clamp(0, 2))
    threshold = kernel._otsu_threshold(torch.stack(distances)) if distances else None
    adjacency = {index: set() for index in range(grid.tokens)}
    if threshold is not None:
        for (left, right), distance in zip(edges, distances):
            if distance <= threshold:
                adjacency[left].add(right)
                adjacency[right].add(left)
    components, remaining = [], set(adjacency)
    while remaining:
        stack, found = [min(remaining)], set()
        while stack:
            node = stack.pop()
            if node not in found:
                found.add(node)
                stack.extend(adjacency[node] - found)
        remaining -= found
        components.append(sorted(found))
    return components


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_sparse_sdc_auxiliary_matches_independent_component_means(dtype):
    generator = torch.Generator().manual_seed(171)
    merges = 0
    for _ in range(12):
        main = torch.randn(16, 5, generator=generator).to(dtype)
        levels = auxiliary(main)
        bank = Qwen3PersistentMemory()
        arrive(bank, main, 0, height=4, width=4, deepstack=levels)
        mask = torch.rand(16, generator=generator) > 0.25
        grid = bank._select(bank._short[0], mask, keep_normalized=True)
        components = reference_components(grid, bank._kernel)
        original_positions = grid.positions.clone()
        original_levels = [level.clone() for level in grid.deepstack]
        merged = bank._merge(grid)
        assert merged.tokens == len(components)
        merges += grid.tokens - merged.tokens
        for output_row, position in enumerate(merged.positions.T):
            anchor = ((original_positions.T == position).all(1)).nonzero().item()
            component = next(component for component in components if anchor in component)
            for original, retained in zip(original_levels, merged.deepstack):
                expected = original[component].float().mean(0).to(dtype)
                # The reference mean and released index_add aggregation have
                # different float32 reduction orders on larger components.
                torch.testing.assert_close(retained[output_row], expected, rtol=2e-6, atol=2e-5)
    assert merges > 0


@pytest.mark.parametrize("case", ["empty", "singleton", "no_neighbors", "constant_distances"])
def test_degenerate_sdc_preserves_all_auxiliary_values(case):
    bank = Qwen3PersistentMemory()
    frame = torch.ones(5, 3)
    levels = auxiliary(frame)
    arrive(bank, frame, 0, height=1, width=5, deepstack=levels)
    indices = {"empty": [], "singleton": [2], "no_neighbors": [0, 2, 4],
               "constant_distances": [0, 1, 2, 3, 4]}[case]
    keep = torch.zeros(5, dtype=torch.bool)
    keep[indices] = True
    merged = bank._merge(bank._select(bank._short[0], keep, keep_normalized=True))
    assert merged.positions[2].tolist() == indices
    for original, retained in zip(levels, merged.deepstack):
        torch.testing.assert_close(retained, original[indices], rtol=0, atol=0)


def distinct_vectors(distances):
    distances = torch.tensor(distances, dtype=torch.float32)
    size = len(distances)
    base = torch.eye(size, size * 2)
    changed = base * (1 - distances).unsqueeze(1)
    changed[:, size:] = torch.diag(torch.sqrt(1 - (1 - distances) ** 2))
    return base, changed


@pytest.mark.parametrize("policy,indices", [("upstream", [4, 5, 6]), ("paper", [1, 4, 5, 6])])
def test_tas_applies_exact_main_mask_to_each_auxiliary_level(policy, indices):
    base, previous = distinct_vectors([0, 0.9, 0.1, 0.15, 0.2, 0.7, 0.05])
    _, following = distinct_vectors([0, 0.05, 0.1, 0.15, 0.2, 0.7, 0.9])
    bank = Qwen3PersistentMemory(short_frames=2, medium_frames=6, tas_policy=policy)
    for index, frame in enumerate((previous, base, following)):
        arrive(bank, frame, index, height=1, width=7)
        bank.advance()
    snapshot = bank.snapshot()
    frame_rows = snapshot["positions"][0] == 6
    assert snapshot["positions"][2, frame_rows].tolist() == indices
    for original, retained in zip(auxiliary(base, 1), snapshot["deepstack"]):
        torch.testing.assert_close(retained[frame_rows], original[indices], rtol=0, atol=0)


@pytest.mark.parametrize("policy", ["upstream", "paper"])
def test_auxiliary_changes_cannot_change_main_outputs_or_trigger(policy):
    banks = [Qwen3PersistentMemory(short_frames=2, medium_frames=2, tas_policy=policy) for _ in range(2)]
    for index, frame in enumerate(sequence()):
        observations = [arrive(bank, frame, index, deepstack=auxiliary(frame, index + variant * 100))
                        for variant, bank in enumerate(banks)]
        assert observations[0] == observations[1]
        for bank in banks:
            if index == 0:
                bank.mark_response()
            bank.advance()
        snapshots = [bank.snapshot() for bank in banks]
        for key in ("embeddings", "positions"):
            torch.testing.assert_close(snapshots[0][key], snapshots[1][key], rtol=0, atol=0)


@pytest.mark.parametrize("long_frames", [0, 2, 4])
@pytest.mark.parametrize("empty_pruned", [False, True])
def test_fifo_eviction_bounds_auxiliary_rows_and_accounts_for_bytes(long_frames, empty_pruned):
    unlimited = Qwen3PersistentMemory(short_frames=2, medium_frames=2)
    bounded = Qwen3PersistentMemory(short_frames=2, medium_frames=2, long_frames=long_frames)
    frames = [torch.ones(9, 8)] * 12 if empty_pruned else sequence(12)
    for index, frame in enumerate(frames):
        for bank in (unlimited, bounded):
            arrive(bank, frame, index)
            bank.advance()
        expected, actual = unlimited.snapshot(), bounded.snapshot()
        capacity = 2 + long_frames // 2
        retained_records = expected["records"][-capacity:]
        keep = torch.isin(expected["positions"][0], torch.tensor([r["time_id"] for r in retained_records]))
        assert actual["records"] == retained_records
        for source, retained in zip(expected["deepstack"], actual["deepstack"]):
            torch.testing.assert_close(retained, source[keep], rtol=0, atol=0)
        stats = actual["stats"]
        auxiliary_bytes = 3 * stats["total_tokens"] * frame.shape[1] * frame.element_size()
        assert stats["deepstack_tensor_bytes"] == auxiliary_bytes
        assert stats["retained_tensor_bytes"] == PersistentMemory.stats(bounded)["retained_tensor_bytes"] + auxiliary_bytes
        assert stats["retained_cpu_bytes"] == stats["retained_tensor_bytes"]
        assert sum(stats["tier_grids"].values()) <= capacity
        assert stats["total_tokens"] <= capacity * 9
    assert bounded.stats()["evicted_grids"] == 12 - capacity


def assert_same_snapshot(actual, expected):
    for key in ("embeddings", "positions"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    for actual_level, expected_level in zip(actual["deepstack"], expected["deepstack"]):
        torch.testing.assert_close(actual_level, expected_level, rtol=0, atol=0)
    assert actual["records"] == expected["records"]
    assert actual["stats"] == expected["stats"]


@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("policy", ["upstream", "paper"])
def test_serialization_resumes_deepstack_in_every_tier_and_pending_phase(pending, policy):
    frames = sequence(12)
    bank = Qwen3PersistentMemory(short_frames=4, medium_frames=4, long_frames=4, tas_policy=policy)
    for index, frame in enumerate(frames[:8]):
        arrive(bank, frame, index)
        if index == 0:
            bank.mark_response()
        if index != 7 or not pending:
            bank.advance()
    checkpoint = bank.state_dict()
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)
    restored = Qwen3PersistentMemory.from_state_dict(torch.load(buffer, weights_only=True))
    assert isinstance(restored, Qwen3PersistentMemory)
    assert_same_snapshot(restored.snapshot(), bank.snapshot())
    checkpoint["queues"]["short"][-1]["deepstack"][0].fill_(999)
    assert_same_snapshot(restored.snapshot(), bank.snapshot())
    if pending:
        restored.advance()
        bank.advance()
    for index in range(8, len(frames)):
        assert arrive(restored, frames[index], index) == arrive(bank, frames[index], index)
        restored.advance()
        bank.advance()
        assert_same_snapshot(restored.snapshot(), bank.snapshot())


def test_input_snapshot_and_checkpoint_auxiliary_are_detached_owned_copies():
    bank = Qwen3PersistentMemory(short_frames=2, medium_frames=0)
    frame = sequence(1)[0]
    levels = [level.requires_grad_() for level in auxiliary(frame)]
    expected_levels = [level.detach().clone() for level in levels]
    arrive(bank, frame, 0, deepstack=levels)
    for level in levels:
        level.detach().fill_(999)
    before = bank.snapshot()
    for expected, actual in zip(expected_levels, before["deepstack"]):
        assert not actual.requires_grad
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    bank.advance()
    arrive(bank, sequence(2)[1], 1)
    bank.advance()
    for expected, actual in zip(expected_levels, before["deepstack"]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    before["deepstack"][0].fill_(888)
    assert not bool((bank.snapshot()["deepstack"][0] == 888).any())
    checkpoint = bank.state_dict()
    checkpoint["queues"]["short"][-1]["deepstack"][0].fill_(777)
    assert not bool((bank.snapshot()["deepstack"][0] == 777).any())


@pytest.mark.parametrize("bad", ["missing", "wrong_count", "wrong_shape", "wrong_dtype", "nan", "not_tensor"])
def test_rejects_invalid_auxiliary_before_mutating_memory(bad):
    bank = Qwen3PersistentMemory()
    frame = sequence(1)[0]
    levels = auxiliary(frame)
    if bad == "missing":
        levels = None
    elif bad == "wrong_count":
        levels.pop()
    elif bad == "wrong_shape":
        levels[1] = levels[1][:-1]
    elif bad == "wrong_dtype":
        levels[1] = levels[1].double()
    elif bad == "nan":
        levels[1][0, 0] = float("nan")
    elif bad == "not_tensor":
        levels[1] = 123
    with pytest.raises(ValueError, match="deepstack"):
        bank.ingest(frame, 3, 3, 0, 0.0, {}, levels)
    assert bank.stats()["total_ingested"] == 0
    assert not bank.stats()["pending_advance"]


def test_empty_checkpoint_and_rejects_checkpoint_without_auxiliary():
    bank = Qwen3PersistentMemory()
    restored = Qwen3PersistentMemory.from_state_dict(bank.state_dict())
    assert_same_snapshot(restored.snapshot(), bank.snapshot())
    assert [tuple(level.shape) for level in restored.snapshot()["deepstack"]] == [(0, 0)] * 3
    frame = sequence(1)[0]
    arrive(restored, frame, 0)
    restored.advance()
    state = restored.state_dict()
    del state["queues"]["short"][0]["deepstack"]
    with pytest.raises(ValueError, match="DeepStack"):
        Qwen3PersistentMemory.from_state_dict(state)
    with pytest.raises(ValueError, match="checkpoint version"):
        Qwen3PersistentMemory.from_state_dict(PersistentMemory().state_dict())
