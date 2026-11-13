"""Deterministic CPU tests for local persistent memory, without model downloads."""
from copy import deepcopy
import io

import pytest
import torch

from streaming.upstream_kernel import FluxMem
from streaming.memory import PersistentMemory


torch.set_num_threads(2)


def frames(count=12, height=3, width=3, dim=8, dtype=torch.float32):
    generator = torch.Generator().manual_seed(419)
    result = []
    for index in range(count):
        frame = torch.randn(height * width, dim, generator=generator)
        # Repeated adjacent content lets both TAS and SDC remove tokens.
        frame[1] = frame[0] * 0.97 + 0.03 * frame[1]
        if index % 3 == 1:
            frame[:3] = result[-1][:3].float()
        result.append(frame.to(dtype))
    return result


def arrive(bank, frame, index, *, height=3, width=3, metadata=None):
    return bank.ingest(frame, height, width, time_id=index * 4 + 2,
                       timestamp_seconds=index * 0.5,
                       metadata={"frame_index": index} if metadata is None else metadata)


def upstream_snapshot(sequence, short, medium, height=3, width=3):
    tokens = height * width
    hidden = torch.cat((torch.zeros(1, sequence[0].shape[1], dtype=sequence[0].dtype),
                        *sequence, torch.zeros(1, sequence[0].shape[1], dtype=sequence[0].dtype))).unsqueeze(0)
    time = torch.tensor([index * 4 + 2 for index in range(len(sequence))]).repeat_interleave(tokens)
    h = torch.arange(height).repeat_interleave(width).repeat(len(sequence))
    w = torch.arange(width).repeat(height).repeat(len(sequence))
    positions = torch.nn.functional.pad(torch.stack((time, h, w)), (1, 1)).unsqueeze(1)
    input_ids = torch.tensor([[90] + [5] * (len(sequence) * tokens) + [91]])
    rotary = torch.zeros((3, 1, hidden.shape[1], 1))
    result = FluxMem(90, 91, short_term_frames=short, mid_term_frames=medium).process_memory_streaming(
        hidden, positions, (rotary, rotary), input_ids,
        video_grid_thw=torch.tensor([[len(sequence), height * 2, width * 2]]),
    )
    return result[0][0, 1:-1], result[2][:, 0, 1:-1]


@pytest.mark.parametrize("short,medium", [(2, 0), (4, 4), (8, 6), (3, 1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_incremental_after_advance_matches_unmodified_upstream_for_each_prefix(short, medium, dtype):
    sequence = frames(dtype=dtype)
    bank = PersistentMemory(short_frames=short, medium_frames=medium)
    for index, frame in enumerate(sequence):
        arrive(bank, frame, index)
        bank.advance()
        actual = bank.snapshot()
        expected_embeddings, expected_positions = upstream_snapshot(sequence[:index + 1], short, medium)
        torch.testing.assert_close(actual["embeddings"], expected_embeddings, rtol=0, atol=0)
        torch.testing.assert_close(actual["positions"], expected_positions, rtol=0, atol=0)
    assert bank.stats()["total_ingested"] == len(sequence)


def test_incremental_work_uses_only_neighbor_grids_and_no_prefix_replay(monkeypatch):
    bank = PersistentMemory(short_frames=4, medium_frames=4)
    match_sizes = []
    calls = {"tas": 0, "sdc": 0}
    original_match = bank._kernel._max_sim_against_frames
    original_tas = bank._kernel._apply_adjacent_pruning
    original_sdc = bank._kernel._long_term_memory_merge_per_frames

    def match(**kwargs):
        match_sizes.append(kwargs["hidden_norm"].shape[0])
        return original_match(**kwargs)

    def tas(**kwargs):
        calls["tas"] += 1
        return original_tas(**kwargs)

    def sdc(**kwargs):
        calls["sdc"] += 1
        assert kwargs["hidden_states"].shape[1] <= 9
        return original_sdc(**kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("Persistent ingestion must not replay the full-prefix upstream API")

    monkeypatch.setattr(bank._kernel, "_max_sim_against_frames", match)
    monkeypatch.setattr(bank._kernel, "_apply_adjacent_pruning", tas)
    monkeypatch.setattr(bank._kernel, "_long_term_memory_merge_per_frames", sdc)
    monkeypatch.setattr(bank._kernel, "process_memory_streaming", forbidden)
    for index, frame in enumerate(frames(20)):
        arrive(bank, frame, index)
        bank.advance()
    assert match_sizes and max(match_sizes) <= 18
    assert calls == {"tas": 18, "sdc": 16}


def test_ordered_chunk_boundaries_do_not_change_memory_or_observations():
    sequence = frames(14)
    outcomes = []
    for chunk_sizes in ([14], [1] * 14, [3, 2, 8, 1]):
        bank = PersistentMemory(short_frames=4, medium_frames=4)
        observations = []
        offset = 0
        for size in chunk_sizes:
            for index in range(offset, offset + size):
                observations.append(arrive(bank, sequence[index], index))
                if index in (0, 6, 11):
                    bank.mark_response()
                bank.advance()
            offset += size
        outcomes.append((bank.snapshot(), observations))
    for snapshot, observations in outcomes[1:]:
        torch.testing.assert_close(snapshot["embeddings"], outcomes[0][0]["embeddings"], rtol=0, atol=0)
        assert snapshot["records"] == outcomes[0][0]["records"]
        assert observations == outcomes[0][1]


def test_query_snapshot_precedes_cascade_and_remains_isolated_after_sdc():
    bank = PersistentMemory(short_frames=2, medium_frames=0)
    sequence = frames(3)
    source_metadata = {"nested": {"source": [1, 2]}}
    arrive(bank, sequence[0].requires_grad_(), 0, metadata=source_metadata)
    source_metadata["nested"]["source"].append(3)
    sequence[0].detach().fill_(123)
    bank.mark_response()
    bank.advance()
    arrive(bank, sequence[1], 1)
    before = bank.snapshot()
    immutable_copy = deepcopy(before)
    assert before["embeddings"].shape == (18, 8)
    assert not before["embeddings"].requires_grad
    assert before["stats"]["tier_grids"] == {"short": 2, "medium": 0, "long": 0}
    assert before["records"][0]["metadata"] == {"nested": {"source": [1, 2]}}
    assert not bool((before["embeddings"] == 123).all())
    bank.advance()
    torch.testing.assert_close(before["embeddings"], immutable_copy["embeddings"], rtol=0, atol=0)
    before["embeddings"].fill_(999)
    before["positions"].fill_(999)
    before["records"][0]["metadata"]["nested"]["source"].append(999)
    after = bank.snapshot()
    assert not bool((after["embeddings"] == 999).any())
    assert not bool((after["positions"] == 999).any())
    assert after["records"][0]["metadata"] == {"nested": {"source": [1, 2]}}
    assert after["stats"]["response_count"] == 1
    assert after["stats"]["total_ingested"] == 2


def test_multiple_questions_never_reset_queues():
    bank = PersistentMemory(short_frames=4, medium_frames=4)
    for index, frame in enumerate(frames(7)):
        arrive(bank, frame, index)
        before = bank.snapshot()
        bank.mark_response()
        after = bank.snapshot()
        torch.testing.assert_close(after["embeddings"], before["embeddings"])
        assert after["records"] == before["records"]
        bank.advance()
    assert len(bank.snapshot()["records"]) == 7
    assert bank.stats()["response_count"] == 7


@pytest.mark.parametrize("long_frames", [0, 2, 4])
def test_long_capacity_evicts_whole_oldest_grids_including_empty_records(long_frames):
    bank = PersistentMemory(short_frames=2, medium_frames=2, long_frames=long_frames)
    # Constant content is removed entirely by upstream TAS, but still uses slots.
    frame = torch.ones((4, 3))
    for index in range(9):
        arrive(bank, frame, index, height=2, width=2)
        bank.advance()
        stats = bank.stats()
        assert stats["tier_grids"]["long"] <= long_frames // 2
        assert stats["tier_grids"]["medium"] <= 1
        assert stats["tier_tokens"]["medium"] == 0
        assert stats["total_tokens"] == 4
        assert stats["retained_cpu_bytes"] == stats["retained_tensor_bytes"]
    records = bank.snapshot()["records"]
    capacity = long_frames // 2 + 2
    assert [r["time_id"] for r in records] == [i * 4 + 2 for i in range(9 - capacity, 9)]
    assert bank.stats()["evicted_grids"] == 9 - capacity


def test_finite_long_capacity_removes_all_tokens_from_only_oldest_grids():
    unlimited = PersistentMemory(short_frames=2, medium_frames=2)
    bounded = PersistentMemory(short_frames=2, medium_frames=2, long_frames=4)
    for index, frame in enumerate(frames(10)):
        for bank in (unlimited, bounded):
            arrive(bank, frame, index)
            bank.advance()
        expected, actual = unlimited.snapshot(), bounded.snapshot()
        kept_ids = [record["time_id"] for record in expected["records"][-4:]]
        keep = torch.isin(expected["positions"][0], torch.tensor(kept_ids))
        torch.testing.assert_close(actual["embeddings"], expected["embeddings"][keep], rtol=0, atol=0)
        torch.testing.assert_close(actual["positions"], expected["positions"][:, keep], rtol=0, atol=0)
        assert actual["records"] == expected["records"][-4:]
    assert bounded.stats()["evicted_tokens"] > 0


def distinct_vectors(distances):
    """Each token's strongest 3x3 reference is its own basis-vector location."""
    distance = torch.tensor(distances, dtype=torch.float32)
    size = len(distances)
    base = torch.eye(size, size * 2)
    changed = base * (1 - distance).unsqueeze(1)
    changed[:, size:] = torch.diag(torch.sqrt(1 - (1 - distance) ** 2))
    return base, changed


@pytest.mark.parametrize("gamma,expected", [(0.499, True), (0.5, False), (1.0, False)])
def test_backward_trigger_stationary_scene_change_exact_gamma_and_prior_answer(gamma, expected):
    base, changed = distinct_vectors([0, 0.1, 0.8, 0.9])
    bank = PersistentMemory(gamma=gamma)
    first = arrive(bank, base, 0, height=1, width=4)
    assert not first["triggered"] and first["backward_score"] == 0
    assert not bank.should_respond()
    bank.mark_response()
    bank.advance()
    stationary = arrive(bank, base.clone(), 1, height=1, width=4)
    assert stationary["backward_score"] == 0 and not stationary["triggered"]
    bank.advance()
    change = arrive(bank, changed, 2, height=1, width=4)
    assert change["backward_score"] == 0.5
    assert change["triggered"] is expected
    assert bank.should_respond() is expected
    bank.mark_response()
    assert not bank.should_respond()
    without_prior = PersistentMemory(gamma=0.1)
    arrive(without_prior, base, 0, height=1, width=4)
    without_prior.advance()
    observation = arrive(without_prior, changed, 1, height=1, width=4)
    assert observation["backward_score"] == 0.5
    assert not observation["triggered"] and not without_prior.should_respond()


def test_constant_distance_fallback_has_no_strictly_above_threshold_tokens():
    base, changed = distinct_vectors([1, 1, 1, 1])
    bank = PersistentMemory(gamma=0)
    arrive(bank, base, 0, height=1, width=4)
    bank.mark_response()
    bank.advance()
    observation = arrive(bank, changed, 1, height=1, width=4)
    assert observation["backward_threshold"] == 1.0
    assert observation["backward_score"] == 0.0
    assert not observation["triggered"]


def test_trigger_cannot_read_future_frames_or_metadata_labels():
    sequence = frames(5)
    observations = []
    for answer in ("future scene: running", "future scene: sitting"):
        bank = PersistentMemory()
        current = []
        for index, frame in enumerate(sequence):
            current.append(arrive(bank, frame, index, metadata={"answer": answer, "future_timestamp": 1000000}))
            if index == 0:
                bank.mark_response()
            bank.advance()
        observations.append(deepcopy(current))
        # An arbitrary future observation cannot change already-returned observations.
        arrive(bank, -sequence[-1], len(sequence))
        assert current == observations[-1]
    assert observations[0] == observations[1]


def test_paper_tas_uses_independent_strict_direction_thresholds_and_no_point8_drop():
    base, previous = distinct_vectors([0, 0.9, 0.1, 0.15, 0.2, 0.7, 0.05])
    _, next_frame = distinct_vectors([0, 0.05, 0.1, 0.15, 0.2, 0.7, 0.9])
    retained = {}
    for policy in ("paper", "upstream"):
        bank = PersistentMemory(short_frames=2, medium_frames=6, tas_policy=policy)
        for index, frame in enumerate((previous, base, next_frame)):
            arrive(bank, frame, index, height=1, width=7)
            bank.advance()
        snapshot = bank.snapshot()
        retained[policy] = snapshot["positions"][2, snapshot["positions"][0] == 6].tolist()
    # The winning 128-bin center is 0.1953125, so distance 0.2 survives too.
    # Token 1 is backward-novel but forward-similar: only paper's OR keeps it.
    assert retained["paper"] == [1, 4, 5, 6]
    assert retained["upstream"] == [4, 5, 6]


def test_paper_tas_strict_constant_threshold_removes_all_tokens():
    bank = PersistentMemory(short_frames=2, medium_frames=2, tas_policy="paper")
    base, changed = distinct_vectors([1, 1, 1, 1])
    arrive(bank, base, 0, height=1, width=4)
    bank.advance()
    arrive(bank, changed, 1, height=1, width=4)
    bank.advance()
    assert bank.snapshot()["records"][0]["retained_tokens"] == 0


def test_validation_rejects_duplicates_order_shape_and_invalid_phases():
    bank = PersistentMemory()
    with pytest.raises(RuntimeError, match="preceding ingest"):
        bank.advance()
    with pytest.raises(RuntimeError, match="before ingesting"):
        bank.mark_response()
    frame = frames(1)[0]
    arrive(bank, frame, 0)
    with pytest.raises(RuntimeError, match="advance"):
        arrive(bank, frame, 1)
    bank.advance()
    with pytest.raises(ValueError, match="time_id must strictly increase"):
        arrive(bank, frame, 0)
    with pytest.raises(ValueError, match="timestamp_seconds must strictly increase"):
        bank.ingest(frame, 3, 3, 5, 0.0, {})
    with pytest.raises(ValueError, match="same|share"):
        bank.ingest(torch.ones(4, 8), 2, 2, 5, 1.0, {})
    with pytest.raises(ValueError, match="finite"):
        bank.ingest(torch.full((9, 8), float("nan")), 3, 3, 5, 1.0, {})
    assert bank.stats()["total_ingested"] == 1


@pytest.mark.parametrize("kwargs", [
    {"short_frames": 1}, {"short_frames": -2}, {"short_frames": True},
    {"medium_frames": -2}, {"medium_frames": 2.5}, {"long_frames": -1},
    {"gamma": float("nan")}, {"gamma": -0.1}, {"gamma": 1.1}, {"tas_policy": "other"},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        PersistentMemory(**kwargs)


@pytest.mark.parametrize("pending", [False, True])
def test_checkpoint_resume_preserves_phase_trigger_queues_and_caches(pending):
    sequence = frames(9)
    bank = PersistentMemory(short_frames=4, medium_frames=4, long_frames=4, tas_policy="paper")
    for index, frame in enumerate(sequence[:6]):
        arrive(bank, frame, index)
        if index == 0:
            bank.mark_response()
        if index != 5 or not pending:
            bank.advance()
    buffer = io.BytesIO()
    torch.save(bank.state_dict(), buffer)
    buffer.seek(0)
    restored = PersistentMemory.from_state_dict(torch.load(buffer, weights_only=True))
    assert restored.stats() == bank.stats()
    assert restored.last_observation == bank.last_observation
    if pending:
        restored.advance()
        bank.advance()
    for index in range(6, len(sequence)):
        assert arrive(restored, sequence[index], index) == arrive(bank, sequence[index], index)
        restored.advance()
        bank.advance()
    actual, expected = restored.snapshot(), bank.snapshot()
    torch.testing.assert_close(actual["embeddings"], expected["embeddings"], rtol=0, atol=0)
    torch.testing.assert_close(actual["positions"], expected["positions"], rtol=0, atol=0)
    assert actual["records"] == expected["records"]
    assert actual["stats"] == expected["stats"]


def test_empty_snapshot_and_checkpoint():
    bank = PersistentMemory()
    snapshot = bank.snapshot()
    assert snapshot["embeddings"].shape == (0, 0)
    assert snapshot["positions"].shape == (3, 0)
    assert snapshot["records"] == []
    restored = PersistentMemory.from_state_dict(bank.state_dict())
    assert restored.stats() == bank.stats()
    arrive(restored, frames(1)[0], 0)
    restored.advance()
