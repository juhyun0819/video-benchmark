"""CPU-only contracts for the bounded writer-process scaling orchestration."""

import copy
from pathlib import Path

import pytest

from rekv_qwen25 import coordinator
from rekv_qwen25.async_scaling import (
    WRITER_MODE, _async_worker_overrides, async_plan, async_worker_command, commit_cadence,
    validate_async_lineage, validate_async_worker_report,
)


def configs(monkeypatch):
    old = dict(n_local=15000, n_init=17, checkpoint="fixed",
               extraction_source_sha256={"attention.py": "attention", "inputs.py": "inputs", "runner.py": "runner"})
    new = copy.deepcopy(old)
    new.update(writer_mode=WRITER_MODE, max_pending_blocks=1)
    new["extraction_source_sha256"].update({"bounded_writer.py": "bounded", "async_runner.py": "async"})
    source = dict(new["extraction_source_sha256"])
    monkeypatch.setattr(coordinator, "sha256_file", lambda path: source[Path(path).name])
    return old, new


def fixture_report(start=100):
    worker = dict(worker_id="gpu0", gpu=0, start_frame=start, end_frame=start+200)
    replay = max(0, start-620)
    if 17+replay*678 <= 15000:
        replay = 0
    report = dict(status="complete", config_id="fixture", worker_id="gpu0", start_frame=start,
                  end_frame=start+200, committed_frames=200, resumed_frames=0,
                  measured_seconds=100.0, read_seconds=1.0, infer_seconds=60.0,
                  atomic_nas_seconds=70.0, serialized_bytes=2, peak_vram_bytes=1000,
                  writer_mode=WRITER_MODE, max_pending_blocks=1, writer_drained=True,
                  writer_exitcode=0, writer_pid=123, parent_pid=122, writer_cuda_initialized=False,
                  max_pending_observed=1, max_inflight_blocks=2, compute_write_overlap_seconds=12.0,
                  overlapped_observation_calls=20, measurement_started_time=0.0, warmup_frames=start-replay,
                  commit_intervals=[dict(chunk_id=start//100+i, started_time=10+i*40,
                                         finished_time=45+i*40, serialized_bytes=1, seconds=35) for i in range(2)])
    return worker, report


def test_plan_runs_only_two_and_four_gpu_two_commit_trials(tmp_path):
    stages = async_plan(tmp_path)
    assert [stage["gpu_count"] for stage in stages] == [2, 4]
    assert [[(w["start_frame"], w["end_frame"]) for w in stage["workers"]] for stage in stages] == [
        [(100, 300), (300, 500)], [(100, 300), (300, 500), (500, 700), (700, 900)]]
    for stage in stages:
        assert all(w["end_frame"]-w["start_frame"] == 200 for w in stage["workers"])


def test_only_process_overlap_fields_and_new_sources_may_change(monkeypatch):
    old, new = configs(monkeypatch)
    result = validate_async_lineage(old, new)
    assert result["model_input_attention_equal"] is True
    assert result["changed_existing_source_files"] == []
    assert result["added_source_files"] == ["async_runner.py", "bounded_writer.py"]
    assert "writer_mode" not in old
    assert "bounded_writer.py" in new["extraction_source_sha256"]


@pytest.mark.parametrize("mutation", [
    lambda cfg: cfg.update(n_local=14999),
    lambda cfg: cfg.update(max_pending_blocks=2),
    lambda cfg: cfg.update(max_pending_blocks=True),
    lambda cfg: cfg["extraction_source_sha256"].update({"attention.py": "changed"}),
    lambda cfg: cfg["extraction_source_sha256"].update({"extra.py": "new"}),
])
def test_lineage_rejects_inference_change_or_unbounded_pending(monkeypatch, mutation):
    old, new = configs(monkeypatch)
    mutation(new)
    with pytest.raises(ValueError):
        validate_async_lineage(old, new)


@pytest.mark.parametrize("start,warmup", [(100, 100), (300, 300), (500, 500), (700, 620)])
def test_worker_validation_accepts_actual_drained_two_block_pipeline(start, warmup):
    worker, report = fixture_report(start)
    assert report["warmup_frames"] == warmup
    assert validate_async_worker_report(report, worker, "fixture", benchmark=True) is report


@pytest.mark.parametrize("mutation", [
    lambda row: row.update(writer_drained=False),
    lambda row: row.update(writer_exitcode=1),
    lambda row: row.update(writer_pid=None),
    lambda row: row.update(commit_intervals=row["commit_intervals"][:1]),
    lambda row: row.update(warmup_frames=0),
    lambda row: row.update(writer_pid=row["parent_pid"]),
    lambda row: row.update(writer_cuda_initialized=True),
    lambda row: row.update(max_pending_observed=2),
    lambda row: row.update(max_inflight_blocks=3),
    lambda row: row.update(compute_write_overlap_seconds=0.0),
])
def test_worker_cannot_finish_before_both_child_commits_are_acknowledged(mutation):
    worker, report = fixture_report()
    mutation(report)
    with pytest.raises(ValueError):
        validate_async_worker_report(report, worker, "fixture", benchmark=True)


def test_command_override_and_failure_restore_original_coordinator_functions(tmp_path):
    worker, _ = fixture_report()
    original_command = coordinator.worker_command
    original_validator = coordinator.validate_worker_report
    with pytest.raises(RuntimeError, match="fixture interruption"):
        with _async_worker_overrides():
            command = coordinator.worker_command(worker, tmp_path, tmp_path/"config", tmp_path/"initial",
                                                 benchmark=True, python_executable="python")
            assert command[command.index("-m")+1] == "rekv_qwen25.async_runner"
            assert coordinator.worker_command is async_worker_command
            assert coordinator.validate_worker_report is validate_async_worker_report
            raise RuntimeError("fixture interruption")
    assert coordinator.worker_command is original_command
    assert coordinator.validate_worker_report is original_validator


def test_commit_cadence_is_separate_from_finite_two_block_end_to_end_fps():
    _, first = fixture_report()
    _, second = fixture_report(300)
    second["worker_id"] = "gpu1"
    second["commit_intervals"][0]["finished_time"] = 50
    second["commit_intervals"][1]["finished_time"] = 100
    value = commit_cadence(dict(gpu_count=2, steady_fps=1.0, workers=[first, second]))
    assert value["conservative_aggregate_fps"] == 200/50
    assert value["measured_first_wave_fill_seconds"] == 50
    assert value["global_wave_aggregate_fps"] == 200/50
    assert value["intervals_per_worker"] == 1


def test_global_wave_does_not_double_count_early_second_receipt():
    _, first = fixture_report()
    _, second = fixture_report(300)
    first["commit_intervals"][0]["finished_time"] = 10
    first["commit_intervals"][1]["finished_time"] = 20
    second["worker_id"] = "gpu1"
    second["commit_intervals"][0]["finished_time"] = 30
    second["commit_intervals"][1]["finished_time"] = 50
    value = commit_cadence(dict(gpu_count=2, workers=[first, second]))
    assert value["global_wave_newly_committed_frames"] == 100
    assert value["global_wave_aggregate_fps"] == 100/20
