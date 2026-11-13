"""Measure explicitly authorized CPU-process overlap on two and four GPUs.

Each worker commits two 100-frame blocks, so the second block's inference can
overlap the first block's CPU-process write. Every output block is compared to
the completed synchronous eight-GPU trial. No asynchronous one/eight-GPU result
is invented, and this module never launches full Day1-7 extraction.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import json
import math
from pathlib import Path
import signal
import sys

from . import coordinator
from .archive import _normalize_config, load_block
from .extension_scaling import compare_transition_chunks


WRITER_MODE = "cpu_process_overlap_v1"
ADDED_SOURCES = ("async_runner.py", "bounded_writer.py")
_BASE_WORKER_COMMAND = coordinator.worker_command
_BASE_WORKER_VALIDATOR = coordinator.validate_worker_report


def async_plan(root):
    return [dict(gpu_count=count, archive_root=str(Path(root)/"benchmarks"/f"async_scaling_n{count}"),
                 workers=[dict(worker_id=f"gpu{gpu}", gpu=gpu, start_frame=100+gpu*200,
                               end_frame=300+gpu*200) for gpu in range(count)]) for count in (2, 4)]


def validate_async_lineage(prior_config, current_config):
    """Only process-overlap fields and its two new source bindings may change."""
    before, after = copy.deepcopy(_normalize_config(prior_config)), copy.deepcopy(_normalize_config(current_config))
    if before.get("writer_mode") is not None or before.get("max_pending_blocks") is not None:
        raise ValueError("Prior reference must be the synchronous writer configuration")
    if after.pop("writer_mode", None) != WRITER_MODE or type(after.get("max_pending_blocks")) is not int or after.pop("max_pending_blocks") != 1:
        raise ValueError("Async configuration must use one bounded pending CPU-process block")
    old_sources, new_sources = before.get("extraction_source_sha256"), after.get("extraction_source_sha256")
    if not isinstance(old_sources, dict) or not isinstance(new_sources, dict):
        raise ValueError("Both configurations must bind extraction sources")
    if set(new_sources)-set(old_sources) != set(ADDED_SOURCES) or set(old_sources)-set(new_sources):
        raise ValueError("Async configuration must add exactly async_runner.py and bounded_writer.py")
    if any(new_sources[name] != digest for name, digest in old_sources.items()):
        raise ValueError("An existing model/input/attention extraction source changed")
    for name in ADDED_SOURCES:
        del new_sources[name]
    if before != after:
        raise ValueError("Async configuration changes model, input, attention or another semantic field")
    for name, digest in current_config["extraction_source_sha256"].items():
        if Path(name).name != name or not name.endswith(".py"):
            raise ValueError("Invalid extraction source filename")
        if coordinator.sha256_file(Path(__file__).with_name(name)) != digest:
            raise ValueError(f"Current extraction source differs from config: {name}")
    return dict(added_source_files=list(ADDED_SOURCES), changed_existing_source_files=[],
                model_input_attention_equal=True,
                added_pipeline_fields=dict(writer_mode=WRITER_MODE, max_pending_blocks=1))


def async_worker_command(*args, **kwargs):
    command = _BASE_WORKER_COMMAND(*args, **kwargs)
    index = command.index("-m")+1
    if command[index] != "rekv_qwen25.runner":
        raise ValueError("Unexpected original worker command module")
    command[index] = "rekv_qwen25.async_runner"
    return command


def validate_async_worker_report(report, worker, expected_config_id, *, benchmark=False):
    _BASE_WORKER_VALIDATOR(report, worker, expected_config_id, benchmark=False)
    if (report.get("writer_mode"), report.get("max_pending_blocks"), report.get("writer_drained"), report.get("writer_exitcode")) != (WRITER_MODE, 1, True, 0):
        raise ValueError("Async writer did not finish and acknowledge the bounded CPU-process pipeline")
    already_complete = (report.get("already_complete") is True and report.get("committed_frames") == 0
                        and report.get("resumed_frames") == worker["end_frame"]-worker["start_frame"])
    if not benchmark and already_complete:
        return report
    if (type(report.get("writer_pid")) is not int or report["writer_pid"] <= 0
            or type(report.get("parent_pid")) is not int or report["parent_pid"] <= 0
            or report["writer_pid"] == report["parent_pid"]):
        raise ValueError("Async report lacks its actual writer process")
    if report.get("writer_cuda_initialized") is not False:
        raise ValueError("Writer process did not demonstrate CPU-only operation")
    if not (type(report.get("max_pending_observed")) is int and 0 < report["max_pending_observed"] <= 1
            and type(report.get("max_inflight_blocks")) is int and 0 < report["max_inflight_blocks"] <= 2):
        raise ValueError("Async worker exceeded its bounded pending/in-flight block contract")
    if benchmark:
        start = worker["start_frame"]
        replay_start = max(0, start-620)
        if 17+replay_start*678 <= 15000:
            replay_start = 0
        if (worker["end_frame"]-start != 200 or report.get("resumed_frames") != 0
                or report.get("committed_frames") != 200 or report.get("warmup_frames") != start-replay_start):
            raise ValueError("Async benchmark requires fresh 200-frame output and the validated warmup")
        if len(report.get("commit_intervals", [])) != 2:
            raise ValueError("Async benchmark requires both real 100-frame commits")
        overlap = report.get("compute_write_overlap_seconds")
        if (not isinstance(overlap, (float, int)) or isinstance(overlap, bool) or not math.isfinite(overlap)
                or overlap <= 0 or report.get("overlapped_observation_calls", 0) < 1):
            raise ValueError("Async benchmark did not demonstrate actual observation/write overlap")
    return report


def commit_cadence(stage):
    """One real pipeline cycle per worker, separate from fill/drain-inclusive FPS."""
    workers = stage.get("workers", [])
    if len(workers) != stage["gpu_count"] or not workers:
        raise ValueError("Commit cadence requires every measured worker")
    observations, starts = [], []
    for worker in workers:
        intervals = sorted(worker.get("commit_intervals", []), key=lambda interval: interval["chunk_id"])
        if len(intervals) != 2 or intervals[1]["chunk_id"] != intervals[0]["chunk_id"]+1:
            raise ValueError("Commit cadence requires two consecutive durable receipts per worker")
        first, second = intervals[0]["finished_time"], intervals[1]["finished_time"]
        start = worker.get("measurement_started_time")
        if (not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                    for value in (first, second, start)) or not start <= first < second):
            raise ValueError("Invalid durable-receipt or measurement timestamps")
        starts.append(start)
        observations.append(dict(worker_id=worker["worker_id"], first_durable_receipt_time=first,
                                 second_durable_receipt_time=second, interval_seconds=second-first,
                                 newly_committed_frames=100, frames_per_second=100/(second-first)))
    longest = max(row["interval_seconds"] for row in observations)
    first_wave = max(row["first_durable_receipt_time"] for row in observations)
    second_wave = max(row["second_durable_receipt_time"] for row in observations)
    # If a fast worker's second receipt predates completion of the first wave,
    # it is not counted again in the global-wave numerator.
    global_new_frames = 100*sum(row["second_durable_receipt_time"] > first_wave for row in observations)
    return dict(status="measured", intervals_per_worker=1, workers=observations,
                max_worker_interval_seconds=longest,
                conservative_aggregate_fps=len(workers)*100/longest,
                first_wave_completion_time=first_wave, second_wave_completion_time=second_wave,
                measured_first_wave_fill_seconds=first_wave-min(starts),
                global_wave_seconds=second_wave-first_wave, global_wave_newly_committed_frames=global_new_frames,
                global_wave_aggregate_fps=global_new_frames/(second_wave-first_wave),
                basis="First-to-second durable receipt interval; conservative aggregate is GPUs*100/max worker interval",
                limitation="Only one observed cycle per worker; not a sustained-run measurement. Exact two-block end-to-end FPS remains separate")


@contextmanager
def _async_worker_overrides():
    """Change only this coordinator process; frozen module files stay intact."""
    command, validator = coordinator.worker_command, coordinator.validate_worker_report
    coordinator.worker_command = async_worker_command
    coordinator.validate_worker_report = validate_async_worker_report
    try:
        yield
    finally:
        coordinator.worker_command = command
        coordinator.validate_worker_report = validator


def prepare_async(root, config_path, prior_config_path, sync_report_path=None, initial_state_root=None):
    root, config_path, prior_config_path = Path(root).resolve(), Path(config_path).resolve(), Path(prior_config_path).resolve()
    sync_report_path = Path(sync_report_path or root/"benchmarks/extended_scaling_report.json").resolve()
    initial = Path(initial_state_root or root/"initial_state").resolve()
    current, validation = coordinator.validate_config(config_path, initial)
    prior_config = json.loads(prior_config_path.read_text())
    lineage = validate_async_lineage(prior_config, current)
    # The previous family's lineage and seven real output comparisons remain
    # mandatory; a status label by itself is insufficient.
    from .preflight import _scaling_evidence
    prior = _scaling_evidence(sync_report_path, prior_config, prior_config_path)
    if prior.get("status") != "complete" or prior.get("config_transition_fidelity", {}).get("status") != "passed":
        raise ValueError("Async benchmark requires the completed verified synchronous eight-GPU trial")
    references = [stage for stage in prior.get("stages", []) if stage.get("gpu_count") == 8]
    if len(references) != 1 or references[0].get("config_id") != coordinator.config_id(prior_config):
        raise ValueError("Synchronous eight-GPU reference uses a different configuration")
    if current["total_frames"] < 900:
        raise ValueError("Async benchmark requires visual frames through 899")
    from .runner import _validate_config
    from .state_io import unflatten_state
    tensors, metadata = load_block(initial, 0, current["initial_state_config_id"])
    canonical = unflatten_state(tensors, metadata["resume_state"])
    receipt = json.loads((initial/"receipts/chunk_00000000.ready.json").read_text())
    binding = {key: receipt[key] for key in ("config_id", "payload_sha256", "index_sha256")}
    _validate_config(current, canonical, binding)
    lineage.update(prior_config_path=str(prior_config_path), prior_config_id=coordinator.config_id(prior_config),
                   prior_config_file_sha256=coordinator.sha256_file(prior_config_path),
                   current_config_path=str(config_path), current_config_id=validation["config_id"],
                   current_config_file_sha256=coordinator.sha256_file(config_path),
                   prior_sync_report_path=str(sync_report_path), prior_sync_report_sha256=coordinator.sha256_file(sync_report_path))
    return dict(validation=validation, config_lineage=lineage, synchronous_reference_report=prior,
                synchronous_reference_stage=references[0], stages=async_plan(root), initial_state_root=str(initial))


def _check_anchors(lineage):
    for path_key, hash_key in (("prior_config_path", "prior_config_file_sha256"),
                               ("current_config_path", "current_config_file_sha256"),
                               ("prior_sync_report_path", "prior_sync_report_sha256")):
        if coordinator.sha256_file(lineage[path_key]) != lineage[hash_key]:
            raise ValueError("Async config or synchronous reference evidence changed during the trial")


def run_async_scaling(root, config_path, prior_config_path, sync_report_path=None,
                      initial_state_root=None, *, python_executable=None):
    prepared = prepare_async(root, config_path, prior_config_path, sync_report_path, initial_state_root)
    output = Path(root).resolve()/"benchmarks/async_scaling_report.json"
    if output.exists():
        raise ValueError(f"Refusing to overwrite prior asynchronous scaling evidence: {output}")
    for stage in prepared["stages"]:
        path = Path(stage["archive_root"])
        if path.exists() and any(path.iterdir()):
            raise ValueError(f"Refusing to overwrite prior asynchronous stage artifacts: {path}")
    summary = dict(schema="rekv_async_scaling_v1", status="running", pipeline_family=WRITER_MODE,
                   writer_mode=WRITER_MODE, max_pending_blocks=1, async_one_gpu_measured=False,
                   validation=prepared["validation"], config_lineage=prepared["config_lineage"],
                   synchronous_reference_report=copy.deepcopy(prepared["synchronous_reference_report"]),
                   stages=[], cross_pipeline_fidelity=dict(status="pending", comparisons=[]),
                   cross_gpu_fidelity=dict(status="pending", comparisons=[]),
                   user_authorization=dict(source="explicit_user_request", scope="bounded_cpu_writer_process_overlap",
                                           measured_gpu_counts=[2, 4], frames_per_gpu=200),
                   started_at=coordinator.utc_now(), full_run_started=False)
    coordinator.write_json(output, summary)
    try:
        for stage in prepared["stages"]:
            _check_anchors(prepared["config_lineage"])
            current = json.loads(Path(config_path).read_text())
            prior_config = json.loads(Path(prior_config_path).read_text())
            validate_async_lineage(prior_config, current)
            with _async_worker_overrides():
                result = coordinator.run_stage(stage, config_path, prepared["validation"]["config_id"],
                    prepared["initial_state_root"], python_executable=python_executable)
            result["pipeline_family"] = WRITER_MODE
            result["commit_cadence"] = commit_cadence(result)
            result["speedup_over_sync_eight_gpu"] = result["steady_fps"]/prepared["synchronous_reference_stage"]["steady_fps"]
            result["cadence_speedup_over_sync_eight_gpu"] = result["commit_cadence"]["conservative_aggregate_fps"]/prepared["synchronous_reference_stage"]["steady_fps"]
            coordinator.write_json(Path(result["archive_root"])/"report.json", result)
            coordinator.write_json(Path(result["archive_root"])/"status.json", result)
            summary["cross_pipeline_fidelity"]["comparisons"].extend(
                compare_transition_chunks(prepared["synchronous_reference_stage"], result))
            for previous in summary["stages"]:
                summary["cross_gpu_fidelity"]["comparisons"].extend(coordinator.compare_shared_chunks(previous, result))
            summary["stages"].append(result)
            coordinator.write_json(output, summary)
        _check_anchors(prepared["config_lineage"])
        if len(summary["cross_pipeline_fidelity"]["comparisons"]) != 12 or len(summary["cross_gpu_fidelity"]["comparisons"]) != 4:
            raise ValueError("Async trial lacks all twelve cross-pipeline and four cross-GPU chunk comparisons")
        summary["cross_pipeline_fidelity"]["status"] = "passed"
        summary["cross_gpu_fidelity"]["status"] = "passed"
        summary.update(status="complete", completed_at=coordinator.utc_now(),
                       fastest_stage_gpu_count=max(summary["stages"], key=lambda stage: stage["steady_fps"])["gpu_count"],
                       total_serialized_bytes=sum(stage["serialized_bytes"] for stage in summary["stages"]))
        coordinator.write_json(output, summary)
        return summary
    except BaseException as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}", updated_at=coordinator.utc_now())
        coordinator.write_json(output, summary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", required=True)
    parser.add_argument("--config", required=True, help="Frozen asynchronous writer configuration v3")
    parser.add_argument("--baseline-config", required=True, help="Preserved synchronous configuration v2")
    parser.add_argument("--synchronous-report")
    parser.add_argument("--initial-state-root")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"Async scaling coordinator received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    function = prepare_async if args.plan_only else run_async_scaling
    kwargs = {} if args.plan_only else {"python_executable": args.python}
    result = function(args.archive_root, args.config, args.baseline_config,
                      args.synchronous_report, args.initial_state_root, **kwargs)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
