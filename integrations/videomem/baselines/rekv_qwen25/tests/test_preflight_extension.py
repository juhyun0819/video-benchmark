"""Prior scaling reuse requires reader-only lineage and real committed equality."""

import copy
import json
from pathlib import Path

import pytest
import torch

from rekv_qwen25.archive import KVArchiveWriter
from rekv_qwen25.coordinator import compare_shared_chunks, config_id
from rekv_qwen25.extension_scaling import compare_transition_chunks, validate_lineage
from rekv_qwen25.async_scaling import WRITER_MODE, commit_cadence, validate_async_lineage
from rekv_qwen25.preflight import SOURCE_FILES, _scaling_evidence, recommend_async_gpu_count, recommend_gpu_count, sha256
import rekv_qwen25.preflight as preflight_module


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+"\n")


def write_stage(root, config, count, *, alter_first=False):
    writer = KVArchiveWriter(root, config)
    workers = []
    for gpu in range(count):
        chunk = gpu+1
        first = chunk*100
        value = float(chunk)+(0.5 if alter_first and chunk == 1 else 0.0)
        writer.commit(chunk, {"layer_00.key": torch.full((1, 4, 1, 128), value, dtype=torch.bfloat16),
                              "position_ids": torch.arange(3, dtype=torch.int64).reshape(3, 1, 1)},
                      dict(frame_ids=list(range(first, first+100)), timestamps=[float(i) for i in range(first, first+100)],
                           config_id=writer.config_id, scope="tiny real archive equality fixture"))
        workers.append(dict(worker_id=f"gpu{gpu}", gpu=gpu, start_frame=first, end_frame=first+100,
                            committed_frames=100))
    return dict(status="complete", gpu_count=count, config_id=writer.config_id, archive_root=str(root),
                committed_frames=count*100, workers=workers, steady_fps={1:.674, 2:1.078, 4:1.465, 8:1.9}[count])


@pytest.fixture
def extension(tmp_path):
    package = Path(preflight_module.__file__).parent
    current = dict(n_local=15000, dtype="bfloat16",
                   extraction_source_sha256={name: sha256(package/name) for name in SOURCE_FILES})
    previous = copy.deepcopy(current)
    previous["extraction_source_sha256"]["inputs.py"] = "0"*64
    old_path, current_path, prior_path, output_path = [tmp_path/name for name in (
        "config.v1.json", "config.v2.json", "scaling.json", "extended.json")]
    write(old_path, previous)
    write(current_path, current)
    old_stages = [write_stage(tmp_path/f"old_n{count}", previous, count) for count in (1, 2, 4)]
    new_stage = write_stage(tmp_path/"new_n8", current, 8)
    cross = []
    for left, right in ((0, 1), (0, 2), (1, 2)):
        cross.extend(compare_shared_chunks(old_stages[left], old_stages[right]))
    prior = dict(status="complete", validation=dict(config_id=config_id(previous), config_file_sha256=sha256(old_path)),
                 stages=old_stages, cross_gpu_fidelity=dict(status="passed", comparisons=cross))
    write(prior_path, prior)
    lineage = validate_lineage(previous, current)
    lineage.update(prior_config_path=str(old_path), prior_config_id=config_id(previous),
                   prior_config_file_sha256=sha256(old_path), current_config_path=str(current_path),
                   current_config_id=config_id(current), current_config_file_sha256=sha256(current_path),
                   prior_scaling_report_path=str(prior_path), prior_scaling_report_sha256=sha256(prior_path))
    transition = []
    for old in old_stages:
        transition.extend(compare_transition_chunks(old, new_stage))
    report = dict(status="complete", validation=dict(config_id=config_id(current), config_file_sha256=sha256(current_path)),
                  config_lineage=lineage, original_scaling_report=copy.deepcopy(prior),
                  stages=copy.deepcopy(old_stages)+[new_stage], cross_gpu_fidelity=copy.deepcopy(prior["cross_gpu_fidelity"]),
                  config_transition_fidelity=dict(status="passed", comparisons=transition),
                  user_authorization=dict(requested_gpu_count=8, frames_per_gpu=100, source="explicit_user_request",
                                          scope="one_additional_scaling_trial"))
    write(output_path, report)
    return current, current_path, output_path, report


def test_reader_only_lineage_and_actual_equal_payloads_allow_measured_eight(extension):
    current, current_path, report_path, _ = extension
    result = _scaling_evidence(report_path, current, current_path)
    assert len(result["config_transition_fidelity"]["comparisons"]) == 7
    recommendation = recommend_gpu_count(result["stages"])
    assert recommendation["gpu_count"] == 8
    assert recommendation["eight_over_four"] == pytest.approx(1.9/1.465)


def test_changed_prior_measurements_cannot_be_silently_reused(extension):
    current, current_path, report_path, report = extension
    report["stages"][0]["steady_fps"] *= 2
    write(report_path, report)
    with pytest.raises(ValueError, match="stage objects"):
        _scaling_evidence(report_path, current, current_path)


def test_false_pass_flags_cannot_hide_actual_different_output_tensors(extension, tmp_path):
    current, current_path, report_path, report = extension
    different = write_stage(tmp_path/"different_n8", current, 8, alter_first=True)
    report["stages"][-1] = different
    for comparison in report["config_transition_fidelity"]["comparisons"]:
        comparison["candidate_archive_root"] = different["archive_root"]
    write(report_path, report)
    with pytest.raises(ValueError, match="tensor values/geometry"):
        _scaling_evidence(report_path, current, current_path)


def test_semantic_change_rejected_even_if_new_file_hashes_are_updated(extension):
    current, current_path, report_path, report = extension
    current["n_local"] = 14999
    write(current_path, current)
    report["validation"].update(config_id=config_id(current), config_file_sha256=sha256(current_path))
    report["config_lineage"].update(current_config_id=config_id(current), current_config_file_sha256=sha256(current_path))
    write(report_path, report)
    with pytest.raises(ValueError, match="semantic"):
        _scaling_evidence(report_path, current, current_path)


@pytest.mark.parametrize("missing", ["user_authorization", "config_transition_fidelity"])
def test_eight_trial_requires_authorization_and_transition_proof(extension, missing):
    current, current_path, report_path, report = extension
    del report[missing]
    write(report_path, report)
    with pytest.raises(ValueError):
        _scaling_evidence(report_path, current, current_path)


def test_running_extension_retains_original_measurements_without_claiming_transition_pass(extension):
    current, current_path, report_path, report = extension
    report["status"] = "running"
    report["stages"] = report["stages"][:3]
    report["config_transition_fidelity"] = dict(status="pending", comparisons=[])
    write(report_path, report)
    result = _scaling_evidence(report_path, current, current_path)
    assert result["status"] == "running"
    assert result["config_transition_fidelity"]["status"] == "pending"
    assert recommend_gpu_count(result["stages"])["gpu_count"] == 4


def write_async_stage(root, config, count):
    writer = KVArchiveWriter(root, config)
    workers, total_bytes = [], 0
    for gpu in range(count):
        start = 100+gpu*200
        intervals = []
        for offset in range(2):
            chunk = start//100+offset
            first = chunk*100
            receipt = writer.commit(chunk, {"layer_00.key": torch.full((1, 4, 1, 128), float(chunk), dtype=torch.bfloat16),
                                           "position_ids": torch.arange(3, dtype=torch.int64).reshape(3, 1, 1)},
                                   dict(frame_ids=list(range(first, first+100)), timestamps=[float(i) for i in range(first, first+100)],
                                        config_id=writer.config_id, scope="tiny real archive equality fixture"))
            finished = 100+gpu*3 if offset == 0 else 180+gpu*5
            intervals.append(dict(chunk_id=chunk, started_time=finished-40, finished_time=finished,
                                  seconds=40, serialized_bytes=receipt["serialized_bytes"]))
        size = sum(row["serialized_bytes"] for row in intervals)
        total_bytes += size
        replay_start = max(0, start-620)
        if 17+replay_start*678 <= 15000:
            replay_start = 0
        workers.append(dict(status="complete", worker_id=f"gpu{gpu}", gpu=gpu, start_frame=start, end_frame=start+200,
                            config_id=writer.config_id, committed_frames=200, resumed_frames=0, warmup_frames=start-replay_start,
                            measured_seconds=intervals[-1]["finished_time"]+5, measurement_started_time=0,
                            measurement_finished_time=intervals[-1]["finished_time"]+5,
                            read_seconds=1, infer_seconds=100, atomic_nas_seconds=80, serialized_bytes=size,
                            peak_vram_bytes=1000, writer_mode=WRITER_MODE, max_pending_blocks=1,
                            writer_drained=True, writer_exitcode=0, writer_pid=2000+gpu, parent_pid=1000+gpu,
                            writer_cuda_initialized=False, max_pending_observed=1, max_inflight_blocks=2,
                            compute_write_overlap_seconds=10, overlapped_observation_calls=15,
                            commit_intervals=intervals))
    result = dict(status="complete", gpu_count=count, config_id=writer.config_id, archive_root=str(root),
                  committed_frames=count*200, serialized_bytes=total_bytes, workers=workers,
                  steady_fps=count*200/max(row["measured_seconds"] for row in workers), pipeline_family=WRITER_MODE)
    result["commit_cadence"] = commit_cadence(result)
    return result


@pytest.fixture
def asynchronous(extension, tmp_path):
    synchronous, sync_config_path, sync_report_path, sync_report = extension
    current = copy.deepcopy(synchronous)
    current.update(writer_mode=WRITER_MODE, max_pending_blocks=1)
    package = Path(preflight_module.__file__).parent
    current["extraction_source_sha256"].update({name: sha256(package/name) for name in ("async_runner.py", "bounded_writer.py")})
    config_path, report_path = tmp_path/"config.v3.json", tmp_path/"async.json"
    write(config_path, current)
    lineage = validate_async_lineage(synchronous, current)
    lineage.update(prior_config_path=str(sync_config_path), prior_config_id=config_id(synchronous),
                   prior_config_file_sha256=sha256(sync_config_path), current_config_path=str(config_path),
                   current_config_id=config_id(current), current_config_file_sha256=sha256(config_path),
                   prior_sync_report_path=str(sync_report_path), prior_sync_report_sha256=sha256(sync_report_path))
    stages = [write_async_stage(tmp_path/f"async_n{count}", current, count) for count in (2, 4)]
    cross_pipeline = []
    for stage in stages:
        cross_pipeline.extend(compare_transition_chunks(sync_report["stages"][-1], stage))
    report = dict(schema="rekv_async_scaling_v1", status="complete", pipeline_family=WRITER_MODE,
                  writer_mode=WRITER_MODE, max_pending_blocks=1, async_one_gpu_measured=False,
                  validation=dict(config_id=config_id(current), config_file_sha256=sha256(config_path)),
                  config_lineage=lineage, synchronous_reference_report=sync_report, stages=stages,
                  cross_pipeline_fidelity=dict(status="passed", comparisons=cross_pipeline),
                  cross_gpu_fidelity=dict(status="passed", comparisons=compare_shared_chunks(stages[0], stages[1])),
                  user_authorization=dict(source="explicit_user_request", scope="bounded_cpu_writer_process_overlap",
                                          measured_gpu_counts=[2, 4], frames_per_gpu=200))
    write(report_path, report)
    return current, config_path, report_path, report


def test_async_report_rechecks_all_actual_pipeline_hashes_and_receipt_cadence(asynchronous):
    current, config_path, report_path, _ = asynchronous
    result = _scaling_evidence(report_path, current, config_path)
    assert len(result["cross_pipeline_fidelity"]["comparisons"]) == 12
    assert len(result["cross_gpu_fidelity"]["comparisons"]) == 4
    assert recommend_async_gpu_count(result["stages"])["gpu_count"] == 4


@pytest.mark.parametrize("mutation", [
    lambda row: row["stages"][0]["commit_cadence"].update(conservative_aggregate_fps=1000),
    lambda row: row["stages"][0]["workers"][0].update(compute_write_overlap_seconds=0),
    lambda row: row["stages"][0]["workers"][0].update(writer_cuda_initialized=True),
    lambda row: row["cross_pipeline_fidelity"].update(comparisons=[]),
])
def test_async_report_rejects_fabricated_cadence_or_missing_process_overlap_proof(asynchronous, mutation):
    current, config_path, report_path, report = asynchronous
    mutation(report)
    write(report_path, report)
    with pytest.raises(ValueError):
        _scaling_evidence(report_path, current, config_path)
