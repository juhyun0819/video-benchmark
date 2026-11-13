"""Preflight must distinguish evidence, estimates, and unmeasured scaling."""

import copy
import json

import pytest

from rekv_qwen25.preflight import (
    REPO_ID, REVISION, _full_run_state, async_runtime_estimate, recommend_async_gpu_count,
    recommend_gpu_count, runtime_estimate,
    storage_estimate, telemetry_summary, validate_smokes,
)


def stage(count, fps):
    return dict(status="complete", gpu_count=count, steady_fps=fps)


def valid_smokes():
    checkpoint = dict(repo_id=REPO_ID, revision=REVISION)
    flags = (
        "instrumentation_kv_exact", "input_embeddings_exact", "position_ids_exact",
        "raw_vs_cached_positions_exact", "raw_vs_cached_postrope_kv_exact",
        "raw_vs_cached_prerope_kv_exact", "reload_exact",
        "disk_prerope_to_native_postrope_exact", "native_cache_reload_continuation_exact",
    )
    samples = [dict(frames=count, **dict.fromkeys(flags, True),
                    instrumentation_logits=dict(exact=True), raw_vs_cached_embeddings=dict(exact=True),
                    raw_vs_cached_logits=dict(exact=True)) for count in (1, 4, 8, 16)]
    passive = dict(status="gpu0_backbone_fidelity_complete", checkpoint=checkpoint,
                   samples=samples, visual_cache_reuse_single_image_exact=True,
                   qa_used=False, full_run_started=False)
    frames = []
    for frame in range(25):
        total = 17+(frame+1)*678
        frames.append(dict(frame=frame, cumulative_tokens=total, bounded_local_tokens=min(total, 15000),
                           all_raw_finite=True, hidden_finite=True, raw_vs_native=dict(exact=True),
                           hidden_vs_native=dict(exact=True), actual_causal_mask_exact=True))
    streaming = dict(status="gpu0_streaming_smoke_complete", checkpoint=checkpoint,
                     qa_used=False, full_run_started=False, n_init=17, n_local=15000,
                     tokens_per_frame=678, production_chunk_frames=1, transformers="4.57.3", torch="2.5.1+cu121",
                     streaming=dict(prefix_raw=dict(exact=True), prefix_logits=dict(exact=True), frames=frames,
                                    window_crossed=True, bounded_state_verified=True, resume_io=dict(all_tensors_exact=True),
                                    resume_continuations=[dict(raw_kv=dict(exact=True), hidden=dict(exact=True)) for _ in range(2)]),
                     chunk_experiments=[dict(chunk_frames=count, exact_vs_native_one_frame=count == 1) for count in (1, 4, 8, 16)])
    return passive, streaming


def test_exact_storage_geometry_and_units():
    value = storage_estimate()
    assert value["raw_kv_bytes_per_frame"] == 38_879_232
    assert value["raw_kv_bytes_per_100_frame_block"] == 3_887_923_200
    assert value["raw_kv_bytes_all_frames"] == 8_620_614_352_896
    assert value["positions_and_input_ids_bytes"] == 4_810_610_688
    assert value["raw_plus_positions_and_ids_bytes"] == 8_625_424_963_584


@pytest.mark.parametrize("stages,expected", [
    ([], None), ([stage(8, 100)], None), ([stage(1, 1)], 1),
    ([stage(1, 2), stage(2, 4), stage(4, 4.8)], 2),
    ([stage(1, 2), stage(2, 4), stage(4, 4.80001)], 4),
    ([stage(1, 5), stage(2, 4), stage(4, 4.9)], 1),
    ([stage(1, 2), stage(4, 10)], 1),
    ([stage(1, .674), stage(2, 1.078), stage(4, 1.465), stage(8, 1.7)], 4),
    ([stage(1, .674), stage(2, 1.078), stage(4, 1.465), stage(8, 1.9)], 8),
    ([stage(1, 1), stage(2, 2), stage(4, 3), stage(8, 3.6)], 4),
    ([stage(1, 1), stage(2, 2), stage(8, 10)], 2),
])
def test_recommendation_requires_measured_comparators_and_meaningful_gain(stages, expected):
    value = recommend_gpu_count(stages)
    assert value["gpu_count"] == expected
    assert value["eight_gpu_recommended"] is (expected == 8)
    if any(row["gpu_count"] == 8 for row in stages) and any(row["gpu_count"] == 4 for row in stages):
        fps = {row["gpu_count"]: row["steady_fps"] for row in stages}
        assert value["eight_over_four"] == fps[8]/fps[4]


def test_runtime_uses_end_to_end_rate_and_parallel_halo_startup():
    measured = dict(stage(4, 2.0), total_stage_seconds=700.0, coordinator_seconds_after_barrier=200.0,
                    workers=[dict(warmup_frames=200, warmup_seconds=200.0),
                             dict(warmup_frames=400, warmup_seconds=400.0)])
    halo = dict(halo_frames=620, verification_frames=4, replay_seconds=400.0)
    value = runtime_estimate(measured, halo)
    assert value["steady_seconds"] == 221728/2
    assert value["startup_fixed_seconds"] == 100
    assert value["estimated_startup_seconds"] == 720
    assert value["estimated_total_seconds"] == 221728/2+720
    measured["gpu_count"] = 1
    one = runtime_estimate(measured, halo)
    assert one["maximum_parallel_halo_frames"] == 0
    assert one["estimated_startup_seconds"] == 100


def test_all_native_and_streaming_fidelity_checks_required():
    passive, streaming = valid_smokes()
    assert validate_smokes(passive, streaming)["next_two_frames_exact"] is True
    broken = copy.deepcopy(streaming)
    broken["streaming"]["frames"][2]["actual_causal_mask_exact"] = False
    with pytest.raises(ValueError, match="mask equality"):
        validate_smokes(passive, broken)
    broken = copy.deepcopy(passive)
    broken["samples"][3]["disk_prerope_to_native_postrope_exact"] = False
    with pytest.raises(ValueError, match="equality"):
        validate_smokes(broken, streaming)
    broken = copy.deepcopy(streaming)
    broken["streaming"]["resume_continuations"][1]["raw_kv"]["exact"] = False
    with pytest.raises(ValueError, match="continuation"):
        validate_smokes(passive, broken)


def test_resource_summary_excludes_startup_and_labels_available_samples(tmp_path):
    logs = tmp_path/"logs"
    logs.mkdir()
    rows = [dict(time=0, cpu_rss=9999, gpu_csv="0, 99, 100"),
            dict(time=10, cpu_rss=100, gpu_csv="0, 20, 100"),
            dict(time=11, cpu_rss=200, gpu_csv="0, 40, 100"),
            dict(time=20, cpu_rss=9999, gpu_csv="0, 99, 100")]
    (logs/"worker_gpu0.telemetry.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    measured = dict(archive_root=str(tmp_path), workers=[dict(worker_id="gpu0", measurement_started_time=10, measurement_finished_time=11)])
    value = telemetry_summary(measured)
    assert value["workers"][0]["timed_samples"] == 2
    worker = value["workers"][0]
    assert worker["sampled_peak_cpu_rss_bytes"] == 200
    assert worker["gpu_utilization_available_sample_mean_percent"] == 30
    assert worker["whole_stage_gpu_utilization_mean_percent"] is None
    assert worker["measured_duration_seconds"] == 1
    assert worker["nominal_sample_count_coverage_fraction"] == 1.0
    assert worker["max_sampling_gap_including_interval_edges_seconds"] == 1
    assert "not a simultaneous host peak or a bound" in value["rss_aggregation_basis"]


def test_sparse_telemetry_exposes_commit_sized_gap_and_missing_coverage(tmp_path):
    logs = tmp_path/"logs"
    logs.mkdir()
    rows = [dict(time=100+i, cpu_rss=1000+i, gpu_csv="0, 71, 100") for i in range(92)]
    (logs/"worker_gpu0.telemetry.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    measured = dict(archive_root=str(tmp_path), workers=[dict(worker_id="gpu0", measurement_started_time=100, measurement_finished_time=373)])
    worker = telemetry_summary(measured)["workers"][0]
    assert worker["measured_duration_seconds"] == 273
    assert worker["expected_samples_at_nominal_interval"] == 273
    assert worker["nominal_sample_count_coverage_fraction"] == pytest.approx(92/273)
    assert worker["max_inter_sample_gap_seconds"] == 1
    assert worker["max_sampling_gap_including_interval_edges_seconds"] == 182
    assert worker["gpu_utilization_available_sample_mean_percent"] == 71
    assert worker["whole_stage_gpu_utilization_mean_percent"] is None
    assert "not whole-stage" in worker["gpu_utilization_mean_basis"]
    assert "true peak" in worker["cpu_peak_basis"]


def test_no_timed_samples_does_not_invent_utilization_or_observed_memory(tmp_path):
    measured = dict(archive_root=str(tmp_path), workers=[dict(worker_id="gpu0", measurement_started_time=0, measurement_finished_time=20)])
    worker = telemetry_summary(measured)["workers"][0]
    assert worker["timed_samples"] == 0
    assert worker["nominal_sample_count_coverage_fraction"] == 0
    assert worker["max_sampling_gap_including_interval_edges_seconds"] == 20
    assert worker["sampled_peak_cpu_rss_bytes"] is None
    assert worker["gpu_utilization_available_sample_mean_percent"] is None


def test_report_cannot_claim_not_started_after_a_supervisor_launch(tmp_path):
    assert _full_run_state(tmp_path)[0] is False
    (tmp_path/"logs").mkdir()
    (tmp_path/"logs/full-plan.fixture.json").write_text("{}")
    assert _full_run_state(tmp_path)[0] is True


def async_stage(count, end_to_end, cadence):
    return dict(stage(count, end_to_end), commit_cadence=dict(conservative_aggregate_fps=cadence))


@pytest.mark.parametrize("four_cadence,expected", [(2.4, 2), (2.40001, 4)])
def test_async_choice_uses_measured_cadence_not_two_block_fill_tax_or_sync_fallback(four_cadence, expected):
    stages = [async_stage(2, 1.0, 2.0), async_stage(4, .9, four_cadence)]
    choice = recommend_async_gpu_count(stages, stage(8, 1.7))
    assert choice["gpu_count"] == expected
    assert choice["four_over_two_end_to_end"] == .9
    assert choice["four_over_two_cadence"] == four_cadence/2
    assert choice["async_one_gpu_measured"] is False
    assert choice["async_eight_gpu_measured"] is False
    assert choice["comparison_to_synchronous_eight"]["synchronous_eight_fps"] == 1.7


def test_async_runtime_counts_startup_and_first_wave_fill_once_then_remaining_cycles():
    measured = dict(stage(2, 1.0), total_stage_seconds=300, coordinator_seconds_after_barrier=200,
                    workers=[dict(warmup_frames=100, warmup_seconds=80)],
                    commit_cadence=dict(conservative_aggregate_fps=4.0,
                                        max_worker_interval_seconds=50.0,
                                        measured_first_wave_fill_seconds=100.0))
    halo = dict(halo_frames=620, verification_frames=4, replay_seconds=400)
    estimate = async_runtime_estimate(measured, halo, total_frames=1000)
    # Startup=20 + 620*0.8, first wave=100, then four 50-second cycles.
    assert estimate["estimated_startup_seconds"] == 516
    assert estimate["estimated_total_seconds"] == 816
    assert estimate["maximum_blocks_per_worker"] == 5
    assert estimate["conservative_end_to_end_estimate"]["estimated_total_seconds"] == 1516
