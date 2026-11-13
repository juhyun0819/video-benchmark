"""Freeze verified extraction inputs and render the required 17-point preflight.

This command never loads a model on a GPU or launches workers. Prepare the
config only after extraction sources are final. A missing scaling report is
reported as pending; it cannot authorize a multi-GPU recommendation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

import torch
import transformers

from .archive import load_block
from .coordinator import config_id, write_json
from .inputs import REPO_ID, REVISION, _header, validate_checkpoint
from .runner import _validate_config
from .smoke import FEATURES, LEDGER, MODEL, ROOT
from .state_io import unflatten_state


TOTAL_FRAMES = 221728
TOKENS_PER_FRAME = 678
SOURCE_FILES = (
    "attention.py", "stream.py", "inputs.py", "state_io.py", "runner.py",
    "smoke.py", "halo_smoke.py",
)
PREFIX_PROTOCOL = "separate_fixed_prefix_then_one_frame"
ASYNC_WRITER_MODE = "cpu_process_overlap_v1"
ASYNC_SOURCE_FILES = ("async_runner.py", "bounded_writer.py")


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_smokes(passive, streaming):
    require(passive.get("status") == "gpu0_backbone_fidelity_complete", "Native fidelity smoke is incomplete")
    require(streaming.get("status") == "gpu0_streaming_smoke_complete", "Streaming smoke is incomplete")
    for report in (passive, streaming):
        checkpoint = report.get("checkpoint", {})
        require((checkpoint.get("repo_id"), checkpoint.get("revision")) == (REPO_ID, REVISION),
                "Smoke checkpoint/revision mismatch")
        require(report.get("qa_used") is False and report.get("full_run_started") is False,
                "Unexpected QA or full-run scope in smoke evidence")
    samples = passive.get("samples", [])
    require([sample.get("frames") for sample in samples] == [1, 4, 8, 16], "Native smoke lacks all 1/4/8/16-frame samples")
    flags = (
        "instrumentation_kv_exact", "input_embeddings_exact", "position_ids_exact",
        "raw_vs_cached_positions_exact", "raw_vs_cached_postrope_kv_exact",
        "raw_vs_cached_prerope_kv_exact", "reload_exact",
        "disk_prerope_to_native_postrope_exact", "native_cache_reload_continuation_exact",
    )
    for sample in samples:
        require(all(sample.get(key) is True for key in flags), "A native/raw/cache/disk equality check failed")
        require(all(sample.get(key, {}).get("exact") is True for key in (
            "instrumentation_logits", "raw_vs_cached_embeddings", "raw_vs_cached_logits")),
            "Native smoke embedding/logit equality failed")
    require(passive.get("visual_cache_reuse_single_image_exact") is True, "Visual cache reuse was not verified")
    require((streaming.get("n_init"), streaming.get("n_local"), streaming.get("tokens_per_frame"),
             streaming.get("production_chunk_frames")) == (17, 15000, 678, 1), "Streaming protocol mismatch")
    require((streaming.get("transformers"), streaming.get("torch")) == ("4.57.3", "2.5.1+cu121"),
            "Streaming runtime mismatch")
    evidence = streaming.get("streaming", {})
    require(evidence.get("prefix_raw", {}).get("exact") is True and
            evidence.get("prefix_logits", {}).get("exact") is True, "Prefix native equality failed")
    frames = evidence.get("frames", [])
    require(len(frames) >= 25 and [row.get("frame") for row in frames] == list(range(len(frames))),
            "Streaming smoke lacks 25 continuous frames")
    for row in frames:
        require(row.get("all_raw_finite") is True and row.get("hidden_finite") is True,
                "Streaming smoke contains non-finite values")
        require(row.get("bounded_local_tokens") == min(row["cumulative_tokens"], 15000),
                "Streaming local state was not bounded")
    for row in frames[:16]:
        require(row.get("raw_vs_native", {}).get("exact") is True and
                row.get("hidden_vs_native", {}).get("exact") is True and
                row.get("actual_causal_mask_exact") is True, "Native one-frame/mask equality failed")
    require(evidence.get("window_crossed") is True and evidence.get("bounded_state_verified") is True,
            "True window boundary was not verified")
    require(evidence.get("resume_io", {}).get("all_tensors_exact") is True, "Atomic resume tensor equality failed")
    continuations = evidence.get("resume_continuations", [])
    require(len(continuations) >= 2 and all(row.get("raw_kv", {}).get("exact") is True and
            row.get("hidden", {}).get("exact") is True for row in continuations), "Disk continuation equality failed")
    chunks = {row.get("chunk_frames"): row for row in streaming.get("chunk_experiments", [])}
    require(set(chunks) == {1, 4, 8, 16} and chunks[1].get("exact_vs_native_one_frame") is True,
            "Chunk experiments or one-frame reproducibility are incomplete")
    return dict(native_samples=[1, 4, 8, 16], native_one_frame_checks=16,
                real_window_frames=len(frames), raw_and_native_disk_roundtrip_exact=True,
                next_two_frames_exact=True, qa_used=False, production_chunk_frames=1,
                larger_chunk_exact={str(n): chunks[n]["exact_vs_native_one_frame"] for n in (4, 8, 16)})


def _canonical_initial(root, attention_hash):
    tensors, metadata = load_block(root, 0)
    state = unflatten_state(tensors, metadata["resume_state"])
    initial_config = read_json(root/"config.json")
    receipt = read_json(root/"receipts/chunk_00000000.ready.json")
    binding = {key: receipt[key] for key in ("config_id", "payload_sha256", "index_sha256")}
    require(config_id(initial_config) == binding["config_id"], "Initial artifact config hash mismatch")
    require((initial_config.get("input_protocol"), initial_config.get("attention_sha256")) ==
            (PREFIX_PROTOCOL, attention_hash), "Canonical prefix uses different code/protocol")
    require((state.get("schema"), state.get("n_init"), state.get("n_local"), state.get("layers"),
             state.get("total_seen")) == (1, 17, 15000, 28, 17), "Canonical prefix geometry mismatch")
    require(initial_config.get("policy") == state.get("policy"), "Canonical prefix policy mismatch")
    require(len(state.get("states", [])) == len(state.get("geometry", [])) == 28, "Canonical prefix layer coverage mismatch")
    native = state.get("native_post_rope_cache")
    require(isinstance(native, (list, tuple)) and len(native) == 28, "Canonical native prefix cache is missing")
    positions = torch.arange(17, dtype=torch.int64).reshape(1, 1, 17).expand(3, 1, 17)
    for index, (layer, geometry) in enumerate(zip(state["states"], state["geometry"])):
        require((geometry.get("query_heads"), geometry.get("kv_heads"), geometry.get("head_dim"),
                 geometry.get("dtype"), geometry.get("mrope_section")) ==
                (28, 4, 128, "torch.bfloat16", [16, 24, 24]), "Canonical Qwen attention geometry mismatch")
        require(torch.equal(layer["positions"], positions), "Canonical prefix positions differ from native text positions")
        for kind in ("key", "value"):
            for name in (kind, f"init_{kind}"):
                tensor = layer[name]
                require(tuple(tensor.shape) == (1, 4, 17, 128) and tensor.dtype == torch.bfloat16
                        and bool(torch.isfinite(tensor).all()), "Invalid canonical prefix tensor")
            require(torch.equal(layer[kind], layer[f"init_{kind}"]), "Initial sinks differ from the original prefix")
        require(len(native[index]) == 2 and all(tuple(t.shape) == (1, 4, 17, 128) and t.dtype == torch.bfloat16
                and bool(torch.isfinite(t).all()) for t in native[index]), "Invalid native post-RoPE prefix cache")
        require(torch.equal(native[index][1], layer["value"]), "Native/raw prefix value mismatch")
    return state, initial_config, binding


def _checked_halo(report, initial, attention_hash):
    require((report.get("status"), report.get("halo_frames"), report.get("adaptation_sha256")) ==
            ("passed", 620, attention_hash), "A passed 620-frame proof for the current attention source is required")
    require((report.get("n_init"), report.get("n_local"), report.get("chunk_frames")) == (17, 15000, 1),
            "Halo protocol mismatch")
    require(report.get("policy") == initial["policy"] and report.get("upstream") == initial["upstream"],
            "Halo and canonical prefix policies differ")
    require(report.get("absolute_positions") is True and report.get("fixed_original_per_layer_sinks") is True,
            "Halo did not preserve positions and original sinks")
    checkpoint = report.get("checkpoint", {})
    require((checkpoint.get("repo_id"), checkpoint.get("revision")) == (REPO_ID, REVISION), "Halo checkpoint mismatch")
    boundary = report.get("boundary_frame")
    require(type(boundary) is int and boundary >= 0, "Halo boundary is missing")
    verified = [row for row in report.get("checks", []) if row.get("frame_id", -1) >= boundary+620]
    require(len(verified) == report.get("verification_frames") and len(verified) > 0,
            "Halo output comparisons are incomplete")
    require(all(row.get("all_kv_exact") is True and row.get("logits", {}).get("exact") is True and
                len(row.get("exact_layers", [])) == 28 and all(value is True for value in row["exact_layers"])
                for row in verified), "Halo KV/logit equality failed")
    require(any(row.get("frame_id") == boundary+620 and row.get("bounded_state_exact") is True for row in verified),
            "Halo full-state equality was not verified")


def evidence_paths(args):
    root = Path(args.root).resolve()
    return dict(root=root, config=Path(args.config or root/"preflight-config.json").resolve(),
                initial=Path(args.initial_state_root or root/"initial_state").resolve(),
                halo=Path(args.halo_report or root/"benchmarks/gpu0_halo_v1/report.json").resolve(),
                passive=Path(args.passive_report or root/"benchmarks/gpu0_fidelity_v1/report.json").resolve(),
                streaming=Path(args.streaming_report or root/"benchmarks/gpu0_streaming_v2/report.json").resolve(),
                scaling=Path(args.scaling_report or root/"benchmarks/scaling_report.json").resolve())


def _full_run_state(root):
    status_path = root/"logs/full-status.json"
    status = read_json(status_path) if status_path.exists() else None
    launched_plan = next((root/"logs").glob("full-plan.*.json"), None)
    started = bool((status or {}).get("full_run_started")) or launched_plan is not None
    manifest_path = root/"manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    return started, dict(supervisor_status_path=str(status_path), supervisor_status=status,
                         launched_plan_path=str(launched_plan) if launched_plan is not None else None,
                         manifest_path=str(manifest_path), manifest_config_id=manifest.get("config_id"),
                         committed_frames=manifest.get("continuity", {}).get("committed_frame_count"))


def prepare_config(args):
    paths = evidence_paths(args)
    require(transformers.__version__ == "4.57.3" and str(torch.__version__) == "2.5.1+cu121",
            "Prepare with the pinned Transformers 4.57.3 / Torch 2.5.1+cu121 environment")
    passive, streaming = read_json(paths["passive"]), read_json(paths["streaming"])
    validate_smokes(passive, streaming)
    attention_hash = sha256(Path(__file__).with_name("attention.py"))
    initial, initial_config, binding = _canonical_initial(paths["initial"], attention_hash)
    halo = read_json(paths["halo"])
    _checked_halo(halo, initial, attention_hash)
    identity = validate_checkpoint(Path(args.model), Path(args.ledger))
    for source in (halo["checkpoint"], passive["checkpoint"], streaming["checkpoint"], initial_config["backbone"]):
        for key in ("repo_id", "revision", "config_sha256", "index_sha256", "preprocessor_sha256"):
            require(source.get(key) == identity.get(key), f"Evidence checkpoint identity differs: {key}")
    header = _header(Path(args.features))
    require(header["inventory"]["total_1fps_frames"] == TOTAL_FRAMES, "Unexpected Day1-7 frame count")
    require(header["inventory"]["fps"] == 1.0, "Unexpected visual sampling rate")
    previous = halo.get("initial_state_binding")
    if previous is not None:
        require(all(previous.get(key) == value for key, value in binding.items()), "Halo binds a different initial artifact")
    else:
        # The running proof predates this field. Its code created this artifact
        # from the same saved prefix it used for both streams. Record the audit
        # explicitly rather than suggesting the original report already bound it.
        halo["initial_state_binding"] = dict(binding,
            recorded_by="postflight audit of canonical artifact created by this halo run",
            recorded_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(paths["halo"], halo)
    config = dict(
        n_init=17, n_local=15000, chunk_frames=1, frames_per_block=100, total_frames=TOTAL_FRAMES,
        prefix_protocol=PREFIX_PROTOCOL, backbone=identity, feature_archive_id=header["archive_id"],
        initial_state_config_id=binding["config_id"], initial_state_payload_sha256=binding["payload_sha256"],
        initial_state_index_sha256=binding["index_sha256"], policy=initial["policy"],
        key_representation="pre_rope", dtype="bfloat16", transformers_version="4.57.3", torch_version="2.5.1+cu121",
        halo_validation=dict(status="passed", report_path=str(paths["halo"]), report_sha256=sha256(paths["halo"]),
                             halo_frames=620, attention_sha256=attention_hash),
        extraction_source_sha256={name: sha256(Path(__file__).with_name(name)) for name in SOURCE_FILES},
        fidelity_validation=dict(passive_report_path=str(paths["passive"]), passive_report_sha256=sha256(paths["passive"]),
                                 streaming_report_path=str(paths["streaming"]), streaming_report_sha256=sha256(paths["streaming"])),
    )
    if getattr(args, "writer_mode", None) is not None:
        require(args.writer_mode == ASYNC_WRITER_MODE, "Unsupported writer mode")
        config.update(writer_mode=ASYNC_WRITER_MODE, max_pending_blocks=1)
        config["extraction_source_sha256"].update({name: sha256(Path(__file__).with_name(name)) for name in ASYNC_SOURCE_FILES})
    _validate_config(config, initial, binding)
    if paths["config"].exists():
        require(read_json(paths["config"]) == config, "Frozen config already exists with different values; do not reinterpret it")
    else:
        write_json(paths["config"], config)
    started, _ = _full_run_state(paths["root"])
    return dict(status="prepared", config_path=str(paths["config"]), config_id=config_id(config),
                config_file_sha256=sha256(paths["config"]), full_run_started=started)


def storage_estimate(total_frames=TOTAL_FRAMES):
    frame = 28*2*4*128*2*TOKENS_PER_FRAME
    raw = frame*total_frames
    positions_and_ids = total_frames*TOKENS_PER_FRAME*4*8
    return dict(bytes_per_token=28*2*4*128*2, raw_kv_bytes_per_frame=frame,
                raw_kv_bytes_per_100_frame_block=frame*100, raw_kv_bytes_all_frames=raw,
                positions_and_input_ids_bytes=positions_and_ids,
                raw_plus_positions_and_ids_bytes=raw+positions_and_ids,
                initial_raw_prefix_bytes=17*28*2*4*128*2,
                budget_decimal_TB=8.63,
                exclusions="indices/receipts, canonical prefix artifact, benchmark artifacts, existing features and model weights")


def recommend_gpu_count(stages, minimum_gain=1.2):
    require(isinstance(minimum_gain, (int, float)) and not isinstance(minimum_gain, bool)
            and math.isfinite(minimum_gain) and minimum_gain > 1, "Minimum scaling gain must exceed one")
    measured = {row["gpu_count"]: row for row in stages if row.get("status") == "complete"
                and row.get("gpu_count") in (1, 2, 4, 8) and isinstance(row.get("steady_fps"), (int, float))
                and not isinstance(row["steady_fps"], bool)
                and math.isfinite(row["steady_fps"]) and row["steady_fps"] > 0}
    candidates = [count for count in (1, 2) if count in measured]
    gain = measured[4]["steady_fps"]/measured[2]["steady_fps"] if 4 in measured and 2 in measured else None
    eight_gain = measured[8]["steady_fps"]/measured[4]["steady_fps"] if 8 in measured and 4 in measured else None
    if gain is not None and gain > minimum_gain:
        candidates.append(4)
    if eight_gain is not None and eight_gain > minimum_gain:
        candidates.append(8)
    selected = max(candidates, key=lambda count: measured[count]["steady_fps"]) if candidates else None
    return dict(gpu_count=selected, four_over_two=gain, eight_over_four=eight_gain,
                minimum_measured_gain=minimum_gain, measured_gpu_counts=sorted(measured),
                rule=f"4 GPUs eligible only above {minimum_gain:g}x measured 2-GPU FPS; explicitly measured 8 GPUs eligible only above {minimum_gain:g}x measured 4-GPU FPS; choose fastest eligible measured stage",
                eight_gpu_measured=8 in measured, eight_gpu_recommended=selected == 8,
                reason="scaling measurements pending" if selected is None else "selected from measured end-to-end throughput including NAS writes")


def runtime_estimate(stage, halo, total_frames=TOTAL_FRAMES):
    fps = stage["steady_fps"]
    require(math.isfinite(fps) and fps > 0, "Invalid aggregate throughput")
    workers = stage.get("workers", [])
    observed_startup = max(0.0, stage.get("total_stage_seconds", 0.0)-stage.get("coordinator_seconds_after_barrier", 0.0))
    max_warmup = max((worker.get("warmup_seconds", 0.0) for worker in workers), default=0.0)
    fixed_startup = max(0.0, observed_startup-max_warmup)
    warmup_per_frame = max((worker.get("warmup_seconds", 0.0)/worker["warmup_frames"]
                           for worker in workers if worker.get("warmup_frames", 0) > 0), default=0.0)
    replay_count = halo["halo_frames"]+halo["verification_frames"]
    warmup_per_frame = max(warmup_per_frame, halo.get("replay_seconds", 0.0)/replay_count)
    halo_frames = 620 if stage["gpu_count"] > 1 else 0
    startup = fixed_startup+halo_frames*warmup_per_frame
    steady = total_frames/fps
    return dict(aggregate_fps=fps, steady_seconds=steady, estimated_startup_seconds=startup,
                startup_fixed_seconds=fixed_startup, maximum_parallel_halo_frames=halo_frames,
                halo_seconds_per_frame_basis=warmup_per_frame, estimated_total_seconds=steady+startup,
                estimated_total_hours=(steady+startup)/3600,
                basis="Measured aggregate FPS includes reads, inference, CPU capture and atomic NAS writes; startup extrapolated from measured load/replay timing",
                uncertainty="Short benchmark extrapolation; source-producer waits, competing NAS traffic, retries and failures are not included")


def recommend_async_gpu_count(stages, synchronous_eight=None, minimum_gain=1.2):
    require(isinstance(minimum_gain, (int, float)) and not isinstance(minimum_gain, bool)
            and math.isfinite(minimum_gain) and minimum_gain > 1, "Minimum scaling gain must exceed one")
    measured = {row["gpu_count"]: row for row in stages if row.get("status") == "complete" and row.get("gpu_count") in (2, 4)}
    for row in measured.values():
        cadence = row.get("commit_cadence", {}).get("conservative_aggregate_fps")
        require(isinstance(cadence, (float, int)) and not isinstance(cadence, bool)
                and math.isfinite(cadence) and cadence > 0, "Async choice requires measured durable-commit cadence")
    ratio = (measured[4]["commit_cadence"]["conservative_aggregate_fps"] /
             measured[2]["commit_cadence"]["conservative_aggregate_fps"]) if {2, 4}.issubset(measured) else None
    selected = 4 if ratio is not None and ratio > minimum_gain else 2 if 2 in measured else None
    chosen = measured.get(selected)
    comparison = None
    if chosen and synchronous_eight:
        comparison = dict(synchronous_eight_fps=synchronous_eight["steady_fps"],
                          async_end_to_end_over_sync_eight=chosen["steady_fps"]/synchronous_eight["steady_fps"],
                          async_cadence_over_sync_eight=chosen["commit_cadence"]["conservative_aggregate_fps"]/synchronous_eight["steady_fps"],
                          basis="Separate measured pipeline families; asynchronous fill/drain-inclusive FPS and one-cycle cadence are both retained")
    return dict(gpu_count=selected, pipeline_family=ASYNC_WRITER_MODE, measured_gpu_counts=sorted(measured),
                async_one_gpu_measured=False, async_eight_gpu_measured=False,
                four_over_two_cadence=ratio,
                four_over_two_end_to_end=measured[4]["steady_fps"]/measured[2]["steady_fps"] if ratio is not None else None,
                minimum_measured_gain=minimum_gain,
                rule=f"Choose among verified asynchronous stages: 4 GPUs only above {minimum_gain:g}x measured 2-GPU durable-commit cadence; otherwise 2",
                comparison_to_synchronous_eight=comparison,
                limitation="Cadence has only one observed interval per worker; asynchronous one/eight-GPU performance is unmeasured")


def async_runtime_estimate(stage, halo, total_frames=TOTAL_FRAMES):
    conservative = runtime_estimate(stage, halo, total_frames)
    cadence = stage["commit_cadence"]
    workers = stage["gpu_count"]
    # Balance 100-frame disk blocks, conservatively charging a full cycle for
    # the final partial block. The first wave is already covered by measured fill.
    max_blocks_per_worker = math.ceil(math.ceil(total_frames/100)/workers)
    after_fill = max(0, max_blocks_per_worker-1)*cadence["max_worker_interval_seconds"]
    total = conservative["estimated_startup_seconds"]+cadence["measured_first_wave_fill_seconds"]+after_fill
    return dict(aggregate_fps=cadence["conservative_aggregate_fps"], throughput_basis="measured durable-commit cadence",
                estimated_startup_seconds=conservative["estimated_startup_seconds"],
                measured_first_wave_fill_seconds=cadence["measured_first_wave_fill_seconds"],
                estimated_after_fill_seconds=after_fill, maximum_blocks_per_worker=max_blocks_per_worker,
                estimated_total_seconds=total, estimated_total_hours=total/3600,
                conservative_end_to_end_estimate=conservative, commit_cadence=cadence,
                basis="Measured startup/halo plus observed first durable wave fill plus balanced remaining disk-block cycles",
                uncertainty="One cadence interval per worker; fill/drain-inclusive end-to-end estimate is also retained. Source waits and shared NAS variation are excluded")


def telemetry_summary(stage):
    workers = []
    for worker in stage.get("workers", []):
        path = Path(stage["archive_root"])/"logs"/f"worker_{worker['worker_id']}.telemetry.jsonl"
        started, finished = worker.get("measurement_started_time"), worker.get("measurement_finished_time")
        bounded = (isinstance(started, (int, float)) and isinstance(finished, (int, float))
                   and math.isfinite(started) and math.isfinite(finished) and finished >= started)
        duration = finished-started if bounded else worker.get("measured_seconds")
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
            duration = None
        rows = []
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (started if bounded else -math.inf) <= row.get("time", -math.inf) <= (finished if bounded else math.inf):
                    rows.append(row)
        times = sorted(row["time"] for row in rows if isinstance(row.get("time"), (int, float))
                       and math.isfinite(row["time"]))
        inter_sample_gaps = [right-left for left, right in zip(times, times[1:])]
        gaps_with_edges = inter_sample_gaps[:]
        if bounded:
            gaps_with_edges += [times[0]-started, finished-times[-1]] if times else [duration]
        expected_samples = max(1, math.ceil(duration)) if duration is not None and duration > 0 else None
        utilization = []
        for row in rows:
            try:
                value = float(row["gpu_csv"].split(",")[1].strip())
                if math.isfinite(value):
                    utilization.append(value)
            except (KeyError, IndexError, ValueError):
                pass
        workers.append(dict(worker_id=worker["worker_id"], telemetry_path=str(path), timed_samples=len(rows),
            measured_duration_seconds=duration, nominal_sampling_interval_seconds=1.0,
            expected_samples_at_nominal_interval=expected_samples,
            nominal_sample_count_coverage_fraction=min(1.0, len(rows)/expected_samples) if expected_samples else None,
            sample_coverage_basis="Sample count / ceil(measured seconds at nominal 1-second cadence); diagnostic only, not observed time coverage",
            max_inter_sample_gap_seconds=max(inter_sample_gaps) if inter_sample_gaps else None,
            max_sampling_gap_including_interval_edges_seconds=max(gaps_with_edges) if gaps_with_edges else None,
            sampled_peak_cpu_rss_bytes=max((row.get("cpu_rss", 0) for row in rows), default=None),
            cpu_peak_basis="Maximum available sample; the true peak may occur during an unsampled interval",
            gpu_utilization_available_sample_mean_percent=statistics.fmean(utilization) if utilization else None,
            gpu_utilization_sampled_max_percent=max(utilization) if utilization else None,
            gpu_utilization_valid_samples=len(utilization),
            gpu_utilization_mean_basis="Unweighted arithmetic mean of available samples, not whole-stage GPU utilization",
            whole_stage_gpu_utilization_mean_percent=None,
            reported_writer_pid=worker.get("writer_pid"),
            reported_writer_peak_rss_bytes=worker.get("peak_writer_rss_bytes"),
            writer_memory_basis="Separate writer-process report; parent RSS samples exclude this process",
            reported_compute_write_overlap_seconds=worker.get("compute_write_overlap_seconds")))
    return dict(workers=workers,
                sum_worker_sampled_rss_peaks_bytes=sum(row["sampled_peak_cpu_rss_bytes"] or 0 for row in workers)
                    if any(row["sampled_peak_cpu_rss_bytes"] is not None for row in workers) else None,
                workers_with_cpu_rss_samples=sum(row["sampled_peak_cpu_rss_bytes"] is not None for row in workers),
                rss_aggregation_basis="Sum of per-worker sampled maxima; not a simultaneous host peak or a bound on the true peak",
                sampling_bias_note="The thread monitor may be delayed during long native calls/serialization; gaps can coincide with commit work and bias the available samples",
                scope="Samples restricted to measured intervals; GPU means describe available samples only, and sampled CPU peaks may miss the true peak")


def _feature_readiness(path, header):
    shards = {int(p.name[6:14]) for p in Path(path).glob("ready_????????.json") if p.name[6:14].isdigit()}
    prefix = 0
    while prefix in shards:
        prefix += 1
    frames = min(TOTAL_FRAMES, prefix*header["frames_per_shard"])
    return dict(contiguous_ready_frames=frames, total_frames=TOTAL_FRAMES, all_receipts_present=frames == TOTAL_FRAMES,
                scope="Receipt coverage only; runner separately verifies each payload/index/frame hash before consumption")


def _validate_eight_authorization(report):
    authorization = report.get("user_authorization", {})
    require((authorization.get("requested_gpu_count"), authorization.get("source"), authorization.get("scope")) ==
            (8, "explicit_user_request", "one_additional_scaling_trial"),
            "Eight-GPU evidence must record the explicit additional-trial request")
    require(authorization.get("frames_per_gpu") in (100, 200), "Unexpected eight-GPU trial frame count")


def _lineage_evidence(report, config, config_path):
    """Verify old evidence reuse, including the actual committed tensor hashes."""
    from .extension_scaling import compare_transition_chunks, validate_lineage

    lineage = report.get("config_lineage", {})
    require(lineage.get("semantic_config_equal") is True and lineage.get("changed_source_files") == ["inputs.py"],
            "Only a proven inputs.py-only source transition may reuse prior scaling evidence")
    for path_field, hash_field in (
        ("prior_config_path", "prior_config_file_sha256"),
        ("current_config_path", "current_config_file_sha256"),
        ("prior_scaling_report_path", "prior_scaling_report_sha256"),
    ):
        value = lineage.get(path_field)
        require(isinstance(value, str) and Path(value).is_absolute(), "Config lineage paths must be absolute")
        require(sha256(value) == lineage.get(hash_field), "Config lineage file hash mismatch")
    require(Path(lineage["current_config_path"]).resolve() == Path(config_path).resolve()
            and read_json(lineage["current_config_path"]) == config, "Lineage current config differs from the selected config")
    previous_config = read_json(lineage["prior_config_path"])
    verified_lineage = validate_lineage(previous_config, config)
    for name in ("changed_source_files", "semantic_config_equal", "prior_inputs_sha256", "current_inputs_sha256"):
        require(lineage.get(name) == verified_lineage[name], f"Unproven config lineage field: {name}")
    prior_id, current_id = config_id(previous_config), config_id(config)
    require((lineage.get("prior_config_id"), lineage.get("current_config_id")) == (prior_id, current_id),
            "Config lineage identifiers mismatch")
    prior = read_json(lineage["prior_scaling_report_path"])
    require(prior.get("status") == "complete" and prior.get("cross_gpu_fidelity", {}).get("status") == "passed",
            "Prior scaling evidence is not complete and verified")
    require(prior.get("validation", {}).get("config_id") == prior_id and
            prior["validation"].get("config_file_sha256") == lineage["prior_config_file_sha256"],
            "Prior scaling report is not bound to the preserved config")
    require(report.get("original_scaling_report") == prior, "Preserved original scaling report differs from its hash-bound artifact")
    old_stages = {stage["gpu_count"]: stage for stage in prior.get("stages", [])}
    require(set(old_stages) == {1, 2, 4} and len(prior["stages"]) == 3,
            "Prior evidence must contain exactly the original 1/2/4 stages")
    stages = {stage["gpu_count"]: stage for stage in report.get("stages", [])}
    require(all(stages.get(count) == old_stages[count] for count in (1, 2, 4)),
            "Prior measured stage objects were changed during evidence reuse")
    require(report.get("cross_gpu_fidelity") == prior["cross_gpu_fidelity"], "Original cross-GPU equality evidence changed")
    transition = report.get("config_transition_fidelity", {})
    if 8 in stages:
        require(stages[8].get("config_id") == current_id, "Eight-GPU outputs use a different config")
        require(transition.get("status") == "passed", "Reader-transition tensor equality is not proven")
        comparisons = []
        for count in (1, 2, 4):
            comparisons.extend(compare_transition_chunks(old_stages[count], stages[8]))
        require(len(comparisons) == 7 and comparisons == transition.get("comparisons"),
                "Reader-transition evidence differs from the seven actual receipt/index-verified tensor comparisons")
    elif report.get("status") == "complete":
        raise ValueError("Completed extension report lacks the requested measured eight-GPU stage")
    return prior_id


def _async_scaling_evidence(report, config, config_path):
    from .async_scaling import commit_cadence, validate_async_lineage, validate_async_worker_report
    from .coordinator import compare_shared_chunks
    from .extension_scaling import compare_transition_chunks

    require((report.get("pipeline_family"), report.get("writer_mode"), report.get("max_pending_blocks")) ==
            (ASYNC_WRITER_MODE, ASYNC_WRITER_MODE, 1), "Async report has a different writer pipeline")
    require(report.get("async_one_gpu_measured") is False, "An unmeasured async one-GPU baseline was claimed")
    authorization = report.get("user_authorization", {})
    require((authorization.get("source"), authorization.get("scope"), authorization.get("measured_gpu_counts"),
             authorization.get("frames_per_gpu")) == ("explicit_user_request", "bounded_cpu_writer_process_overlap", [2, 4], 200),
            "Async trial scope differs from the authorized two-block CPU-process experiment")
    require(report.get("validation", {}).get("config_id") == config_id(config) and
            report["validation"].get("config_file_sha256") == sha256(config_path), "Async report config identity mismatch")
    lineage = report.get("config_lineage", {})
    for path_key, hash_key in (("prior_config_path", "prior_config_file_sha256"),
                               ("current_config_path", "current_config_file_sha256"),
                               ("prior_sync_report_path", "prior_sync_report_sha256")):
        path = lineage.get(path_key)
        require(isinstance(path, str) and Path(path).is_absolute() and sha256(path) == lineage.get(hash_key),
                "Async lineage artifact path/hash mismatch")
    require(Path(lineage["current_config_path"]).resolve() == Path(config_path).resolve()
            and read_json(lineage["current_config_path"]) == config, "Async lineage current config differs")
    prior_config = read_json(lineage["prior_config_path"])
    verified = validate_async_lineage(prior_config, config)
    require(all(lineage.get(key) == value for key, value in verified.items()), "Async semantic/source lineage was not proven")
    require((lineage.get("prior_config_id"), lineage.get("current_config_id")) == (config_id(prior_config), config_id(config)),
            "Async lineage configuration identifiers differ")
    prior = _scaling_evidence(Path(lineage["prior_sync_report_path"]), prior_config, Path(lineage["prior_config_path"]))
    require(prior.get("status") == "complete" and report.get("synchronous_reference_report") == prior,
            "Async report changed its completed synchronous reference evidence")
    references = [stage for stage in prior.get("stages", []) if stage.get("gpu_count") == 8]
    require(len(references) == 1, "Async report requires the actual synchronous eight-GPU reference")
    stages = report.get("stages", [])
    require(len({row.get("gpu_count") for row in stages}) == len(stages), "Duplicate async stages")
    actual_pipeline, actual_cross = [], []
    for index, stage in enumerate(stages):
        count = stage.get("gpu_count")
        require(count in (2, 4) and stage.get("status") == "complete" and stage.get("config_id") == config_id(config),
                "Unexpected async stage identity/count/status")
        require(stage.get("steady_fps", 0) > 0 and math.isfinite(stage["steady_fps"]), "Invalid async end-to-end throughput")
        require(len(stage.get("workers", [])) == count and {row.get("gpu") for row in stage["workers"]} == set(range(count)),
                "Async stage lacks its assigned GPUs")
        for worker in stage["workers"]:
            gpu = worker["gpu"]
            expected = dict(worker_id=f"gpu{gpu}", gpu=gpu, start_frame=100+gpu*200, end_frame=300+gpu*200)
            validate_async_worker_report(worker, expected, config_id(config), benchmark=True)
        require(stage.get("committed_frames") == count*200 and stage.get("commit_cadence") == commit_cadence(stage),
                "Async frame coverage or measured durable-commit cadence differs from actual worker receipts")
        actual_pipeline.extend(compare_transition_chunks(references[0], stage))
        for previous in stages[:index]:
            actual_cross.extend(compare_shared_chunks(previous, stage))
    require(report.get("cross_pipeline_fidelity", {}).get("comparisons") == actual_pipeline,
            "Cross-pipeline equality differs from actual committed output tensors")
    require(report.get("cross_gpu_fidelity", {}).get("comparisons") == actual_cross,
            "Async cross-GPU equality differs from actual committed output tensors")
    if report.get("status") == "complete":
        require({stage["gpu_count"] for stage in stages} == {2, 4} and len(actual_pipeline) == 12 and len(actual_cross) == 4
                and report["cross_pipeline_fidelity"].get("status") == report["cross_gpu_fidelity"].get("status") == "passed",
                "Completed async trial lacks all twelve pipeline and four GPU equality comparisons")
    return report


def _scaling_evidence(path, config, config_path):
    if not path.exists():
        return dict(status="pending", stages=[], cross_gpu_fidelity=dict(status="pending"))
    report = read_json(path)
    if report.get("schema") == "rekv_async_scaling_v1":
        return _async_scaling_evidence(report, config, config_path)
    require(report.get("validation", {}).get("config_id") == config_id(config), "Scaling used a different frozen config")
    require(report["validation"].get("config_file_sha256") == sha256(config_path), "Scaling config file hash mismatch")
    stages = report.get("stages", [])
    require(len({stage.get("gpu_count") for stage in stages}) == len(stages), "Duplicate scaling stages")
    extended = "config_lineage" in report
    if extended or any(stage.get("gpu_count") == 8 for stage in stages):
        _validate_eight_authorization(report)
    previous_id = _lineage_evidence(report, config, config_path) if extended else None
    for stage in stages:
        expected_id = previous_id if extended and stage.get("gpu_count") in (1, 2, 4) else config_id(config)
        require(stage.get("status") == "complete" and stage.get("config_id") == expected_id, "Invalid scaling stage identity/status")
        require(stage.get("gpu_count") in (1, 2, 4, 8), "Unapproved scaling GPU count")
        require(stage.get("steady_fps", 0) > 0 and math.isfinite(stage["steady_fps"]), "Invalid measured scaling throughput")
        if stage["gpu_count"] == 8:
            count = report["user_authorization"]["frames_per_gpu"]
            require(len(stage.get("workers", [])) == 8 and stage.get("committed_frames") == count*8,
                    "Eight-GPU stage does not contain the requested actual frame workload")
            require({worker.get("gpu") for worker in stage["workers"]} == set(range(8)) and all(
                worker.get("end_frame", 0)-worker.get("start_frame", 0) == count
                and worker.get("committed_frames") == count for worker in stage["workers"]),
                "Eight-GPU worker coverage differs from the explicit trial")
    cross = report.get("cross_gpu_fidelity", {})
    if cross.get("status") == "passed":
        comparisons = cross.get("comparisons", [])
        require(len(comparisons) >= 4 and all(row.get("status") == "passed" and
                row.get("all_tensor_sha256_exact") is True and row.get("all_tensor_geometry_exact") is True
                and row.get("metadata_exact") is True for row in comparisons),
                "Cross-GPU fidelity report lacks the actual four exact comparisons")
    return report


def build_report(args):
    paths = evidence_paths(args)
    config = read_json(paths["config"])
    initial, _, binding = _canonical_initial(paths["initial"], sha256(Path(__file__).with_name("attention.py")))
    _validate_config(config, initial, binding)
    for kind in ("passive", "streaming"):
        record = config["fidelity_validation"]
        require(str(paths[kind]) == record[f"{kind}_report_path"] and sha256(paths[kind]) == record[f"{kind}_report_sha256"],
                "Fidelity evidence changed after config freeze")
    passive, streaming, halo = read_json(paths["passive"]), read_json(paths["streaming"]), read_json(paths["halo"])
    require(str(paths["halo"]) == config["halo_validation"]["report_path"], "Halo report path differs from frozen config")
    fidelity = validate_smokes(passive, streaming)
    scaling = _scaling_evidence(paths["scaling"], config, paths["config"])
    stages = {row["gpu_count"]: row for row in scaling.get("stages", [])}
    async_family = scaling.get("schema") == "rekv_async_scaling_v1"
    synchronous_report = scaling.get("synchronous_reference_report", {}) if async_family else scaling
    synchronous_stages = {row["gpu_count"]: row for row in synchronous_report.get("stages", [])}
    baseline_one = synchronous_stages.get(1)
    recommendation = (recommend_async_gpu_count(list(stages.values()), synchronous_stages.get(8)) if async_family
                      else recommend_gpu_count(list(stages.values())))
    selected = stages.get(recommendation["gpu_count"])
    runtime = (async_runtime_estimate(selected, halo) if async_family else runtime_estimate(selected, halo)) if selected else None
    resources = {str(count): telemetry_summary(stage) for count, stage in stages.items()}
    header = _header(Path(args.features))
    require(header["archive_id"] == config["feature_archive_id"], "Feature archive changed")
    readiness = _feature_readiness(args.features, header)
    disk = storage_estimate()
    disk["nas_free_bytes_at_report"] = shutil.disk_usage(paths["root"]).free
    long_frames = [row for row in streaming["streaming"]["frames"] if row["mode"] == "rekv_local_plus_sink"]
    infer_fps = len(long_frames)/sum(row["seconds"] for row in long_frames)
    peak = baseline_one["peak_worker_vram_bytes"] if baseline_one else max(row["peak_vram_bytes"] for row in streaming["streaming"]["frames"])
    io = streaming["streaming"]["resume_io"]
    bottleneck = None
    if selected:
        components = {key: sum(worker.get(key, 0.0) for worker in selected["workers"])
                      for key in ("read_seconds", "infer_seconds", "capture_seconds", "atomic_nas_seconds")}
        total = sum(components.values())
        bottleneck = dict(dominant_measured_component=max(components, key=components.get),
                          summed_worker_seconds=components,
                          fractions={key: value/total if total else 0.0 for key, value in components.items()},
                          caveat="Worker component durations, not hardware-counter attribution; async compute/write durations overlap and are not additive elapsed-time fractions. Atomic NAS includes copying, hashing, serialization and fsync")
    def item(number, title, text, status="verified", evidence=None):
        return dict(number=number, title=title, status=status, summary=text, evidence=evidence)
    gpu0_fps_text = (f"{baseline_one['steady_fps']:.3f} frames/s synchronous GPU0 end-to-end including NAS writes" if baseline_one else
                     f"End-to-end measurement pending; long-window smoke {infer_fps:.3f} frames/s including CPU KV capture but excluding NAS writes")
    if async_family:
        gpu0_fps_text += "; asynchronous one-GPU performance was not measured"
    nas_text = (f"Synchronous GPU0 atomic commit {baseline_one['atomic_nas_effective_bytes_per_second']/1e6:.2f} MB/s" if baseline_one else
                f"910 MB resume smoke commit {io['commit_MB_per_second']:.2f} MB/s; full raw-block throughput pending")
    if async_family and selected:
        nas_text += f"; selected async {selected['gpu_count']}-GPU aggregate atomic commit {selected['atomic_nas_effective_bytes_per_second']/1e6:.2f} MB/s"
    def scaling_text(count):
        if count not in stages:
            return "Not yet measured"
        row = stages[count]
        if async_family:
            return (f"Async {count} GPUs: {row['steady_fps']:.3f} frames/s measured two-block end-to-end; "
                    f"{row['commit_cadence']['conservative_aggregate_fps']:.3f} frames/s durable-commit cadence "
                    f"(one interval/worker); atomic NAS {row['atomic_nas_effective_bytes_per_second']/1e6:.2f} MB/s")
        if 1 not in stages:
            return "GPU0 reference not yet measured"
        return (f"{row['steady_fps']:.3f} aggregate frames/s; {row['steady_fps']/stages[1]['steady_fps']:.3f}x GPU0; "
                f"atomic NAS {row['atomic_nas_effective_bytes_per_second']/1e6:.2f} MB/s")
    requested_eight = not async_family and ("config_lineage" in scaling or 8 in stages)
    if async_family:
        eight_text = f"Synchronous 8-GPU reference: {synchronous_stages[8]['steady_fps']:.3f} frames/s; asynchronous 8-GPU performance unmeasured"
        best_text = (f"{recommendation['gpu_count']} asynchronous GPUs; choose 4 only above 1.2x async 2-GPU durable-commit cadence, otherwise 2. "
                     + (f"Observed 4/2 ratio: cadence {recommendation['four_over_two_cadence']:.6f}x, end-to-end {recommendation['four_over_two_end_to_end']:.6f}x"
                        if recommendation["four_over_two_cadence"] is not None else "Async 4-GPU comparison pending")) if selected else "Asynchronous scaling pending"
    else:
        eight_text = (f"Requested 8-GPU trial: {scaling_text(8)}; {recommendation['eight_over_four']:.6f}x measured 4-GPU throughput"
                      if 8 in stages and recommendation["eight_over_four"] is not None else
                      "Requested 8-GPU trial pending" if requested_eight else "8 GPUs have not been measured")
        best_text = (f"{recommendation['gpu_count']} GPUs; 4 requires >1.2x measured 2-GPU throughput, and 8 requires >1.2x measured 4-GPU throughput. "
                     + (f"Measured 8/4 ratio: {recommendation['eight_over_four']:.6f}" if recommendation["eight_over_four"] is not None
                        else "No unmeasured 8-GPU recommendation") if selected else "Pending scaling; no unmeasured GPU-count recommendation")
    runtime_text = "Pending end-to-end scaling measurements"
    if runtime:
        runtime_text = (f"{runtime['estimated_total_hours']:.2f} hours with {selected['gpu_count']} GPUs, including estimated startup/halo; source waiting and shared-resource variation excluded")
        if async_family:
            runtime_text = (f"{runtime['estimated_total_hours']:.2f} hours from measured cadence with startup/halo and first-wave fill; "
                            f"{runtime['conservative_end_to_end_estimate']['estimated_total_hours']:.2f} hours from conservative two-block end-to-end FPS. "
                            "One cadence interval/worker; source waiting and NAS variation excluded")
    items = [
        item(1, "Checkpoint / revision", f"{REPO_ID} @ {REVISION}; frozen official Instruct weights", evidence=config["backbone"]),
        item(2, "Official ReKV retained", "Verified upstream TorchMultiStage attention; pre-RoPE KV, 15K causal local window, fixed sinks, joint softmax and chunk-end sink exposure retained", evidence=initial["upstream"]),
        item(3, "Qwen2.5 adaptation", "Three-axis mRoPE local rebasing; sink queries rotated at [14999,14999,14999], raw sink keys; native HF short-cache/mask dispatch; atomic NAS storage added. Retrieval/QA remains unimplemented", evidence=initial["policy"]),
        item(4, "KV extraction boundary", "Actual k_proj/v_proj outputs in every decoder layer; keys captured before mRoPE, never obtained by inverting post-RoPE cache", evidence=fidelity),
        item(5, "Sampling / chunk", "1 FPS, 221728 ordered Day1-7 frames; separate 17-token prefix, exactly one 678-token frame per model call; 100-frame disk blocks; 620-frame validated halo", evidence=dict(source_duration_seconds=header["inventory"]["total_duration_seconds"], readiness=readiness)),
        item(6, "Dtype", "BF16 K/V, lossless uncompressed safetensors; int64 positions/input IDs; Transformers 4.57.3, Torch 2.5.1+cu121"),
        item(7, "KV shape", "28 layers × K and V [1,4,T,128]; T=678 per frame, 67800 per 100-frame disk block"),
        item(8, "KV bytes per frame / chunk", f"{disk['raw_kv_bytes_per_frame']:,} bytes/frame; {disk['raw_kv_bytes_per_100_frame_block']:,} bytes/100-frame block", evidence=disk),
        item(9, "Full storage", f"{disk['raw_kv_bytes_all_frames']/1e12:.6f} TB raw KV + {disk['positions_and_input_ids_bytes']/1e9:.3f} GB positions/IDs; budget about 8.63 TB plus auxiliary artifacts", evidence=disk),
        item(10, "GPU0 peak VRAM", f"{peak/2**30:.3f} GiB ({peak:,} bytes), PyTorch max_memory_allocated for the synchronous one-frame reference; excludes reserved allocator memory and other device allocations", evidence=dict(peak_bytes=peak, metric="torch.cuda.max_memory_allocated", basis="synchronous scaling_n1 worker" if baseline_one else "streaming_v2 one-frame/window crossing", selected_async_peak_worker_vram_bytes=selected.get("peak_worker_vram_bytes") if async_family and selected else None)),
        item(11, "GPU0 processing FPS", gpu0_fps_text, "measured" if baseline_one else "partial", evidence=dict(long_window_smoke_fps=infer_fps, synchronous_scaling=baseline_one)),
        item(12, "NAS write throughput", nas_text+"; commit includes CPU copy/hash/serialization/fsync", "measured", evidence=dict(resume_smoke=io)),
        item(13, "GPU0 → 2 GPUs", scaling_text(2), "measured" if 2 in stages else "pending", evidence=stages.get(2)),
        item(14, "GPU0 → 4 GPUs / synchronous 8-GPU reference" if async_family else "GPU0 → 4 GPUs / requested 8-GPU trial", scaling_text(4)+"; "+eight_text,
             "measured" if 4 in stages else "pending", evidence=dict(four_gpu=stages.get(4), synchronous_eight_gpu=synchronous_stages.get(8))),
        item(15, "Estimated total runtime", runtime_text, "estimate" if runtime else "pending", evidence=runtime),
        item(16, "Measured bottleneck", (f"Largest timed component: {bottleneck['dominant_measured_component']}. GPU utilization is an available-sample mean, not a whole-stage mean; sampled CPU peaks may miss commit-time activity (coverage/gaps attached). Source readiness may add waiting" if bottleneck else "Pending timed worker stage; long attention and atomic NAS writes are measured separately, not yet attributed as the full-run bottleneck"), "measured" if bottleneck else "pending", evidence=dict(timing=bottleneck, telemetry=resources, feature_readiness=readiness)),
        item(17, "Best measured GPU count", best_text, "recommendation" if selected else "pending", evidence=recommendation),
    ]
    cross = scaling.get("cross_gpu_fidelity", {})
    required_stages = {2, 4} if async_family else {1, 2, 4, 8} if requested_eight else {1, 2, 4}
    transition_passed = (scaling.get("cross_pipeline_fidelity", {}).get("status") == "passed" if async_family else
                         not scaling.get("config_lineage") or scaling.get("config_transition_fidelity", {}).get("status") == "passed")
    complete = (scaling.get("status") == "complete" and set(stages) == required_stages
                and cross.get("status") == "passed" and transition_passed)
    recommendation["validated_for_production"] = bool(complete and selected)
    started, production_state = _full_run_state(paths["root"])
    status = "complete" if complete else "scaling_failed" if scaling.get("status") == "failed" else "fidelity_passed_scaling_pending"
    report = dict(schema="rekv_qwen25_preflight_report_v1", status=status,
                  created_at_utc=datetime.now(timezone.utc).isoformat(), full_run_started=started,
                  full_run_status_basis=production_state,
                  config_path=str(paths["config"]), config_id=config_id(config), config_sha256=sha256(paths["config"]),
                  halo_report_path=str(paths["halo"]), halo_report_sha256=sha256(paths["halo"]),
                  initial_state_binding=binding, fidelity=fidelity, source_readiness=readiness,
                  scaling_status=scaling.get("status"), scaling_report_path=str(paths["scaling"]),
                  scaling_report_sha256=sha256(paths["scaling"]) if paths["scaling"].exists() else None,
                  cross_gpu_fidelity=cross, config_lineage=scaling.get("config_lineage"),
                  config_transition_fidelity=scaling.get("config_transition_fidelity"),
                  cross_pipeline_fidelity=scaling.get("cross_pipeline_fidelity"),
                  pipeline_family=ASYNC_WRITER_MODE if async_family else "synchronous",
                  measured_pipeline_families=dict(synchronous=list(synchronous_stages.values()), asynchronous=list(stages.values()) if async_family else []),
                  eight_gpu_trial=synchronous_stages.get(8), eight_gpu_trial_authorization=synchronous_report.get("user_authorization"),
                  recommendation=recommendation, runtime_estimate=runtime,
                  resource_measurements=resources, storage=disk, items=items)
    write_json(paths["root"]/"preflight-report.json", report)
    launch_text = "A full-run supervisor launch is recorded under this root." if started else "No full-run supervisor launch is recorded under this root."
    previous_run = production_state.get("supervisor_status") or {}
    if previous_run.get("status") == "failed":
        launch_text = (f"The previous full-run startup/run failed: {previous_run.get('error', 'see supervisor report')}. "
                       f"Committed archive frames recorded under this root: {production_state.get('committed_frames', 'unknown')}.")
    markdown = [f"ReKV (Qwen2.5-VL-7B adaptation) preflight — {report['status']}", "",
                launch_text+" Native/raw/cache/disk equality and the continuous-state halo proof passed; scaling availability is stated below.", ""]
    markdown += [f"{row['number']}. **{row['title']}** — {row['summary']}" for row in items]
    markdown += ["", "Runtime is an extrapolation from short measured stages, not a completion promise. It excludes source-producer waits, retries, and changes in shared NAS traffic.", ""]
    path = paths["root"]/"preflight-report.md"
    temp = path.with_suffix(".md.partial")
    temp.write_text("\n".join(markdown))
    temp.replace(path)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-config", action="store_true")
    action.add_argument("--report", action="store_true")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--config")
    parser.add_argument("--initial-state-root")
    parser.add_argument("--halo-report")
    parser.add_argument("--passive-report")
    parser.add_argument("--streaming-report")
    parser.add_argument("--scaling-report")
    parser.add_argument("--model", default=str(MODEL))
    parser.add_argument("--ledger", default=str(LEDGER))
    parser.add_argument("--features", default=str(FEATURES))
    parser.add_argument("--writer-mode", choices=[ASYNC_WRITER_MODE], help="Explicitly freeze the bounded CPU-process overlap pipeline")
    args = parser.parse_args()
    result = prepare_config(args) if args.prepare_config else build_report(args)
    print(json.dumps({key: result[key] for key in ("status", "config_path", "config_id", "full_run_started")}, indent=2))


if __name__ == "__main__":
    main()
