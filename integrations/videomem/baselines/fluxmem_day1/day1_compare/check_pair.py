"""Independently audit saved Day1 paired runs; never load a model or use a GPU.

Exit codes: 0 passed, 1 failed, 2 pending. An in-progress run cannot pass.
Each --output must be a new JSON path. Generation configuration sidecars are
reconstructed from the local checkpoint with the installed Transformers
GenerationConfig defaults and the explicit runner overrides, never fed back
to either model. Extra source files added after launch are permitted; the
runner, model source and every common recorded source hash must agree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODEL_PREFIX = "models/qwen2-5-vl/"
MODEL_FILE = MODEL_PREFIX + "src/qwen2_5_vl_fluxmem/modeling_qwen2_5_vl_fluxmem.py"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def generation_sidecar(provenance: dict, root: Path) -> dict:
    """Configuration only: no pretrained model construction or network access."""
    import transformers
    from transformers import GenerationConfig

    settings = provenance["settings"]
    model_dir = Path(settings["model"])
    if not model_dir.is_absolute():
        model_dir = root / model_dir
    source = model_dir / "generation_config.json"
    inherited = json.loads(source.read_text())
    overrides = {"do_sample": False, "max_new_tokens": settings["max_new_tokens"]}
    config = GenerationConfig.from_dict(inherited)
    for key, value in overrides.items():
        setattr(config, key, value)
    effective = config.to_dict()
    # These bookkeeping fields do not affect generation.
    effective.pop("_commit_hash", None)
    return {"derivation": "Local checkpoint generation_config.json plus installed GenerationConfig defaults and explicit runner overrides",
            "checkpoint_path": str(source.resolve()), "checkpoint_sha256": sha(source),
            "checkpoint_generation_config": inherited, "explicit_overrides": overrides,
            "effective_generation_config": effective,
            "transformers_version": transformers.__version__,
            "model_revision": provenance.get("model_revision")}


def check_pair(base: Path, on: Path, off: Path, *, root: Path = ROOT,
               runner: str = "day1_compare/run.py") -> tuple[dict, dict]:
    checks: list[dict] = []
    pending: list[str] = []
    sidecars: dict[str, dict] = {}

    def check(name: str, condition: bool, detail: Any = None) -> None:
        checks.append({"check": name, "passed": bool(condition), **({"detail": detail} if detail is not None else {})})

    def read(path: Path, *, required: bool = False):
        if not path.is_file():
            if required:
                check(f"exists:{path}", False)
            else:
                pending.append(f"Not yet available: {path}")
            return None
        try:
            return json.loads(path.read_text())
        except (ValueError, OSError) as exc:
            check(f"readable:{path}", False, str(exc))
            return None

    completed = read(base / "completed.json", required=True)
    protocol = read(base / "protocol.json", required=True)
    questions = read(base / "public_questions.json", required=True)
    manifest = read(base / "source_frames_manifest.json", required=True)
    arm_data = {}
    expected_ids, frames = [], []
    try:
        if all(x is not None for x in (completed, protocol, questions, manifest)):
            check("preparation_complete", completed.get("status") == "complete")
            required_files = {"protocol.json", "public_questions.json", "source_frames_manifest.json", "eval_labels.json"}
            check("completion_records_required_hashes", required_files <= set(completed["files"]))
            for name, digest in completed["files"].items():
                path = base / name
                check(f"frozen_hash:{name}", path.is_file() and sha(path) == digest)
            expected_ids = [q["question_id"] for q in questions]
            frames = manifest["frames"]
            count = protocol["selection"]["count"]
            check("selected_ids_and_order", expected_ids == protocol["selection"]["selected_ids"] and len(set(expected_ids)) == count)
            check("question_counts", len(questions) == count == completed["questions"])
            check("frame_counts", len(frames) == completed["frames"] == manifest["total_frames"] == protocol["sampling"]["frame_count"])
            check("manifest_binds_protocol", manifest["protocol_sha256"] == sha(base / "protocol.json"))
            duration = float(protocol["query_time_seconds"])
            check("shared_day_end", duration > 0 and manifest["day_duration_seconds"] == duration and all(q["query_time_seconds"] == duration for q in questions))
            check("sample_order", [f["sample_index"] for f in frames] == list(range(len(frames))))
            times = [f["actual_global_seconds"] for f in frames]
            check("chronological_before_cutoff", bool(times) and 0 <= times[0] and times[-1] < duration and all(a < b for a, b in zip(times, times[1:])))
            check("uniform_midpoint_requests", all(math.isclose(f["requested_global_seconds"], duration * (2 * i + 1) / (2 * len(frames)), abs_tol=1e-8) for i, f in enumerate(frames)))
            bad_frames = [f["sample_index"] for f in frames if not Path(f["path"]).is_file() or sha(Path(f["path"])) != f["sha256"]]
            check("cached_frame_hashes", not bad_frames, bad_frames)
            check("cached_frame_dimensions", all(f["width"] == protocol["sampling"]["image_width"] and f["height"] == protocol["sampling"]["image_height"] for f in frames))
            fps = len(frames) / duration
            for name, directory in (("on", on), ("off", off)):
                data = {key: read(directory / filename) for key, filename in (
                    ("status", "status.json"), ("provenance", "provenance.json"),
                    ("video", "processed_video.json"), ("attention", "attention_runtime.json"),
                    ("predictions", "predictions.json"))}
                arm_data[name] = data
                prov, status, video, attention, rows = (data[key] for key in ("provenance", "status", "video", "attention", "predictions"))
                if prov is not None:
                    check(f"{name}:input_hashes", prov["public_questions_sha256"] == sha(base / "public_questions.json") and prov["frames_manifest_sha256"] == sha(base / "source_frames_manifest.json"))
                    check(f"{name}:input_budget", prov["frames"] == len(frames) and math.isclose(prov["effective_fps"], fps))
                    check(f"{name}:inference_arm", prov["settings"]["fluxmem"] == name and prov["training"] is False and prov["no_label_file_opened"] is True)
                    frozen = prov["source_hashes"]
                    required = {runner, MODEL_FILE, "local_videomem/run.py"}
                    check(f"{name}:required_source_records", required <= set(frozen))
                    important = {key for key in frozen if key.startswith(MODEL_PREFIX)} | required
                    bad = [key for key in sorted(important) if key not in frozen or not (root / key).is_file() or sha(root / key) != frozen[key]]
                    check(f"{name}:frozen_runner_model_source", not bad, bad)
                    sidecars[name] = generation_sidecar(prov, root)
                    check(f"{name}:generation_library_version", sidecars[name]["transformers_version"] == prov["transformers"])
                if attention is not None:
                    check(f"{name}:full_attention", attention.get("checkpoint_use_sliding_window") is False and "effective_sliding_window" in attention and attention["effective_sliding_window"] is None)
                if video is not None:
                    check(f"{name}:video_fps", math.isclose(video["effective_fps"], fps))
                    check(f"{name}:native_video_fields", {"pixel_values_videos", "video_grid_thw"} <= set(video["hashes"]) and video["shapes"].get("video_grid_thw") == [1, 3])
                if status is not None:
                    check(f"{name}:not_stopped_on_error", status.get("phase") != "stopped_on_error")
                    if status.get("phase") != "finished":
                        pending.append(f"{name} has not finished: {status.get('phase')}")
                if rows is not None:
                    row_ids = [r["question_id"] for r in rows]
                    check(f"{name}:prediction_order", row_ids == expected_ids[:len(rows)] and len(rows) <= len(expected_ids))
                    if len(rows) != len(expected_ids):
                        pending.append(f"{name}: {len(rows)}/{len(expected_ids)} predictions available")
                    check(f"{name}:complete_rows_only", all(r.get("status") == "complete" for r in rows), {r["question_id"]: r.get("status") for r in rows if r.get("status") != "complete"})
                    if status is not None and status.get("phase") == "finished":
                        check(f"{name}:finished_counts", status.get("completed") == status.get("attempted") == status.get("total") == len(expected_ids) == len(rows))
                wires = []
                for i, qid in enumerate(expected_ids):
                    wire = read(directory / f"wire_{i:04d}.json")
                    wires.append(wire)
                    if wire is None:
                        continue
                    check(f"{name}:wire:{i}:identity", wire.get("question_id") == qid)
                    check(f"{name}:wire:{i}:prompt_hash", wire.get("prompt_sha256") == hashlib.sha256(wire["prompt"].encode()).hexdigest())
                    check(f"{name}:wire:{i}:times", wire.get("second_per_grid_ts") == [2 / fps])
                    if video is not None:
                        check(f"{name}:wire:{i}:video", wire.get("processed_video_hashes") == video["hashes"])
                    if rows is not None and i < len(rows):
                        row = rows[i]
                        check(f"{name}:wire:{i}:tokens", row.get("input_tokens") == wire["input_tokens"])
                        events = row.get("memory_events")
                        check(f"{name}:row:{i}:native_memory_switch", isinstance(events, list) and (len(events) == 1 if name == "on" else len(events) == 0))
                        if name == "on" and isinstance(events, list) and len(events) == 1:
                            event = events[0]
                            check(f"{name}:row:{i}:memory_counts", event["input_tokens"] == wire["input_tokens"] and 0 <= event["visual_tokens_after"] <= event["visual_tokens_before"] and event["input_tokens"] - event["retained_tokens"] == event["dropped_tokens"] == event["visual_tokens_before"] - event["visual_tokens_after"])
                            check(f"{name}:row:{i}:compression_paths", event["temporal_calls"] > 0 and event["spatial_calls"] > 0)
                        if row.get("status") == "complete" and name in sidecars:
                            eos = sidecars[name]["effective_generation_config"]["eos_token_id"]
                            eos = eos if isinstance(eos, list) else [eos]
                            tokens = row.get("output_token_ids", [])
                            check(f"{name}:row:{i}:complete_eos", bool(tokens) and tokens[-1] in eos and len(tokens) == row.get("output_tokens") and bool(row.get("prediction", "").strip()))
                data["wires"] = wires
                check(f"{name}:no_extra_wires", len(list(directory.glob("wire_*.json"))) <= len(expected_ids))
            left, right = arm_data["on"], arm_data["off"]
            if left["provenance"] is not None and right["provenance"] is not None:
                a, b = left["provenance"], right["provenance"]
                for field in ("upstream_commit", "model_revision", "torch", "transformers", "dtype", "attention", "decoder", "video_chunk_t", "frames", "effective_fps"):
                    check(f"paired_provenance:{field}", field in a and field in b and a[field] == b[field])
                excluded = {"gpu", "fluxmem", "output"}
                check("paired_settings", {k: v for k, v in a["settings"].items() if k not in excluded} == {k: v for k, v in b["settings"].items() if k not in excluded})
                ah, bh = a["source_hashes"], b["source_hashes"]
                common = set(ah) & set(bh)
                check("paired_common_source_hashes", all(ah[key] == bh[key] for key in common))
                check("paired_model_source_set", {k for k in ah if k.startswith(MODEL_PREFIX)} == {k for k in bh if k.startswith(MODEL_PREFIX)})
            for field in ("video", "attention"):
                if left[field] is not None and right[field] is not None:
                    check(f"paired_{field}", left[field] == right[field])
            for i, (a, b) in enumerate(zip(left["wires"], right["wires"])):
                if a is not None and b is not None:
                    check(f"paired_wire:{i}", a == b)
            if set(sidecars) == {"on", "off"}:
                check("paired_effective_generation_config", sidecars["on"] == sidecars["off"])
    except (KeyError, TypeError, ValueError, OSError, ImportError, ZeroDivisionError) as exc:
        check("artifact_schema_and_configuration", False, f"{type(exc).__name__}: {exc}")
    failures = [entry for entry in checks if not entry["passed"]]
    status = "failed" if failures else "pending" if pending else "passed"
    return {
        "schema_version": 1, "status": status, "valid_for_paired_accuracy": status == "passed",
        "expected_questions": len(expected_ids), "expected_frames": len(frames),
        "runner_source": runner, "checks": checks, "failures": failures, "pending": pending,
        "input_directory": str(base.resolve()), "on_directory": str(on.resolve()), "off_directory": str(off.resolve()),
        "note": "Technical pairing audit only; no labels are parsed or accuracy calculated. Failed or pending runs are not valid completed comparisons.",
    }, sidecars


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "on", "off", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--runner", default="day1_compare/run.py", help="Repo-relative frozen runner source, e.g. day1_compare/run_v2.py")
    args = parser.parse_args(argv)
    report, sidecars = check_pair(args.input, args.on, args.off, runner=args.runner)
    targets = {args.output: report}
    for name, sidecar in sidecars.items():
        target = args.output.with_name(args.output.stem + f"_generation_config_{name}.json")
        targets[target] = sidecar
    if any(path.exists() for path in targets):
        raise FileExistsError("Choose a new report path; existing audit artifacts are immutable")
    payloads = {path: json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n" for path, value in targets.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path, value in payloads.items():
        with path.open("x", encoding="utf-8") as stream:
            stream.write(value)
    print(json.dumps({"status": report["status"], "valid_for_paired_accuracy": report["valid_for_paired_accuracy"], "failures": len(report["failures"]), "pending": len(report["pending"]), "report": str(args.output)}, ensure_ascii=False))
    return {"passed": 0, "failed": 1, "pending": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
