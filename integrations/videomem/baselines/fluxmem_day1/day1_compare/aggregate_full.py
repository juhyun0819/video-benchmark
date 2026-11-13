"""Combine the frozen 16-question v2 pilot and 233-question continuation.

This is CPU-only postprocessing. Incomplete or inconsistent artifacts produce
no output. Finished attempts with generation failures retain their original
records and all-question denominators, but cannot pass the technical audit.
An existing output directory is never reused. No combined run provenance is
invented: every row points to its original run and an immutable local snapshot.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys
from typing import Any

from day1_compare.check_pair import MODEL_FILE, MODEL_PREFIX, generation_sidecar
from day1_compare.evaluate import evaluate
from day1_compare.run import public_prompt
from local_videomem.run import verify_weights

ROOT = Path(__file__).resolve().parents[1]
CORE = {"day1_compare/run.py", "day1_compare/run_v2.py", "local_videomem/run.py", MODEL_FILE}
PACKAGE_FILES = {"protocol.json", "public_questions.json", "source_frames_manifest.json", "eval_labels.json"}
RUN_FILES = ("provenance.json", "status.json", "processed_video.json", "attention_runtime.json", "predictions.json")
PAIRED_FIELDS = ("upstream_commit", "model_revision", "torch", "transformers", "dtype", "attention",
                 "decoder", "video_chunk_t", "frames", "effective_fps", "checkpoint", "temporal_positions")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class ValidationError(ValueError):
    def __init__(self, message: str, *, pending: bool = False):
        super().__init__(message)
        self.pending = pending


def require(condition: bool, message: str, *, pending: bool = False) -> None:
    if not condition:
        raise ValidationError(message, pending=pending)


def index(rows: list[dict], description: str) -> dict[str, dict]:
    require(isinstance(rows, list), f"{description}: expected a list")
    result = {}
    for row in rows:
        qid = row.get("question_id")
        require(isinstance(qid, str) and bool(qid), f"{description}: invalid question ID")
        require(qid not in result, f"{description}: duplicate question ID {qid}")
        result[qid] = row
    return result


class Inputs:
    """Capture exact source bytes once and detect changes before output creation."""
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.raw: dict[Path, bytes] = {}

    def path(self, path: str | Path) -> Path:
        path = Path(path)
        return (path if path.is_absolute() else self.root / path).resolve()

    def read(self, path: str | Path, expected: str | None = None) -> Any:
        path = self.path(path)
        require(path.is_file(), f"Missing artifact: {path}")
        raw = self.raw.setdefault(path, path.read_bytes())
        require(expected is None or digest(raw) == expected, f"Frozen hash mismatch: {path}")
        return json.loads(raw)

    def hash(self, path: str | Path) -> str:
        path = self.path(path)
        if path not in self.raw:
            require(path.is_file(), f"Missing artifact: {path}")
            self.raw[path] = path.read_bytes()
        return digest(self.raw[path])

    def source(self, path: str | Path) -> dict:
        return {"path": str(self.path(path)), "sha256": self.hash(path)}

    def unchanged(self) -> None:
        for path, raw in self.raw.items():
            require(path.is_file() and digest(path.read_bytes()) == digest(raw), f"Artifact changed during aggregation: {path}")


def load_package(inputs: Inputs, spec: dict) -> dict:
    base = inputs.path(spec["path"])
    completed = inputs.read(base / "completed.json", spec["completed_sha256"])
    require(completed.get("status") == "complete", f"Package is incomplete: {base}")
    require(PACKAGE_FILES <= set(spec["files"]) and PACKAGE_FILES <= set(completed["files"]), f"Package hashes missing: {base}")
    for name, checksum in completed["files"].items():
        require(inputs.hash(base / name) == checksum, f"Frozen package file changed: {base / name}")
    values = {name: inputs.read(base / name, spec["files"][name]) for name in PACKAGE_FILES}
    questions = values["public_questions.json"]
    ids = list(index(questions, str(base)))
    protocol, manifest = values["protocol.json"], values["source_frames_manifest.json"]
    require(ids == protocol["selection"]["selected_ids"], f"Question order differs from protocol: {base}")
    require(len(ids) == completed["questions"] == protocol["selection"]["count"], f"Question count mismatch: {base}")
    require(manifest["protocol_sha256"] == inputs.hash(base / "protocol.json"), f"Manifest does not bind its protocol: {base}")
    require(len(manifest["frames"]) == manifest["total_frames"] == completed["frames"] == protocol["sampling"]["frame_count"], f"Frame count mismatch: {base}")
    return {"base": base, "questions": questions, "ids": ids, "protocol": protocol,
            "manifest": manifest, "labels": values["eval_labels.json"]}


def aggregate(package: Path, pilot_on: Path, pilot_off: Path, new_root: Path,
              output: Path, *, root: Path = ROOT, expected_questions: int = 249,
              expected_reused: int = 16) -> dict:
    """Validate, preserve original records, and publish one immutable result set."""
    inputs = Inputs(root)
    package, output = inputs.path(package), inputs.path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {output}")
    frozen = inputs.read(package / "run_manifest.json")
    require(frozen.get("status") == "prepared", "Full package is not prepared")
    counts = {"all": expected_questions, "reused": expected_reused, "remaining": expected_questions - expected_reused}
    require(frozen["counts"] == counts, f"Expected partition counts {counts}")
    batches = {"all": load_package(inputs, frozen["packages"]["all"]),
               "remaining": load_package(inputs, frozen["packages"]["remaining"]),
               "pilot": load_package(inputs, frozen["pilot_package"])}
    shards = frozen["shards"]
    require(bool(shards) and len({shard["name"] for shard in shards}) == len(shards), "Shard names are missing or duplicated")
    for shard in shards:
        require(shard["name"] not in batches, "Shard name conflicts with a package name")
        batches[shard["name"]] = load_package(inputs, shard["package"])
    ids, pilot_ids, new_ids = (batches[key]["ids"] for key in ("all", "pilot", "remaining"))
    for name, sequence in (("all", ids), ("reused", pilot_ids), ("remaining", new_ids)):
        require(sequence == frozen["question_ids"][name] and len(sequence) == counts[name], f"Frozen {name} IDs differ")
    require(not set(pilot_ids) & set(new_ids), "Pilot and continuation IDs overlap")
    require(set(pilot_ids) | set(new_ids) == set(ids), "Pilot and continuation do not exactly cover all questions")
    require([qid for qid in ids if qid not in set(pilot_ids)] == new_ids, "Continuation order differs from all-question order")
    cursor = 0
    for shard in shards:
        require(shard["remaining_start"] == cursor and shard["remaining_stop"] == cursor + len(shard["question_ids"]), f"{shard['name']}: shard boundaries differ")
        require(batches[shard["name"]]["ids"] == shard["question_ids"] == new_ids[cursor:shard["remaining_stop"]], f"{shard['name']}: shard question IDs differ")
        require(shard["gpu"] in range(1, 6) and sorted(shard["arm_order"]) == ["off", "on"], f"{shard['name']}: invalid hardware/arm schedule")
        cursor = shard["remaining_stop"]
    require(cursor == len(new_ids), "Shards do not cover all remaining questions")
    full_index = index(batches["all"]["questions"], "All questions")
    full_labels = index(batches["all"]["labels"], "All labels")
    require(set(full_labels) == set(ids), "Labels must cover all questions exactly")
    frames = batches["pilot"]["manifest"]["frames"]
    duration = batches["pilot"]["protocol"]["query_time_seconds"]
    require(duration > 0 and bool(frames), "Invalid frame budget or cutoff")
    fps = len(frames) / duration
    for name, batch in batches.items():
        require(batch["manifest"]["frames"] == frames, f"{name}: sampled frame records differ from pilot")
        require(batch["manifest"]["day_duration_seconds"] == duration == batch["protocol"]["query_time_seconds"], f"{name}: day cutoff differs")
        labels = index(batch["labels"], f"{name} labels")
        require(set(labels) == set(batch["ids"]), f"{name}: labels differ from selected IDs")
        for q in batch["questions"]:
            require(q == full_index[q["question_id"]] and q["query_time_seconds"] == duration, f"{name}: public question changed: {q['question_id']}")
            require(labels[q["question_id"]] == full_labels[q["question_id"]], f"{name}: label changed: {q['question_id']}")
    require(digest(canonical_bytes(frames)) == frozen["frame_content_sha256"], "Frozen frame record hash differs")
    times = [frame["actual_global_seconds"] for frame in frames]
    require(0 <= times[0] and times[-1] < duration and all(a < b for a, b in zip(times, times[1:])), "Frames are not chronological before cutoff")
    require([frame["sample_index"] for frame in frames] == list(range(len(frames))), "Frame sample order differs")
    for i, frame in enumerate(frames):
        require(inputs.hash(frame["path"]) == frame["sha256"], f"Cached frame changed: {i}")
        require(frame["width"] == frame["height"] == 224, f"Frame dimensions differ: {i}")
        require(math.isclose(frame["requested_global_seconds"], duration * (2 * i + 1) / (2 * len(frames)), abs_tol=1e-8), f"Frame midpoint request differs: {i}")
    for spec in frozen["source_files"].values():
        require(inputs.hash(spec["path"]) == spec["sha256"], f"Frozen source changed: {spec['path']}")
    sources = frozen["runner"]["source_hashes"]
    require(CORE <= set(sources), "Frozen runner/model source records are missing")
    require(frozen["runner"]["module"] == "day1_compare.run_multi_gpu", "Expected the v2 multi-GPU inference runner")
    for name, checksum in sources.items():
        require(inputs.hash(name) == checksum, f"Frozen runner/model source changed: {name}")
    transition = frozen["runtime_source_transition"]
    old_runner_path = transition["old_source"]["path"]
    require(inputs.hash(old_runner_path) == transition["old_sha256"], "Original runner snapshot changed")
    require(transition["old_source"]["sha256"] == transition["old_sha256"], "Original runner snapshot hash differs")
    require(inputs.hash(transition["diff"]["path"]) == transition["diff"]["sha256"], "Authorized runner patch changed")
    old_runner = inputs.raw[inputs.path(old_runner_path)].decode("utf-8")
    new_runner = inputs.raw[inputs.path("day1_compare/run.py")].decode("utf-8")
    authorized = old_runner.replace("def main():", "def main(*, allowed_gpus=(4, 5)):", 1).replace("choices=(4,5), required=True", "choices=allowed_gpus, required=True", 1)
    require(old_runner.count("def main():") == 1 and old_runner.count("choices=(4,5), required=True") == 1 and new_runner == authorized, "Runner transition exceeds the authorized GPU argument change")
    require(transition["new_sha256"] == sources["day1_compare/run.py"] == inputs.hash("day1_compare/run.py"), "New runner transition hash differs")
    reuse = inputs.read(frozen["reuse_manifest"]["path"], frozen["reuse_manifest"]["sha256"])
    require(reuse["question_ids"] == pilot_ids, "Reused question IDs differ")
    saved_pilot_audit = inputs.read(frozen["pilot_pair_audit"]["path"], frozen["pilot_pair_audit"]["sha256"])
    require(saved_pilot_audit["status"] == "passed" and saved_pilot_audit["valid_for_paired_accuracy"] is True, "Frozen pilot audit did not pass")
    saved_generation = inputs.read(frozen["generation_config"]["path"], frozen["generation_config"]["sha256"])
    marker_spec = frozen["checkpoint_verification"]
    marker = inputs.read(marker_spec["path"], marker_spec["sha256"])
    require(marker["revision"] == frozen["paired_provenance"]["model_revision"], "Checkpoint revision differs")

    directories = {"pilot_on": inputs.path(pilot_on), "pilot_off": inputs.path(pilot_off)}
    shard_index = {shard["name"]: shard for shard in shards}
    for shard in shards:
        for arm in ("on", "off"):
            directory = inputs.path(shard["outputs"][arm])
            require(directory == inputs.path(new_root) / shard["name"] / arm, f"{shard['name']}: continuation root was reassigned")
            directories[f"{shard['name']}_{arm}"] = directory
    runs, technical_failures = {}, []
    baseline = None
    for label, directory in directories.items():
        partition, arm = label.rsplit("_", 1)
        batch = batches[partition]
        expected = batch["ids"]
        if partition == "pilot":
            spec = reuse["arms"][arm]
            require(directory == inputs.path(spec["directory"]), f"{label}: pilot directory was reassigned")
            require(set(RUN_FILES) | {f"wire_{i:04d}.json" for i in range(len(expected))} <= set(spec["files"]), f"{label}: missing frozen run files")
            for name, checksum in spec["files"].items():
                require(inputs.hash(directory / name) == checksum, f"{label}: frozen pilot artifact changed: {name}")
        else:
            spec = shard_index[partition]
            require(directory == inputs.path(spec["outputs"][arm]), f"{label}: continuation directory was reassigned")
        data = {name.removesuffix(".json"): inputs.read(directory / name) for name in RUN_FILES}
        prov, status, rows = (data[name] for name in ("provenance", "status", "predictions"))
        require(status.get("phase") == "finished", f"{label}: job has not finished ({status.get('phase')})", pending=status.get("phase") != "stopped_on_error")
        require(list(index(rows, label)) == expected, f"{label}: predictions are missing, extra, or out of order")
        require(status["attempted"] == status["total"] == len(expected), f"{label}: final attempt counts differ")
        require(status["completed"] == sum(row["status"] == "complete" for row in rows), f"{label}: completed count differs")
        require(status.get("arm") == arm and prov["settings"]["fluxmem"] == arm, f"{label}: arm differs")
        if partition != "pilot":
            require(prov["settings"]["gpu"] == prov["gpu"]["physical_index"] == spec["gpu"], f"{label}: physical GPU differs from assigned shard")
        require(inputs.path(prov["settings"]["input"]) == batch["base"], f"{label}: original input path differs")
        require(inputs.path(prov["settings"]["output"]) == directory, f"{label}: original output path differs")
        require(prov["public_questions_sha256"] == inputs.hash(batch["base"] / "public_questions.json") and prov["frames_manifest_sha256"] == inputs.hash(batch["base"] / "source_frames_manifest.json"), f"{label}: original input hashes differ")
        require(prov["training"] is False and prov["no_label_file_opened"] is True, f"{label}: inference-only constraints differ")
        require(prov["frames"] == len(frames) and math.isclose(prov["effective_fps"], fps), f"{label}: frame budget differs")
        settings = {k: v for k, v in prov["settings"].items() if k not in {"input", "output", "gpu", "fluxmem"}}
        require(settings == frozen["inference_settings"], f"{label}: inference settings differ beyond FluxMem/GPU/input/output")
        for key, value in frozen["paired_provenance"].items():
            require(prov.get(key) == value, f"{label}: frozen provenance differs: {key}")
        require(CORE <= set(prov["source_hashes"]), f"{label}: missing core source hashes")
        for key in CORE | {key for key in sources if key.startswith(MODEL_PREFIX)}:
            expected_hash = transition["old_sha256"] if partition == "pilot" and key == "day1_compare/run.py" else sources[key]
            require(prov["source_hashes"].get(key) == expected_hash, f"{label}: core source differs: {key}")
        if partition != "pilot":
            require(prov["source_hashes"].get("day1_compare/run_multi_gpu.py") == sources["day1_compare/run_multi_gpu.py"], f"{label}: multi-GPU wrapper changed")
        video, attention = data["processed_video"], data["attention_runtime"]
        require({"pixel_values_videos", "video_grid_thw"} <= set(video["hashes"]) and video["shapes"]["video_grid_thw"] == [1, 3] and math.isclose(video["effective_fps"], fps), f"{label}: invalid native video")
        require(attention.get("checkpoint_use_sliding_window") is False and "effective_sliding_window" in attention and attention["effective_sliding_window"] is None, f"{label}: attention correction differs")
        current_generation = generation_sidecar(prov, inputs.root)
        require(current_generation == saved_generation and current_generation["transformers_version"] == prov["transformers"], f"{label}: effective generation settings differ")
        if baseline is None:
            baseline = data
        else:
            for key in PAIRED_FIELDS:
                require(key in prov and prov[key] == baseline["provenance"][key], f"{label}: paired provenance differs: {key}")
            require(video == baseline["processed_video"] and attention == baseline["attention_runtime"], f"{label}: processed video or attention differs")
            shared = set(prov["source_hashes"]) & set(baseline["provenance"]["source_hashes"])
            if partition != "pilot":
                shared.discard("day1_compare/run.py")  # Exact two-line transition checked above.
            require(all(prov["source_hashes"][k] == baseline["provenance"]["source_hashes"][k] for k in shared), f"{label}: common source hashes differ")
        require(set(directory.glob("wire_*.json")) == {directory / f"wire_{i:04d}.json" for i in range(len(expected))}, f"{label}: wire files missing or extra")
        data["wires"] = []
        for ordinal, row in enumerate(rows):
            qid = row["question_id"]
            wire = inputs.read(directory / f"wire_{ordinal:04d}.json")
            data["wires"].append(wire)
            prompt = public_prompt(full_index[qid], duration=duration, frames=len(frames))
            require(wire["question_id"] == qid and wire["prompt"] == prompt and wire["prompt_sha256"] == digest(prompt.encode()), f"{label}: prompt mismatch for {qid}")
            token_hash = wire["input_ids_sha256"]
            require(isinstance(token_hash, str) and len(token_hash) == 64 and all(c in "0123456789abcdef" for c in token_hash), f"{label}: invalid input tensor hash for {qid}")
            require(wire["processed_video_hashes"] == video["hashes"] and wire["second_per_grid_ts"] == [2 / fps], f"{label}: video/timing wire mismatch for {qid}")
            require(row["status"] in {"complete", "truncated", "execution_error"}, f"{label}: unsupported status for {qid}")
            if row["status"] != "complete":
                technical_failures.append({"run": label, "question_id": qid, "status": row["status"]})
            if row["status"] != "execution_error":
                require(row["input_tokens"] == wire["input_tokens"], f"{label}: input token count mismatch for {qid}")
                events = row["memory_events"]
                require(isinstance(events, list) and len(events) == (1 if arm == "on" else 0), f"{label}: native FluxMem switch mismatch for {qid}")
                if arm == "on":
                    event = events[0]
                    require(event["input_tokens"] == wire["input_tokens"] and 0 <= event["visual_tokens_after"] <= event["visual_tokens_before"] and event["input_tokens"] - event["retained_tokens"] == event["dropped_tokens"] == event["visual_tokens_before"] - event["visual_tokens_after"] and event["temporal_calls"] > 0 and event["spatial_calls"] > 0, f"{label}: native memory event mismatch for {qid}")
                tokens = row["output_token_ids"]
                require(len(tokens) == row["output_tokens"], f"{label}: output token count mismatch for {qid}")
                if row["status"] == "complete":
                    eos = saved_generation["effective_generation_config"]["eos_token_id"]
                    require(bool(tokens) and tokens[-1] in (eos if isinstance(eos, list) else [eos]) and bool(row.get("prediction", "").strip()), f"{label}: invalid complete generation for {qid}")
            if partition == "pilot":
                frozen_row = reuse["arms"][arm]["rows"][ordinal]
                require(frozen_row["question_id"] == qid and frozen_row["pilot_ordinal"] == ordinal and frozen_row["full_ordinal"] == ids.index(qid) and frozen_row["prediction_row_sha256"] == digest(canonical_bytes(row)) and frozen_row["wire_filename"] == f"wire_{ordinal:04d}.json" and frozen_row["wire_sha256"] == inputs.hash(directory / frozen_row["wire_filename"]), f"{label}: reused row origin differs for {qid}")
        runs[label] = data
    partitions = ["pilot", *(shard["name"] for shard in shards)]
    for partition in partitions:
        require(runs[f"{partition}_on"]["wires"] == runs[f"{partition}_off"]["wires"], f"{partition}: ON/OFF question text/tensor/frame wires differ")
        left, right = (runs[f"{partition}_{arm}"]["provenance"] for arm in ("on", "off"))
        require(left["source_hashes"] == right["source_hashes"], f"{partition}: paired source hashes differ")
        if partition != "pilot":
            require(left["gpu"] == right["gpu"], f"{partition}: ON/OFF physical GPUs differ")
    # Recompute official checkpoint digests once, using CPU file I/O only.
    verified = verify_weights(inputs.path(frozen["inference_settings"]["model"]), str(inputs.path(marker_spec["path"])))
    require(verified == marker, "Checkpoint verification marker changed")
    combined, origins = {}, []
    for arm in ("on", "off"):
        by_id = {}
        for partition in partitions:
            label = f"{partition}_{arm}"
            directory = directories[label]
            for ordinal, row in enumerate(runs[label]["predictions"]):
                qid = row["question_id"]
                require(qid not in by_id, f"{arm}: duplicate merged question ID {qid}")
                by_id[qid] = row
                origins.append({"question_id": qid, "arm": arm, "partition": partition,
                                "source_ordinal": ordinal, "combined_ordinal": ids.index(qid),
                                "prediction_row_sha256": digest(canonical_bytes(row)),
                                "predictions_file": inputs.source(directory / "predictions.json"),
                                "provenance_file": inputs.source(directory / "provenance.json"),
                                "wire_file": inputs.source(directory / f"wire_{ordinal:04d}.json"),
                                "source_snapshot": f"sources/{label}"})
        require(set(by_id) == set(ids), f"{arm}: merged coverage differs")
        combined[arm] = [by_id[qid] for qid in ids]
    summary, comparison = evaluate(batches["all"]["questions"], batches["all"]["labels"], combined["on"], combined["off"], batches["all"]["protocol"])
    valid = not technical_failures
    audit = {"schema_version": 1, "status": "passed" if valid else "failed", "valid_for_paired_accuracy": valid,
             "expected_questions": expected_questions, "reused_questions": expected_reused,
             "new_questions": counts["remaining"], "failures": technical_failures,
             "provenance_and_exact_coverage_verified": True,
             "note": "Generation failures remain in all-question denominators and prevent a technically successful comparison."}
    summary["technical_audit"] = audit
    summary["input_files"] = {"questions": inputs.source(batches["all"]["base"] / "public_questions.json"),
                              "labels": inputs.source(batches["all"]["base"] / "eval_labels.json"),
                              "protocol": inputs.source(batches["all"]["base"] / "protocol.json"),
                              "run_manifest": inputs.source(package / "run_manifest.json")}
    summary["aggregation"] = {"reused_questions": expected_reused, "new_questions": counts["remaining"],
                              "row_origins": "row_origins.json", "original_sources": "sources/"}
    summary["hardware_strata"] = []
    for partition in partitions:
        selected = batches[partition]["ids"]
        group_summary, _ = evaluate([full_index[qid] for qid in selected], [full_labels[qid] for qid in selected],
                                    runs[f"{partition}_on"]["predictions"], runs[f"{partition}_off"]["predictions"])
        summary["hardware_strata"].append({"partition": partition, "selected_questions": len(selected),
            "gpu": {arm: runs[f"{partition}_{arm}"]["provenance"]["gpu"] for arm in ("on", "off")},
            "arms": group_summary["arms"], "paired": group_summary["paired"]})
    summary["hardware_timing_note"] = "Continuation ON/OFF pairs use the same physical GPU per shard. Pilot ON/OFF used separate GPUs. GPU models differ across shards; pooled observed times do not estimate a hardware-controlled speedup."
    summary["limitations"][0] = "Results cover every question in this Day1 package and the supplied sparse full-day visual input."
    summary["limitations"][2] = "This adapted Day1 comparison does not establish general benchmark accuracy or a speed advantage."
    targets: dict[str, bytes] = {}
    def emit(name: str, value: Any) -> None:
        targets[name] = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    emit("summary.json", summary)
    emit("comparison.json", comparison)
    emit("technical_audit.json", audit)
    emit("row_origins.json", sorted(origins, key=lambda row: (row["combined_ordinal"], row["arm"])))
    for arm, rows in combined.items():
        emit(f"{arm}/predictions.json", rows)
    table = io.StringIO(newline="")
    writer = csv.DictWriter(table, fieldnames=["question_id", "question", "gold_letter", "on_status", "on_prediction", "on_parsed_answer", "on_correct", "off_status", "off_prediction", "off_parsed_answer", "off_correct", "paired_correctness"])
    writer.writeheader()
    for row in comparison:
        flat = {key: row[key] for key in ("question_id", "question", "gold_letter", "paired_correctness")}
        for arm in ("on", "off"):
            flat.update({f"{arm}_{key}": row[arm][key] for key in ("status", "prediction", "parsed_answer", "correct")})
        writer.writerow(flat)
    targets["comparison.csv"] = table.getvalue().encode("utf-8")
    for label, directory in directories.items():
        for name in (*RUN_FILES, *(f"wire_{i:04d}.json" for i in range(len(runs[label]["predictions"])))):
            targets[f"sources/{label}/{name}"] = inputs.raw[directory / name]
    emit("aggregation_manifest.json", {"schema_version": 1, "run_manifest": inputs.source(package / "run_manifest.json"),
        "question_ids": ids, "sources": {label: {"directory": str(directory), "files": {name: inputs.source(directory / name) for name in RUN_FILES}} for label, directory in directories.items()},
        "provenance_policy": "Raw records and original run provenance are preserved unchanged; row_origins.json identifies each source row. Combined prediction files are derived artifacts, not new inference runs."})
    inputs.unchanged()
    # Exclusive directory creation protects even empty preexisting directories.
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in targets.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw)
    completed = {"status": "complete", "technical_status": audit["status"], "questions": expected_questions,
                 "files": {name: digest(raw) for name, raw in targets.items()}}
    with (output / "completed.json").open("x", encoding="utf-8") as stream:
        json.dump(completed, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("package", "pilot-on", "pilot-off", "new-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = aggregate(**vars(args))
    except (ValidationError, ValueError, KeyError, TypeError, OSError) as exc:
        pending = isinstance(exc, ValidationError) and exc.pending
        print(json.dumps({"status": "pending" if pending else "failed", "error": str(exc), "output_written": False}), file=sys.stderr)
        return 2 if pending else 1
    print(json.dumps({"status": summary["technical_audit"]["status"], "selected_questions": summary["selected_questions"],
                      "on_correct": summary["arms"]["on"]["correct"], "off_correct": summary["arms"]["off"]["correct"],
                      "output": str(args.output)}, ensure_ascii=False))
    return 0 if summary["technical_audit"]["valid_for_paired_accuracy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
