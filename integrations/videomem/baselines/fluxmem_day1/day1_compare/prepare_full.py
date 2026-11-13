"""Freeze all Day1 QA plus an exact continuation of the completed v2 pilot.

CPU-only preparation: reuse verified PNGs and raw pilot results; never decode
media, load a model, run inference, or select questions using labels or scores.
Existing output paths are refused, including empty directories.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from day1_compare.check_pair import check_pair
from day1_compare.evaluate import parse_answer
from day1_compare.prepare import (
    EXPECTED_DAY_SECONDS, SELECTION_SALT, canonical_bytes, evaluation_labels,
    public_question, select_questions, selected_provenance, sha256_file,
)
from day1_compare.run import public_prompt

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_POPULATION = 249
EXPECTED_REUSED = 16
EXPECTED_FRAMES = 1024
PACKAGE_FILES = (
    "protocol.json", "eval_labels.json", "source_frames_manifest.json", "public_questions.json",
)
INFERENCE_KEYS = (
    "model", "verification", "max_new_tokens", "short_frames", "medium_frames", "video_chunk_t",
)
PROVENANCE_KEYS = (
    "upstream_commit", "model_revision", "torch", "transformers", "dtype", "attention",
    "decoder", "video_chunk_t", "frames", "effective_fps", "checkpoint", "temporal_positions",
)
GPU_REPLACEMENTS = (
    ("def main():", "def main(*, allowed_gpus=(4, 5)):"),
    ("p.add_argument('--gpu', type=int, choices=(4,5), required=True)",
     "p.add_argument('--gpu', type=int, choices=allowed_gpus, required=True)"),
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def record(path: Path) -> dict:
    return {"path": str(path.resolve(strict=True)), "sha256": sha256_file(path)}


def object_sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def save_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def package_record(directory: Path) -> dict:
    return {
        "path": str(directory.resolve(strict=True)),
        "completed_sha256": sha256_file(directory / "completed.json"),
        "files": {name: sha256_file(directory / name) for name in PACKAGE_FILES},
    }


def runtime_transition(provenance: dict, root: Path) -> dict | None:
    """Permit only the two explicitly authorized GPU allowlist source edits."""
    source = "day1_compare/run.py"
    original_hash = provenance["source_hashes"][source]
    current = root / source
    if sha256_file(current) == original_hash:
        return None
    original = root / "reports/day1_full_v2_runtime/run_pilot_v2.py.txt"
    patch = root / "reports/day1_full_v2_runtime/gpu_allowlist_only.patch"
    if sha256_file(original) != original_hash:
        raise ValueError("Saved original runtime does not match pilot source hash")
    expected = original.read_text(encoding="utf-8")
    for old, new in GPU_REPLACEMENTS:
        if expected.count(old) != 1:
            raise ValueError("GPU allowlist replacement is not uniquely identified")
        expected = expected.replace(old, new, 1)
    if current.read_bytes() != expected.encode("utf-8"):
        raise ValueError("Runtime change exceeds authorized GPU allowlist edits")
    return {
        "source": source, "old_sha256": original_hash, "new_sha256": sha256_file(current),
        "old_source": record(original), "diff": record(patch),
        "allowed_replacements": [{"old": old, "new": new} for old, new in GPU_REPLACEMENTS],
        "scope": "GPU allowlist only; all inference semantics unchanged",
    }


def freeze_reuse(questions: list[dict], on: Path, off: Path, *, root: Path,
                 transition: dict | None = None) -> dict:
    """Bind each reused row and wire to immutable original raw files.

    Parsing establishes technical eligibility only; gold labels are never
    consulted here. A single invalid row rejects the entire reuse package.
    """
    ids = [q["question_id"] for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate pilot question IDs")
    arms = {}
    for name, directory in (("on", on), ("off", off)):
        rows = read_json(directory / "predictions.json")
        status = read_json(directory / "status.json")
        provenance = read_json(directory / "provenance.json")
        video = read_json(directory / "processed_video.json")
        if [row["question_id"] for row in rows] != ids:
            raise ValueError(f"{name}: reused predictions must exactly match pilot question order")
        if status.get("phase") != "finished" or any(
            status.get(key) != len(ids) for key in ("total", "attempted", "completed")
        ):
            raise ValueError(f"{name}: reused run is not complete")
        if provenance["settings"]["fluxmem"] != name:
            raise ValueError(f"{name}: reused run has wrong arm")
        for source in ("day1_compare/run.py", "day1_compare/run_v2.py", "local_videomem/run.py"):
            expected_hash = sha256_file(root / source)
            if transition is not None and source == transition["source"]:
                expected_hash = transition["old_sha256"]
            if provenance["source_hashes"].get(source) != expected_hash:
                raise ValueError(f"{name}: frozen runtime source changed: {source}")
        mappings = []
        filenames = ["predictions.json", "provenance.json", "processed_video.json", "status.json", "attention_runtime.json"]
        for ordinal, (question, row) in enumerate(zip(questions, rows)):
            if row.get("status") != "complete" or parse_answer(row.get("prediction"), len(question["choices"])) is None:
                raise ValueError(f"{name}: reused answer must be complete and parse-valid: {question['question_id']}")
            filename = f"wire_{ordinal:04d}.json"
            wire = read_json(directory / filename)
            prompt = public_prompt(question, duration=question["query_time_seconds"], frames=provenance["frames"])
            if wire.get("question_id") != question["question_id"] or wire.get("prompt") != prompt:
                raise ValueError(f"{name}: reused wire does not match public question: {ordinal}")
            if wire.get("processed_video_hashes") != video["hashes"]:
                raise ValueError(f"{name}: reused wire video mismatch: {ordinal}")
            filenames.append(filename)
            mappings.append({
                "question_id": question["question_id"], "full_ordinal": ordinal,
                "pilot_ordinal": ordinal, "prediction_row_sha256": object_sha(row),
                "wire_filename": filename, "wire_sha256": sha256_file(directory / filename),
            })
        if len(list(directory.glob("wire_*.json"))) != len(ids):
            raise ValueError(f"{name}: unexpected reused wire files")
        arms[name] = {
            "directory": str(directory.resolve(strict=True)),
            "files": {filename: sha256_file(directory / filename) for filename in filenames},
            "rows": mappings,
        }
    return {
        "schema_version": "day1_full_v2_pilot_reuse_v1", "question_ids": ids,
        "eligibility": "Every frozen v2 pilot row must be complete and parse-valid in both arms; no gold-dependent selection or retry",
        "arms": arms,
    }


def prepare_full(pilot: Path, on: Path, off: Path, output: Path, *, root: Path = ROOT) -> dict:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace an existing preparation: {output}")
    pilot, on, off = (path.resolve(strict=True) for path in (pilot, on, off))
    protocol = read_json(pilot / "protocol.json")
    frames_manifest = read_json(pilot / "source_frames_manifest.json")
    pilot_questions = read_json(pilot / "public_questions.json")
    frames = frames_manifest["frames"]
    if len(pilot_questions) != EXPECTED_REUSED or len(frames) != EXPECTED_FRAMES:
        raise ValueError("Requires the frozen 16-question, 1024-frame pilot")
    if protocol["selection"]["salt"] != SELECTION_SALT or protocol["query_time_seconds"] != EXPECTED_DAY_SECONDS:
        raise ValueError("Frozen pilot ranking or Day1 cutoff differs")
    source_files = {}
    for key in ("qa", "timeline"):
        source = Path(protocol[f"source_{key}_path"])
        source_files[key] = record(source)
        if source_files[key]["sha256"] != protocol[f"source_{key}_sha256"]:
            raise ValueError(f"Original source {key} hash changed")
    rows = read_json(Path(source_files["qa"]["path"]))
    if len(rows) != EXPECTED_POPULATION:
        raise ValueError("Requires the entire audited 249-question Day1 population")
    ranked = select_questions(rows, len(rows), salt=SELECTION_SALT)
    ids = [row["qa_id"] for row in ranked]
    if object_sha(sorted(ids)) != protocol["selection"]["population_ids_sha256"]:
        raise ValueError("Day1 population ID hash differs from the pilot")
    questions = [public_question(row, protocol["query_time_seconds"], frames) for row in ranked]
    if questions[:EXPECTED_REUSED] != pilot_questions:
        raise ValueError("First 16 public questions must exactly equal the frozen pilot")
    if evaluation_labels(ranked[:EXPECTED_REUSED]) != read_json(pilot / "eval_labels.json"):
        raise ValueError("Pilot evaluator labels differ from original source")
    for frame in frames:
        if not Path(frame["path"]).is_absolute():
            raise ValueError("Reused PNG paths must be absolute")

    # Fresh independent CPU audit verifies every pilot PNG, source hash, wire,
    # tensor hash, EOS termination, native switch and completed paired status.
    audit, sidecars = check_pair(pilot, on, off, root=root, runner="day1_compare/run_v2.py")
    if audit["status"] != "passed":
        raise ValueError(f"Pilot reuse audit did not pass: {audit['status']}; failures={audit['failures']}; pending={audit['pending']}")
    provenance = read_json(on / "provenance.json")
    transition = runtime_transition(provenance, root)
    reuse = freeze_reuse(pilot_questions, on, off, root=root, transition=transition)
    settings = {key: provenance["settings"][key] for key in INFERENCE_KEYS}
    expected_settings = {"max_new_tokens": 16, "short_frames": 8, "medium_frames": 512, "video_chunk_t": 1}
    if any(settings[key] != value for key, value in expected_settings.items()):
        raise ValueError("Pilot inference settings differ from the frozen v2 condition")
    checkpoint_verification = Path(settings["verification"])
    if not checkpoint_verification.is_absolute():
        checkpoint_verification = root / checkpoint_verification
    checkpoint_record = record(checkpoint_verification)
    marker = read_json(checkpoint_verification)
    if marker["revision"] != provenance["model_revision"]:
        raise ValueError("Checkpoint verification revision differs from pilot")
    frame_content_sha256 = object_sha(frames)
    core_hashes = {name: value for name, value in provenance["source_hashes"].items()
                   if name.startswith("models/qwen2-5-vl/") or name in {
                       "day1_compare/run.py", "day1_compare/run_v2.py", "local_videomem/run.py"}}
    core_hashes["day1_compare/run.py"] = sha256_file(root / "day1_compare/run.py")
    core_hashes["day1_compare/run_multi_gpu.py"] = sha256_file(root / "day1_compare/run_multi_gpu.py")
    remaining_count = len(ids) - EXPECTED_REUSED
    shard_specs = []
    cursor = 0
    for index in range(5):
        size = remaining_count // 5 + (index < remaining_count % 5)
        if size <= 0:
            raise ValueError("Five GPU shards require at least five remaining questions")
        name = f"shard_{index + 1:02d}"
        shard_specs.append({
            "name": name, "remaining_start": cursor, "remaining_stop": cursor + size,
            "question_ids": ids[EXPECTED_REUSED + cursor:EXPECTED_REUSED + cursor + size],
            "gpu": index + 1, "arm_order": ["on", "off"] if index % 2 == 0 else ["off", "on"],
            "outputs": {arm: str(root / "local_runs/day1_full_v2" / name / arm) for arm in ("on", "off")},
        })
        cursor += size
    # All validation above occurs before publication; x-mode refuses overwrites.
    output.mkdir(parents=True, exist_ok=False)
    packages = {}
    selections = [("all", 0, len(ids)), ("remaining", EXPECTED_REUSED, len(ids))]
    selections.extend((f"shards/{s['name']}", EXPECTED_REUSED + s["remaining_start"],
                       EXPECTED_REUSED + s["remaining_stop"]) for s in shard_specs)
    for name, start, stop in selections:
        directory = output / name
        directory.mkdir(parents=True)
        subset = ranked[start:stop]
        public = questions[start:stop]
        current = copy.deepcopy(protocol)
        current["schema_version"] = "day1_original_qwen_native_fluxmem_full_v2"
        current["prepare_script_sha256"] = sha256_file(Path(__file__))
        current["selection"].update(count=len(subset), selected_ids=[row["qa_id"] for row in subset])
        current["selected_records"] = [selected_provenance(row) for row in subset]
        current["augmentation"] = {
            "package": name, "population_count": len(ranked), "reused_count": EXPECTED_REUSED,
            "remaining_count": len(ranked) - EXPECTED_REUSED,
            "source_pilot_protocol_sha256": sha256_file(pilot / "protocol.json"),
            "frame_content_sha256": frame_content_sha256,
            "content_order": "video_then_question", "runner_module": "day1_compare.run_multi_gpu",
            "new_frame_extraction": False, "question_ranking_uses_answers_or_model_scores": False,
            "pilot_reuse": "First 16 questions reuse every completed, parse-valid v2 pilot result in both arms",
            "new_inference_scope": "Only the remaining 233 question IDs, in unchanged SHA ranking",
        }
        save_new(directory / "protocol.json", current)
        save_new(directory / "eval_labels.json", evaluation_labels(subset))
        manifest = copy.deepcopy(frames_manifest)
        manifest["protocol_sha256"] = sha256_file(directory / "protocol.json")
        save_new(directory / "source_frames_manifest.json", manifest)
        save_new(directory / "public_questions.json", public)
        completion = {
            "status": "complete", "questions": len(public), "frames": len(frames),
            "questions_with_zero_in_scope_sampled_frames": sum(
                all(alias["sampled_frame_count"] == 0 for alias in q["input_aliases"].values()) for q in public),
            "files": {filename: sha256_file(directory / filename) for filename in PACKAGE_FILES},
        }
        save_new(directory / "completed.json", completion)
        packages[name] = package_record(directory)
    for spec in shard_specs:
        spec["package"] = packages.pop(f"shards/{spec['name']}")
    save_new(output / "reuse_manifest.json", reuse)
    save_new(output / "pilot_pair_check.json", audit)
    save_new(output / "pilot_generation_config.json", sidecars["on"])
    result = {
        "schema_version": "day1_full_v2_augmentation_v1", "status": "prepared",
        "counts": {"all": len(ids), "reused": EXPECTED_REUSED, "remaining": len(ids) - EXPECTED_REUSED},
        "question_ids": {"all": ids, "reused": ids[:EXPECTED_REUSED], "remaining": ids[EXPECTED_REUSED:]},
        "packages": packages, "pilot_package": package_record(pilot),
        "reuse_manifest": record(output / "reuse_manifest.json"),
        "pilot_pair_audit": record(output / "pilot_pair_check.json"),
        "generation_config": record(output / "pilot_generation_config.json"),
        "source_files": source_files, "checkpoint_verification": checkpoint_record,
        "frame_content_sha256": frame_content_sha256,
        "inference_settings": settings,
        "paired_provenance": {key: provenance[key] for key in PROVENANCE_KEYS},
        "runner": {"module": "day1_compare.run_multi_gpu", "source_hashes": core_hashes},
        "runtime_source_transition": transition, "shards": shard_specs,
        "gpu_schedule": "Five contiguous deterministic shards; both arms sequential on each physical GPU 1-5; first arm alternates by shard",
        "runtime_inputs": ["public_questions.json", "source_frames_manifest.json"],
        "evaluator_only_files": ["eval_labels.json"],
        "policy": "Reuse all 16 valid v2 pilot answers and run each of the remaining 233 once per arm; no retries, selection by answers, or v1 substitution",
    }
    save_new(output / "run_manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=ROOT / "artifacts/day1_pilot_v1")
    parser.add_argument("--on", type=Path, default=ROOT / "local_runs/day1_v2_on")
    parser.add_argument("--off", type=Path, default=ROOT / "local_runs/day1_v2_off")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/day1_full_v2")
    args = parser.parse_args()
    manifest = prepare_full(args.pilot, args.on, args.off, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "status": manifest["status"],
                      "counts": manifest["counts"], "new_frame_extraction": False}, indent=2))


if __name__ == "__main__":
    main()
