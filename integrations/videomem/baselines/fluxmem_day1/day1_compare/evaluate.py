"""Score a preselected Day1 paired run without loading a model.

The two arms must use the same original Qwen2.5-VL weights, prompts, sampled
frames and generation settings, with only ``use_fluxmem`` changed. The native
switch is in models/qwen2-5-vl/src/qwen2_5_vl_fluxmem/
modeling_qwen2_5_vl_fluxmem.py (Qwen2_5_VLModel.forward); visual chunking is in
Qwen2_5_VLForConditionalGeneration.forward and must be identical in both arms.
The released reduction runs during each question's prefill, not as persistent
memory across questions. This evaluator does not infer unsampled evidence or
claim that a small selected subset measures general benchmark performance.

Example: python -m day1_compare.evaluate --questions public_questions.json
  --labels eval_labels.json --on on/predictions.json --off off/predictions.json
  --protocol protocol.json --output comparison

Only this post-inference process reads labels. A valid answer is a single A-E
letter (optionally surrounded by punctuation, or prefixed by ``Answer:``).
Incomplete generation is never scored as correct, even when its raw text can
be parsed. Missing predictions remain in the all-selected denominator.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import string
from typing import Any


_ANSWER = re.compile(r"(?:answer\s*:\s*)?([A-Ea-e])", re.IGNORECASE)
_PUNCTUATION = string.punctuation + "“”‘’。！？：，、（）【】"
_STATUSES = {"complete", "truncated", "execution_error"}


def parse_answer(prediction: Any, num_choices: int = 5) -> str | None:
    """Parse the whole response; never extract a letter from narrative text."""
    if not isinstance(prediction, str):
        return None
    text = prediction.strip().strip(_PUNCTUATION).strip()
    match = _ANSWER.fullmatch(text)
    if match is None:
        return None
    letter = match.group(1).upper()
    return letter if ord(letter) - ord("A") < num_choices else None


def _rows(value: Any, key: str, description: str) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        value = value.get(key)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{description} must be a list of objects or an object containing '{key}'")
    return value


def _index(rows: list[dict[str, Any]], description: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(f"{description} has a missing or invalid question_id")
        if question_id in result:
            raise ValueError(f"{description} has duplicate question_id: {question_id}")
        result[question_id] = row
    return result


def _flags(value: Any) -> Any:
    """Keep coverage/alignment facts where supplied, without manufacturing flags."""
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            if any(part in key.lower() for part in (
                "coverage", "provisional", "verified", "alignment", "sampled_frame_count",
                "timeline_basis", "mapping_basis", "sampling_basis", "sampling_note",
            )):
                result[key] = child
            else:
                selected = _flags(child)
                if selected:
                    result[key] = selected
        return result
    if isinstance(value, list):
        return [selected for child in value if (selected := _flags(child))]
    return {}


def _arm_row(row: dict[str, Any] | None, choices: int, gold: str) -> dict[str, Any]:
    if row is None:
        return {"status": "missing", "prediction": None, "parsed_answer": None,
                "complete": False, "parsed_complete": False, "correct": False,
                "raw_prediction_record": None}
    status = row.get("status")
    if status not in _STATUSES:
        raise ValueError(f"Unsupported prediction status for {row['question_id']}: {status!r}")
    # Deliberately ignore any supplied parsed_answer. The raw response is authoritative.
    answer = parse_answer(row.get("prediction"), choices)
    complete = status == "complete"
    return {"status": status, "prediction": row.get("prediction"), "parsed_answer": answer,
            "complete": complete, "parsed_complete": complete and answer is not None,
            "correct": complete and answer == gold, "raw_prediction_record": row}


def _arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    complete = sum(row["complete"] for row in rows)
    parsed = sum(row["parsed_complete"] for row in rows)
    correct = sum(row["correct"] for row in rows)
    durations = []
    for row in rows:
        raw = row["raw_prediction_record"] or {}
        elapsed = raw.get("generation_seconds")
        if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) and math.isfinite(elapsed) and elapsed >= 0:
            durations.append(float(elapsed))
    return {
        "selected": total, "present_rows": total - sum(row["status"] == "missing" for row in rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "complete": complete, "parsed_complete": parsed,
        "parsed_any_status": sum(row["parsed_answer"] is not None for row in rows),
        "complete_unparseable": complete - parsed, "correct": correct,
        "accuracy_all_selected": correct / total,
        "accuracy_completed": correct / complete if complete else None,
        "accuracy_parsed_completed": correct / parsed if parsed else None,
        "generation_timing": {
            "recorded_rows": len(durations),
            "sum_seconds": sum(durations) if durations else None,
            "mean_seconds": sum(durations) / len(durations) if durations else None,
            "note": "Descriptive observed times; output lengths and completion status may differ.",
        },
    }


def evaluate(questions: Any, labels: Any, on: Any, off: Any,
             protocol: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return aggregate and raw paired comparison; never drop a selected question."""
    question_rows = _rows(questions, "questions", "Questions")
    question_index = _index(question_rows, "Questions")
    if not question_index:
        raise ValueError("No questions were selected")
    label_index = _index(_rows(labels, "labels", "Labels"), "Labels")
    if set(label_index) != set(question_index):
        raise ValueError("Label question IDs must exactly match the selected question IDs")
    arms = {"on": _index(_rows(on, "predictions", "ON predictions"), "ON predictions"),
            "off": _index(_rows(off, "predictions", "OFF predictions"), "OFF predictions")}
    for name, index in arms.items():
        extra = set(index) - set(question_index)
        if extra:
            raise ValueError(f"{name.upper()} predictions contain unselected question IDs: {sorted(extra)}")
    if protocol is not None and not isinstance(protocol, dict):
        raise ValueError("Protocol must be a JSON object")
    protocol = protocol or {}
    selected_records = protocol.get("selected_records", [])
    if isinstance(selected_records, dict):
        protocol_records = selected_records
    elif isinstance(selected_records, list):
        protocol_records = {row.get("question_id", row.get("qa_id")): row
                            for row in selected_records if isinstance(row, dict)}
    else:
        protocol_records = {}
    comparison = []
    for question in question_rows:
        qid = question["question_id"]
        choices = question.get("choices")
        if not isinstance(choices, list) or not 2 <= len(choices) <= 5:
            raise ValueError(f"Question {qid} must have between 2 and 5 choices")
        correct_idx = label_index[qid].get("correct_idx")
        if type(correct_idx) is not int or not 0 <= correct_idx < len(choices):
            raise ValueError(f"Invalid correct_idx for {qid}")
        gold = chr(ord("A") + correct_idx)
        row = {"question_id": qid, "question": question.get("question"), "choices": choices,
               "gold_index": correct_idx, "gold_letter": gold,
               "coverage_and_provisional_flags": {
                   "question": _flags(question), "label": _flags(label_index[qid]),
                   "protocol_record": _flags(protocol_records.get(qid, {})),
               }}
        for name, index in arms.items():
            row[name] = _arm_row(index.get(qid), len(choices), gold)
        row["paired_correctness"] = (
            "both_correct" if row["on"]["correct"] and row["off"]["correct"] else
            "fixed_by_on" if row["on"]["correct"] else
            "broken_by_on" if row["off"]["correct"] else "neither_correct")
        comparison.append(row)
    paired = Counter(row["paired_correctness"] for row in comparison)
    summary = {
        "schema_version": 1, "selected_questions": len(comparison),
        "denominator_policy": "Every selected question; missing, failed, truncated and unparseable answers count as incorrect.",
        "answer_parser": "Full response only: A-E, optional Answer: prefix and surrounding punctuation; valid choice index required.",
        "arms": {name: _arm_summary([row[name] for row in comparison]) for name in arms},
        "paired": {key: paired[key] for key in ("both_correct", "fixed_by_on", "broken_by_on", "neither_correct")},
        "coverage_and_provisional_flags": _flags(protocol),
        "limitations": [
            "Results describe only this preselected sample and the supplied sampled visual input.",
            "Sampling coverage does not verify that the evidence needed to answer is visible.",
            "A small paired run does not establish a general accuracy or speed advantage.",
            "Correctness labels and source timeline alignment are used as supplied; this script does not reverify them.",
        ],
    }
    summary["paired"].update({
        "both_complete": sum(row["on"]["complete"] and row["off"]["complete"] for row in comparison),
        "both_parsed_complete": sum(row["on"]["parsed_complete"] and row["off"]["parsed_complete"] for row in comparison),
        "correct_count_delta_on_minus_off": paired["fixed_by_on"] - paired["broken_by_on"],
        "accuracy_delta_on_minus_off": (paired["fixed_by_on"] - paired["broken_by_on"]) / len(comparison),
    })
    summary["coverage_flags_available"] = bool(summary["coverage_and_provisional_flags"]) or any(
        any(row["coverage_and_provisional_flags"].values()) for row in comparison)
    return summary, comparison


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("questions", "labels", "on", "off", "output"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, help="Optional public protocol and coverage provenance")
    args = parser.parse_args(argv)
    values, sources = {}, {}
    for name in ("questions", "labels", "on", "off", "protocol"):
        path = getattr(args, name)
        if path is not None:
            raw = path.read_bytes()
            values[name] = json.loads(raw)
            sources[name] = {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}
    summary, comparison = evaluate(**values)
    summary["input_files"] = sources
    targets = [args.output / "summary.json", args.output / "comparison.json"]
    encoded = [json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
               for value in (summary, comparison)]
    if any(path.exists() for path in targets):
        raise FileExistsError("Comparison output exists; choose a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    for path, text in zip(targets, encoded):
        # Exclusive creation protects previously written results.
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(encoded[0], end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
