"""Freeze a Day1 pilot and decode a common, question-independent PNG cache.

This is an adapted end-of-Day1 evaluation, not the unchanged HD-EPIC task.
No annotations, model packages, inference, or GPU access are used here.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_QA_DIR = Path(
    "/home/kai/video_timetable/videomem_20261107/"
    "daywise_annotations_qa_v2_20261108/Day1"
)
DEFAULT_MEDIA_ROOT = Path("/home/kai/lifelong_mem/routine_7days")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "artifacts/day1_pilot_v1"
SELECTION_SALT = "day1-pilot-v1"
EXPECTED_DAY_SECONDS = 39016.8


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def select_questions(rows: list[dict], count: int, salt: str = SELECTION_SALT) -> list[dict]:
    """Select using IDs alone: SHA256(UTF8(salt + qa_id)), no separator."""
    ids = [row["qa_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("QA IDs must be unique")
    if not 0 < count <= len(rows):
        raise ValueError("Pilot count must be between one and the population size")
    return sorted(
        rows,
        key=lambda row: (
            hashlib.sha256((salt + row["qa_id"]).encode("utf-8")).hexdigest(),
            row["qa_id"],
        ),
    )[:count]


def sampling_plan(segments: list[dict], frame_count: int) -> list[dict]:
    """Uniform temporal midpoints over the entire ordered concatenation."""
    if not segments or frame_count <= 0:
        raise ValueError("At least one segment and one sample are required")
    cursor = Decimal(0)
    ends = []
    for segment in segments:
        start = Decimal(str(segment["concat_start_seconds"]))
        end = Decimal(str(segment["concat_end_seconds"]))
        duration = Decimal(str(segment["video_duration_seconds"]))
        if start != cursor or end <= start or end - start != duration:
            raise ValueError("Timeline must be contiguous, ordered, and duration preserving")
        cursor = end
        ends.append(end)
    result = []
    for index in range(frame_count):
        position = cursor * (2 * index + 1) / (2 * frame_count)
        segment_index = bisect_right(ends, position)
        segment = segments[segment_index]
        local = position - Decimal(str(segment["concat_start_seconds"]))
        result.append({
            "sample_index": index,
            "segment_index": segment_index,
            "source_clip_key": segment["clip_key"],
            "requested_global_seconds": float(position),
            "requested_source_seconds": float(local),
        })
    return result


def public_question(row: dict, cutoff: float, frames: list[dict]) -> dict:
    """Explicit allowlist; original QA, statistics, and labels are never copied."""
    aliases = {}
    for name, source in row["mapped_inputs"].items():
        start, end = source["day_start_seconds"], source["day_end_seconds"]
        count = sum(start <= frame["actual_global_seconds"] < end for frame in frames)
        aliases[name] = {
            "source_id": source["original_source_video_id"],
            "source_start_seconds": source["source_start_seconds"],
            "source_end_seconds": source["source_end_seconds"],
            "day_start_seconds": start,
            "day_end_seconds": end,
            "sampled_frame_count": count,
        }
    return {
        "question_id": row["qa_id"],
        "question": row["day_time_view"]["question"],
        "choices": row["day_time_view"]["choices"],
        "query_time_seconds": cutoff,
        "input_aliases": aliases,
    }


def evaluation_labels(rows: list[dict]) -> list[dict]:
    return [{"question_id": row["qa_id"], "correct_idx": row["correct_idx"]} for row in rows]


def selected_provenance(row: dict) -> dict:
    return {
        "question_id": row["qa_id"],
        "original_qa_sha256": row["original_qa_sha256"],
        "source_revision": row["source_revision"],
        "source_file": row["source_file"],
        "source_url": row["source_url"],
        "status": row["status"],
        "timeline_basis": row["timeline_basis"],
        "edit_alignment_status": row["edit_alignment_status"],
        "actual_render_alignment_verified": row["actual_render_alignment_verified"],
        "answer_reverified_on_rendered_video": row["answer_reverified_on_rendered_video"],
        "one_hz_visual_evidence_reverified": row["one_hz_visual_evidence_reverified"],
        "original_input_scope_policy": row["input_scope_policy"],
    }


def collect_sources(segments: list[dict], media_root: Path) -> list[dict]:
    import av

    sources = []
    for segment in segments:
        path = (media_root / segment["prepared_clip"]).resolve(strict=True)
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            if stream.duration is None:
                raise ValueError(f"Missing video stream duration: {path}")
            duration = float(stream.duration * stream.time_base)
            if abs(duration - segment["video_duration_seconds"]) > 0.0001:
                raise ValueError(f"Current source duration differs from timeline: {path}")
            if stream.width != 720 or stream.height != 720:
                raise ValueError(f"Expected square 720-pixel source: {path}")
            metadata = {
                "stream_time_base": str(stream.time_base),
                "stream_start_pts": stream.start_time or 0,
                "stream_duration_seconds": duration,
                "width": stream.width,
                "height": stream.height,
                "average_rate": str(stream.average_rate),
            }
        sources.append({
            "clip_key": segment["clip_key"],
            "source_id": segment["source_id"],
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "day_start_seconds": segment["concat_start_seconds"],
            "day_end_seconds": segment["concat_end_seconds"],
            "video_duration_seconds": segment["video_duration_seconds"],
            "probe": metadata,
        })
        print(f"Verified source {len(sources)}/{len(segments)}: {segment['clip_key']}", flush=True)
    return sources


def extract_frames(plan: list[dict], sources: list[dict], output: Path, size: int) -> list[dict]:
    """Seek to each position, then take the first decoded PTS at/after it.

    One video decoder with one codec thread is active at a time. No full video
    or list of decoded images is retained in memory.
    """
    import av
    from PIL import Image

    frame_dir = output / "frames"
    frame_dir.mkdir()
    result = []
    for segment_index, source in enumerate(sources):
        samples = [item for item in plan if item["segment_index"] == segment_index]
        with av.open(source["path"]) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            stream.thread_type = "NONE"
            base = stream.time_base
            origin_pts = stream.start_time or 0
            origin_seconds = float(origin_pts * base)
            for item in samples:
                requested = item["requested_source_seconds"]
                target_pts = origin_pts + int(requested / float(base))
                container.seek(target_pts, stream=stream, backward=True, any_frame=False)
                chosen = None
                for frame in container.decode(stream):
                    if frame.pts is None:
                        continue
                    actual = float(frame.pts * frame.time_base) - origin_seconds
                    if actual + 1e-9 >= requested:
                        chosen = frame
                        break
                if chosen is None:
                    raise RuntimeError(f"No frame at requested source time {requested}: {source['path']}")
                actual = float(chosen.pts * chosen.time_base) - origin_seconds
                if actual - requested > 0.040001:
                    raise RuntimeError(f"Decode PTS more than one source frame late: {requested} -> {actual}")
                global_seconds = source["day_start_seconds"] + actual
                if not source["day_start_seconds"] <= global_seconds < source["day_end_seconds"]:
                    raise RuntimeError("Decoded frame lies outside its timeline segment")
                path = frame_dir / f"frame_{item['sample_index']:04d}.png"
                image = chosen.to_image().convert("RGB")
                image = image.resize((size, size), resample=Image.Resampling.LANCZOS)
                image.save(path, format="PNG", optimize=False)
                result.append({
                    **item,
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "width": size,
                    "height": size,
                    "actual_global_seconds": global_seconds,
                    "actual_source_seconds": actual,
                    "source_pts": chosen.pts,
                    "source_time_base": str(chosen.time_base),
                    "source_start_pts": origin_pts,
                    "source_path": source["path"],
                    "source_sha256": source["sha256"],
                    "source_id": source["source_id"],
                })
                if len(result) % 64 == 0 or len(result) == len(plan):
                    print(f"Decoded {len(result)}/{len(plan)} frames", flush=True)
    return sorted(result, key=lambda frame: frame["sample_index"])


def prepare(args: argparse.Namespace) -> dict:
    qa_path = (args.qa_dir / "qa.json").resolve(strict=True)
    timeline_path = (args.qa_dir / "timeline.json").resolve(strict=True)
    rows = json.loads(qa_path.read_text(encoding="utf-8"))
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    segments = timeline["segments"]
    cutoff = segments[-1]["concat_end_seconds"]
    if cutoff != EXPECTED_DAY_SECONDS or len(segments) != 10:
        raise ValueError("This pilot requires the audited ten-clip Day1 timeline")
    if len(rows) != 249:
        raise ValueError("This pilot requires the audited 249-question Day1 population")
    selected = select_questions(rows, args.questions)
    plan = sampling_plan(segments, args.frames)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to replace an existing preparation: {output}")
    output.mkdir(parents=True, exist_ok=True)
    sources = collect_sources(segments, args.media_root)
    ids = sorted(row["qa_id"] for row in rows)
    import av
    import PIL

    protocol = {
        "schema_version": "day1_original_qwen_native_fluxmem_compare_v1",
        "evaluation_task": "adapted_offline_end_of_Day1_QA",
        "official_task_unchanged": False,
        "task_adaptation": (
            "All selected original questions are asked after the full Day1 concatenation. "
            "Both arms receive the same full-Day uniform frame sample. Original video 1 "
            "scope is stated in each question, but full-Day visual access differs from "
            "the official input-scoped task and may expose later events."
        ),
        "query_time_seconds": cutoff,
        "query_time_basis": "end_of_independently_probed_ten_clip_concatenation",
        "query_time_in_original_qa": False,
        "gold_or_evidence_used_for_cutoff_or_sampling": False,
        "annotations_used": False,
        "fine_tuning": False,
        "model_comparison": "same original Qwen2.5-VL checkpoint; native FluxMem enabled versus disabled",
        "source_qa_path": str(qa_path),
        "source_qa_sha256": sha256_file(qa_path),
        "source_timeline_path": str(timeline_path),
        "source_timeline_sha256": sha256_file(timeline_path),
        "prepare_script_sha256": sha256_file(Path(__file__)),
        "selection": {
            "population_count": len(rows),
            "population_ids_sha256": hashlib.sha256(canonical_bytes(ids)).hexdigest(),
            "salt": SELECTION_SALT,
            "ranking": "ascending SHA256(UTF8(salt + qa_id)), no separator; ID breaks ties",
            "count": len(selected),
            "selected_ids": [row["qa_id"] for row in selected],
            "selection_uses_answers_or_model_scores": False,
        },
        "selected_records": [selected_provenance(row) for row in selected],
        "sampling": {
            "frame_count": args.frames,
            "positions": "duration * (2 * sample_index + 1) / (2 * frame_count)",
            "decode_rule": "first decoded source frame at or after requested midpoint",
            "interval_membership": "start inclusive, end exclusive, using actual decoded PTS",
            "image_width": args.size,
            "image_height": args.size,
            "resampling": "Pillow LANCZOS; RGB PNG",
            "decoder_threads": 1,
            "pyav_version": av.__version__,
            "pillow_version": PIL.__version__,
            "plan_sha256": hashlib.sha256(canonical_bytes(plan)).hexdigest(),
        },
        "sources": sources,
        "limitations": [
            "Original answers have not been visually revalidated on edited 1 Hz cooking clips.",
            "Frame sampling is sparse; individual supplied input intervals may contain no sampled frame.",
            "All Day1 official questions concern the final cooking sources, not all daily activities.",
            "Only source concatenation offsets are established; a separate edited render is not verified.",
        ],
    }
    # Freeze protocol and evaluator labels before decoding or running a model.
    write_json(output / "protocol.json", protocol)
    write_json(output / "eval_labels.json", evaluation_labels(selected))
    frames = extract_frames(plan, sources, output, args.size)
    protocol_sha = sha256_file(output / "protocol.json")
    manifest = {
        "schema_version": "day1_uniform_source_frames_v1",
        "protocol_sha256": protocol_sha,
        "day_duration_seconds": cutoff,
        "total_frames": len(frames),
        "frames": frames,
    }
    questions = [public_question(row, cutoff, frames) for row in selected]
    write_json(output / "source_frames_manifest.json", manifest)
    write_json(output / "public_questions.json", questions)
    completion = {
        "status": "complete",
        "questions": len(questions),
        "frames": len(frames),
        "questions_with_zero_in_scope_sampled_frames": sum(
            all(alias["sampled_frame_count"] == 0 for alias in q["input_aliases"].values())
            for q in questions
        ),
        "files": {
            name: sha256_file(output / name)
            for name in ["protocol.json", "eval_labels.json", "source_frames_manifest.json", "public_questions.json"]
        },
    }
    write_json(output / "completed.json", completion)
    print(json.dumps({"output": str(output), **completion}, indent=2), flush=True)
    return completion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-dir", type=Path, default=DEFAULT_QA_DIR)
    parser.add_argument("--media-root", type=Path, default=DEFAULT_MEDIA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--questions", type=int, default=16)
    parser.add_argument("--frames", type=int, default=1024)
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()
    if args.frames <= 0 or args.size <= 0:
        parser.error("Frame count and image size must be positive")
    prepare(args)


if __name__ == "__main__":
    main()
