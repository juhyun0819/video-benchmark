"""Remap real source captions onto concatenated daily videos without inventing text.

The caller supplies measured segment boundaries and writes the returned JSON.
Only per-video annotation files are read; the global Ego4D narration dump is never
loaded. Point annotations stay points rather than becoming arbitrary subtitles.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _path(value, manifest_path, mappings):
    path = Path(value)
    if not path.is_absolute():
        if manifest_path is None:
            return None
        path = Path(manifest_path).resolve().parent / path
    path = path.resolve()
    for old, new in sorted(mappings, key=lambda pair: len(str(pair[0])), reverse=True):
        try:
            return (Path(new) / path.relative_to(Path(old))).resolve()
        except ValueError:
            pass
    return path


def _annotation_files(slot, manifest_path, mappings):
    evidence, source = slot.get("evidence") or {}, slot.get("source") or {}
    paths = []
    for data in (evidence, source):
        for key in ("full_caption_file", "caption_path", "annotations_path"):
            if isinstance(data.get(key), str):
                candidate = _path(data[key], manifest_path, mappings)
                if candidate is not None:
                    paths.append(candidate)
    uid = source.get("video_id")
    if uid and source.get("dataset") == "ego4d" and manifest_path:
        paths.append(Path(manifest_path).resolve().parent / "caption-review" / f"{uid}.json")
    if uid and source.get("dataset") == "egoexo4d" and source.get("media_path"):
        # Existing take files live below <dataset>/fps1 or <dataset>/takes/...
        video = _path(source["media_path"], manifest_path, mappings)
        if video:
            for ancestor in video.parents:
                if ancestor.name == "egoexo4d":
                    paths.append(ancestor / "synced_annotations" / "annotations" / f"{uid}.jsonl")
                    break
    return list(dict.fromkeys(paths))


def _canonical(record, provenance, fallback_id):
    """Return a timed caption or None; leave text exactly as supplied."""
    text = record.get("text", record.get("narration_text", record.get("summary_text")))
    if not isinstance(text, str) or not text.strip():
        return None
    time = record.get("time")
    if not isinstance(time, dict):
        if "timestamp_sec" in record:
            time = {"kind": "point", "t": record["timestamp_sec"]}
        elif "start_sec" in record and "end_sec" in record:
            time = {"kind": "interval", "start": record["start_sec"], "end": record["end_sec"]}
        elif "timestamp" in record:
            time = {"kind": "point", "t": record["timestamp"]}
        else:
            return None
    if time.get("kind") == "point" or "t" in time:
        time = {"kind": "point", "t": _number(time.get("t"), "caption point")}
    elif "start" in time and "end" in time:
        a, b = _number(time["start"], "caption start"), _number(time["end"], "caption end")
        if b <= a:
            raise ValueError("Caption interval end must exceed start")
        time = {"kind": "interval", "start": a, "end": b}
    else:
        return None
    origin = {**provenance}
    for key in ("annotation_uid", "track", "level", "pass_name"):
        if record.get(key) is not None:
            origin[key] = record[key]
    original = record.get("original") or {}
    for key in ("ego_visible", "subject", "unsure", "best_exo"):
        if key in original:
            origin[key] = original[key]
    if record.get("id"):
        origin["annotation_id"] = record["id"]
    return {"id": str(record.get("id") or fallback_id), "text": text,
            "source_time": time, "provenance": origin}


def _rows_from_document(document, origin):
    if isinstance(document, list):
        for index, row in enumerate(document):
            if isinstance(row, dict):
                yield row, origin, f"caption:{index}"
    elif isinstance(document, dict):
        passes = [key for key in document if key.startswith("narration_pass_")]
        if passes:
            for pass_name in sorted(passes):
                for track, level in (("narrations", "atomic"), ("summaries", "summary")):
                    for index, row in enumerate(document[pass_name].get(track, [])):
                        yield row, {**origin, "pass_name": pass_name, "track": track, "level": level}, \
                            f"{pass_name}:{track}:{index}"
        elif isinstance(document.get("captions"), list):
            yield from _rows_from_document(document["captions"], origin)
        elif "text" in document or "narration_text" in document:
            yield document, origin, "caption:0"


def _load_captions(slot, manifest_path, mappings):
    evidence = slot.get("evidence") or {}
    alignment = str(evidence.get("alignment", ""))
    if "unknown" in alignment or "video_level_only" in alignment:
        return [], {"status": "unaligned", "alignment": alignment}, []
    warnings = []
    embedded = evidence.get("captions")
    embedded_location = "manifest.evidence.captions"
    if not isinstance(embedded, list):
        embedded = (evidence.get("original") or {}).get("captions")
        embedded_location = "manifest.evidence.original.captions"
    if not isinstance(embedded, list):
        embedded = (slot.get("source", {}).get("metadata") or {}).get("captions")
        embedded_location = "manifest.source.metadata.captions"
    if isinstance(embedded, list):
        document, origin = embedded, {"location": embedded_location}
    else:
        document = None
        origin = None
        for candidate in _annotation_files(slot, manifest_path, mappings):
            if not candidate.is_file():
                continue
            # Per-video inputs are small. Never accidentally load a global dump.
            if candidate.stat().st_size > 128 * 1024 * 1024:
                warnings.append(f"Skipped oversized annotation file: {candidate}")
                continue
            try:
                if candidate.suffix == ".jsonl":
                    with candidate.open(encoding="utf-8") as handle:
                        document = [json.loads(line) for line in handle if line.strip()]
                else:
                    document = json.loads(candidate.read_text(encoding="utf-8"))
                origin = {"path": str(candidate)}
                break
            except (OSError, ValueError) as exc:
                warnings.append(f"Cannot read {candidate}: {exc}")
        if document is None:
            summaries = evidence.get("full_source_summaries")
            if isinstance(summaries, list) and summaries:
                document, origin = summaries, {"location": "manifest.evidence.full_source_summaries", "level": "summary"}
            elif evidence.get("evidence_kind") == "caption" and isinstance(evidence.get("original"), dict):
                document, origin = [evidence["original"]], {"location": "manifest.evidence.original", "scope": "single_evidence_caption_only"}
            elif evidence.get("text") and evidence.get("time") and evidence.get("evidence_kind") == "caption":
                document, origin = [evidence], {"location": "manifest.evidence", "scope": "single_evidence_caption_only"}
            else:
                return [], {"status": "no_aligned_captions_found"}, warnings
    captions = []
    for row, provenance, identifier in _rows_from_document(document, origin):
        try:
            normalized = _canonical(row, provenance, identifier)
            if normalized is not None:
                captions.append(normalized)
        except (TypeError, ValueError) as exc:
            warnings.append(f"Invalid annotation {identifier}: {exc}")
    return captions, {"status": "loaded", **origin, "loaded_caption_count": len(captions)}, warnings


def _iso(base, seconds):
    return (base + dt.timedelta(seconds=seconds)).isoformat(sep=" ")


def _shift(time, offset, transform=None):
    transform = transform or (lambda value: value)
    if time["kind"] == "point":
        return {"kind": "point", "t": transform(time["t"] + offset)}
    return {"kind": "interval", "start": transform(time["start"] + offset),
            "end": transform(time["end"] + offset)}


def _union_duration(intervals):
    total, end = 0.0, None
    for a, b in sorted(intervals):
        if end is None or a >= end:
            total += b - a
        elif b > end:
            total += b - end
        end = b if end is None else max(end, b)
    return total


def build_day_captions(day_mapping, manifest, manifest_path=None, path_maps=()):
    """Return captions with source, per-clip, concatenated-video and wall clocks.

    day_mapping: {day, date, timezone, segments:[{slot_id, output_start_s,
      output_end_s, synthetic_start, synthetic_start_s, source_start_s,
      source_end_s, input_duration_s}]}. All numeric times are seconds.
    ``manifest`` is the loaded caption-pipeline manifest, not a file path.
    Intervals use half-open boundaries [start,end); text is never regenerated.
    """
    slots = {slot["slot_id"]: slot for slot in manifest["slots"]}
    day, date = day_mapping["day"], dt.date.fromisoformat(day_mapping["date"])
    captions, unaligned, coverage, warnings = [], [], [], []
    for segment in day_mapping["segments"]:
        identifier = segment["slot_id"]
        slot = slots[identifier]
        if slot.get("day") != day or not slot.get("proposed_observation"):
            raise ValueError(f"{identifier} is not an observed slot in Day{day}")
        if slot.get("status") in ("excluded_by_user", "removed_from_video_plan"):
            raise ValueError(f"Cannot map excluded slot {identifier}")
        start = _number(segment["source_start_s"], "source_start_s")
        end = _number(segment["source_end_s"], "source_end_s")
        out_start = _number(segment["output_start_s"], "output_start_s")
        out_end = _number(segment["output_end_s"], "output_end_s")
        input_duration = _number(segment["input_duration_s"], "input_duration_s")
        if start < 0 or end <= start or out_start < 0 or out_end <= out_start or input_duration <= 0:
            raise ValueError(f"Invalid interval in {identifier}")
        source = slot["source"]
        if abs(start - source["start_s"]) > 1e-6 or abs(end - source["end_s"]) > 1e-6:
            raise ValueError(f"Source range differs from manifest for {identifier}")
        clock = dt.datetime.fromisoformat(segment["synthetic_start"])
        expected_clock = dt.datetime.combine(date, dt.time()) + dt.timedelta(seconds=segment["synthetic_start_s"])
        if clock != expected_clock:
            raise ValueError(f"Synthetic date/time mismatch for {identifier}")
        if abs(segment["synthetic_start_s"] - slot["proposed_observation"]["synthetic_start_s"]) > 1e-6:
            raise ValueError(f"Synthetic placement differs from manifest for {identifier}")
        # Frame rounding can shorten or extend the media by less than one second.
        effective_end = min(end, start + input_duration, start + out_end - out_start)
        loaded, origin, notes = _load_captions(slot, manifest_path, path_maps)
        warnings.extend({"slot_id": identifier, "message": note} for note in notes)
        rows, interval_ranges = [], []
        for item in loaded:
            original = item["source_time"]
            if original["kind"] == "point":
                if not start <= original["t"] < effective_end:
                    continue
                selected = dict(original)
                boundary = {"start": False, "end": False}
            else:
                a, b = max(start, original["start"]), min(effective_end, original["end"])
                if b <= a:
                    continue
                selected = {"kind": "interval", "start": a, "end": b}
                boundary = {"start": a > original["start"], "end": b < original["end"]}
                interval_ranges.append((a - start, b - start))
            local = _shift(selected, -start)
            rows.append({
                "id": f"{identifier}:{item['id']}", "slot_id": identifier,
                "dataset": source.get("dataset"), "video_id": source.get("video_id"),
                "text": item["text"], "kind": original["kind"],
                "source_time": original, "selected_source_time": selected,
                "clip_time": local, "output_time": _shift(local, out_start),
                "synthetic_time": _shift(local, 0, lambda t: _iso(clock, t)),
                "boundary_clipped": boundary, "provenance": item["provenance"],
            })
        captions.extend(rows)
        if not rows:
            evidence = slot.get("evidence") or {}
            unaligned.append({"slot_id": identifier, "dataset": source.get("dataset"),
                "video_id": source.get("video_id"), "text": evidence.get("text"),
                "evidence_kind": evidence.get("evidence_kind"),
                "alignment": evidence.get("alignment"), "caption_scope": evidence.get("caption_scope"),
                "reason": origin["status"] if origin["status"] != "loaded" else "no_captions_in_selected_interval",
                "note": "No caption timestamp is inferred from the slot label or whole-video duration."})
        coverage.append({"slot_id": identifier, "source": origin,
            "status": "aligned_annotations_available" if rows else "no_aligned_annotations",
            "point_count": sum(row["kind"] == "point" for row in rows),
            "interval_count": sum(row["kind"] == "interval" for row in rows),
            "interval_annotation_coverage_s": _union_duration(interval_ranges),
            "output_duration_s": out_end - out_start,
            "effective_source_start_s": start, "effective_source_end_s": effective_end})
    captions.sort(key=lambda row: (row["output_time"].get("t", row["output_time"].get("start")), row["id"]))
    return {"schema_version": 1, "day": day, "date": date.isoformat(),
        "timezone": day_mapping["timezone"],
        "time_mapping": "output_start + (original_caption_time - source_start); synthetic_start + same elapsed seconds",
        "caption_semantics": "Points remain points; intervals intersect the selected source and actual output bounds. Boundary-clipped text is unchanged and may describe context outside the retained boundary.",
        "captions": captions, "unaligned_evidence": unaligned, "warnings": warnings,
        "coverage": {"segments": coverage, "segment_count": len(coverage),
            "aligned_segment_count": sum(c["status"] == "aligned_annotations_available" for c in coverage),
            "unaligned_segment_count": len(unaligned), "caption_count": len(captions),
            "total_output_duration_s": sum(c["output_duration_s"] for c in coverage),
            "note": "Annotation presence and interval coverage are not visual or semantic certification; point annotations do not imply continuous coverage."}}
