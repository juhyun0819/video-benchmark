from collections import Counter
from datetime import datetime
from math import isfinite
from src.common import probe

def validate(data, qa=None):
    errors, warnings, manual = [], [], []
    cache = {}
    events = data["events"]
    ids = [e["event_id"] for e in events]
    if len(ids) != len(set(ids)):
        errors.append("Duplicate event IDs")
    cursor = 0
    previous = {}
    for e in events:
        try:
            source = e["source_video"]
            if source not in cache:
                cache[source] = probe(source)
            if not all(isfinite(e[k]) for k in ["start_time", "end_time", "stream_start", "stream_end"]):
                raise ValueError("Non-finite timing")
            if not 0 <= e["start_time"] < e["end_time"] <= cache[source]["duration"]:
                errors.append(f"Invalid source interval: {e['event_id']}")
            if abs(e["stream_start"]-cursor) > 1e-6 or e["stream_end"] <= cursor:
                errors.append(f"Invalid stream timing: {e['event_id']}")
            cursor = e["stream_end"]
            timestamp = datetime.strptime(e["synthetic_time"], "%H:%M")
            day = e["synthetic_day"]
            if not 1 <= day <= data["days"]:
                errors.append("Day outside configured range")
            if day in previous and timestamp < previous[day]:
                errors.append(f"Virtual time order conflict on day {day}")
            previous[day] = timestamp
            if e["verified"] and not e["evidence"].strip():
                errors.append("Verified label missing provenance")
        except (OSError, RuntimeError, ValueError, KeyError, StopIteration) as error:
            errors.append(f"{e.get('event_id')}: {error}")
    if not isfinite(data["duration"]) or abs(cursor - data["duration"]) > 1e-6:
        errors.append("Total duration inconsistent with events")
    for interval in data.get("day_intervals", []):
        daily = [e for e in events if e["synthetic_day"] == interval["synthetic_day"]]
        if daily and (abs(daily[0]["stream_start"]-interval["stream_start"]) > 1e-6 or
                      abs(daily[-1]["stream_end"]-interval["stream_end"]) > 1e-6):
            errors.append("Day interval inconsistent with events")
    counts = Counter((e["source_video"], e["start_time"], e["end_time"]) for e in events)
    if any(n > 1 for n in counts.values()):
        warnings.append("Source intervals reused across events")
    identities = {(e["source_dataset"], e["source_participant"]) for e in events if e["source_participant"]}
    if len(identities) > 1:
        warnings.append("Multiple source participants/datasets combined into synthetic persona")
        if not data.get("synthetic_persona"):
            errors.append("Multiple real identities require synthetic_persona=true")
    if any(not e["source_participant"] for e in events):
        manual.append("Source participant identity unknown")
    manual += ["Manual Review Required: spatial/viewpoint continuity and source chronology",
               "Manual Review Required: cross-day object identity; no object-state QA is generated"]
    if any(not e["verified"] for e in events):
        manual.append("Unverified activity labels excluded from QA")
    if qa is not None:
        from src.qa_generator import generate_qa
        expected = generate_qa(data, check=False)
        if qa != expected:
            errors.append("QA differs from deterministic evidence-backed manifest QA")
    return {"valid": not errors, "errors": errors, "warnings": warnings, "manual_review_required": manual}
