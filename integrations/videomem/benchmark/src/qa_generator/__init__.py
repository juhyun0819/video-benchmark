from collections import defaultdict
from typing import Protocol

class QAGenerator(Protocol):
    def generate(self, manifest: dict) -> list[dict]: ...

def generate_qa(data, check=True):
    if check:
        from src.validation import validate
        report = validate(data)
        if report["errors"]:
            raise ValueError(str(report["errors"]))
    result = []
    events = data["events"]
    eligible = [e for e in events if e["verified"] and e["evidence"].strip()]
    def add(kind, question, answer, evidence):
        result.append({"qa_id": f"qa_{len(result)+1:04d}", "question_type": kind,
                       "question": question, "answer": answer,
                       "evidence": [{"event_id": e["event_id"], "clip_id": e["clip_id"],
                                     "stream_start": e["stream_start"], "stream_end": e["stream_end"]} for e in evidence]})
    for e in eligible:
        add("episodic_recall", f"What activity appears in Day {e['synthetic_day']} slot {e['routine_slot']} at virtual {e['synthetic_time']}, repetition {e['repetition']}?", e["activity"], [e])
    # Counts are scoped to selected clips, never to unobserved real-world behavior.
    if len(eligible) == len(events):
        for activity in sorted({e["activity"] for e in eligible}):
            matching = [e for e in eligible if e["activity"] == activity]
            add("routine_frequency", f"How many selected clip occurrences show {activity} in this stream?", len(matching), matching)
    for day in range(1, data["days"]+1):
        daily = [e for e in events if e["synthetic_day"] == day]
        for a,b in zip(daily, daily[1:]):
            if a in eligible and b in eligible:
                add("temporal_order", f"What activity immediately follows event {a['event_id']} in Day {day}'s selected stream?", b["activity"], [a,b])
    groups = defaultdict(list)
    for e in eligible:
        groups[(e["routine_slot"], e["repetition"])].append(e)
    for (slot, rep), group in sorted(groups.items()):
        for a,b in zip(group, group[1:]):
            if a["synthetic_day"] != b["synthetic_day"]:
                add("cross_day_comparison", f"Which activities appear in slot {slot}, repetition {rep}, on Day {a['synthetic_day']} and Day {b['synthetic_day']}?", {str(a["synthetic_day"]): a["activity"], str(b["synthetic_day"]): b["activity"]}, [a,b])
    return result
