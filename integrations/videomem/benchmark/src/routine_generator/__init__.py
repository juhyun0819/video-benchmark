import random
from copy import deepcopy
from src.persona_builder import persona

TEMPLATE = [{"routine_slot": s, "synthetic_time": t, "activity": a} for s,t,a in [
    ("wake_up", "08:00", "wake_up"), ("breakfast", "08:30", "cooking"),
    ("working", "09:00", "working"), ("dinner", "18:30", "cleaning"), ("hobby", "20:00", "reading")]]
VARIATIONS = {"stable_routine", "episodic_deviation", "activity_replacement", "activity_omission",
              "temporal_order_change", "cross_day_state_change", "gradual_drift"}

def build(clips, persona_id="remote_worker_01", days=7, repeats=1, seed=42, template=None):
    if days < 1 or repeats < 1:
        raise ValueError("Days and repeats must be positive")
    rng = random.Random(seed)
    events = []
    for day in range(1, days+1):
        for slot in (template if template is not None else TEMPLATE):
            choices = [c for c in clips if c["activity"] == slot["activity"]]
            if not choices:
                raise ValueError(f"No clips for activity: {slot['activity']}")
            for repetition in range(repeats):
                events.append({"event_id": f"day{day}_event{len(events)+1}", "synthetic_day": day,
                               **slot, "repetition": repetition+1, "variation": "stable_routine",
                               "clip_id": rng.choice(choices)["clip_id"]})
    return {"schema_version": "1.0", **persona(persona_id), "seed": seed, "days": days,
            "repeats": repeats, "events": events}

def change(plan, event_id, clip, variation="activity_replacement"):
    if variation not in {"stable_routine", "episodic_deviation", "activity_replacement"}:
        raise NotImplementedError("Reserved variation requires a future evidence-aware implementation")
    result = deepcopy(plan)
    event = next(e for e in result["events"] if e["event_id"] == event_id)
    event.update(clip_id=clip["clip_id"], activity=clip["activity"], variation=variation)
    return result
