import hashlib
import json
import math
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def public_question(row):
    """Whitelist, never derive observation time from answer/evidence/input ranges."""
    cutoff = row.get("query_time")
    if not isinstance(cutoff, dict):
        raise ValueError("Explicit query_time {day_id, day_order, seconds} required")
    order = cutoff.get("day_order")
    seconds = cutoff.get("seconds")
    if not isinstance(order, int) or isinstance(order, bool) or order < 1:
        raise ValueError("day_order must be an explicit positive integer")
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("query seconds must be finite and nonnegative")
    if not isinstance(cutoff.get("day_id"), str) or not cutoff["day_id"]:
        raise ValueError("query day_id required")
    if not isinstance(row.get("question"), str) or not row["question"].strip():
        raise ValueError("question text required")
    choices = row.get("choices", [])
    if not isinstance(choices, list) or any(not isinstance(x, str) for x in choices):
        raise ValueError("choices must be strings; structured original choices need explicit adaptation")
    return {"question_id": str(row["question_id"]), "question": row["question"],
            "choices": choices, "query_time": {"day_id": cutoff["day_id"],
                "day_order": order, "seconds": float(seconds)}}


def visible(record, cutoff):
    return (record["day_order"], record["day_end_seconds"]) <= (cutoff["day_order"], cutoff["seconds"])


def timecode(seconds):
    """Upstream HHMMSSFF; FF is centiseconds here, NOT a virtual wall clock."""
    cs = int(round(seconds * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, f = divmod(cs, 100)
    if h >= 24:
        raise ValueError("Single Day playback >=24h cannot use upstream HHMMSSFF")
    return f"{h:02d}{m:02d}{s:02d}{f:02d}"


def seconds_from_code(code):
    s = str(code).zfill(8)
    if len(s) != 8 or not s.isdigit():
        raise ValueError("Invalid HHMMSSFF")
    h,m,sec,f = (int(s[i:i+2]) for i in (0,2,4,6))
    if h >= 24 or m >= 60 or sec >= 60:
        raise ValueError("Invalid HHMMSSFF")
    return h*3600 + m*60 + sec + f/100
