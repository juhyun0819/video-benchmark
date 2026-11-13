import uuid
from pathlib import Path
from src.common import load, save, probe, run

def create_clip(video, start, end, activity, participant="", location="", attributes=None,
                verified=False, evidence="", source_recorded_at=None):
    info = probe(video["path"])
    if not 0 <= start < end <= info["duration"]:
        raise ValueError("Clip interval must satisfy 0 <= start < end <= video duration")
    if not activity.strip():
        raise ValueError("Activity is required")
    if attributes and not verified:
        raise ValueError("Attributes require human or annotation verification")
    if verified and not evidence.strip():
        raise ValueError("Verified labels require human review or annotation provenance")
    return {"clip_id": "clip_" + uuid.uuid4().hex[:12], "source_video": info["path"],
            "source_dataset": video["source_dataset"], "source_participant": participant,
            "start_time": start, "end_time": end, "activity": activity.strip(), "location": location,
            "source_recorded_at": source_recorded_at, "attributes": attributes or {},
            "verified": verified, "evidence": evidence, "is_test_data": video["source_dataset"] == "TEST_ONLY"}

class ClipStore:
    def __init__(self, path):
        self.path = Path(path)
    def all(self):
        return load(self.path, [])
    def add(self, clip):
        rows = self.all()
        if any(c["clip_id"] == clip["clip_id"] for c in rows):
            raise ValueError("Duplicate clip ID")
        save(self.path, rows + [clip])
    def export_csv(self, path):
        import pandas as pd
        pd.DataFrame(self.all()).to_csv(path, index=False)

def thumbnail(clip, path):
    import cv2
    capture = cv2.VideoCapture(clip["source_video"])
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, (clip["start_time"] + clip["end_time"]) * 500)
        ok, frame = capture.read()
        if not ok:
            raise ValueError("Could not decode thumbnail")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), frame):
            raise ValueError("Could not save thumbnail")
    finally:
        capture.release()
