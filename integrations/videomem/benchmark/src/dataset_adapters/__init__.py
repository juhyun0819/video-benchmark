from pathlib import Path
from src.common import probe

class LocalAdapter:
    """Recursive fallback; participant/location remain unknown until reviewed.

    Subclasses may override discover() and metadata() for release-specific layouts.
    """
    def __init__(self, root, dataset):
        self.root, self.dataset = Path(root).expanduser(), dataset

    def metadata(self, path):
        return {"source_dataset": self.dataset, "source_participant": "", "location": "",
                "source_recorded_at": None}

    def discover(self):
        if not self.root.is_dir():
            raise ValueError(f"Local dataset directory missing: {self.root}")
        rows, errors = [], []
        for path in sorted(self.root.rglob("*")):
            if path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}:
                try:
                    rows.append({**probe(path), **self.metadata(path)})
                except (RuntimeError, ValueError, KeyError, StopIteration) as error:
                    errors.append({"path": str(path), "error": str(error)})
        return rows, errors

REGISTRY = {name: LocalAdapter for name in ["EgoLife", "HD-EPIC", "EPIC-KITCHENS-100", "Ego4D", "Ego-Exo4D", "TEST_ONLY"]}

def register(name, adapter):
    REGISTRY[name] = adapter

def adapter(root, dataset):
    return REGISTRY.get(dataset, LocalAdapter)(root, dataset)
