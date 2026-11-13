"""Reuse only local-video manifest validation from the existing Qwen3 variant."""
from pathlib import Path
import sys

BASELINE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = BASELINE_ROOT.parents[1]
SHARED_ROOT = BASELINE_ROOT.parent / "fluxmem_qwen3vl"


def load_shared():
    """Expose exactly one sibling variant, rejecting an already mixed runtime.

    Run this baseline in its own Python process. The repository keeps several
    incompatible packages named ``streaming`` and ``local_videomem``.
    """
    for name in ("streaming", "local_videomem"):
        if not (SHARED_ROOT / name).is_dir():
            raise RuntimeError(f"Missing shared Qwen3 package: {SHARED_ROOT / name}")
        for key, module in tuple(sys.modules.items()):
            if key == name or key.startswith(name + "."):
                origin = getattr(module, "__file__", None)
                paths = ([origin] if origin is not None else list(getattr(module, '__path__', [])))
                if not paths or any(not Path(path).resolve().is_relative_to(SHARED_ROOT) for path in paths):
                    raise RuntimeError(f"{key} was loaded from another baseline; use a fresh Python process")
    root = str(SHARED_ROOT)
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    return SHARED_ROOT
