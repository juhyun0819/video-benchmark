"""Load unchanged Torch-only FluxMem kernels without importing a Qwen2.5 model.

The original package __init__ imports its Transformers-4.51-specific model.
Qwen3-VL uses Transformers 4.57.3, so give only the unchanged kernel/utils files
an isolated package namespace. No global Qwen/Transformers monkeypatching.
"""
import importlib
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIRECTORY = ROOT/'models/qwen2-5-vl/src/qwen2_5_vl_fluxmem'
PACKAGE = '_fluxmem_torch_only_upstream'
if PACKAGE not in sys.modules:
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(KERNEL_DIRECTORY)]
    sys.modules[PACKAGE] = package
FluxMem = importlib.import_module(PACKAGE+'.fluxmem').FluxMem
