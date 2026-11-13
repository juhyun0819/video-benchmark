"""Pinned, unchanged StreamForest source with a SigLIP-only inference dispatcher.

The upstream package initializers load training code, and its vision dispatcher
imports unused CUDA-only encoders. Bypass those entrypoints, while using the
original Qwen2 class, SigLIP encoder, projector and MemoryManager unchanged.
"""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

VENDOR_ROOT = Path(__file__).resolve().parents[1] / 'vendor' / 'streamforest'
UPSTREAM_COMMIT = 'f9d14670ff451ff5865638e17c9958e99bf2fec5'


def verify_source():
    manifest = json.loads((VENDOR_ROOT / 'UPSTREAM.json').read_text())
    if manifest['commit'] != UPSTREAM_COMMIT or manifest.get('modifications'):
        raise RuntimeError('Unexpected StreamForest source revision or modifications')
    for entry in manifest['files']:
        path = (VENDOR_ROOT / entry['path']).resolve()
        if not path.is_relative_to(VENDOR_ROOT.resolve()):
            raise ValueError('Invalid upstream source path')
        raw = path.read_bytes()
        if len(raw) != entry['bytes'] or hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise RuntimeError('Vendored source changed: ' + entry['path'])
    return manifest


def load_memory_manager():
    """Import the original Torch-only memory module without model dependencies."""
    name = '_videomem_streamforest_memory'
    if name not in sys.modules:
        path = VENDOR_ROOT / 'llava/model/multimodal_projector/memory_manager.py'
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name].MemoryManager


def _namespace(name):
    path = VENDOR_ROOT.joinpath(*name.split('.'))
    if name in sys.modules:
        existing = sys.modules[name]
        origin = getattr(existing, '__file__', None)
        paths = [origin] if origin else list(getattr(existing, '__path__', []))
        if not paths or any(not Path(p).resolve().is_relative_to(VENDOR_ROOT) for p in paths):
            raise RuntimeError(f'{name} already loaded from another project; use a fresh process')
        return existing
    module = ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    module.__spec__ = importlib.util.spec_from_loader(name, loader=None, is_package=True)
    sys.modules[name] = module
    return module


def load_native():
    """Load unchanged native inference classes, selecting only the SigLIP branch."""
    for name in ('llava', 'llava.model', 'llava.model.language_model',
                 'llava.model.multimodal_encoder', 'llava.model.multimodal_projector'):
        _namespace(name)
    name = 'llava.model.multimodal_encoder.builder'
    if name not in sys.modules:
        module = ModuleType(name)
        module.__file__ = __file__

        def build_vision_tower(config, **kwargs):
            from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower, SigLipVisionModel
            path = getattr(config, 'mm_vision_tower', None)
            if not isinstance(path, str) or not Path(path).is_dir():
                raise ValueError('StreamForest requires an explicit local checkpoint directory')
            if not getattr(config, '_videomem_embedded_vision', False):
                raise ValueError('Load the released checkpoint with its embedded SigLIP weights')
            # Construct the same architecture as upstream load_model(), without
            # first loading an external copy of weights which the full checkpoint
            # would replace. Transformers then loads the embedded vision keys.
            from copy import copy
            import torch.nn as nn
            vision_options = copy(config)
            vision_options.unfreeze_mm_vision_tower = False
            vision_options.mm_tunable_parts = ''
            tower = SigLipVisionTower(path, vision_tower_cfg=vision_options, delay_load=True)
            tower.vision_tower = SigLipVisionModel(tower.config)
            del tower.vision_tower.vision_model.encoder.layers[-1:]
            tower.vision_tower.vision_model.head = nn.Identity()
            tower.vision_tower.requires_grad_(False)
            tower.is_loaded = True
            return tower

        module.build_vision_tower = build_vision_tower
        module._videomem_siglip_dispatch = True
        sys.modules[name] = module
    elif not getattr(sys.modules[name], '_videomem_siglip_dispatch', False):
        raise RuntimeError('Another vision dispatcher is already loaded; use a fresh process')
    return importlib.import_module('llava.model.language_model.llava_qwen')
