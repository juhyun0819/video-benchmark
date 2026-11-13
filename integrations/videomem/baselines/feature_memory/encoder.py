"""Qwen2.5-VL image features, after the spatial merger and before the LLM.

Only ``visual.*`` checkpoint tensors are read. Each arriving frame is processed
as one image using the checkpoint's native image processor. Its temporal patch
padding repeats this same image; no later video frame is inspected. Features
therefore differ from encoding pairs through the model's native video path.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any


DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4}
DTYPE_ALIASES = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}


def _file_state(path: Path) -> list[int]:
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def _json_file(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _checkpoint_info(model_path: Path, required_files: list[str]) -> dict[str, Any]:
    """Reuse existing download hash verification when file identities still match."""
    manifest_path = model_path.parent / ".downloads" / model_path.name / "verification.json"
    result: dict[str, Any] = {
        "model_path": str(model_path),
        "verification_manifest": str(manifest_path) if manifest_path.exists() else None,
        "verified_checkpoint_files_unchanged": False,
        "config_sha256": hashlib.sha256((model_path / "config.json").read_bytes()).hexdigest(),
        "index_sha256": hashlib.sha256((model_path / "model.safetensors.index.json").read_bytes()).hexdigest(),
    }
    if manifest_path.exists():
        manifest = _json_file(manifest_path)
        states = manifest.get("materialized_file_states", {})
        for name in required_files:
            if states.get(name) != _file_state(model_path / name):
                raise ValueError(f"Checkpoint file changed since its hash verification: {name}")
        if not manifest.get("remote_revision_verified"):
            raise ValueError("Checkpoint manifest does not verify a pinned remote revision")
        result.update({"repo_id": manifest["repo_id"], "revision": manifest["revision"],
                       "verified_checkpoint_files_unchanged": True,
                       "verification_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()})
    return result


class Qwen25VisionEncoder:
    """A vision-only, independent-frame representation with explicit timings.

    CPU feature payload is owned by the returned tensor. The class keeps no
    frame/feature history. Frozen parameters are runtime model cost, separately
    reported from retained memory. The returned feature tensor is NOT LLM KV.
    """

    def __init__(self, model_path: str | Path, device: str = "cuda:0",
                 dtype: str = "bfloat16") -> None:
        import torch
        import transformers
        from accelerate import init_empty_weights
        from safetensors import safe_open
        from transformers import Qwen2VLImageProcessor
        from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel

        started = time.perf_counter()
        dtype = DTYPE_ALIASES.get(dtype, dtype)
        if dtype not in DTYPE_BYTES:
            raise ValueError("dtype must be fp16/float16, bf16/bfloat16, or fp32/float32")
        self.dtype_name = dtype
        self.dtype = getattr(torch, dtype)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        model_path = Path(model_path).resolve(strict=True)
        config_dict = _json_file(model_path / "config.json")
        if config_dict.get("model_type") != "qwen2_5_vl":
            raise ValueError("This representation requires a Qwen2.5-VL checkpoint")
        index = _json_file(model_path / "model.safetensors.index.json")
        weight_map = index.get("weight_map", {})
        selected = {name: filename for name, filename in weight_map.items() if name.startswith("visual.")}
        if not selected:
            raise ValueError("No visual.* parameters found in the safetensors index")
        by_shard: dict[str, list[str]] = defaultdict(list)
        for name, filename in selected.items():
            shard = (model_path / filename).resolve(strict=True)
            if not shard.is_relative_to(model_path) or not shard.is_file():
                raise ValueError("A visual checkpoint shard escapes the model directory")
            by_shard[filename].append(name)
        required_files = ["config.json", "preprocessor_config.json", "model.safetensors.index.json", *by_shard]
        provenance = _checkpoint_info(model_path, required_files)
        states_before = {name: _file_state(model_path / name) for name in required_files}

        # Preserve the processor's resolution policy rather than imposing a
        # memory-budget-dependent resize. This selects the checkpoint's exact
        # declared slow image processor, not a version-dependent fast fallback.
        self.processor = Qwen2VLImageProcessor.from_pretrained(str(model_path), local_files_only=True)
        self.vision_config = Qwen2_5_VLVisionConfig(**config_dict["vision_config"])
        self.vision_config._attn_implementation = "sdpa"
        # Nonpersistent rotary frequencies remain real fp32 buffers. Parameters
        # start on meta so no random full-size vision weights are allocated.
        with init_empty_weights(include_buffers=False):
            self.model = Qwen2_5_VisionTransformerPretrainedModel(self.vision_config)
        state = {}
        for filename, names in sorted(by_shard.items()):
            with safe_open(str(model_path / filename), framework="pt", device="cpu") as handle:
                for name in names:
                    state[name[len("visual."):]] = handle.get_tensor(name).to(dtype=self.dtype)
        self.model.load_state_dict(state, strict=True, assign=True)
        del state
        self.model.requires_grad_(False).eval().to(device=self.device)
        if any(parameter.is_meta for parameter in self.model.parameters()):
            raise RuntimeError("Vision parameters remain unmaterialized after strict checkpoint loading")
        self._sync()
        for name in required_files:
            if states_before[name] != _file_state(model_path / name):
                raise ValueError(f"Checkpoint file changed during vision loading: {name}")

        parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in self.model.parameters())
        buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in self.model.buffers())
        self.metadata = {
            "representation": "qwen2_5_vl_post_spatial_merger_image_features",
            "feature_location": "visual(image_pixels, image_grid_thw), after merger, before LLM",
            "checkpoint": provenance,
            "transformers_version": transformers.__version__, "torch_version": torch.__version__,
            "loaded_parameter_prefix": "visual.", "loaded_tensor_count": len(selected),
            "loaded_shards": sorted(by_shard), "llm_loaded": False, "qa_inference": False,
            "dtype": dtype, "device": str(self.device),
            "frozen_parameter_bytes": parameter_bytes, "frozen_buffer_bytes": buffer_bytes,
            "frozen_model_bytes": parameter_bytes + buffer_bytes,
            "hidden_dimension": int(self.vision_config.out_hidden_size),
            "encoder_hidden_dimension": int(self.vision_config.hidden_size),
            "spatial_merge_size": int(self.vision_config.spatial_merge_size),
            "patch_size": int(self.vision_config.patch_size),
            "temporal_patch_size": int(self.vision_config.temporal_patch_size),
            "temporal_policy": "each frame independently as image; native padding duplicates only current image",
            "native_video_pair_encoding": False,
            "processor_class": type(self.processor).__name__,
            "processor_min_pixels": self.processor.min_pixels,
            "processor_max_pixels": self.processor.max_pixels,
            "resolution_policy": "checkpoint image-processor defaults; no external resize or token cap",
            "attention_implementation": "sdpa",
            "bytes_per_token_by_dtype": {
                name: int(self.vision_config.out_hidden_size) * size for name, size in DTYPE_BYTES.items()
            },
            "model_load_seconds": time.perf_counter() - started,
            "raw_frames_persisted": False,
        }

    def _sync(self) -> None:
        if self.device.type == "cuda":
            import torch
            torch.cuda.synchronize(self.device)

    def encode(self, image) -> tuple[Any, dict[str, Any]]:
        """Return owned CPU features plus metadata; all GPU stages synchronized."""
        import torch
        from PIL import Image

        if not isinstance(image, Image.Image) or image.mode != "RGB":
            raise ValueError("encode expects one decoded PIL RGB image")
        started = time.perf_counter()
        prepared = self.processor(images=[image], return_tensors="pt")
        grid = prepared["image_grid_thw"]
        grid_values = [int(value) for value in grid[0].tolist()]
        if grid.shape != (1, 3) or grid_values[0] != 1:
            raise ValueError("Single-image processing must produce exactly one temporal grid")
        preprocess_seconds = time.perf_counter() - started
        self._sync()
        started = time.perf_counter()
        pixels = prepared["pixel_values"].to(device=self.device, dtype=self.dtype)
        device_grid = grid.to(device=self.device)
        self._sync()
        h2d_seconds = time.perf_counter() - started
        with torch.inference_mode():
            started = time.perf_counter()
            features_device = self.model(pixels, grid_thw=device_grid)
            self._sync()
            encode_seconds = time.perf_counter() - started
            started = time.perf_counter()
            features = features_device.detach().to(device="cpu").contiguous()
            # CPU mode can alias output without .to() allocating a copy. An
            # owned compact tensor keeps storage accounting independent of any
            # intermediate views. CUDA copies already own compact CPU storage.
            if features.device == features_device.device:
                features = features.clone()
            self._sync()
            d2h_seconds = time.perf_counter() - started
        tokens = grid_values[0] * grid_values[1] * grid_values[2] // self.vision_config.spatial_merge_size ** 2
        hidden = int(self.vision_config.out_hidden_size)
        if tuple(features.shape) != (tokens, hidden) or features.dtype != self.dtype:
            raise RuntimeError("Vision output shape/dtype disagrees with native grid and merger configuration")
        if features.untyped_storage().nbytes() != features.numel() * features.element_size():
            raise RuntimeError("Returned features must own a compact tensor allocation")
        # This CPU check is outside synchronized encoder and transfer timings.
        validation_started = time.perf_counter()
        if not torch.isfinite(features).all().item():
            raise RuntimeError("Vision features contain NaN or Inf")
        validation_seconds = time.perf_counter() - validation_started
        metadata = {
            "input_width": image.width, "input_height": image.height,
            "resized_width": grid_values[2] * self.vision_config.patch_size,
            "resized_height": grid_values[1] * self.vision_config.patch_size,
            "grid_thw": grid_values, "tensor_shape": list(features.shape),
            "dtype": self.dtype_name, "tokens_per_frame": tokens,
            "hidden_dimension": hidden, "hidden_dim": hidden,
            "encoder_patch_tokens": grid_values[0] * grid_values[1] * grid_values[2],
            "encoder_hidden_dimension": int(self.vision_config.hidden_size),
            "bytes_per_element": features.element_size(), "element_size": features.element_size(),
            "bytes_per_token": hidden * features.element_size(),
            "feature_bytes": features.numel() * features.element_size(),
            "tensor_storage_bytes": features.untyped_storage().nbytes(),
            "feature_bytes_by_dtype": {name: tokens * hidden * size for name, size in DTYPE_BYTES.items()},
            "preprocess_seconds": preprocess_seconds, "h2d_seconds": h2d_seconds,
            "encode_seconds": encode_seconds, "d2h_seconds": d2h_seconds,
            "feature_validation_seconds": validation_seconds,
            "gpu_timing_synchronized": self.device.type == "cuda",
        }
        return features, metadata
