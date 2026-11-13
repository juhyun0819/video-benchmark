"""Lazy, offline Qwen3-VL inference restricted to physical NVIDIA GPUs 4 and 5.

Importing this module or validating a configuration never imports torch, queries
CUDA, or loads a checkpoint. The numeric launch mask is resolved to GPU UUIDs
before CUDA initialization, so CUDA's device ordering cannot select another GPU.
"""

from __future__ import annotations

import hashlib
import io
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from .backends import BackendConfigurationError, ModelExecutionError


class LocalHF:
    def __init__(self):
        self._loaded = {}
        self._requested_mask = None

    @staticmethod
    def validate_settings(role, settings):
        model = settings.get("model")
        if not isinstance(model, str) or not Path(model).is_absolute() or not Path(model).is_dir():
            raise BackendConfigurationError("Role %s hf_local model must be an existing absolute directory" % role)
        if not isinstance(settings.get("revision"), str) or not re.fullmatch(r"[0-9a-f]{40}", settings["revision"]):
            raise BackendConfigurationError("hf_local requires an immutable 40-character revision")
        if settings.get("device") not in {"cuda:0", "cuda:1"}:
            raise BackendConfigurationError("hf_local device must be logical cuda:0 or cuda:1 (physical GPUs 4,5)")
        if settings.get("dtype", "bfloat16") not in {"bfloat16", "float16"}:
            raise BackendConfigurationError("hf_local dtype must be bfloat16 or float16")
        if settings.get("attn_implementation", "sdpa") not in {"sdpa", "eager"}:
            raise BackendConfigurationError("hf_local attn_implementation must be sdpa or eager")
        limit = settings.get("max_tokens", 1024)
        if type(limit) is not int or limit <= 0:
            raise BackendConfigurationError("hf_local max_tokens must be a positive integer")
        temperature = settings.get("temperature", 0)
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature < 0:
            raise BackendConfigurationError("hf_local temperature must be a nonnegative finite number")
        for key in ("min_pixels", "max_pixels"):
            if key in settings and (type(settings[key]) is not int or settings[key] <= 0):
                raise BackendConfigurationError("hf_local %s must be a positive integer" % key)
        if settings.get("min_pixels", 1) > settings.get("max_pixels", float("inf")):
            raise BackendConfigurationError("hf_local min_pixels must not exceed max_pixels")

    def _guard_devices(self):
        mask = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if mask != "4,5" and not re.fullmatch(r"GPU-[0-9a-fA-F-]+,GPU-[0-9a-fA-F-]+", mask):
            raise ModelExecutionError("hf_local requires CUDA_VISIBLE_DEVICES=4,5 (or the exact UUIDs of those GPUs)")
        try:
            result = subprocess.run(
                ["nvidia-smi", "--id=4,5", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
                check=True, capture_output=True, text=True, timeout=15,
            )
            devices = {}
            for line in result.stdout.strip().splitlines():
                index, uuid = (part.strip() for part in line.split(","))
                if index in devices or not re.fullmatch(r"GPU-[0-9a-fA-F-]+", uuid):
                    raise ValueError("Invalid GPU identity")
                devices[index] = uuid
            if set(devices) != {"4", "5"} or devices["4"] == devices["5"]:
                raise ValueError("Missing GPU identity")
        except (OSError, subprocess.SubprocessError, ValueError):
            raise ModelExecutionError("Cannot verify physical GPU 4,5 UUIDs with nvidia-smi") from None
        uuid_mask = devices["4"] + "," + devices["5"]
        if mask not in {"4,5", uuid_mask}:
            raise ModelExecutionError("hf_local CUDA mask does not match physical GPUs 4,5 in that order")
        # Even cuda.is_available() can initialize the driver before torch reports
        # is_initialized(). Never change a numeric mask after torch was imported.
        if mask != uuid_mask and "torch" in sys.modules:
            raise ModelExecutionError("torch was imported before the physical GPU guard; restart with the GPU 4,5 UUID mask")
        if self._requested_mask is None:
            self._requested_mask = mask
        os.environ["CUDA_VISIBLE_DEVICES"] = uuid_mask
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        return uuid_mask, devices

    def _load(self, settings, trace):
        effective_mask, physical_devices = self._guard_devices()
        logical_index = int(settings["device"].split(":")[1])
        trace["local_runtime"] = {
            "requested_cuda_visible_devices": self._requested_mask,
            "effective_cuda_visible_devices": effective_mask,
            "device": settings["device"], "physical_gpu_index": 4 + logical_index,
            "physical_gpu_uuid": physical_devices[str(4 + logical_index)],
            "dtype": settings.get("dtype", "bfloat16"),
            "attn_implementation": settings.get("attn_implementation", "sdpa"),
            "local_files_only": True, "trust_remote_code": False,
            "image_decode_device": "cpu", "image_preprocess_device": "cpu",
        }
        key = tuple(settings.get(name, default) for name, default in (
            ("model", None), ("revision", None), ("device", None),
            ("dtype", "bfloat16"), ("attn_implementation", "sdpa"),
            ("min_pixels", None), ("max_pixels", None),
        ))
        if key not in self._loaded:
            import torch
            import transformers
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

            if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
                raise ModelExecutionError("hf_local requires both allowed GPUs to be visible")
            # Also reject a process whose CUDA runtime was initialized under an
            # earlier mask, even if its environment was subsequently changed.
            actual_uuids = [str(torch.cuda.get_device_properties(index).uuid) for index in range(2)]
            if [uuid.lower() for uuid in actual_uuids] != [physical_devices[str(index)].lower() for index in (4, 5)]:
                raise ModelExecutionError("CUDA runtime devices do not match allowed physical GPUs 4,5")
            started = time.perf_counter()
            common = {"revision": settings["revision"], "local_files_only": True, "trust_remote_code": False}
            pixels = {name: settings[name] for name in ("min_pixels", "max_pixels") if name in settings}
            processor = AutoProcessor.from_pretrained(settings["model"], use_fast=True, **common, **pixels)
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                settings["model"], dtype=getattr(torch, settings.get("dtype", "bfloat16")),
                device_map={"": settings["device"]},
                attn_implementation=settings.get("attn_implementation", "sdpa"), **common,
            )
            model.eval()
            provenance = {"torch_version": torch.__version__, "transformers_version": transformers.__version__,
                          "cuda_verified_uuids": actual_uuids,
                          "load_elapsed_seconds": time.perf_counter() - started, "config_sha256": {}}
            for name in ("config.json", "generation_config.json", "preprocessor_config.json", "processor_config.json"):
                path = Path(settings["model"]) / name
                if path.is_file():
                    provenance["config_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
            self._loaded[key] = (torch, processor, model, provenance)
            trace["local_runtime"]["loaded_this_call"] = True
        else:
            trace["local_runtime"]["loaded_this_call"] = False
        torch, processor, model, provenance = self._loaded[key]
        trace["local_runtime"].update(provenance)
        return torch, processor, model

    def generate(self, settings, system, prompt, decoded, image_metadata, trace):
        from PIL import Image

        images = []
        content = [{"type": "text", "text": prompt}]
        trace["image_decode_device"] = "cpu"
        trace["decoded_frame_count"] = 0
        try:
            for raw, metadata in zip(decoded, image_metadata):
                try:
                    with Image.open(io.BytesIO(raw)) as opened:
                        image = opened.convert("RGB")
                except Exception:
                    raise ModelExecutionError("hf_local could not decode actual image pixels on CPU") from None
                images.append(image)
                trace["decoded_frame_count"] += 1
                content.extend([{"type": "text", "text": metadata["wire_text"]},
                                {"type": "image", "image": image}])
            messages = [{"role": "system", "content": [{"type": "text", "text": system}]},
                        {"role": "user", "content": content}]
            torch, processor, model = self._load(settings, trace)
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", images_kwargs={"device": "cpu"},
            )
            inputs = inputs.to(settings["device"])
            prompt_tokens = len(inputs["input_ids"][0])
            temperature = settings.get("temperature", 0)
            parameters = {
                "max_new_tokens": settings.get("max_tokens", 1024), "do_sample": temperature > 0,
                "temperature": temperature if temperature > 0 else 1.0,
                "top_p": 1.0, "top_k": 0 if temperature > 0 else 50, "repetition_penalty": 1.0,
                "num_beams": 1, "return_dict_in_generate": False,
            }
            trace["parameters"] = parameters.copy()
            with torch.inference_mode():
                generated = model.generate(**inputs, **parameters)
            continuation = generated[0][prompt_tokens:]
            completion_tokens = len(continuation)
            eos = model.generation_config.eos_token_id
            eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
            ended = bool(completion_tokens and int(continuation[-1]) in eos)
            finish = "length" if completion_tokens >= parameters["max_new_tokens"] and not ended else "stop"
            text = processor.batch_decode([continuation], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return {
                "choices": [{"message": {"content": text}, "finish_reason": finish}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                          "total_tokens": prompt_tokens + completion_tokens},
            }
        finally:
            for image in images:
                image.close()
