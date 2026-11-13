"""Local native StreamForest Qwen2/SigLip inference from persistent memory.

The released ``generate_online`` route encodes raw SigLip features, applies the
memory manager, then applies only ``mm_projector.mlp``. This bridge preserves
that route without re-encoding video or rebuilding a memory manager at query
time. It does not run the batch projector's positional additions or compression.
"""
from __future__ import annotations

from copy import deepcopy
import json
from numbers import Integral
from pathlib import Path
import time

from PIL import Image
import torch

from .upstream import load_native

PROMPT_CONTRACT = "streamforest_qwen_1_5_public_observed_prefix_v1"


def _local_directory(value, name):
    path = Path(value).expanduser().resolve()
    if not path.is_dir() or not (path / "config.json").is_file():
        raise ValueError(f"{name} must be a local checkpoint directory with config.json")
    return path


def _check_prompt_config(config):
    if getattr(config, "mm_patch_merge_type", "flat") != "flat":
        raise ValueError("The native StreamForest online bridge requires flat patch merging")
    if "anyres" in getattr(config, "frame_aspect_ratio", "square"):
        raise ValueError("The native StreamForest online bridge requires square SigLip preprocessing")
    if getattr(config, "mm_llm_compress", False):
        raise ValueError("Additional LLM-side compression is unsupported by this baseline")


def load_model(model_path, vision_path=None, *, device="cuda:0", dtype=None,
               attn_implementation="sdpa"):
    """Load the native language/projector/embedded-SigLip checkpoint locally.

    The caller verifies checkpoint revision/hashes before calling this helper.
    No remote repository identifiers, automatic downloads, training loader or
    Qwen3 conversion are used. Returns ``(model, tokenizer, image_processor)``.
    """
    model_path = _local_directory(model_path, "model_path")
    if vision_path is not None:
        raise ValueError("The released checkpoint embeds SigLip weights; do not supply an external vision_path")
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise ValueError("The released sharded checkpoint requires model.safetensors.index.json")
    index = json.loads(index_path.read_text())
    vision_keys = [key for key in index.get("weight_map", {})
                   if key.startswith("model.vision_tower.vision_tower.")]
    if len(vision_keys) != 421:
        raise ValueError("Checkpoint index lacks the complete released 26-layer SigLip weights")
    if attn_implementation not in {"eager", "sdpa"}:
        raise ValueError("attn_implementation must be eager or sdpa")
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only explicit CPU or CUDA devices are supported")
    dtype = dtype or (torch.float32 if device.type == "cpu" else torch.bfloat16)
    if dtype not in {torch.float32, torch.float16, torch.bfloat16}:
        raise ValueError("dtype must be a supported floating-point Torch dtype")
    native = load_native()
    from transformers import AutoTokenizer

    config = native.LlavaQwenConfig.from_pretrained(str(model_path), local_files_only=True)
    # The published config identifies its language backbone as qwen2; the
    # native LlavaQwen class sets llava_qwen while constructing the model.
    if config.model_type not in {"qwen2", "llava_qwen"}:
        raise ValueError("Expected the released native Qwen2/LlavaQwen checkpoint")
    if getattr(config, "mm_projector_type", None) != "tome729_fstw_pemf":
        raise ValueError("Expected the released StreamForest tome729_fstw_pemf projector")
    if getattr(config, "mm_hidden_size", None) != 1152:
        raise ValueError("The released StreamForest SigLip feature width must be 1152")
    _check_prompt_config(config)
    # The published model contains the trained vision tower as well as Qwen2
    # and the MLP. The local inference dispatcher constructs that exact tower
    # shape, avoiding the upstream hardcoded external SigLip download.
    config.mm_vision_tower = str(model_path)
    config._videomem_embedded_vision = True
    config.delay_load = True
    config.unfreeze_mm_vision_tower = False
    config.mm_tunable_parts = ""
    model, loading = native.LlavaQwenForCausalLM.from_pretrained(
        str(model_path), config=config, local_files_only=True,
        torch_dtype=dtype, attn_implementation=attn_implementation,
        device_map={"": str(device)}, low_cpu_mem_usage=True, output_loading_info=True,
    )
    if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise RuntimeError("Native StreamForest language/projector weights did not load completely")
    if loading.get("unexpected_keys"):
        raise RuntimeError("Unexpected native checkpoint weights; refusing to discard trained components")
    tower = model.get_vision_tower()
    model.eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True,
                                               trust_remote_code=False, use_fast=False)
    return model, tokenizer, tower.image_processor


def _require_eval(model):
    if model.training:
        raise ValueError("StreamForest inference requires model.eval()")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def encode_frame(model, processor, image):
    """Encode one observed RGB PIL frame into raw native [729, 1152] features."""
    _require_eval(model)
    if not isinstance(image, Image.Image) or image.mode != "RGB":
        raise ValueError("encode_frame requires one RGB PIL image")
    tower = model.get_vision_tower()
    parameter = next(tower.parameters())
    pixels = processor.preprocess(image, return_tensors="pt")["pixel_values"]
    if tuple(pixels.shape) != (1, 3, 384, 384):
        raise ValueError("The released SigLip processor must produce one 384x384 RGB frame")
    features = tower(pixels.to(device=parameter.device, dtype=parameter.dtype))
    if tuple(features.shape) != (1, 729, 1152):
        raise ValueError("The native SigLip tower must return [1, 729, 1152] raw features")
    if not features.is_floating_point() or not bool(torch.isfinite(features).all()):
        raise ValueError("SigLip produced invalid visual features")
    return features[0].detach().clone()


def _prepare_prompt(model, tokenizer, snapshot, question, max_new_tokens):
    load_native()
    from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token

    _check_prompt_config(model.config)
    if not isinstance(question, str) or not question.strip():
        raise ValueError("A nonempty public question is required")
    if DEFAULT_IMAGE_TOKEN in question:
        raise ValueError("Public questions cannot contain an additional image placeholder")
    features = snapshot["features"]
    if not isinstance(features, torch.Tensor) or not features.is_floating_point():
        raise ValueError("snapshot features must be a floating-point tensor")
    if features.ndim == 2:
        features = features.unsqueeze(0)
    if features.ndim != 3 or features.shape[0] != 1 or features.shape[1] < 1:
        raise ValueError("snapshot features must contain nonempty [1, tokens, channels] memory")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("snapshot features must be finite")
    projector = model.get_model().mm_projector.mlp
    parameter = next(projector.parameters())
    if features.shape[-1] != projector[0].in_features:
        raise ValueError("snapshot feature width does not match the native projector")
    placeholder = DEFAULT_IMAGE_TOKEN
    if getattr(model.config, "mm_use_im_start_end", False):
        placeholder = DEFAULT_IM_START_TOKEN + placeholder + DEFAULT_IM_END_TOKEN
    conversation = conv_templates["qwen_1_5"].copy()
    conversation.append_message(conversation.roles[0], placeholder + "\n" + question)
    conversation.append_message(conversation.roles[1], None)
    prompt = conversation.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).to(model.get_input_embeddings().weight.device)
    if int((input_ids == IMAGE_TOKEN_INDEX).sum()) != 1:
        raise ValueError("The native prompt must contain exactly one image placeholder")
    memory_tokens = int(features.shape[1])
    expected_tokens = int(input_ids.shape[1]) - 1 + memory_tokens
    for name in ("tokenizer_model_max_length", "max_position_embeddings"):
        limit = getattr(model.config, name, None)
        if isinstance(limit, Integral) and limit > 0 and expected_tokens + max_new_tokens > limit:
            raise ValueError(f"Prompt plus generation exceeds {name}; reduce the declared memory budget")
    projected = projector(features.to(device=parameter.device, dtype=parameter.dtype))
    # Only the native insertion routine examines the image container. Flat,
    # square mode does not read its pixels: no historical frames are retained.
    sentinel = [projected.new_empty((0, 3, 384, 384))]
    prepared = model.prepare_inputs_labels_for_LLM(
        input_ids, None, torch.ones_like(input_ids), None, None,
        sentinel, [projected], modalities=["video"], image_sizes=None,
    )
    _, position_ids, attention_mask, _, inputs_embeds, _ = prepared
    if inputs_embeds is None or inputs_embeds.shape[1] != expected_tokens:
        raise RuntimeError("Native insertion unexpectedly truncated or omitted visual memory")
    return {"inputs_embeds": inputs_embeds, "position_ids": position_ids,
            "attention_mask": attention_mask, "input_tokens": expected_tokens,
            "memory_tokens": memory_tokens, "prompt": prompt}


@torch.inference_mode()
def generate_from_memory(model, tokenizer, snapshot, question, max_new_tokens=192):
    """Greedy native Qwen2 decoding with a fresh KV cache and no vision replay."""
    _require_eval(model)
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, Integral) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    from transformers import Qwen2ForCausalLM
    device = model.get_input_embeddings().weight.device
    _sync(device)
    started = time.perf_counter()
    prepared = _prepare_prompt(model, tokenizer, snapshot, question, int(max_new_tokens))
    generation = deepcopy(model.generation_config)
    generation.do_sample = False
    generation.num_beams = 1
    generation.num_beam_groups = 1
    generation.num_return_sequences = 1
    generation.temperature, generation.top_p, generation.top_k = 1.0, 1.0, 50
    generation.use_cache = True
    if generation.pad_token_id is None:
        generation.pad_token_id = tokenizer.pad_token_id
    if generation.pad_token_id is None:
        eos = generation.eos_token_id
        generation.pad_token_id = eos[0] if isinstance(eos, (list, tuple)) else eos
    # Calling the native base implementation bypasses LlavaQwen.generate and
    # generate_online, both of which reconstruct visual inputs/memory.
    output = Qwen2ForCausalLM.generate(
        model, inputs_embeds=prepared["inputs_embeds"],
        position_ids=prepared["position_ids"], attention_mask=prepared["attention_mask"],
        past_key_values=None, generation_config=generation,
        max_new_tokens=int(max_new_tokens), return_dict_in_generate=True,
    )
    _sync(device)
    token_ids = output.sequences[0].detach().cpu().tolist()
    eos = generation.eos_token_id
    eos_ids = set(eos) if isinstance(eos, (list, tuple)) else ({eos} if eos is not None else set())
    return {
        "prediction": tokenizer.decode(token_ids, skip_special_tokens=True).strip(),
        "token_ids": token_ids, "stopped_on_eos": bool(token_ids and token_ids[-1] in eos_ids),
        "input_tokens": prepared["input_tokens"], "output_tokens": len(token_ids),
        "memory_tokens": prepared["memory_tokens"], "empty_memory": False,
        "memory_truncated": bool(snapshot.get("memory_truncated", False)),
        "elapsed_sec": time.perf_counter() - started,
        "decode_backend": "native_qwen2_cached_greedy", "prompt": prepared["prompt"],
    }
