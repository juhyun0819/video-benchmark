"""Inference adapters around the unchanged official Qwen/FluxMem model.

Each ingestion call encodes exactly two available frames, i.e. one Qwen temporal
patch. Answering consumes already projected memory tokens and does not invoke
vision or FluxMem again. The explicit greedy loop is intentional: the official
``prepare_inputs_for_generation`` discards user-supplied ``position_ids``.
Keeping rotary coordinates separate from consecutive KV-cache offsets preserves
sparse memory coordinates during both prefill and cached decoding.
"""
from __future__ import annotations

from collections.abc import Mapping
import time
from typing import Any

from PIL import Image
import torch


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _require_eval(model: Any) -> None:
    if model.training:
        raise ValueError("Streaming inference requires model.eval().")


@torch.inference_mode()
def encode_pair(
    model: Any,
    processor: Any,
    frames: list[Image.Image],
    max_pixels: int = 256 * 28 * 28,
) -> dict[str, Any]:
    """Project two available RGB PIL frames with the official processor/ViT.

    Output embeddings are detached on the visual encoder's device. ``height``
    and ``width`` are the spatial dimensions after Qwen's spatial merger;
    ``grid_thw`` is the processor's original, unmerged CPU grid. This method
    takes neither text nor a complete video and retains no frame history. The
    caller owns both images; this adapter neither copies nor closes them.
    """
    _require_eval(model)
    if not isinstance(frames, list) or len(frames) != 2:
        raise ValueError("encode_pair requires a list of exactly two available frames.")
    if any(not isinstance(frame, Image.Image) for frame in frames):
        raise TypeError("encode_pair frames must be PIL images.")
    if any(frame.mode != "RGB" for frame in frames):
        raise ValueError("encode_pair requires RGB frames from the streaming iterator.")
    if not isinstance(max_pixels, int) or isinstance(max_pixels, bool) or max_pixels <= 0:
        raise ValueError("max_pixels must be a positive integer.")
    image_processor = processor.image_processor
    temporal_patch_size = int(model.config.vision_config.temporal_patch_size)
    if temporal_patch_size != 2 or int(image_processor.temporal_patch_size) != 2:
        raise ValueError("encode_pair requires Qwen's two-frame temporal patch size.")
    processed = image_processor(
        images=None,
        videos=[frames],
        return_tensors="pt",
        max_pixels=max_pixels,
    )
    grid = processed["video_grid_thw"]
    if tuple(grid.shape) != (1, 3) or int(grid[0, 0]) != 1:
        raise ValueError("Two frames must produce exactly one temporal grid.")
    merge_size = int(model.config.vision_config.spatial_merge_size)
    _, raw_height, raw_width = (int(value) for value in grid[0].tolist())
    if raw_height % merge_size or raw_width % merge_size:
        raise ValueError("Processed spatial grid is incompatible with Qwen's merger.")
    visual_parameter = next(model.visual.parameters())
    pixels = processed["pixel_values_videos"].to(
        device=visual_parameter.device, dtype=visual_parameter.dtype,
    )
    embeddings = model.visual(pixels, grid_thw=grid.to(visual_parameter.device))
    height, width = raw_height // merge_size, raw_width // merge_size
    if embeddings.ndim != 2 or embeddings.shape[0] != height * width:
        raise ValueError("Visual encoder output does not match the merged spatial grid.")
    return {
        "embeddings": embeddings.detach(),
        "height": height,
        "width": width,
        "grid_thw": grid.detach().cpu().clone(),
    }


def _prompt_ids(model: Any, tokenizer_or_processor: Any, question: str, has_memory: bool):
    tokenizer = getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)
    content = []
    if has_memory:
        content.append({"type": "video"})
    content.append({"type": "text", "text": question})
    template_owner = tokenizer_or_processor
    if not hasattr(template_owner, "apply_chat_template"):
        template_owner = tokenizer
    prompt = template_owner.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    ids = encoded["input_ids"]
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
        raise ValueError("The chat template must produce one nonempty prompt.")
    markers = (ids[0] == model.config.video_token_id).nonzero(as_tuple=True)[0]
    if markers.numel() != int(has_memory):
        raise ValueError("The chat template must contain exactly one video placeholder when memory is present.")
    return tokenizer, ids, int(markers[0]) if has_memory else ids.shape[1]


def _memory_ids(snapshot: Mapping[str, Any]) -> list[Any]:
    if "used_memory_ids" in snapshot:
        return list(snapshot["used_memory_ids"])
    ids = []
    for record in snapshot.get("records", []):
        if isinstance(record, Mapping):
            if record.get("retained_tokens", 1) == 0:
                continue
            metadata = record.get("metadata", {})
            if "memory_id" in metadata:
                ids.append(metadata["memory_id"])
                continue
            for key in ("memory_id", "frame_id", "id", "time_id"):
                if key in record:
                    ids.append(record[key])
                    break
        else:
            if getattr(record, "retained_tokens", 1) == 0:
                continue
            metadata = getattr(record, "metadata", {})
            if "memory_id" in metadata:
                ids.append(metadata["memory_id"])
                continue
            for key in ("memory_id", "frame_id", "id", "time_id"):
                if hasattr(record, key):
                    ids.append(getattr(record, key))
                    break
    return ids


@torch.inference_mode()
def generate_from_memory(
    model: Any,
    tokenizer_or_processor: Any,
    snapshot: Mapping[str, Any],
    question: str,
    max_new_tokens: int = 128,
) -> dict[str, Any]:
    """Greedily answer from a snapshot without changing its tensors or records.

    ``embeddings`` must be projected ``[N, D]`` tokens. ``positions`` must be
    nonnegative integer ``[3, N]`` temporal/height/width coordinates relative to
    the start of the visual segment; sparse/global temporal values are retained.
    All three axes receive the ordinary text-prefix offset. Suffix and generated
    text follow the largest preceding rotary coordinate, while cache offsets
    count actual retained tokens. Every call starts with a fresh local KV cache.

    There is no automatic context truncation. ``memory_truncated`` reports the
    snapshot's declared truncation/eviction state. Empty snapshots use a text-only
    prompt and are explicitly marked in the result. This is greedy argmax with
    EOS stopping, not the full Hugging Face logits-processor/sampling interface.
    """
    _require_eval(model)
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string.")
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer.")
    embeddings = snapshot["embeddings"]
    positions = snapshot["positions"]
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2 or not embeddings.is_floating_point():
        raise ValueError("snapshot embeddings must be a floating-point [N, D] tensor.")
    memory_tokens = embeddings.shape[0]
    if not isinstance(positions, torch.Tensor) or tuple(positions.shape) != (3, memory_tokens):
        raise ValueError("snapshot positions must have shape [3, N].")
    if positions.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("snapshot positions must contain integer coordinates.")
    if positions.numel() and bool((positions < 0).any()):
        raise ValueError("snapshot positions must be nonnegative.")
    input_embedding = model.get_input_embeddings()
    device, dtype = input_embedding.weight.device, input_embedding.weight.dtype
    if memory_tokens and embeddings.shape[1] != input_embedding.weight.shape[1]:
        raise ValueError("snapshot embedding width does not match the model.")
    _sync(device)
    started = time.perf_counter()
    tokenizer, template_ids, prefix_length = _prompt_ids(
        model, tokenizer_or_processor, question, bool(memory_tokens),
    )
    template_ids = template_ids.to(device)
    if memory_tokens:
        input_ids = torch.cat((
            template_ids[:, :prefix_length],
            template_ids.new_full((1, memory_tokens), model.config.video_token_id),
            template_ids[:, prefix_length + 1:],
        ), dim=1)
    else:
        input_ids = template_ids
    input_tokens = input_ids.shape[1]
    inputs_embeds = input_embedding(input_ids)
    prefix_positions = torch.arange(prefix_length, device=device).expand(3, -1)
    if memory_tokens:
        inputs_embeds[:, prefix_length:prefix_length + memory_tokens] = embeddings.to(device=device, dtype=dtype)
        visual_positions = positions.to(device=device, dtype=torch.long) + prefix_length
        suffix_start = int(visual_positions.max()) + 1
        suffix_length = input_tokens - prefix_length - memory_tokens
        suffix_positions = torch.arange(suffix_start, suffix_start + suffix_length, device=device).expand(3, -1)
        prefill_positions = torch.cat((prefix_positions, visual_positions, suffix_positions), dim=1).unsqueeze(1)
    else:
        prefill_positions = prefix_positions.unsqueeze(1)
    next_rotary_position = int(prefill_positions.max()) + 1
    attention_mask = torch.ones_like(input_ids)
    common = {
        "use_cache": True,
        "use_fluxmem": False,
        "return_dict": True,
        "pixel_values": None,
        "pixel_values_videos": None,
        "image_grid_thw": None,
        "video_grid_thw": None,
    }
    # Passing explicit positions avoids reading or writing model.rope_deltas.
    output = model(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=prefill_positions,
        cache_position=torch.arange(input_tokens, device=device),
        past_key_values=None,
        **common,
    )
    _sync(device)
    prefill_finished = time.perf_counter()
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos is None:
        eos = getattr(model.config, "eos_token_id", None)
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) if eos is not None else set()
    generated = []
    stopped_on_eos = False
    for step in range(max_new_tokens):
        next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        token_id = int(next_token[0, 0])
        generated.append(token_id)
        if token_id in eos_ids:
            stopped_on_eos = True
            break
        if step + 1 == max_new_tokens:
            break
        cache = output.past_key_values
        if cache is None:
            raise RuntimeError("The model did not return a KV cache for cached decoding.")
        attention_mask = torch.cat((attention_mask, attention_mask.new_ones((1, 1))), dim=1)
        output = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            position_ids=torch.full((3, 1, 1), next_rotary_position + step, dtype=torch.long, device=device),
            cache_position=torch.tensor([input_tokens + step], dtype=torch.long, device=device),
            past_key_values=cache,
            **common,
        )
    _sync(device)
    finished = time.perf_counter()
    prediction = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return {
        "prediction": prediction,
        "token_ids": generated,
        "input_tokens": input_tokens,
        "output_tokens": len(generated),
        "memory_tokens": memory_tokens,
        "empty_memory": memory_tokens == 0,
        "memory_truncated": bool(snapshot.get("memory_truncated", snapshot.get("truncated", False))
                                 or snapshot.get("stats", {}).get("evicted_grids", 0)),
        "used_memory_ids": _memory_ids(snapshot),
        "elapsed_sec": finished - started,
        "prefill_sec": prefill_finished - started,
        "decode_sec": finished - prefill_finished,
        "decode_backend": "explicit_cached_greedy",
        "stopped_on_eos": stopped_on_eos,
    }
