"""Training-free Qwen3-VL adapters for persistent, sparse visual memory.

Ingestion uses the official video processor and returns the final merger plus
all three DeepStack mergers. Answering reconstructs Qwen3's timestamp-separated
video prompt from stored grids, obtains native full-grid rotary coordinates,
and deletes exactly the discarded visual positions. It never calls vision or
FluxMem. A fresh explicit greedy KV cache is used for each question.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Integral, Real
import time
from typing import Any

from PIL import Image
import torch


def _require_eval(model: Any) -> None:
    if model.training:
        raise ValueError("Streaming inference requires model.eval().")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


@torch.inference_mode()
def encode_pair(
    model: Any,
    processor: Any,
    frames: list[Image.Image],
    max_pixels: int = 256 * 32 * 32,
) -> dict[str, Any]:
    """Encode exactly two available RGB frames, without sampling or history.

    ``max_pixels`` is a per-frame area bound. The official Qwen3 video processor
    instead accepts a total video-area bound, so its upper bound is twice this
    value. Preprocessing stays on CPU; only the resulting patches move to the
    vision encoder. Outputs remain detached on the encoder's device.
    """
    _require_eval(model)
    max_pixels = _positive_integer(max_pixels, "max_pixels")
    if not isinstance(frames, list) or len(frames) != 2:
        raise ValueError("encode_pair requires a list of exactly two available frames.")
    if any(not isinstance(frame, Image.Image) for frame in frames):
        raise TypeError("encode_pair frames must be PIL images.")
    if any(frame.mode != "RGB" for frame in frames):
        raise ValueError("encode_pair requires RGB frames from the streaming iterator.")
    video_processor = processor.video_processor
    vision_config = model.config.vision_config
    if int(vision_config.temporal_patch_size) != 2 or int(video_processor.temporal_patch_size) != 2:
        raise ValueError("encode_pair requires Qwen3's two-frame temporal patch size.")
    merge_size = int(vision_config.spatial_merge_size)
    if int(video_processor.merge_size) != merge_size or int(video_processor.patch_size) != int(vision_config.patch_size):
        raise ValueError("Video processor patch and merger sizes must match the model.")
    if max_pixels < (int(vision_config.patch_size) * merge_size) ** 2:
        raise ValueError("max_pixels must fit at least one merged spatial patch.")
    if len(vision_config.deepstack_visual_indexes) != 3:
        raise ValueError("This Qwen3-VL adapter requires all three DeepStack levels.")
    minimum = int(video_processor.size["shortest_edge"])
    processed = video_processor(
        videos=[frames],
        do_sample_frames=False,
        device="cpu",
        size={"shortest_edge": min(minimum, 2 * max_pixels), "longest_edge": 2 * max_pixels},
        return_tensors="pt",
    )
    grid = processed["video_grid_thw"]
    if tuple(grid.shape) != (1, 3) or int(grid[0, 0]) != 1:
        raise ValueError("Two frames must produce exactly one temporal grid.")
    _, raw_height, raw_width = (int(value) for value in grid[0].tolist())
    if raw_height % merge_size or raw_width % merge_size:
        raise ValueError("Processed spatial grid is incompatible with Qwen3's merger.")
    visual = model.model.visual
    parameter = next(visual.parameters())
    pixels = processed["pixel_values_videos"].to(device=parameter.device, dtype=parameter.dtype)
    embeddings, deepstack = visual(pixels, grid_thw=grid.to(parameter.device))
    height, width = raw_height // merge_size, raw_width // merge_size
    expected = (height * width, int(vision_config.out_hidden_size))
    if tuple(embeddings.shape) != expected:
        raise ValueError("Visual encoder output does not match the merged spatial grid.")
    if len(deepstack) != 3 or any(tuple(level.shape) != expected for level in deepstack):
        raise ValueError("Visual encoder must return three aligned DeepStack tensors.")
    return {
        "embeddings": embeddings.detach(),
        "deepstack": [level.detach() for level in deepstack],
        "height": height,
        "width": width,
        "grid_thw": grid.detach().cpu().clone(),
    }


def _validate_snapshot(model: Any, snapshot: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, list]:
    embeddings, positions = snapshot["embeddings"], snapshot["positions"]
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2 or not embeddings.is_floating_point():
        raise ValueError("snapshot embeddings must be a floating-point [N, D] tensor.")
    count = embeddings.shape[0]
    if not isinstance(positions, torch.Tensor) or tuple(positions.shape) != (3, count):
        raise ValueError("snapshot positions must have shape [3, N].")
    if positions.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("snapshot positions must contain integer coordinates.")
    if positions.numel() and bool((positions < 0).any()):
        raise ValueError("snapshot positions must be nonnegative.")
    if count and embeddings.shape[1] != model.get_input_embeddings().weight.shape[1]:
        raise ValueError("snapshot embedding width does not match the model.")
    deepstack = snapshot.get("deepstack", [])
    if not isinstance(deepstack, (tuple, list)):
        raise ValueError("snapshot deepstack must be a list of three aligned tensors.")
    if count or deepstack:
        if len(deepstack) != 3 or any(
            not isinstance(level, torch.Tensor) or tuple(level.shape) != tuple(embeddings.shape)
            or not level.is_floating_point() for level in deepstack
        ):
            raise ValueError("snapshot deepstack must contain three floating-point [N, D] tensors.")
    if len(model.config.vision_config.deepstack_visual_indexes) != 3:
        raise ValueError("This Qwen3-VL adapter requires all three DeepStack levels.")
    return embeddings, positions, list(deepstack)


def _memory_prompt(model: Any, processor: Any, snapshot: Mapping[str, Any], question: str) -> dict[str, Any]:
    """Build native grid RoPE then select sparse positions, without image work.

    Stored time IDs identify memory groups only; Qwen3 encodes time in text and
    resets local rotary T to zero for every group. ``native_timestamp_seconds``
    in record metadata is preferred. Without it, ``timestamp_seconds`` is used
    verbatim; callers must provide their intended timeline in that field.
    Zero-token records contribute no timestamp or vision markers.
    """
    embeddings, positions, _ = _validate_snapshot(model, snapshot)
    count = embeddings.shape[0]
    tokenizer = getattr(processor, "tokenizer", processor)
    template_owner = processor if hasattr(processor, "apply_chat_template") else tokenizer
    video_token = getattr(processor, "video_token", "<|video_pad|>")
    vision_start = getattr(processor, "vision_start_token", "<|vision_start|>")
    vision_end = getattr(processor, "vision_end_token", "<|vision_end|>")
    positions_cpu = positions.detach().to(device="cpu", dtype=torch.long)
    grids, segments, keep_visual, used_ids = [], [], [], []
    token_offset = dense_offset = 0
    previous_time = -1
    merge_size = int(model.config.vision_config.spatial_merge_size)
    for record in snapshot.get("records", []):
        if not isinstance(record, Mapping):
            raise ValueError("snapshot records must be mappings.")
        retained = record.get("retained_tokens")
        time_id = record.get("time_id")
        if isinstance(retained, bool) or not isinstance(retained, Integral) or retained < 0:
            raise ValueError("Each record needs a nonnegative retained_tokens count.")
        if isinstance(time_id, bool) or not isinstance(time_id, Integral) or time_id <= previous_time:
            raise ValueError("Record time_id values must be nonnegative and strictly increasing.")
        previous_time = time_id
        height = _positive_integer(record.get("height"), "record height")
        width = _positive_integer(record.get("width"), "record width")
        if token_offset + retained > count:
            raise ValueError("Record token counts exceed the snapshot's tokens.")
        coords = positions_cpu[:, token_offset:token_offset + retained]
        token_offset += retained
        if retained == 0:
            continue
        flat = coords[1] * width + coords[2]
        if (bool((coords[0] != time_id).any()) or bool((coords[1] >= height).any())
                or bool((coords[2] >= width).any())):
            raise ValueError("Snapshot coordinates do not match their record's grid/time ID.")
        if flat.numel() > 1 and bool((flat[1:] <= flat[:-1]).any()):
            raise ValueError("Retained grid coordinates must be unique and in row-major order.")
        metadata = record.get("metadata", {})
        timestamp = metadata.get("native_timestamp_seconds", record.get("timestamp_seconds"))
        if isinstance(timestamp, bool) or not isinstance(timestamp, Real) or not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Each retained record needs a finite, nonnegative timestamp.")
        segments.append(f"<{timestamp:.1f} seconds>{vision_start}{video_token * (height * width)}{vision_end}")
        grids.append([1, height * merge_size, width * merge_size])
        keep_visual.append(flat + dense_offset)
        dense_offset += height * width
        used_ids.append(metadata.get("memory_id", record.get("memory_id", record.get("frame_id", time_id))))
    if token_offset != count:
        raise ValueError("Record token counts must equal the snapshot's token count.")
    content = ([{"type": "video"}] if count else []) + [{"type": "text", "text": question}]
    prompt = template_owner.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True,
    )
    if count:
        placeholder = vision_start + video_token + vision_end
        if prompt.count(placeholder) != 1 or prompt.count(video_token) != 1:
            raise ValueError("The chat template must contain exactly one native video placeholder.")
        prompt = prompt.replace(placeholder, "".join(segments), 1)
    encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    dense_ids = encoded["input_ids"]
    if dense_ids.ndim != 2 or dense_ids.shape[0] != 1 or dense_ids.shape[1] == 0:
        raise ValueError("The chat template must produce one nonempty prompt.")
    dense_visual_indices = (dense_ids[0] == model.config.video_token_id).nonzero(as_tuple=True)[0]
    if dense_visual_indices.numel() != dense_offset:
        raise ValueError("The prompt's video-token count does not match its full grids.")
    if bool((dense_ids == model.config.image_token_id).any()):
        raise ValueError("Memory questions cannot contain image placeholders.")
    grid_thw = torch.tensor(grids, dtype=torch.long) if grids else None
    native_positions, _ = model.model.get_rope_index(
        input_ids=dense_ids, video_grid_thw=grid_thw, attention_mask=torch.ones_like(dense_ids),
    )
    keep = torch.ones(dense_ids.shape[1], dtype=torch.bool)
    keep[dense_visual_indices] = False
    if keep_visual:
        keep[dense_visual_indices[torch.cat(keep_visual)]] = True
    input_ids = dense_ids[:, keep]
    rotary_positions = native_positions[:, :, keep]
    # The first axis is the causal sequence position. Flash Attention's packed
    # sequence detection must never receive repeated per-grid rotary T values.
    causal_positions = torch.arange(input_ids.shape[1]).view(1, 1, -1)
    return {
        "tokenizer": tokenizer,
        "input_ids": input_ids,
        "position_ids": torch.cat((causal_positions, rotary_positions), dim=0),
        "visual_pos_masks": input_ids == model.config.video_token_id,
        "next_rotary_position": int(native_positions.max()) + 1,
        "used_memory_ids": list(snapshot.get("used_memory_ids", used_ids)),
    }


@torch.inference_mode()
def generate_from_memory(
    model: Any,
    processor: Any,
    snapshot: Mapping[str, Any],
    question: str,
    max_new_tokens: int = 128,
) -> dict[str, Any]:
    """Greedily answer from immutable projected memory, with DeepStack prefill.

    Sparse spatial coordinates retain their native full-grid rotary values;
    timestamp text and grid boundaries remain separate native Qwen3 segments.
    Cache offsets count retained tokens. DeepStack is added once during prefill,
    at the model's original decoder layers. No model-global rope delta is read
    or written, and every answer starts with a fresh KV cache.
    """
    _require_eval(model)
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string.")
    max_new_tokens = _positive_integer(max_new_tokens, "max_new_tokens")
    embeddings, _, deepstack = _validate_snapshot(model, snapshot)
    memory_tokens = embeddings.shape[0]
    input_embedding = model.get_input_embeddings()
    device, dtype = input_embedding.weight.device, input_embedding.weight.dtype
    _sync(device)
    started = time.perf_counter()
    prompt = _memory_prompt(model, processor, snapshot, question)
    input_ids = prompt["input_ids"].to(device)
    position_ids = prompt["position_ids"].to(device)
    visual_mask = prompt["visual_pos_masks"].to(device)
    input_tokens = input_ids.shape[1]
    inputs_embeds = input_embedding(input_ids)
    if memory_tokens:
        inputs_embeds[visual_mask] = embeddings.to(device=device, dtype=dtype)
    attention_mask = torch.ones_like(input_ids)
    language_model = model.model.language_model
    output = language_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache_position=torch.arange(input_tokens, device=device),
        past_key_values=None,
        visual_pos_masks=visual_mask if memory_tokens else None,
        deepstack_visual_embeds=[level.to(device=device, dtype=dtype) for level in deepstack] if memory_tokens else None,
        use_cache=True,
        return_dict=True,
    )
    logits = model.lm_head(output.last_hidden_state[:, -1:])
    _sync(device)
    prefill_finished = time.perf_counter()
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos is None:
        eos = getattr(model.config.text_config, "eos_token_id", None)
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) if eos is not None else set()
    generated, stopped_on_eos = [], False
    for step in range(max_new_tokens):
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        token_id = int(next_token[0, 0])
        generated.append(token_id)
        if token_id in eos_ids:
            stopped_on_eos = True
            break
        if step + 1 == max_new_tokens:
            break
        if output.past_key_values is None:
            raise RuntimeError("The model did not return a KV cache for cached decoding.")
        attention_mask = torch.cat((attention_mask, attention_mask.new_ones((1, 1))), dim=1)
        next_positions = torch.full(
            (4, 1, 1), prompt["next_rotary_position"] + step, dtype=torch.long, device=device,
        )
        next_positions[0] = input_tokens + step
        output = language_model(
            input_ids=next_token,
            attention_mask=attention_mask,
            position_ids=next_positions,
            cache_position=torch.tensor([input_tokens + step], dtype=torch.long, device=device),
            past_key_values=output.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        logits = model.lm_head(output.last_hidden_state[:, -1:])
    _sync(device)
    finished = time.perf_counter()
    return {
        "prediction": prompt["tokenizer"].decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False),
        "token_ids": generated,
        "input_tokens": input_tokens,
        "output_tokens": len(generated),
        "memory_tokens": memory_tokens,
        "empty_memory": memory_tokens == 0,
        "memory_truncated": bool(snapshot.get("memory_truncated", snapshot.get("truncated", False))
                                 or snapshot.get("stats", {}).get("evicted_grids", 0)),
        "used_memory_ids": prompt["used_memory_ids"],
        "elapsed_sec": finished - started,
        "prefill_sec": prefill_finished - started,
        "decode_sec": finished - prefill_finished,
        "decode_backend": "explicit_cached_greedy",
        "stopped_on_eos": stopped_on_eos,
    }
