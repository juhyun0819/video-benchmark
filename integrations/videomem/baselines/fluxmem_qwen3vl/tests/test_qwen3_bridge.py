"""CPU parity checks against the actual Transformers Qwen3-VL implementation."""
from __future__ import annotations

import copy
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import Qwen2TokenizerFast, Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

from streaming.qwen3_bridge import _memory_prompt, encode_pair, generate_from_memory


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def processor():
    words = ["[PAD]", "[UNK]", "[EOS]", "<|vision_start|>", "<|vision_end|>",
             "<|image_pad|>", "<|video_pad|>", "<|im_start|>", "<|im_end|>",
             "user", "assistant", "What", "happened?", "<0.5", "<2.5", "<4.5", "seconds>",
             "<80.5", "<20.5"]
    words.extend(f"word{index}" for index in range(len(words), 64))
    backend = Tokenizer(WordLevel({word: index for index, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = Qwen2TokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]",
        additional_special_tokens=words[3:9],
    )
    template = (
        "{% for message in messages %}<|im_start|> {{ message['role'] }} "
        "{% for item in message['content'] %}"
        "{% if item['type'] == 'video' %}<|vision_start|><|video_pad|><|vision_end|>"
        "{% elif item['type'] == 'text' %}{{ item['text'] }} {% endif %}"
        "{% endfor %}<|im_end|> {% endfor %}"
        "{% if add_generation_prompt %}<|im_start|> assistant {% endif %}"
    )
    return Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessorFast(patch_size=2, temporal_patch_size=2, merge_size=2),
        video_processor=Qwen3VLVideoProcessor(
            patch_size=2, temporal_patch_size=2, merge_size=2,
            size={"shortest_edge": 32, "longest_edge": 128},
        ),
        tokenizer=tokenizer, chat_template=template,
    )


@pytest.fixture
def model():
    torch.manual_seed(13)
    config = Qwen3VLConfig(
        text_config={"vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
                     "num_hidden_layers": 3, "num_attention_heads": 2,
                     "num_key_value_heads": 1, "head_dim": 16,
                     "max_position_embeddings": 2048,
                     "rope_scaling": {"rope_type": "default", "mrope_section": [2, 3, 3]},
                     "eos_token_id": None},
        vision_config={"depth": 3, "hidden_size": 32, "intermediate_size": 64,
                       "num_heads": 4, "patch_size": 2, "spatial_merge_size": 2,
                       "temporal_patch_size": 2, "out_hidden_size": 32,
                       "num_position_embeddings": 16, "deepstack_visual_indexes": [0, 1, 2]},
        image_token_id=5, video_token_id=6, vision_start_token_id=3, vision_end_token_id=4,
    )
    config._attn_implementation = "eager"
    instance = Qwen3VLForConditionalGeneration(config).eval()
    instance.generation_config.eos_token_id = None
    return instance


def frames(count=6):
    # Distinct spatial/temporal pixels prevent degenerate all-identical features.
    return [Image.fromarray((np.arange(8 * 8 * 3).reshape(8, 8, 3) + i * 31).astype(np.uint8))
            for i in range(count)]


def native_case(model, processor, groups=3):
    images = frames(2 * groups)
    question = "What happened?"
    template = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": question}]}],
        tokenize=False, add_generation_prompt=True,
    )
    native = processor(
        text=template, videos=[images], do_sample_frames=False, device="cpu",
        size={"shortest_edge": 32, "longest_edge": 128 * groups},
        video_metadata=[{"total_num_frames": 2 * groups, "fps": 1., "frames_indices": list(range(2 * groups))}],
        return_tensors="pt",
    )
    with torch.inference_mode():
        main, aux = model.model.visual(native["pixel_values_videos"], grid_thw=native["video_grid_thw"])
    h = torch.arange(2).repeat_interleave(2)
    w = torch.arange(2).repeat(2)
    snapshot = {
        "embeddings": main.detach().clone(), "deepstack": [level.detach().clone() for level in aux],
        "positions": torch.cat([torch.stack((torch.full_like(h, i), h, w)) for i in range(groups)], dim=1),
        "records": [{"time_id": i, "timestamp_seconds": i * 2., "height": 2, "width": 2,
                     "retained_tokens": 4, "metadata": {"memory_id": f"pair-{i}",
                     "native_timestamp_seconds": i * 2. + .5,
                     "frame_metadata": [{"timestamp_seconds": 1000. + i * 2.},
                                        {"timestamp_seconds": 1001. + i * 2.}]}} for i in range(groups)],
    }
    return question, native, snapshot


def select_snapshot(snapshot, selections):
    selected = copy.deepcopy(snapshot)
    keep = torch.cat([torch.tensor(indices, dtype=torch.long) + 4 * i for i, indices in enumerate(selections)])
    selected["embeddings"] = selected["embeddings"][keep]
    selected["positions"] = selected["positions"][:, keep]
    selected["deepstack"] = [level[keep] for level in selected["deepstack"]]
    for record, indices in zip(selected["records"], selections):
        record["retained_tokens"] = len(indices)
    return selected


def test_encode_pair_uses_native_cpu_video_processor_and_all_deepstack(model, processor):
    images = frames(2)
    with patch.object(processor.video_processor, "preprocess", wraps=processor.video_processor.preprocess) as process, \
         patch.object(processor.video_processor, "sample_frames", side_effect=AssertionError("sampled")), \
         patch.object(processor.image_processor, "preprocess", side_effect=AssertionError("image processor")), \
         patch.object(model.model.visual, "forward", wraps=model.model.visual.forward) as visual:
        encoded = encode_pair(model, processor, images, max_pixels=64)
    assert process.call_count == visual.call_count == 1
    args = process.call_args.kwargs
    assert process.call_args.args[0][0] is images
    assert args["do_sample_frames"] is False and args["device"] == "cpu"
    assert args["size"] == {"shortest_edge": 32, "longest_edge": 128}
    assert encoded["grid_thw"].tolist() == [[1, 4, 4]]
    assert (encoded["height"], encoded["width"]) == (2, 2)
    assert encoded["embeddings"].shape == (4, 32)
    assert len(encoded["deepstack"]) == 3
    direct = processor.video_processor(videos=[images], do_sample_frames=False, device="cpu", return_tensors="pt")
    with torch.inference_mode():
        main, aux = model.model.visual(direct["pixel_values_videos"], grid_thw=direct["video_grid_thw"])
    for actual, expected in zip([encoded["embeddings"], *encoded["deepstack"]], [main, *aux]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert not actual.requires_grad and actual.device.type == "cpu"
    assert not torch.cuda.is_initialized()


def test_dense_native_prompt_rope_logits_and_deepstack_parity(model, processor):
    question, native, snapshot = native_case(model, processor)
    prompt = _memory_prompt(model, processor, snapshot, question)
    assert torch.equal(prompt["input_ids"], native["input_ids"])
    native_rope, _ = model.model.get_rope_index(native["input_ids"], video_grid_thw=native["video_grid_thw"])
    torch.testing.assert_close(prompt["position_ids"][1:], native_rope, rtol=0, atol=0)
    assert torch.equal(prompt["position_ids"][0, 0], torch.arange(native["input_ids"].shape[1]))
    with torch.inference_mode():
        native_out = model(**native, use_cache=False, logits_to_keep=1)
    seen_logits, deepstack_calls = [], []
    hook = model.lm_head.register_forward_hook(lambda module, args, out: seen_logits.append(out.detach().clone()))
    real_deepstack = model.model.language_model._deepstack_process

    def observed_deepstack(hidden, mask, aux):
        deepstack_calls.append(aux.detach().clone())
        return real_deepstack(hidden, mask, aux)

    try:
        with patch.object(model.model.visual, "forward", side_effect=AssertionError("query called vision")), \
             patch.object(model.model.language_model, "_deepstack_process", side_effect=observed_deepstack):
            result = generate_from_memory(model, processor, snapshot, question, max_new_tokens=3)
    finally:
        hook.remove()
    torch.testing.assert_close(seen_logits[0][:, -1], native_out.logits[:, -1], rtol=0, atol=0)
    assert len(deepstack_calls) == 3  # Prefill only, across three native decoder layers.
    for actual, expected in zip(deepstack_calls, snapshot["deepstack"]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert result["token_ids"][0] == int(native_out.logits[0, -1].argmax())
    assert result["memory_tokens"] == 12 and result["used_memory_ids"] == ["pair-0", "pair-1", "pair-2"]


def test_sparse_positions_keep_full_grid_extents_and_native_timestamp_segments(model, processor):
    question, native, snapshot = native_case(model, processor)
    full = _memory_prompt(model, processor, snapshot, question)
    sparse = select_snapshot(snapshot, [[0], [1, 2], [0]])
    # IDs are memory identities, not rotary temporal coordinates.
    for i, record in enumerate(sparse["records"]):
        sparse["positions"][0, sparse["positions"][0] == i] = 40 + 10 * i
        record["time_id"] = 40 + 10 * i
    actual = _memory_prompt(model, processor, sparse, question)
    visual_indices = (native["input_ids"][0] == model.config.video_token_id).nonzero(as_tuple=True)[0]
    keep = torch.ones(native["input_ids"].shape[1], dtype=torch.bool)
    keep[visual_indices] = False
    keep[visual_indices[torch.tensor([0, 5, 6, 8])]] = True
    torch.testing.assert_close(actual["input_ids"], full["input_ids"][:, keep], rtol=0, atol=0)
    torch.testing.assert_close(actual["position_ids"][1:], full["position_ids"][1:, :, keep], rtol=0, atol=0)
    assert actual["next_rotary_position"] == full["next_rotary_position"]
    # Native markers include one timestamp and one segment per temporal group.
    text = processor.tokenizer.decode(actual["input_ids"][0], skip_special_tokens=False)
    assert "<0.5 seconds>" in text and "<2.5 seconds>" in text and "<4.5 seconds>" in text
    assert text.count("<|vision_start|>") == 3
    assert "1000" not in text  # Actual PTS remain provenance, not the compact video timeline.


@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
def test_sparse_cached_greedy_matches_full_recompute_with_deepstack(model, processor, implementation):
    model.config.text_config._attn_implementation = implementation
    question, _, snapshot = native_case(model, processor)
    snapshot = select_snapshot(snapshot, [[0], [0, 2], [1]])
    prompt = _memory_prompt(model, processor, snapshot, question)
    cached_logits = []
    hook = model.lm_head.register_forward_hook(
        lambda module, args, out: cached_logits.append(out.detach().clone())
    )
    try:
        cached = generate_from_memory(model, processor, snapshot, question, max_new_tokens=4)
    finally:
        hook.remove()
    with torch.inference_mode():
        embeds = model.get_input_embeddings()(prompt["input_ids"])
        embeds[prompt["visual_pos_masks"]] = snapshot["embeddings"]
        positions, visual_mask = prompt["position_ids"], prompt["visual_pos_masks"]
        expected = []
        for step in range(4):
            output = model.model.language_model(
                inputs_embeds=embeds, position_ids=positions, attention_mask=torch.ones(embeds.shape[:2], dtype=torch.long),
                visual_pos_masks=visual_mask, deepstack_visual_embeds=snapshot["deepstack"], use_cache=False,
            )
            logits = model.lm_head(output.last_hidden_state[:, -1:])
            torch.testing.assert_close(cached_logits[step], logits, rtol=1e-5, atol=1e-6)
            token = logits[:, -1].argmax(-1, keepdim=True)
            expected.append(int(token[0, 0]))
            embeds = torch.cat((embeds, model.get_input_embeddings()(token)), dim=1)
            next_positions = torch.full((4, 1, 1), prompt["next_rotary_position"] + step, dtype=torch.long)
            next_positions[0] = positions.shape[-1]
            positions = torch.cat((positions, next_positions), dim=-1)
            visual_mask = torch.cat((visual_mask, torch.zeros((1, 1), dtype=torch.bool)), dim=1)
    assert cached["token_ids"] == expected


def test_whole_group_drops_omit_timestamp_markers_and_preserve_snapshot(model, processor):
    question, _, snapshot = native_case(model, processor)
    snapshot = select_snapshot(snapshot, [[0, 3], [], [1]])
    original = copy.deepcopy(snapshot)
    sentinel = torch.tensor([[8765]])
    model.model.rope_deltas = sentinel.clone()
    observed = []

    def pre_hook(module, args, kwargs):
        observed.append({key: value.detach().clone() if isinstance(value, torch.Tensor) else value
                         for key, value in kwargs.items() if key in {"position_ids", "cache_position", "past_key_values",
                                                                   "deepstack_visual_embeds", "visual_pos_masks"}})

    hook = model.model.language_model.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        first = generate_from_memory(model, processor, snapshot, question, max_new_tokens=3)
        second = generate_from_memory(model, processor, snapshot, question, max_new_tokens=3)
    finally:
        hook.remove()
    assert first["token_ids"] == second["token_ids"]
    assert first["used_memory_ids"] == ["pair-0", "pair-2"]
    assert observed[0]["past_key_values"] is None and observed[3]["past_key_values"] is None
    for i in (1, 2, 4, 5):
        assert "deepstack_visual_embeds" not in observed[i] and "visual_pos_masks" not in observed[i]
        assert int(observed[i]["position_ids"][0, 0, 0]) == int(observed[i]["cache_position"][0])
    for key in ("embeddings", "positions"):
        torch.testing.assert_close(snapshot[key], original[key], rtol=0, atol=0)
    for actual, saved in zip(snapshot["deepstack"], original["deepstack"]):
        torch.testing.assert_close(actual, saved, rtol=0, atol=0)
    assert snapshot["records"] == original["records"]
    assert torch.equal(model.model.rope_deltas, sentinel)
    prompt = _memory_prompt(model, processor, snapshot, question)
    decoded = processor.tokenizer.decode(prompt["input_ids"][0], skip_special_tokens=False)
    assert decoded.count("<|vision_start|>") == 2 and "<2.5 seconds>" not in decoded


def test_empty_memory_and_timestamp_fallback(model, processor):
    empty = {"embeddings": torch.empty((0, 0)), "positions": torch.empty((3, 0), dtype=torch.long), "records": []}
    result = generate_from_memory(model, processor, empty, "What happened?", max_new_tokens=2)
    assert result["empty_memory"] and result["used_memory_ids"] == []
    question, _, snapshot = native_case(model, processor, groups=1)
    del snapshot["records"][0]["metadata"]["native_timestamp_seconds"]
    snapshot["records"][0]["timestamp_seconds"] = 80.5
    prompt = _memory_prompt(model, processor, snapshot, question)
    assert "<80.5 seconds>" in processor.tokenizer.decode(prompt["input_ids"][0])


@pytest.mark.parametrize("invalid", ["deepstack", "coordinates", "count", "order"])
def test_invalid_alignment_fails_before_language_model(model, processor, invalid):
    question, _, snapshot = native_case(model, processor)
    if invalid == "deepstack":
        snapshot["deepstack"] = snapshot["deepstack"][:2]
    elif invalid == "coordinates":
        snapshot["positions"][1, 0] = 8
    elif invalid == "count":
        snapshot["records"][0]["retained_tokens"] = 3
    else:
        snapshot["positions"][:, [0, 1]] = snapshot["positions"][:, [1, 0]]
    with patch.object(model.model.language_model, "forward", side_effect=AssertionError("must validate first")):
        with pytest.raises(ValueError):
            generate_from_memory(model, processor, snapshot, question, max_new_tokens=2)
