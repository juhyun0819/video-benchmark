"""CPU execution/position checks with the actual tiny official Qwen model."""
from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

from test_fluxmem_runtime import tiny_model
from streaming.model_bridge import encode_pair, generate_from_memory


def tiny_tokenizer():
    words = ["[PAD]", "[BOS]", "[EOS]", "<|vision_start|>", "<|vision_end|>",
             "<|image_pad|>", "<|video_pad|>", "<|im_start|>", "<|im_end|>",
             "user", "assistant", "What", "happened?", "[UNK]"]
    words.extend(f"word{index}" for index in range(len(words), 64))
    backend = Tokenizer(WordLevel({word: index for index, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]",
        additional_special_tokens=words[3:9],
    )
    tokenizer.chat_template = (
        "{% for message in messages %}<|im_start|> {{ message['role'] }} "
        "{% for item in message['content'] %}"
        "{% if item['type'] == 'video' %}<|vision_start|> <|video_pad|> <|vision_end|> "
        "{% elif item['type'] == 'text' %}{{ item['text'] }} {% endif %}"
        "{% endfor %}<|im_end|> {% endfor %}"
        "{% if add_generation_prompt %}<|im_start|> assistant {% endif %}"
    )
    return tokenizer


@pytest.fixture
def model():
    instance = tiny_model()
    instance.config.eos_token_id = None
    instance.generation_config.eos_token_id = None
    return instance


def dense_inputs(model, tokenizer, temporal=3):
    question = "What happened?"
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": question}]}],
        tokenize=False, add_generation_prompt=True,
    )
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    prefix = int((ids[0] == model.config.video_token_id).nonzero()[0])
    ids = torch.cat((ids[:, :prefix], ids.new_full((1, temporal * 4), 6), ids[:, prefix + 1:]), dim=1)
    grid = torch.tensor([[temporal, 4, 4]])
    pixels = torch.randn(temporal * 16, 24)
    positions, _ = model.get_rope_index(ids, video_grid_thw=grid, second_per_grid_ts=[1.0])
    with torch.inference_mode():
        embeddings = torch.cat([
            model.visual(pixels[index * 16:(index + 1) * 16], grid_thw=torch.tensor([[1, 4, 4]]))
            for index in range(temporal)
        ])
    snapshot = {
        "embeddings": embeddings,
        "positions": positions[:, 0, prefix:prefix + temporal * 4] - prefix,
        "records": [{"memory_id": f"pair-{index}"} for index in range(temporal)],
    }
    return question, ids, pixels, grid, positions, snapshot, prefix


def test_encode_pair_runs_official_processor_and_visual_once(model):
    image_processor = Qwen2VLImageProcessor(
        min_pixels=16, max_pixels=64, patch_size=2, temporal_patch_size=2, merge_size=2,
    )
    frames = [Image.new("RGB", (8, 8), color) for color in ("red", "blue")]
    processor_calls = []

    class ObservedProcessor:
        temporal_patch_size = 2

        def __call__(self, **kwargs):
            processor_calls.append(kwargs)
            return image_processor(**kwargs)

    visual_calls = []
    hook = model.visual.register_forward_pre_hook(
        lambda module, args, kwargs: visual_calls.append((args[0].clone(), kwargs["grid_thw"].clone())),
        with_kwargs=True,
    )
    try:
        encoded = encode_pair(model, SimpleNamespace(image_processor=ObservedProcessor()), frames, max_pixels=64)
    finally:
        hook.remove()
    assert len(processor_calls) == len(visual_calls) == 1
    assert set(processor_calls[0]) == {"images", "videos", "return_tensors", "max_pixels"}
    assert processor_calls[0]["images"] is None
    assert len(processor_calls[0]["videos"]) == 1
    assert len(processor_calls[0]["videos"][0]) == 2
    assert processor_calls[0]["videos"][0] is frames
    assert frames[0].getpixel((0, 0)) == (255, 0, 0)
    assert frames[1].getpixel((0, 0)) == (0, 0, 255)
    assert encoded["grid_thw"].tolist() == [[1, 4, 4]]
    assert (encoded["height"], encoded["width"]) == (2, 2)
    assert encoded["embeddings"].shape == (4, 64)
    assert not encoded["embeddings"].requires_grad
    with torch.inference_mode():
        direct = image_processor(images=None, videos=[frames], return_tensors="pt", max_pixels=64)
        expected = model.visual(direct["pixel_values_videos"], grid_thw=direct["video_grid_thw"])
    torch.testing.assert_close(encoded["embeddings"], expected, rtol=0, atol=0)
    assert not torch.cuda.is_initialized()


def test_sparse_visual_tokens_positions_and_fresh_caches_reach_forward(model):
    tokenizer = tiny_tokenizer()
    snapshot = {
        "embeddings": torch.randn(3, 64),
        "positions": torch.tensor([[0, 43, 101], [0, 1, 0], [1, 0, 1]]),
        "records": [{"memory_id": "old"}, {"memory_id": "recent"}],
        "truncated": True,
    }
    original = copy.deepcopy(snapshot)
    model.rope_deltas = torch.tensor([[8765]])
    observed = []

    def observe(module, args, kwargs):
        observed.append({
            key: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for key, value in kwargs.items()
            if key in {"inputs_embeds", "input_ids", "position_ids", "cache_position", "past_key_values",
                       "attention_mask", "use_fluxmem", "pixel_values", "pixel_values_videos", "video_grid_thw"}
        })

    hook = model.register_forward_pre_hook(observe, with_kwargs=True)
    try:
        with patch.object(model.visual, "forward", side_effect=AssertionError("Query called vision")), \
             patch.object(model.model.fluxmem, "process_memory_streaming", side_effect=AssertionError("Query compressed memory")):
            result = generate_from_memory(model, tokenizer, snapshot, "What happened?", max_new_tokens=4)
            repeated = generate_from_memory(model, tokenizer, snapshot, "What happened?", max_new_tokens=4)
    finally:
        hook.remove()
    assert result["token_ids"] == repeated["token_ids"]
    assert result["used_memory_ids"] == ["old", "recent"]
    assert result["memory_truncated"] and not result["empty_memory"]
    assert result["memory_tokens"] == 3 and result["output_tokens"] == 4
    assert result["decode_backend"] == "explicit_cached_greedy"
    assert result["elapsed_sec"] >= result["prefill_sec"] >= 0
    assert len(observed) == 8
    assert observed[0]["past_key_values"] is observed[4]["past_key_values"] is None
    assert observed[1]["past_key_values"] is not observed[5]["past_key_values"]
    prefill = observed[0]
    prefix = int((prefill["input_ids"][0] == 6).nonzero()[0])
    torch.testing.assert_close(prefill["inputs_embeds"][0, prefix:prefix + 3], original["embeddings"], rtol=0, atol=0)
    torch.testing.assert_close(prefill["position_ids"][:, 0, prefix:prefix + 3], original["positions"] + prefix)
    assert prefill["position_ids"][0, 0, prefix + 3] == int(original["positions"].max()) + prefix + 1
    expected_cache = torch.arange(result["input_tokens"])
    torch.testing.assert_close(prefill["cache_position"], expected_cache)
    for step, call in enumerate(observed[1:4]):
        assert call["input_ids"].shape == (1, 1)
        assert call["cache_position"].tolist() == [result["input_tokens"] + step]
        assert call["position_ids"].unique().tolist() == [int(prefill["position_ids"].max()) + 1 + step]
        assert call["attention_mask"].shape[1] == result["input_tokens"] + step + 1
    for call in observed:
        assert call["use_fluxmem"] is False
        assert call["pixel_values"] is call["pixel_values_videos"] is call["video_grid_thw"] is None
    torch.testing.assert_close(snapshot["embeddings"], original["embeddings"], rtol=0, atol=0)
    torch.testing.assert_close(snapshot["positions"], original["positions"], rtol=0, atol=0)
    assert snapshot["records"] == original["records"]
    assert model.rope_deltas.tolist() == [[8765]]
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("side", [224, 448])
def test_encoder_preserves_supported_input_resolution_without_upscaling(model, side):
    config = copy.deepcopy(model.config)
    config.vision_config.patch_size = 14
    config.vision_config.window_size = 112
    resized_model = type(model)(config).eval()
    processor = SimpleNamespace(image_processor=Qwen2VLImageProcessor())
    encoded = encode_pair(resized_model, processor, [Image.new("RGB", (side, side)) for _ in range(2)])
    assert encoded["grid_thw"].tolist() == [[1, side // 14, side // 14]]
    assert encoded["embeddings"].shape == ((side // 28) ** 2, 64)
    assert encoded["height"] == encoded["width"] == side // 28


def test_cached_answer_matches_native_chunked_dense_reference(model):
    tokenizer = tiny_tokenizer()
    question, ids, pixels, grid, native_positions, snapshot, _ = dense_inputs(model, tokenizer)
    native_logits, bridge_logits, bridge_positions = [], [], []
    native_hook = model.register_forward_hook(lambda module, args, output: native_logits.append(output.logits[:, -1].clone()))
    try:
        with torch.inference_mode():
            native_tokens = model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values_videos=pixels,
                video_grid_thw=grid, second_per_grid_ts=[1.0], video_chunk_t=1,
                use_fluxmem=False, do_sample=False, max_new_tokens=4,
            )
    finally:
        native_hook.remove()
    bridge_hook = model.register_forward_hook(lambda module, args, output: bridge_logits.append(output.logits[:, -1].clone()))
    position_hook = model.register_forward_pre_hook(
        lambda module, args, kwargs: bridge_positions.append(kwargs["position_ids"].clone()), with_kwargs=True,
    )
    try:
        with patch.object(model.visual, "forward", side_effect=AssertionError("Query called vision")):
            result = generate_from_memory(model, tokenizer, snapshot, question, max_new_tokens=4)
    finally:
        bridge_hook.remove()
        position_hook.remove()
    assert result["input_tokens"] == ids.shape[1]
    assert result["token_ids"] == native_tokens[0, ids.shape[1]:].tolist()
    torch.testing.assert_close(bridge_positions[0], native_positions, rtol=0, atol=0)
    assert len(native_logits) == len(bridge_logits) == 4
    for actual, expected in zip(bridge_logits, native_logits):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_sparse_cached_logits_match_uncached_full_prefix(model):
    tokenizer = tiny_tokenizer()
    snapshot = {"embeddings": torch.randn(2, 64), "positions": torch.tensor([[0, 130], [0, 2], [1, 0]])}
    observed = []
    logits = []
    hook = model.register_forward_pre_hook(
        lambda module, args, kwargs: observed.append({key: value.clone() for key, value in kwargs.items() if isinstance(value, torch.Tensor)}),
        with_kwargs=True,
    )
    output_hook = model.register_forward_hook(lambda module, args, output: logits.append(output.logits[:, -1].clone()))
    try:
        result = generate_from_memory(model, tokenizer, snapshot, "What happened?", max_new_tokens=4)
    finally:
        hook.remove()
        output_hook.remove()
    original = observed[0]
    with torch.inference_mode():
        for step in range(1, 4):
            generated = torch.tensor([result["token_ids"][:step]])
            ids = torch.cat((original["input_ids"], generated), dim=1)
            embeds = torch.cat((original["inputs_embeds"], model.get_input_embeddings()(generated)), dim=1)
            start = int(original["position_ids"].max()) + 1
            pos = torch.cat((original["position_ids"], torch.arange(start, start + step).view(1, 1, -1).expand(3, 1, -1)), dim=2)
            reference = model(
                input_ids=ids, inputs_embeds=embeds, position_ids=pos, attention_mask=torch.ones_like(ids),
                cache_position=torch.arange(ids.shape[1]), past_key_values=None, use_cache=False,
                use_fluxmem=False, return_dict=True,
            )
            torch.testing.assert_close(logits[step], reference.logits[:, -1], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("width", [0, 64])
def test_empty_memory_is_explicit_and_text_only(model, width):
    snapshot = {"embeddings": torch.empty(0, width), "positions": torch.empty(3, 0, dtype=torch.long), "records": []}
    with patch.object(model.visual, "forward", side_effect=AssertionError("Query called vision")):
        result = generate_from_memory(model, tiny_tokenizer(), snapshot, "What happened?", max_new_tokens=1)
    assert result["empty_memory"]
    assert result["memory_tokens"] == 0
    assert result["used_memory_ids"] == []
    assert not result["memory_truncated"]


def test_memory_bank_ids_skip_empty_records_and_report_eviction(model):
    snapshot = {
        "embeddings": torch.randn(2, 64),
        "positions": torch.tensor([[0, 4], [0, 0], [0, 0]]),
        "records": [
            {"time_id": 0, "retained_tokens": 1, "metadata": {"memory_id": "file:pair0"}},
            {"time_id": 2, "retained_tokens": 0, "metadata": {}},
            {"time_id": 4, "retained_tokens": 1, "metadata": {}},
        ],
        "stats": {"evicted_grids": 3},
    }
    result = generate_from_memory(model, tiny_tokenizer(), snapshot, "What happened?", max_new_tokens=1)
    assert result["used_memory_ids"] == ["file:pair0", 4]
    assert result["memory_truncated"]


def test_eos_stops_cached_greedy_generation(model):
    model.generation_config.eos_token_id = [2, 8]
    snapshot = {"embeddings": torch.empty(0, 64), "positions": torch.empty(3, 0, dtype=torch.long)}

    def force_eos(module, args, output):
        scores = torch.zeros_like(output)
        scores[..., 8] = 10
        return scores

    hook = model.lm_head.register_forward_hook(force_eos)
    try:
        result = generate_from_memory(model, tiny_tokenizer(), snapshot, "What happened?", max_new_tokens=5)
    finally:
        hook.remove()
    assert result["token_ids"] == [8]
    assert result["output_tokens"] == 1 and result["stopped_on_eos"]


@pytest.mark.parametrize("frames", [[], [Image.new("RGB", (8, 8))], [None, None],
                                    [Image.new("L", (8, 8)), Image.new("L", (8, 8))]])
def test_invalid_pairs_fail_before_visual_encoding(model, frames):
    with patch.object(model.visual, "forward", side_effect=AssertionError("Invalid pair reached vision")):
        with pytest.raises((ValueError, TypeError)):
            encode_pair(model, SimpleNamespace(), frames)


@pytest.mark.parametrize("positions", [torch.zeros(2, 1, dtype=torch.long), torch.zeros(3, 1), -torch.ones(3, 1, dtype=torch.long)])
def test_invalid_memory_positions_fail_explicitly(model, positions):
    with pytest.raises(ValueError):
        generate_from_memory(model, tiny_tokenizer(), {"embeddings": torch.randn(1, 64), "positions": positions}, "What happened?")
