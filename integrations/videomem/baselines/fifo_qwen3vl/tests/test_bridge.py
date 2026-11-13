"""Native tiny-Qwen3 integration; no pretrained weights or network are used."""
from copy import deepcopy
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest
import torch

# The memory/scheduler tests can run with Torch alone. This optional integration
# needs the documented Qwen3 inference environment, including torchvision.
pytest.importorskip("transformers", minversion="4.57.3", exc_type=ImportError,
                    reason="Native Qwen3 integration requires the Qwen3 inference dependencies")
pytest.importorskip("torchvision", exc_type=ImportError)
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import Qwen2TokenizerFast, Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

from fifo_videomem.memory import Qwen3FIFOMemory
from fifo_videomem.shared import load_shared

load_shared()
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
             "user", "assistant", "What", "happened?", "<0.5", "<2.5", "<4.5", "seconds>"]
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


@pytest.mark.parametrize("budget", [{"memory_frames": 4}, {"memory_tokens": 9}])
def test_evicted_fifo_snapshot_decodes_with_native_deepstack_and_no_vision_replay(model, processor, budget):
    bank = Qwen3FIFOMemory(**budget)
    encoded = []
    for index in range(3):
        images = [Image.fromarray(
            (np.arange(8 * 8 * 3).reshape(8, 8, 3) + (index * 2 + offset) * 31).astype(np.uint8)
        ) for offset in (0, 1)]
        try:
            projection = encode_pair(model, processor, images, max_pixels=64)
        finally:
            for image in images:
                image.close()
        encoded.append(projection)
        bank.ingest(
            projection["embeddings"], projection["height"], projection["width"],
            index, index * 2.,
            {"memory_id": f"pair-{index}", "native_timestamp_seconds": index * 2. + .5},
            deepstack=projection["deepstack"],
        )
        bank.advance()

    snapshot = bank.snapshot()
    original = deepcopy(snapshot)
    assert [record["time_id"] for record in snapshot["records"]] == [1, 2]
    assert snapshot["stats"]["total_tokens"] == 8
    torch.testing.assert_close(snapshot["embeddings"],
                               torch.cat([item["embeddings"] for item in encoded[1:]]), rtol=0, atol=0)
    prompt = _memory_prompt(model, processor, snapshot, "What happened?")
    text = processor.tokenizer.decode(prompt["input_ids"][0], skip_special_tokens=False)
    assert "<0.5 seconds>" not in text
    assert "<2.5 seconds>" in text and "<4.5 seconds>" in text
    assert text.count("<|vision_start|>") == 2
    assert prompt["used_memory_ids"] == ["pair-1", "pair-2"]

    # Independent dense forward of the retained bank checks bridge logits and
    # all three native DeepStack injections after deleting the oldest grid.
    with torch.inference_mode():
        embeddings = model.get_input_embeddings()(prompt["input_ids"])
        embeddings[prompt["visual_pos_masks"]] = snapshot["embeddings"]
        direct = model.model.language_model(
            inputs_embeds=embeddings, position_ids=prompt["position_ids"],
            attention_mask=torch.ones(embeddings.shape[:2], dtype=torch.long),
            visual_pos_masks=prompt["visual_pos_masks"],
            deepstack_visual_embeds=snapshot["deepstack"], use_cache=False,
        )
        expected_logits = model.lm_head(direct.last_hidden_state[:, -1:])
    logits, injected = [], []
    hook = model.lm_head.register_forward_hook(
        lambda module, args, result: logits.append(result.detach().clone()))
    native_deepstack = model.model.language_model._deepstack_process

    def observe_deepstack(hidden, mask, auxiliary):
        injected.append(auxiliary.detach().clone())
        return native_deepstack(hidden, mask, auxiliary)

    try:
        with patch.object(model.model.visual, "forward", side_effect=AssertionError("vision replay")), \
             patch.object(model.model.language_model, "_deepstack_process", side_effect=observe_deepstack):
            result = generate_from_memory(model, processor, snapshot, "What happened?", max_new_tokens=2)
    finally:
        hook.remove()
    torch.testing.assert_close(logits[0], expected_logits, rtol=1e-5, atol=1e-6)
    assert len(injected) == 3
    for actual, expected in zip(injected, snapshot["deepstack"]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert result["token_ids"][0] == int(expected_logits[0, -1].argmax())
    assert result["memory_tokens"] == 8
    assert result["used_memory_ids"] == ["pair-1", "pair-2"]
    assert result["memory_truncated"] is True
    for key in ("embeddings", "positions"):
        torch.testing.assert_close(snapshot[key], original[key], rtol=0, atol=0)
    for actual, expected in zip(snapshot["deepstack"], original["deepstack"]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert snapshot["records"] == original["records"]
