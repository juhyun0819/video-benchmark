"""CPU native Qwen2/projector checks, without pretrained downloads or training."""
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast, Qwen2ForCausalLM

from streamforest_videomem import bridge
from streamforest_videomem.upstream import load_native

native = load_native()
from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import tokenizer_image_token
from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor, SigLipVisionModel
from llava.model.multimodal_projector.projector_FSTW_PEMF import ToMe_FSTW_PEMF


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tokenizer():
    words = ["[PAD]", "[UNK]", "[EOS]", "<|im_start|>", "<|im_end|>",
             "system", "user", "assistant", "You", "are", "a", "helpful", "assistant.",
             "What", "happened?", "A", "B", "<im_start>", "<im_end>"]
    words.extend(f"word{i}" for i in range(len(words), 64))
    backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]",
        additional_special_tokens=["<|im_start|>", "<|im_end|>", "<im_start>", "<im_end>"],
    )


class IsolatedVision(torch.nn.Module):
    """Tests preprocessing/inference boundaries, not pretrained SigLip quality."""

    def __init__(self, hidden=1152, tokens=729):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor([7.]))
        self.hidden, self.tokens = hidden, tokens
        self.calls = []
        self.image_processor = SigLipImageProcessor()

    def forward(self, pixels):
        self.calls.append(pixels.detach().clone())
        return pixels.mean().expand(1, self.tokens, self.hidden).clone()


@pytest.fixture
def model():
    torch.manual_seed(311)
    config = native.LlavaQwenConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
        pad_token_id=0, eos_token_id=None, tie_word_embeddings=False,
        transformers_version="4.46.0",
        mm_hidden_size=16, mm_pos_num_frames=1, mm_patch_merge_type="flat",
        frame_aspect_ratio="square", mm_llm_compress=False,
    )
    config._attn_implementation = "eager"
    instance = native.LlavaQwenForCausalLM(config).eval()
    # No mm_vision_tower in the tiny config, so construction never touches a
    # checkpoint. Attach the actual released projector and a non-reading tower.
    instance.model.mm_projector = ToMe_FSTW_PEMF(
        config, SimpleNamespace(image_size=28, patch_size=14, num_attention_heads=4))
    instance.model.vision_tower = IsolatedVision()
    instance.eval().requires_grad_(False)
    instance.generation_config.eos_token_id = None
    instance.generation_config.pad_token_id = 0
    return instance


@pytest.fixture
def snapshot():
    generator = torch.Generator().manual_seed(901)
    return {"features": torch.randn(1, 8, 16, generator=generator), "memory_truncated": True}


def test_native_projected_memory_insertion_and_cached_decode_match_dense_reference(model, tokenizer, snapshot):
    original = deepcopy(snapshot)
    prepared = bridge._prepare_prompt(model, tokenizer, snapshot, "What happened?", 3)
    ids = tokenizer_image_token(prepared["prompt"], tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    image_index = int((ids == IMAGE_TOKEN_INDEX).nonzero()[0])
    # Construct a separate dense embedding sequence from public text plus the
    # released MLP output; native LLM insertion must preserve every memory row.
    projected = model.model.mm_projector.mlp(snapshot["features"])[0]
    dense = torch.cat((model.get_input_embeddings()(ids[:image_index]), projected,
                       model.get_input_embeddings()(ids[image_index + 1:]))).unsqueeze(0)
    torch.testing.assert_close(prepared["inputs_embeds"], dense, rtol=0, atol=0)
    assert prepared["input_tokens"] == ids.numel() - 1 + 8
    assert "<|im_start|>system" in prepared["prompt"]
    assert prepared["prompt"].count("<image>") == 1
    with torch.inference_mode():
        reference = Qwen2ForCausalLM.forward(
            model, inputs_embeds=dense, attention_mask=torch.ones(dense.shape[:2], dtype=torch.long),
            use_cache=False, return_dict=True,
        ).logits[:, -1]
    logits, prefill_cache_lengths = [], []
    output_hook = model.lm_head.register_forward_hook(
        lambda module, args, output: logits.append(output.detach().clone()))

    def observe_inputs(module, args, kwargs):
        if kwargs.get("inputs_embeds") is not None:
            cache = kwargs.get("past_key_values")
            prefill_cache_lengths.append(0 if cache is None else cache.get_seq_length())

    input_hook = model.model.register_forward_pre_hook(observe_inputs, with_kwargs=True)
    try:
        with patch.object(model, "generate", side_effect=AssertionError("multimodal generate replay")), \
             patch.object(model, "generate_online", side_effect=AssertionError("memory reconstruction")), \
             patch.object(model.model.mm_projector, "forward", side_effect=AssertionError("batch memory rebuild")), \
             patch.object(model.get_vision_tower(), "forward", side_effect=AssertionError("vision replay")):
            first = bridge.generate_from_memory(model, tokenizer, snapshot, "What happened?", 3)
            second = bridge.generate_from_memory(model, tokenizer, snapshot, "What happened?", 3)
    finally:
        output_hook.remove()
        input_hook.remove()
    torch.testing.assert_close(logits[0][:, -1], reference, rtol=1e-5, atol=1e-6)
    assert first["token_ids"][0] == int(reference[0].argmax())
    assert first["token_ids"] == second["token_ids"]
    assert prefill_cache_lengths == [0, 0]
    assert first["memory_tokens"] == 8 and first["output_tokens"] == 3
    assert first["input_tokens"] == prepared["input_tokens"]
    assert first["stopped_on_eos"] is False
    assert first["memory_truncated"] is True
    assert first["decode_backend"] == "native_qwen2_cached_greedy"
    torch.testing.assert_close(snapshot["features"], original["features"], rtol=0, atol=0)


def test_actual_eos_stops_generation_before_token_limit(model, tokenizer, snapshot):
    # Zero logits make token zero the deterministic greedy choice.
    model.lm_head.weight.data.zero_()
    model.generation_config.eos_token_id = [0, 2]
    result = bridge.generate_from_memory(model, tokenizer, snapshot, "What happened?", 5)
    assert result["token_ids"] == [0]
    assert result["output_tokens"] == 1
    assert result["stopped_on_eos"] is True


def test_one_frame_uses_native_siglip_preprocessing_once_without_projection(model):
    image = Image.new("RGB", (31, 17), (40, 80, 120))
    tower = model.get_vision_tower()
    try:
        with patch.object(model.model.mm_projector, "forward", side_effect=AssertionError("projected during encode")), \
             patch.object(model.model.mm_projector.mlp, "forward", side_effect=AssertionError("MLP during encode")):
            features = bridge.encode_frame(model, tower.image_processor, image)
        assert len(tower.calls) == 1
        expected_pixels = tower.image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
        torch.testing.assert_close(tower.calls[0], expected_pixels, rtol=0, atol=0)
        assert features.shape == (729, 1152)
        torch.testing.assert_close(features, expected_pixels.mean().expand_as(features), rtol=0, atol=0)
        assert not features.requires_grad
        assert image.getpixel((0, 0)) == (40, 80, 120)
    finally:
        image.close()


@pytest.mark.parametrize("failure", ["empty", "width", "nonfinite", "reserved_token", "context", "train"])
def test_invalid_memory_or_prompt_is_rejected_before_generation(model, tokenizer, snapshot, failure):
    question = "What happened?"
    if failure == "empty":
        snapshot["features"] = torch.empty(1, 0, 16)
    elif failure == "width":
        snapshot["features"] = torch.zeros(1, 8, 17)
    elif failure == "nonfinite":
        snapshot["features"][0, 0, 0] = float("nan")
    elif failure == "reserved_token":
        question = "What does <image> contain?"
    elif failure == "context":
        model.config.tokenizer_model_max_length = 5
    else:
        model.train()
    with patch.object(Qwen2ForCausalLM, "generate", side_effect=AssertionError("generation started")):
        with pytest.raises(ValueError):
            bridge.generate_from_memory(model, tokenizer, snapshot, question, 3)


@pytest.fixture
def local_checkpoint(tmp_path):
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {f"model.vision_tower.vision_tower.weight_{i}": "model-1.safetensors"
                       for i in range(421)},
    }))
    return checkpoint


def test_loader_uses_embedded_native_vision_and_only_local_reads(local_checkpoint, model, tokenizer):
    config = SimpleNamespace(model_type="llava_qwen", mm_projector_type="tome729_fstw_pemf",
                             mm_hidden_size=1152, mm_patch_merge_type="flat")
    with patch.object(native.LlavaQwenConfig, "from_pretrained", return_value=config) as config_load, \
         patch.object(native.LlavaQwenForCausalLM, "from_pretrained", return_value=(model, {})) as model_load, \
         patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer) as tokenizer_load, \
         patch.object(SigLipVisionModel, "from_pretrained", side_effect=AssertionError("external vision load")):
        loaded, loaded_tokenizer, processor = bridge.load_model(local_checkpoint, device="cpu")
    assert loaded is model and loaded_tokenizer is tokenizer
    assert processor is model.get_vision_tower().image_processor
    assert config.mm_vision_tower == str(local_checkpoint.resolve())
    assert config._videomem_embedded_vision is True
    assert config.delay_load is True
    for call in (config_load.call_args, model_load.call_args, tokenizer_load.call_args):
        assert call.args[0] == str(local_checkpoint.resolve())
        assert call.kwargs["local_files_only"] is True
    assert model_load.call_args.kwargs["device_map"] == {"": "cpu"}
    assert model_load.call_args.kwargs["torch_dtype"] == torch.float32
    assert model_load.call_args.kwargs["output_loading_info"] is True
    assert tokenizer_load.call_args.kwargs["trust_remote_code"] is False
    assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())
    assert float(model.get_vision_tower().anchor[0]) == 7.0


@pytest.mark.parametrize("field", ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"])
def test_loader_refuses_partial_or_discarded_weights(local_checkpoint, model, field):
    config = SimpleNamespace(model_type="llava_qwen", mm_projector_type="tome729_fstw_pemf",
                             mm_hidden_size=1152, mm_patch_merge_type="flat")
    with patch.object(native.LlavaQwenConfig, "from_pretrained", return_value=config), \
         patch.object(native.LlavaQwenForCausalLM, "from_pretrained", return_value=(model, {field: ["bad"]})), \
         patch("transformers.AutoTokenizer.from_pretrained", side_effect=AssertionError("tokenizer after bad weights")):
        with pytest.raises(RuntimeError):
            bridge.load_model(local_checkpoint, device="cpu")


def test_loader_rejects_external_vision_and_incomplete_embedded_index(local_checkpoint):
    with pytest.raises(ValueError, match="embeds SigLip"):
        bridge.load_model(local_checkpoint, vision_path=local_checkpoint, device="cpu")
    (local_checkpoint / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    with pytest.raises(ValueError, match="complete released"):
        bridge.load_model(local_checkpoint, device="cpu")


def test_published_general_checkpoint_geometry_and_complete_key_set_on_meta(tmp_path):
    # Public config from MCG-NJU/StreamForest-Qwen2-7B revision
    # 6840f91e17346b4c406ba03b57d1a63582765c91/config.json. The expected key
    # digest is computed from that revision's model.safetensors.index.json,
    # independently of this adapter. No weight shards are needed for this test.
    published_config = json.loads(r'''{
  "architectures": [
    "LlavaQwenForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "enable_depth_vision_tower": false,
  "eos_token_id": 151645,
  "frame_aspect_ratio": "square",
  "frame_grid_pinpoints": null,
  "hidden_act": "silu",
  "hidden_size": 3584,
  "image_aspect_ratio": "square",
  "image_crop_resolution": null,
  "image_grid_pinpoints": null,
  "image_split_resolution": null,
  "initializer_range": 0.02,
  "intermediate_size": 18944,
  "llm_compress_layer_list": [
    8,
    16,
    24
  ],
  "llm_compress_type": "attention",
  "llm_image_token_ratio_list": [
    1.0,
    0.5,
    0.25,
    0.125
  ],
  "max_num_pixels": 14745600000,
  "max_position_embeddings": 32768,
  "max_window_layers": 28,
  "min_slow_num_frames": 4,
  "mm_close_init": true,
  "mm_hidden_size": 1152,
  "mm_llm_compress": false,
  "mm_local_num_frames": 1,
  "mm_newline_position": "one_token",
  "mm_num_compress_latents": 128,
  "mm_num_compress_query_type": "pooling",
  "mm_patch_merge_type": "flat",
  "mm_pos_num_frames": 1,
  "mm_projector_lr": null,
  "mm_projector_type": "tome729_fstw_pemf",
  "mm_resampler_type": null,
  "mm_spatial_pool_mode": "bilinear",
  "mm_tunable_parts": "mm_mlp_adapter,mm_language_model",
  "mm_use_im_patch_token": false,
  "mm_use_im_start_end": false,
  "mm_vision_select_feature": "patch",
  "mm_vision_select_layer": -2,
  "mm_vision_tower": "/mnt/petrelfs/share/videointern/siglip/siglip-so400m-patch14-384",
  "mm_vision_tower_lr": 2e-06,
  "model_type": "qwen2",
  "num_attention_heads": 28,
  "num_hidden_layers": 28,
  "num_key_value_heads": 4,
  "pos_skipping_range": 4096,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000.0,
  "sliding_window": null,
  "tie_word_embeddings": false,
  "time_msg_per_frame_token": null,
  "time_msg_type": "short_online_v2",
  "tokenizer_model_max_length": 32768,
  "tokenizer_padding_side": "right",
  "torch_dtype": "bfloat16",
  "transformers_version": "4.46.0",
  "use_cache": false,
  "use_mm_proj": true,
  "use_pos_skipping": false,
  "use_sliding_window": false,
  "vision_encode_type": "image_video_memory_batch",
  "vision_tower_pretrained": null,
  "vocab_size": 152064
}''')
    (tmp_path / "config.json").write_text(json.dumps(published_config))
    config = native.LlavaQwenConfig.from_pretrained(str(tmp_path), local_files_only=True)
    assert config.model_type == "qwen2"
    bridge._check_prompt_config(config)
    config.mm_vision_tower = str(tmp_path)
    config._videomem_embedded_vision = True
    config.delay_load = True
    config.unfreeze_mm_vision_tower = False
    config.mm_tunable_parts = ""
    config._attn_implementation = "eager"
    with torch.device("meta"):
        instance = native.LlavaQwenForCausalLM(config)
    state = instance.state_dict()
    import hashlib
    digest = hashlib.sha256("\n".join(sorted(state)).encode()).hexdigest()
    assert digest == "59354adf2cff235b8106f16ede458a2f7e2e2fadf9db402feca1d0ab6fe9ad7c"
    assert len(state) == 764
    assert sum(name.startswith("model.vision_tower.") for name in state) == 421
    assert all(value.device.type == "meta" for value in state.values())
    assert state["model.embed_tokens.weight"].shape == (152064, 3584)
    assert state["model.layers.0.self_attn.q_proj.weight"].shape == (3584, 3584)
    assert state["model.mm_projector.mlp.0.weight"].shape == (3584, 1152)
    assert state["model.vision_tower.vision_tower.vision_model.encoder.layers.25.self_attn.q_proj.weight"].shape == (1152, 1152)
    assert len(instance.get_vision_tower().vision_tower.vision_model.encoder.layers) == 26
    assert isinstance(instance.get_vision_tower().vision_tower.vision_model.head, torch.nn.Identity)
