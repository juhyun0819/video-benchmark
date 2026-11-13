"""CPU checks using real native Qwen processor and a tiny vision-only model."""
import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import Qwen2VLImageProcessor
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel

from baselines.feature_memory.encoder import Qwen25VisionEncoder, _checkpoint_info, _file_state


@pytest.fixture
def tiny_checkpoint(tmp_path):
    config = Qwen2_5_VLVisionConfig(depth=1, hidden_size=16, intermediate_size=24,
                                  num_heads=4, patch_size=2, spatial_merge_size=2,
                                  temporal_patch_size=2, out_hidden_size=24,
                                  window_size=8, fullatt_block_indexes=[0])
    config._attn_implementation = "sdpa"
    torch.manual_seed(123)
    reference = Qwen2_5_VisionTransformerPretrainedModel(config).eval()
    state = {"visual." + name: value.contiguous() for name, value in reference.state_dict().items()}
    save_file(state, str(tmp_path / "visual.safetensors"))
    # The unneeded LLM shard deliberately does not exist: loading it would fail.
    weight_map = {name: "visual.safetensors" for name in state}
    weight_map["model.language_model.weight"] = "DO_NOT_LOAD_LLM.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen2_5_vl", "vision_config": config.to_dict()}))
    processor = Qwen2VLImageProcessor(min_pixels=16, max_pixels=4096, patch_size=2,
                                    temporal_patch_size=2, merge_size=2)
    processor.save_pretrained(tmp_path)
    return tmp_path, reference, processor


def test_real_native_processor_and_vision_output_agree_without_llm(tiny_checkpoint):
    path, reference, processor = tiny_checkpoint
    image = Image.new("RGB", (12, 8), (33, 75, 121))
    encoder = Qwen25VisionEncoder(path, device="cpu", dtype="fp32")
    features, metadata = encoder.encode(image)
    prepared = processor(images=[image], return_tensors="pt")
    with torch.inference_mode():
        expected = reference(prepared["pixel_values"], grid_thw=prepared["image_grid_thw"])
    torch.testing.assert_close(features, expected)
    assert features.shape == (6, 24)
    assert metadata["grid_thw"] == [1, 4, 6]
    assert metadata["feature_bytes"] == 6 * 24 * 4
    assert metadata["tensor_storage_bytes"] == metadata["feature_bytes"]
    assert metadata["feature_bytes_by_dtype"] == {"float16": 288, "bfloat16": 288, "float32": 576}
    assert not encoder.metadata["llm_loaded"]
    assert encoder.metadata["loaded_shards"] == ["visual.safetensors"]
    assert not any(parameter.requires_grad for parameter in encoder.model.parameters())
    assert not any(parameter.is_meta for parameter in encoder.model.parameters())
    assert encoder.model.rotary_pos_emb.inv_freq.dtype == torch.float32
    assert all(metadata[key] >= 0 for key in ("preprocess_seconds", "encode_seconds", "h2d_seconds", "d2h_seconds"))
    json.dumps(metadata)
    json.dumps(encoder.metadata)
    # Neither model nor processor retains the caller-owned image/feature tensor.
    later, _ = encoder.encode(Image.new("RGB", (12, 8), (80, 20, 3)))
    assert later.data_ptr() != features.data_ptr()
    torch.testing.assert_close(features, expected)


def test_native_720_resolution_and_same_frame_temporal_padding():
    processor = Qwen2VLImageProcessor(min_pixels=3136, max_pixels=12845056,
                                    patch_size=14, temporal_patch_size=2, merge_size=2)
    image = Image.new("RGB", (720, 720), (23, 124, 254))
    prepared = processor(images=[image], return_tensors="pt")
    assert prepared["image_grid_thw"].tolist() == [[1, 52, 52]]
    patches = prepared["pixel_values"].reshape(2704, 3, 2, 14, 14)
    torch.testing.assert_close(patches[:, :, 0], patches[:, :, 1], rtol=0, atol=0)
    assert 52 * 52 // 4 == 676


def test_refuse_changed_verified_checkpoint(tiny_checkpoint):
    path, _, _ = tiny_checkpoint
    # Test manifest validation directly to avoid loading the model twice.
    names = ["config.json", "model.safetensors.index.json", "visual.safetensors"]
    marker = path.parent / ".downloads" / path.name / "verification.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({"remote_revision_verified": True, "repo_id": "fixture/qwen",
                                  "revision": "0" * 40,
                                  "materialized_file_states": {name: _file_state(path / name) for name in names}}))
    assert _checkpoint_info(path, names)["verified_checkpoint_files_unchanged"]
    with (path / "config.json").open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="changed since"):
        _checkpoint_info(path, names)
