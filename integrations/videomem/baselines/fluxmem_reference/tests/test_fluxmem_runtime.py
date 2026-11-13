"""CPU-only compatibility checks for official FluxMem inference.

Random tiny-model outputs are execution checks, not model-quality evidence.
All FluxMem calls below delegate to the unmodified official methods.
Run directly to also write report/runtime_versions.json.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from qwen2_5_vl_fluxmem import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from qwen2_5_vl_fluxmem.fluxmem import FluxMem

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def _git_metadata(*arguments):
    """Export metadata is optional for source ZIPs and an unborn checkout."""
    try:
        return subprocess.check_output(
            ['git', *arguments], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@contextmanager
def observe_memory(memory):
    """Observe calls and lengths while returning each original result unchanged."""
    names = ("process_memory_streaming", "_apply_adjacent_pruning", "_long_term_memory_merge_per_frames")
    originals = {name: getattr(memory, name) for name in names}
    observation = {"memory_calls": [], "tas_calls": 0, "sdc_calls": 0, "sdc_tokens_removed": 0}

    def process(*args, **kwargs):
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        before = int(hidden.shape[1])
        result = originals["process_memory_streaming"](*args, **kwargs)
        observation["memory_calls"].append({"input_tokens": before, "retained_tokens": int(result[0].shape[1])})
        return result

    def tas(*args, **kwargs):
        observation["tas_calls"] += 1
        return originals["_apply_adjacent_pruning"](*args, **kwargs)

    def sdc(*args, **kwargs):
        observation["sdc_calls"] += 1
        keep_mask = kwargs["keep_mask"]
        before = int(keep_mask.sum())
        result = originals["_long_term_memory_merge_per_frames"](*args, **kwargs)
        observation["sdc_tokens_removed"] += before - int(keep_mask.sum())
        return result

    memory.process_memory_streaming = process
    memory._apply_adjacent_pruning = tas
    memory._long_term_memory_merge_per_frames = sdc
    try:
        yield observation
    finally:
        for name, method in originals.items():
            setattr(memory, name, method)


def tiny_model():
    torch.manual_seed(42)
    config = Qwen2_5_VLConfig(
        vocab_size=64, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=1024, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        vision_start_token_id=3, vision_end_token_id=4, image_token_id=5, video_token_id=6,
        rope_scaling={"type": "mrope", "mrope_section": [2, 2, 4]},
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64,
                       "num_heads": 4, "in_channels": 3, "patch_size": 2,
                       "spatial_merge_size": 2, "temporal_patch_size": 2,
                       "tokens_per_second": 4, "window_size": 8, "out_hidden_size": 64,
                       "fullatt_block_indexes": [0]},
        attn_implementation="sdpa",
    )
    return Qwen2_5_VLForConditionalGeneration(config).eval()


def check_tiny_generation():
    model = tiny_model()
    assert model.config._attn_implementation == "sdpa"
    assert model.visual.config._attn_implementation == "sdpa"
    input_ids = torch.tensor([[1, 9, 3] + [6] * 32 + [4, 10, 11]])
    inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids),
              "pixel_values_videos": torch.randn(8 * 4 * 4, 3 * 2 * 2 * 2),
              "video_grid_thw": torch.tensor([[8, 4, 4]]), "second_per_grid_ts": [0.5]}
    results = {}
    for enabled in (False, True):
        decoder_lengths = []
        hook = model.model.layers[0].register_forward_pre_hook(
            lambda module, args: decoder_lengths.append(int(args[0].shape[1]))
        )
        try:
            with observe_memory(model.model.fluxmem) as observation, torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=3, min_new_tokens=3,
                                        do_sample=False, use_fluxmem=enabled,
                                        short_frames=2, medium_frames=2)
        finally:
            hook.remove()
        assert output.shape == (1, input_ids.shape[1] + 3)
        assert decoder_lengths[1:] == [1, 1], "Cached autoregressive decode did not run"
        assert len(observation["memory_calls"]) == int(enabled)
        if enabled:
            assert observation["tas_calls"] > 0 and observation["sdc_calls"] > 0
            assert decoder_lengths[0] < input_ids.shape[1], "FluxMem was not applied to prefill"
        else:
            assert decoder_lengths[0] == input_ids.shape[1]
            assert observation["tas_calls"] == observation["sdc_calls"] == 0
        results[str(enabled).lower()] = {**observation, "decoder_sequence_lengths": decoder_lengths,
                                         "generated_token_ids": output[0, -3:].tolist()}
    assert not torch.cuda.is_initialized()
    return {"status": "passed", "device": "cpu", "attention_implementation": "sdpa",
            "vision_attention_implementation": "sdpa", "random_parameters": sum(p.numel() for p in model.parameters()),
            "input_tokens": int(input_ids.shape[1]), "synthetic_temporal_grids": 8,
            "short_frames": 2, "medium_frames": 2, "cases": results}


def controlled_memory_input(kind):
    temporal, height, width, dimension = 8, 3, 3, 64
    input_ids = torch.tensor([[1, 3] + [6] * (temporal * height * width) + [4, 10]])
    length = input_ids.shape[1]
    positions = torch.arange(length).repeat(3, 1).unsqueeze(1)
    hidden = torch.zeros(1, length, dimension)
    for frame in range(temporal):
        for row in range(height):
            for column in range(width):
                index = 2 + frame * height * width + row * width + column
                positions[:, 0, index] = torch.tensor([frame, row, column])
                axis = (0 if kind == "repeated_frames" else frame * 2) + (0 if row < 2 else 1)
                hidden[0, index, axis] = 1
    return {"hidden_states": hidden, "position_ids": positions,
            "position_embeddings": (torch.zeros(3, 1, length, 16), torch.ones(3, 1, length, 16)),
            "input_ids": input_ids, "video_grid_thw": torch.tensor([[temporal, height * 2, width * 2]])}


def check_controlled_memory(kind):
    inputs = controlled_memory_input(kind)
    memory = FluxMem(vision_start_token_id=3, vision_end_token_id=4,
                     short_term_frames=2, mid_term_frames=2)
    original_length = inputs["hidden_states"].shape[1]
    with observe_memory(memory) as observation, torch.inference_mode():
        output = memory.process_memory_streaming(**inputs)
    hidden, position_embeddings, positions, attention_mask, kept = output
    assert observation["tas_calls"] == 7 and observation["sdc_calls"] == 6
    assert hidden.shape[1] == (13 if kind == "repeated_frames" else 34)
    assert all(index in kept[0].tolist() for index in [0, 1, original_length - 2, original_length - 1])
    assert set(range(65, 74)).issubset(kept[0].tolist()), "Latest frame was compressed"
    assert positions.shape[-1] == hidden.shape[1] == attention_mask.shape[-1]
    assert position_embeddings[0].shape[2] == hidden.shape[1]
    assert torch.isfinite(hidden).all()
    assert observation["sdc_tokens_removed"] == (0 if kind == "repeated_frames" else 42)
    assert not torch.cuda.is_initialized()
    return {"status": "passed", "fixture": kind, "input_tokens": original_length,
            "retained_tokens": hidden.shape[1], "all_text_and_latest_frame_tokens_preserved": True,
            **observation}


def test_sdpa_tiny_model_generation_with_fluxmem_enabled_and_disabled():
    check_tiny_generation()


def test_tas_prunes_repeated_frames():
    check_controlled_memory("repeated_frames")


def test_sdc_merges_spatial_groups_after_temporal_survival():
    check_controlled_memory("orthogonal_frames")


def write_runtime_report():
    import av
    import torchvision
    import transformers
    import qwen_vl_utils_fluxmem

    boxes = torch.tensor([[0., 0., 2., 2.], [0., 0., 2., 2.]])
    assert torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8]), 0.5).tolist() == [0]
    source = Path("/home/kai/lifelong_mem/routine_7days/Day3/Day3_04-04_식사후업무_01분30초.mp4")
    with av.open(str(source)) as container:
        frame = next(container.decode(video=0))
        pixels = frame.to_ndarray(format="rgb24")
        assert pixels.ndim == 3 and pixels.shape[-1] == 3 and pixels.size > 0
        media_probe = {"status": "passed", "path": str(source), "decoded_frames": 1,
                       "frame_pts": frame.pts, "frame_time_base": str(frame.time_base),
                       "rgb_shape": list(pixels.shape), "device": "cpu"}
    assert subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "models/"], cwd=ROOT).returncode == 0
    report = {
        "schema_version": "fluxmem_inference_runtime_v1", "status": "passed",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": str(ROOT / ".venv/bin/python"),
        "upstream_commit": '6fdbf65b6f2af1a94ca41d542be5be86a565d338', 'repository_commit': _git_metadata('rev-parse','HEAD'),
        "versions": {name: metadata.version(name) for name in
                     ["torch", "torchvision", "transformers", "tokenizers", "av", "accelerate",
                      "qwen2-5-vl-fluxmem", "qwen-vl-utils-fluxmem"]},
        "torch_build": torch.__version__, "torch_cuda_build": torch.version.cuda,
        "compiled_cuda_architectures": torch._C._cuda_getArchFlags(),
        "transformers_source": transformers.__file__, "av_source": av.__file__,
        "qwen_utils_source": qwen_vl_utils_fluxmem.__file__,
        "torch_source": torch.__file__, "torchvision_cpu_binary_operation": "passed",
        "bounded_pyav_video_decode": media_probe,
        "cuda_initialized": False, "pretrained_weights_loaded": False, "training_performed": False,
        "upstream_model_code_modified": False, "compatibility_shim_required": False,
        "scope": "CPU execution compatibility with random tiny weights and controlled token fixtures; no quality evaluation",
        "runtime_pin_deviations": {
            "torch": {"upstream": "2.6.0", "runtime": "2.9.0+cu128", "reason": "Blackwell sm_120 support"},
            "torchvision": {"upstream": "0.21.0", "runtime": "0.24.0+cu128", "reason": "matching torch binary ABI"},
            "av": {"upstream": "14.4.0", "runtime": "18.1.0", "reason": "available working wheel; bounded CPU media decode passed"},
        },
        "tiny_generation": check_tiny_generation(),
        "controlled_memory": [check_controlled_memory("repeated_frames"), check_controlled_memory("orthogonal_frames")],
    }
    assert not torch.cuda.is_initialized()
    destination = ROOT / "report/runtime_versions.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    write_runtime_report()
