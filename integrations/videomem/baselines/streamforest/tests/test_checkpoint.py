"""Local inventory contracts using tiny fake files, never actual model weights."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from streamforest_videomem import checkpoint


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def model(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    write_json(root / "config.json", {"model_type": "qwen2",
               "architectures": ["LlavaQwenForCausalLM"],
               "mm_projector_type": "tome729_fstw_pemf", "mm_hidden_size": 1152})
    for name in ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"):
        write_json(root / name, {})
    (root / "merges.txt").write_text("#version: 0.2\na b\n")
    (root / "model-00001-of-00002.safetensors").write_bytes(b"fake language shard for inventory tests")
    (root / "model-00002-of-00002.safetensors").write_bytes(b"fake embedded vision shard for inventory tests")
    write_json(root / "model.safetensors.index.json", {"weight_map": {
        "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
        "model.vision_tower.vision_tower.vision_model.embeddings.patch_embedding.weight":
            "model-00002-of-00002.safetensors",
    }})
    return root


def inventory(model, tmp_path):
    target = tmp_path / "verification.json"
    record = checkpoint.create_verification_record(model, target)
    return target, record


def test_create_and_verify_local_hashes_without_claiming_remote_authentication(model, tmp_path):
    write_json(model / "special_tokens_map.json", {"eos_token": "<|im_end|>"})
    (model / "chat_template.jinja").write_text("{{ messages }}")
    (model / "README.md").write_text("Not a live inference artifact")
    target, record = inventory(model, tmp_path)
    result = checkpoint.verify_checkpoint(model, target)
    assert result["repo_id"] == "MCG-NJU/StreamForest-Qwen2-7B"
    assert result["revision"] == "6840f91e17346b4c406ba03b57d1a63582765c91"
    assert result["verification_kind"] == "local_sha256_inventory"
    assert result["remote_revision_verified"] is False
    assert result["verification_status"] == "local_bytes_match_inventory"
    names = {entry["path"] for entry in result["verified_files"]}
    assert {"special_tokens_map.json", "chat_template.jinja", "vocab.json", "merges.txt"} <= names
    assert "README.md" not in names
    for entry in result["verified_files"]:
        raw = (model / entry["path"]).read_bytes()
        assert entry["bytes"] == len(raw)
        assert entry["hash"] == hashlib.sha256(raw).hexdigest()
    assert result["record_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert result["artifact_set_sha256"] == record["artifact_set_sha256"]
    with pytest.raises(FileExistsError):
        checkpoint.create_verification_record(model, target)


@pytest.mark.parametrize("artifact", ["tokenizer.json", "merges.txt", "generation_config.json",
                                      "model-00002-of-00002.safetensors"])
def test_missing_required_artifact_prevents_inventory_creation(model, tmp_path, artifact):
    (model / artifact).unlink()
    target = tmp_path / "verification.json"
    with pytest.raises(FileNotFoundError):
        checkpoint.create_verification_record(model, target)
    assert not target.exists()


@pytest.mark.parametrize("same_size", [False, True])
def test_modified_shard_fails_even_when_file_size_is_unchanged(model, tmp_path, same_size):
    target, _ = inventory(model, tmp_path)
    shard = model / "model-00001-of-00002.safetensors"
    before = shard.read_bytes()
    shard.write_bytes(b"X" * len(before) if same_size else before + b"changed")
    with pytest.raises(RuntimeError, match="SHA256 changed" if same_size else "byte count changed"):
        checkpoint.verify_checkpoint(model, target)


@pytest.mark.parametrize("name", ["added_tokens.json", "special_tokens_map.json",
                                 "model.safetensors", "chat_template.jinja", "tokenizer.alt.json"])
def test_added_live_inference_artifact_requires_record_coverage(model, tmp_path, name):
    target, _ = inventory(model, tmp_path)
    (model / name).write_text("{}")
    with pytest.raises(ValueError, match="missing="):
        checkpoint.verify_checkpoint(model, target)


def test_optional_template_directory_is_covered(model, tmp_path):
    (model / "chat_templates").mkdir()
    (model / "chat_templates" / "tool_use.jinja").write_text("{{ messages }}")
    target, record = inventory(model, tmp_path)
    assert "chat_templates/tool_use.jinja" in {entry["path"] for entry in record["verified_files"]}
    checkpoint.verify_checkpoint(model, target)
    (model / "chat_templates" / "extra.jinja").write_text("{{ tools }}")
    with pytest.raises(ValueError, match="missing="):
        checkpoint.verify_checkpoint(model, target)


@pytest.mark.parametrize("bad_path", ["../outside.safetensors", "/tmp/weights.safetensors",
                                     "./model-00001-of-00002.safetensors", "sub//weights.safetensors",
                                     "sub\\weights.safetensors", "weights\x00.safetensors"])
def test_index_rejects_unsafe_or_noncanonical_paths(model, tmp_path, bad_path):
    index = {"weight_map": {"model.vision_tower.vision_tower.weight": bad_path}}
    write_json(model / "model.safetensors.index.json", index)
    with pytest.raises(ValueError, match="paths"):
        checkpoint.create_verification_record(model, tmp_path / "verification.json")


def test_outside_symlink_is_rejected_even_if_target_has_same_bytes(model, tmp_path):
    target, _ = inventory(model, tmp_path)
    shard = model / "model-00001-of-00002.safetensors"
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(shard.read_bytes())
    shard.unlink()
    shard.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        checkpoint.verify_checkpoint(model, target)
    with pytest.raises(ValueError, match="escapes"):
        checkpoint.create_verification_record(model, tmp_path / "new_inventory.json")


def test_internal_symlink_is_allowed_and_hashed(model, tmp_path):
    storage = model / "storage"
    storage.mkdir()
    shard = model / "model-00001-of-00002.safetensors"
    real = storage / "language.payload"
    shard.rename(real)
    shard.symlink_to("storage/language.payload")
    target, _ = inventory(model, tmp_path)
    checkpoint.verify_checkpoint(model, target)


def test_non_streamforest_config_or_missing_embedded_vision_is_rejected(model, tmp_path):
    path = model / "config.json"
    good = json.loads(path.read_text())
    write_json(path, {**good, "model_type": "qwen3_vl"})
    with pytest.raises(ValueError, match="architecture"):
        checkpoint.create_verification_record(model, tmp_path / "verification.json")
    write_json(path, good)
    write_json(model / "model.safetensors.index.json", {"weight_map": {
        "model.embed_tokens.weight": "model-00001-of-00002.safetensors"}})
    with pytest.raises(ValueError, match="embedded SigLip"):
        checkpoint.create_verification_record(model, tmp_path / "verification.json")


@pytest.mark.parametrize("model_type", ["qwen2", "llava_qwen"])
def test_published_qwen2_and_native_llava_qwen_config_names_are_supported(model, tmp_path, model_type):
    config = json.loads((model / "config.json").read_text())
    config["model_type"] = model_type
    write_json(model / "config.json", config)
    target, _ = inventory(model, tmp_path)
    assert checkpoint.verify_checkpoint(model, target)["verification_status"] == "local_bytes_match_inventory"


@pytest.mark.parametrize("invalid", ["identity", "revision", "remote_claim", "duplicate",
                                      "digest", "bytes", "entry_path", "missing_shard", "aggregate"])
def test_invalid_inventory_record_is_rejected(model, tmp_path, invalid):
    target, record = inventory(model, tmp_path)
    changed = deepcopy(record)
    if invalid == "identity":
        changed["repo_id"] = "someone/another-model"
    elif invalid == "revision":
        changed["revision"] = "main"
    elif invalid == "remote_claim":
        changed["remote_revision_verified"] = True
    elif invalid == "duplicate":
        changed["verified_files"].append(deepcopy(changed["verified_files"][0]))
    elif invalid == "digest":
        changed["verified_files"][0]["hash"] = "a" * 40
    elif invalid == "bytes":
        changed["verified_files"][0]["bytes"] = True
    elif invalid == "entry_path":
        changed["verified_files"][0]["path"] = "../config.json"
    elif invalid == "missing_shard":
        changed["verified_files"] = [entry for entry in changed["verified_files"]
                                     if entry["path"] != "model-00002-of-00002.safetensors"]
    elif invalid == "aggregate":
        changed["artifact_set_sha256"] = "0" * 64
    write_json(target, changed)
    with pytest.raises(ValueError):
        checkpoint.verify_checkpoint(model, target)


def test_inventory_can_live_inside_model_but_cannot_shadow_an_inference_file(model):
    target = model / "verification.json"
    checkpoint.create_verification_record(model, target)
    checkpoint.verify_checkpoint(model, target)
    with pytest.raises(ValueError, match="artifact name"):
        checkpoint.create_verification_record(model, model / "model.safetensors")


def test_cli_and_import_require_no_torch_or_transformers(model, tmp_path, capsys):
    target = tmp_path / "verification.json"
    assert checkpoint.main(["--model-path", str(model), "--output", str(target)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["remote_revision_verified"] is False
    assert output["verified_files"] == len(checkpoint.verify_checkpoint(model, target)["verified_files"])
    command = "import sys; import streamforest_videomem.checkpoint; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
    subprocess.run([sys.executable, "-c", command], check=True, cwd=Path(__file__).resolve().parents[1])
