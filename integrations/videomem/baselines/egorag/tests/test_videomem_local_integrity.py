"""Tiny CPU checkpoint fixtures; no torch, inference, network, or downloads."""

import builtins
import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest

from videomem_adapter.backends import BackendConfigurationError
from videomem_adapter import local_integrity
from videomem_adapter.local_integrity import CheckpointIntegrity


REVISION = "a" * 40
SHARD = "model-00001-of-00001.safetensors"


def source_hash(data, git_blob=False):
    if git_blob:
        return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()
    return hashlib.sha256(data).hexdigest()


def write_marker(root, marker, names=None):
    names = names or sorted(path.name for path in root.iterdir() if path.is_file())
    files = []
    for name in names:
        data = (root / name).read_bytes()
        files.append({"path": name, "bytes": len(data), "hash": source_hash(data, git_blob=name.endswith(".json"))})
    contents = {
        "repo_id": "Qwen/Qwen3-VL-8B-Instruct", "revision": REVISION,
        "file_count": len(files), "verified_files": files,
        "total_bytes": sum(item["bytes"] for item in files), "shard_count": 1,
        "indexed_tensors": 1, "gpu_loaded": False, "inference_tested": False,
    }
    marker.write_text(json.dumps(contents), encoding="utf-8")
    return contents


def checkpoint(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"qwen3_vl"}', encoding="utf-8")
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"tiny.weight": SHARD}}), encoding="utf-8")
    (root / SHARD).write_bytes(b"tiny CPU-only weight fixture")
    (root / "tokenizer.json").write_text('{"tokens":["tiny"]}', encoding="utf-8")
    (root / "README.md").write_text("test checkpoint\n", encoding="utf-8")
    marker = tmp_path / "verification.json"
    write_marker(root, marker)
    return {"model": str(root), "revision": REVISION, "verified_manifest": str(marker)}


def edit_marker(settings, edit):
    marker = Path(settings["verified_manifest"])
    contents = json.loads(marker.read_text())
    edit(contents)
    marker.write_text(json.dumps(contents), encoding="utf-8")


def test_verification_is_opt_in_and_never_claims_unconfigured_files():
    verifier = CheckpointIntegrity()
    assert verifier.register({}) == {"verification": "not_configured"}
    verifier.assert_unchanged()


def test_verifies_git_blobs_and_sha256_without_importing_model_libraries(tmp_path, monkeypatch):
    settings = checkpoint(tmp_path)
    original_import = builtins.__import__

    def cpu_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "transformers", "open_clip"}
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", cpu_import)
    identity = CheckpointIntegrity().register(settings)
    assert identity["verification"] == "verified"
    assert identity["revision"] == REVISION
    assert identity["manifest_sha256"] == hashlib.sha256(Path(settings["verified_manifest"]).read_bytes()).hexdigest()
    root = Path(settings["model"])
    for item in identity["files"]:
        data = (root / item["path"]).read_bytes()
        assert item == {"path": item["path"], "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    serialized = json.dumps(identity)
    assert str(tmp_path) not in serialized
    assert not any(field in serialized for field in ("mtime", "ctime", "st_ino", "st_dev"))


def test_moving_identical_checkpoint_preserves_identity(tmp_path):
    settings = checkpoint(tmp_path)
    identity = CheckpointIntegrity().register(settings)
    copy_root = tmp_path / "moved"
    shutil.copytree(settings["model"], copy_root)
    copy_marker = tmp_path / "moved-verification.json"
    shutil.copyfile(settings["verified_manifest"], copy_marker)
    copied = {**settings, "model": str(copy_root), "verified_manifest": str(copy_marker)}
    assert CheckpointIntegrity().register(copied) == identity


def test_roles_and_replicas_share_one_hash_pass_and_cannot_mutate_identity(tmp_path, monkeypatch):
    settings = checkpoint(tmp_path)
    calls = []
    original = local_integrity._read_file

    def counted(path, *args, **kwargs):
        calls.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(local_integrity, "_read_file", counted)
    verifier = CheckpointIntegrity()
    first = verifier.register(settings)
    for role in range(9):
        assert verifier.register({**settings, "device": "cuda:%d" % (role % 2)}) == first
    verifier.assert_unchanged()
    assert len(calls) == len(first["files"]) + 1
    assert len(set(calls)) == len(calls)
    first["files"][0]["sha256"] = "mutated return value"
    assert verifier.register(settings)["files"][0]["sha256"] != "mutated return value"


@pytest.mark.parametrize("field,value,match", [
    ("repo_id", "untrusted/model", "repository or revision"),
    ("revision", "b" * 40, "repository or revision"),
    ("file_count", 99, "file_count"),
    ("file_count", True, "file_count"),
    ("total_bytes", 1, "total_bytes"),
    ("shard_count", 2, "shard_count"),
    ("indexed_tensors", 2, "indexed_tensors"),
    ("verified_files", [], "nonempty list"),
])
def test_rejects_invalid_manifest_metadata(tmp_path, field, value, match):
    settings = checkpoint(tmp_path)
    edit_marker(settings, lambda marker: marker.update({field: value}))
    with pytest.raises(BackendConfigurationError, match=match):
        CheckpointIntegrity().register(settings)


@pytest.mark.parametrize("field,value,match", [
    ("bytes", True, "byte count"),
    ("bytes", -1, "byte count"),
    ("hash", "not a digest", "manifest hash"),
    ("hash", "0" * 40, "hash mismatch"),
])
def test_rejects_invalid_file_metadata_and_wrong_source_digest(tmp_path, field, value, match):
    settings = checkpoint(tmp_path)
    edit_marker(settings, lambda marker: marker["verified_files"][0].update({field: value}))
    with pytest.raises(BackendConfigurationError, match=match):
        CheckpointIntegrity().register(settings)


@pytest.mark.parametrize("unsafe", ["../outside", "/outside", "nested/../outside", "nested//file", "./file", "nested\\file", ""])
def test_rejects_unsafe_manifest_paths(tmp_path, unsafe):
    settings = checkpoint(tmp_path)
    edit_marker(settings, lambda marker: marker["verified_files"][0].update({"path": unsafe}))
    with pytest.raises(BackendConfigurationError, match="unsafe manifest path"):
        CheckpointIntegrity().register(settings)


def test_rejects_duplicate_files_and_duplicate_json_fields(tmp_path):
    settings = checkpoint(tmp_path)
    edit_marker(settings, lambda marker: marker["verified_files"][1].update(marker["verified_files"][0]))
    with pytest.raises(BackendConfigurationError, match="duplicate manifest file"):
        CheckpointIntegrity().register(settings)
    marker = Path(settings["verified_manifest"])
    marker.write_text('{"repo_id":"first","repo_id":"second"}', encoding="utf-8")
    with pytest.raises(BackendConfigurationError, match="invalid JSON"):
        CheckpointIntegrity().register(settings)


@pytest.mark.parametrize("filename", ["config.json", "model.safetensors.index.json"])
def test_requires_config_and_shard_index_in_manifest(tmp_path, filename):
    settings = checkpoint(tmp_path)
    root = Path(settings["model"])
    write_marker(root, Path(settings["verified_manifest"]), [p.name for p in root.iterdir() if p.name != filename])
    with pytest.raises(BackendConfigurationError, match="must cover"):
        CheckpointIntegrity().register(settings)


def test_index_cannot_reference_unverified_or_unsafe_shard(tmp_path):
    settings = checkpoint(tmp_path)
    root = Path(settings["model"])
    (root / SHARD).unlink()
    write_marker(root, Path(settings["verified_manifest"]))
    with pytest.raises(BackendConfigurationError, match="unverified weight shard"):
        CheckpointIntegrity().register(settings)
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"tiny.weight": "../outside.safetensors"}}), encoding="utf-8")
    write_marker(root, Path(settings["verified_manifest"]))
    with pytest.raises(BackendConfigurationError, match="unsafe manifest path"):
        CheckpointIntegrity().register(settings)


@pytest.mark.parametrize("name", ["model.safetensors", "pytorch_model.bin", "tokenizer_config.json", "processor_config.json", "chat_template.jinja", "merges.txt", "additional_chat_templates/extra.jinja"])
def test_rejects_unverified_files_that_can_change_loading(tmp_path, name):
    settings = checkpoint(tmp_path)
    extra = Path(settings["model"]) / name
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_bytes(b"injected alternate model input")
    with pytest.raises(BackendConfigurationError, match="unverified model files"):
        CheckpointIntegrity().register(settings)


def test_rejects_missing_and_corrupted_files(tmp_path):
    settings = checkpoint(tmp_path)
    path = Path(settings["model"]) / SHARD
    data = path.read_bytes()
    path.write_bytes(b"X" * len(data))
    with pytest.raises(BackendConfigurationError, match="hash mismatch"):
        CheckpointIntegrity().register(settings)
    path.write_bytes(data[:-1])
    with pytest.raises(BackendConfigurationError, match="byte count mismatch"):
        CheckpointIntegrity().register(settings)
    path.unlink()
    with pytest.raises(BackendConfigurationError, match="missing or unreadable file"):
        CheckpointIntegrity().register(settings)


def test_missing_marker_and_nonabsolute_marker_fail_clearly(tmp_path):
    settings = checkpoint(tmp_path)
    Path(settings["verified_manifest"]).unlink()
    with pytest.raises(BackendConfigurationError, match="missing or unreadable file.*verification.json"):
        CheckpointIntegrity().register(settings)
    with pytest.raises(BackendConfigurationError, match="verified_manifest must be an absolute path"):
        CheckpointIntegrity().register({**settings, "verified_manifest": "verification.json"})


def test_rejects_symlinked_files_and_directories(tmp_path):
    settings = checkpoint(tmp_path)
    path = Path(settings["model"]) / SHARD
    original = tmp_path / "outside.safetensors"
    path.replace(original)
    path.symlink_to(original)
    with pytest.raises(BackendConfigurationError, match="symlink"):
        CheckpointIntegrity().register(settings)
    path.unlink()
    original.replace(path)
    (Path(settings["model"]) / "nested").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(BackendConfigurationError, match="symlink directory"):
        CheckpointIntegrity().register(settings)


@pytest.mark.parametrize("mutation", ["restored_mtime_replacement", "same_bytes_replacement", "marker_rewrite", "delete", "injected_weight", "injected_template"])
def test_mutations_after_verification_fail_closed_without_rehashing(tmp_path, monkeypatch, mutation):
    settings = checkpoint(tmp_path)
    verifier = CheckpointIntegrity()
    verifier.register(settings)
    path = Path(settings["model"]) / SHARD
    if mutation == "restored_mtime_replacement":
        before = path.stat()
        replacement = tmp_path / "replacement"
        replacement.write_bytes(b"X" * before.st_size)
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
        replacement.replace(path)
    elif mutation == "same_bytes_replacement":
        replacement = tmp_path / "replacement"
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
    elif mutation == "marker_rewrite":
        marker = Path(settings["verified_manifest"])
        before = marker.stat()
        marker.write_bytes(marker.read_bytes())
        os.utime(marker, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    elif mutation == "delete":
        path.unlink()
    else:
        name = "model.safetensors" if mutation == "injected_weight" else "chat_template.jinja"
        (path.parent / name).write_bytes(b"unverified")

    def no_rehash(*args, **kwargs):
        pytest.fail("A changed checkpoint must fail, not silently rehash/rebaseline")

    monkeypatch.setattr(local_integrity, "_read_file", no_rehash)
    with pytest.raises(BackendConfigurationError):
        verifier.assert_unchanged()
    with pytest.raises(BackendConfigurationError):
        verifier.register(settings)


def test_marker_mutation_during_weight_verification_is_detected(tmp_path, monkeypatch):
    settings = checkpoint(tmp_path)
    original = local_integrity._read_file

    def mutate_marker(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if path.name == SHARD:
            marker = Path(settings["verified_manifest"])
            before = marker.stat()
            marker.write_bytes(marker.read_bytes())
            os.utime(marker, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
        return result

    monkeypatch.setattr(local_integrity, "_read_file", mutate_marker)
    with pytest.raises(BackendConfigurationError, match="changed after verification"):
        CheckpointIntegrity().register(settings)


def test_openclip_file_uses_same_hash_and_mutation_guards(tmp_path, monkeypatch):
    path = tmp_path / "open_clip_pytorch_model.bin"
    path.write_bytes(b"small CPU embedding checkpoint")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    verifier = CheckpointIntegrity()
    identity = verifier.register_file(str(path), expected, revision="pinned-revision")
    assert identity == {
        "verification": "verified", "sha256": expected, "content_sha256": expected,
        "bytes": path.stat().st_size, "revision": "pinned-revision",
    }
    with pytest.raises(BackendConfigurationError, match="hash mismatch"):
        CheckpointIntegrity().register_file(str(path), "0" * 64)

    def no_rehash(*args, **kwargs):
        pytest.fail("Repeated standalone registration must reuse verification")

    monkeypatch.setattr(local_integrity, "_read_file", no_rehash)
    assert verifier.register_file(str(path), expected, revision="pinned-revision") == identity
    before = path.stat()
    path.write_bytes(b"X" * before.st_size)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    with pytest.raises(BackendConfigurationError, match="changed after verification"):
        verifier.assert_unchanged()
