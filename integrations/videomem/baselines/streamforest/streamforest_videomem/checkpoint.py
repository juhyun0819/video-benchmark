"""Create and verify a local SHA256 inventory for the released StreamForest model.

This module uses only the standard library and never downloads weights. The
fixed repository/revision identify the intended checkpoint; a locally generated
inventory does not authenticate that the files came from that remote revision.
Fetch the pinned snapshot separately, into a materialized local directory.

API:
  create_verification_record(model_path, output_path) -> inventory dictionary
  verify_checkpoint(model_path, verification_path) -> verified provenance/hashes

CLI:
  python -m streamforest_videomem.checkpoint --model-path /local/model \
      --output /local/verification.json
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


MODEL_ID = "MCG-NJU/StreamForest-Qwen2-7B"
REVISION = "6840f91e17346b4c406ba03b57d1a63582765c91"
UPSTREAM_COMMIT = "f9d14670ff451ff5865638e17c9958e99bf2fec5"
SCHEMA_VERSION = "streamforest_checkpoint_inventory_v1"
VERIFICATION_KIND = "local_sha256_inventory"
_INDEX = "model.safetensors.index.json"
_REQUIRED = {
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "vocab.json", "merges.txt", _INDEX,
}
_OPTIONAL = {
    "special_tokens_map.json", "added_tokens.json", "chat_template.json", "chat_template.jinja",
    "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json",
    "tokenizer.model", "sentencepiece.bpe.model", "spiece.model",
}


def _model_root(model_path) -> Path:
    root = Path(model_path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("model_path must be a local directory")
    return root


def _relative_name(value) -> str:
    if (not isinstance(value, str) or not value or "\\" in value
            or any(ord(char) < 32 for char in value)):
        raise ValueError("checkpoint paths must be nonempty canonical relative POSIX paths")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in (".", "..") for part in path.parts) or str(path) != value:
        raise ValueError("checkpoint paths must not be absolute or contain traversal")
    return value


def _checked_path(root: Path, relative: str) -> Path:
    relative = _relative_name(relative)
    resolved = (root / relative).resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("checkpoint symlink or path escapes the model directory: " + relative)
    if not resolved.is_file():
        raise ValueError("checkpoint artifact is not a regular file: " + relative)
    return resolved


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON artifact: {path}") from exc


def _looks_like_artifact(name: str) -> bool:
    basename = PurePosixPath(name).name
    return (name in _REQUIRED or name in _OPTIONAL
            or basename.endswith(".safetensors")
            or (basename.startswith("pytorch_model") and basename.endswith((".bin", ".json")))
            or (basename.startswith(("tokenizer", "vocab")) and basename.endswith(".json"))
            or (basename.startswith("merges") and basename.endswith(".txt"))
            or (basename.startswith("chat_template") and basename.endswith((".json", ".jinja"))))


def _artifact_names(root: Path) -> set[str]:
    # The native slow Qwen2 tokenizer uses vocab.json and merges.txt. Record the
    # fast tokenizer too, as well as optional files Transformers may consult.
    names = set(_REQUIRED)
    config = _read_json(_checked_path(root, "config.json"))
    # The published release serializes model_type='qwen2'; the native local
    # LlavaQwenConfig class uses 'llava_qwen'. Both describe this architecture.
    if (not isinstance(config, dict) or config.get("model_type") not in {"qwen2", "llava_qwen"}
            or config.get("architectures") != ["LlavaQwenForCausalLM"]):
        raise ValueError("Expected the released native LlavaQwenForCausalLM architecture")
    index = _read_json(_checked_path(root, _INDEX))
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if (not isinstance(weight_map, dict) or not weight_map
            or any(not isinstance(key, str) or not key for key in weight_map)):
        raise ValueError("checkpoint index must contain a nonempty weight_map")
    if not any(key.startswith("model.vision_tower.vision_tower.") for key in weight_map):
        raise ValueError("checkpoint index lacks embedded SigLip vision weights")
    for shard in weight_map.values():
        shard = _relative_name(shard)
        if not shard.endswith(".safetensors"):
            raise ValueError("the released index must reference safetensors shards")
        names.add(shard)
    # In addition to indexed shards, detect alternative local weights and
    # tokenizer files so adding an unrecorded live artifact fails verification.
    for path in root.iterdir():
        if _looks_like_artifact(path.name):
            names.add(path.name)
    template_directory = root / "chat_templates"
    if template_directory.exists():
        resolved = template_directory.resolve(strict=True)
        if not resolved.is_relative_to(root) or not resolved.is_dir():
            raise ValueError("chat_templates must be a directory contained in the model root")
        for path in resolved.iterdir():
            if path.suffix == ".jinja":
                names.add("chat_templates/" + path.name)
    for name in names:
        _checked_path(root, name)
    return names


def _file_state(path: Path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _inventory_entry(root: Path, name: str) -> dict:
    path = _checked_path(root, name)
    before = _file_state(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if _checked_path(root, name) != path or _file_state(path) != before:
        raise RuntimeError("checkpoint artifact changed during hashing: " + name)
    if before[2] == 0:
        raise ValueError("checkpoint artifact is empty: " + name)
    return {"path": name, "bytes": before[2], "hash": digest.hexdigest()}


def _artifact_digest(entries):
    serialized = json.dumps(sorted(entries, key=lambda entry: entry["path"]),
                            sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


def create_verification_record(model_path, output_path) -> dict:
    """Inventory local files without asserting remote origin or overwriting output."""
    root = _model_root(model_path)
    output = Path(output_path).expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError("verification output already exists: " + str(output))
    resolved_output = output.resolve()
    if resolved_output.is_relative_to(root) and _looks_like_artifact(resolved_output.relative_to(root).as_posix()):
        raise ValueError("verification output must not use a model/tokenizer artifact name")
    names = _artifact_names(root)
    entries = [_inventory_entry(root, name) for name in sorted(names)]
    if _artifact_names(root) != names:
        raise RuntimeError("checkpoint artifact set changed during inventory")
    record = {
        "schema_version": SCHEMA_VERSION, "verification_kind": VERIFICATION_KIND,
        "repo_id": MODEL_ID, "revision": REVISION, "source_revision": UPSTREAM_COMMIT,
        "remote_revision_verified": False,
        "identity_note": "Repository/revision are declared targets; these local SHA256 hashes do not authenticate remote origin.",
        "hash_algorithm": "sha256", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "verified_files": entries, "artifact_set_sha256": _artifact_digest(entries),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    return record


def verify_checkpoint(model_path, verification_path) -> dict:
    """Check exact local inventory coverage/bytes/SHA256 and return provenance.

    The result confirms local integrity relative to this record. It explicitly
    preserves ``remote_revision_verified=False``; no network evidence is added.
    """
    root = _model_root(model_path)
    verification = Path(verification_path).expanduser().resolve(strict=True)
    record_bytes = verification.read_bytes()
    try:
        record = json.loads(record_bytes)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("verification record is not valid JSON") from exc
    if (not isinstance(record, dict) or record.get("schema_version") != SCHEMA_VERSION
            or record.get("verification_kind") != VERIFICATION_KIND
            or record.get("hash_algorithm") != "sha256"
            or record.get("remote_revision_verified") is not False):
        raise ValueError("Expected a local SHA256 inventory without an asserted remote verification")
    if (record.get("repo_id") != MODEL_ID or record.get("revision") != REVISION
            or record.get("source_revision") != UPSTREAM_COMMIT):
        raise ValueError("checkpoint inventory identity is not the pinned official StreamForest release")
    entries = record.get("verified_files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("verified_files must be a nonempty list")
    expected = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "hash"}:
            raise ValueError("each inventory entry must contain path, bytes and hash")
        name = _relative_name(entry["path"])
        if name in expected:
            raise ValueError("duplicate checkpoint inventory path: " + name)
        if type(entry["bytes"]) is not int or entry["bytes"] < 1:
            raise ValueError("inventory byte counts must be positive integers")
        if not isinstance(entry["hash"], str) or re.fullmatch(r"[0-9a-f]{64}", entry["hash"]) is None:
            raise ValueError("inventory hashes must be lowercase SHA256 digests")
        _checked_path(root, name)
        expected[name] = entry
    required = _artifact_names(root)
    if required != set(expected):
        missing = sorted(required - set(expected))
        extra = sorted(set(expected) - required)
        raise ValueError(f"inventory does not match live inference artifacts; missing={missing}, extra={extra}")
    if record.get("artifact_set_sha256") != _artifact_digest(entries):
        raise ValueError("inventory artifact-set digest is inconsistent")
    for name in sorted(expected):
        observed = _inventory_entry(root, name)
        if observed["bytes"] != expected[name]["bytes"]:
            raise RuntimeError("checkpoint byte count changed: " + name)
        if observed["hash"] != expected[name]["hash"]:
            raise RuntimeError("checkpoint SHA256 changed: " + name)
    if _artifact_names(root) != required:
        raise RuntimeError("checkpoint artifact set changed during verification")
    return {**record, "model_path": str(root), "verification_path": str(verification),
            "record_sha256": hashlib.sha256(record_bytes).hexdigest(),
            "verification_status": "local_bytes_match_inventory"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Materialized local pinned model directory")
    parser.add_argument("--output", required=True, help="New local SHA256 inventory JSON path")
    args = parser.parse_args(argv)
    record = create_verification_record(args.model_path, args.output)
    print(json.dumps({"output": str(Path(args.output).expanduser()),
                      "verified_files": len(record["verified_files"]),
                      "artifact_set_sha256": record["artifact_set_sha256"],
                      "remote_revision_verified": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
