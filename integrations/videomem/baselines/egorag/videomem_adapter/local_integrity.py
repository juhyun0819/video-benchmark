"""CPU-only verification and mutation guards for local model checkpoints.

The downloader's marker is evidence to check, never proof by itself. Every
listed file is hashed before an identity is returned. Later cache/model use
must call ``assert_unchanged``; replacing even identical bytes requires a new
verifier, while a newly verified copy has the same content identity.

The later guard checks filesystem stats rather than rehashing large weights.
An in-place rewrite preserving every recorded stat, including timestamps within
one filesystem clock tick, is outside that guard's detection capability.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat

from .backends import BackendConfigurationError


_REPO = "Qwen/Qwen3-VL-8B-Instruct"
_CHUNK_BYTES = 8 * 1024 * 1024
_LOADABLE_SUFFIXES = {
    ".json", ".safetensors", ".bin", ".pt", ".pth", ".h5", ".msgpack",
    ".gguf", ".model", ".tiktoken", ".vocab", ".merges", ".jinja",
}


def _fail(message):
    raise BackendConfigurationError("Local checkpoint integrity: " + message)


def _anchor(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _file_anchor(path):
    try:
        value = path.lstat()
    except OSError:
        _fail("missing or unreadable file: %s" % path)
    if not stat.S_ISREG(value.st_mode):
        _fail("file must be regular and cannot be a symlink: %s" % path)
    return _anchor(value)


def _absolute_path(value, name):
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        _fail("%s must be an absolute path" % name)
    # Preserve the requested path so later checks follow the same path as the
    # model loader, including changes to any ancestor directory.
    return Path(value)


def _relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        _fail("unsafe manifest path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        _fail("unsafe manifest path: %s" % value)
    return path


def _model_file(root, relative):
    path = root
    for part in _relative_path(relative).parts:
        path = path / part
        if path.is_symlink():
            _fail("symlink in manifest path: %s" % relative)
    return path


def _json(data, description):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        result = json.loads(data, object_pairs_hook=unique_pairs)
    except (ValueError, UnicodeError):
        _fail("invalid JSON in %s" % description)
    if not isinstance(result, dict):
        _fail("%s must contain a JSON object" % description)
    return result


def _read_file(path, expected_size=None, source_hash=None, capture=False):
    """Hash one stream while checking both its descriptor and named inode."""
    before = _file_anchor(path)
    if expected_size is not None and before[2] != expected_size:
        _fail("byte count mismatch: %s" % path)
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1() if source_hash is not None and len(source_hash) == 40 else None
    if git_blob is not None:
        git_blob.update(b"blob " + str(before[2]).encode("ascii") + b"\0")
    chunks = [] if capture else None
    size = 0
    try:
        with path.open("rb") as stream:
            if _anchor(os.fstat(stream.fileno())) != before:
                _fail("file changed while opening: %s" % path)
            for chunk in iter(lambda: stream.read(_CHUNK_BYTES), b""):
                size += len(chunk)
                sha256.update(chunk)
                if git_blob is not None:
                    git_blob.update(chunk)
                if chunks is not None:
                    chunks.append(chunk)
            if _anchor(os.fstat(stream.fileno())) != before:
                _fail("file changed while hashing: %s" % path)
    except OSError:
        _fail("could not read file: %s" % path)
    if _file_anchor(path) != before or size != before[2]:
        _fail("file changed while hashing: %s" % path)
    digest = sha256.hexdigest()
    actual_source = git_blob.hexdigest() if git_blob is not None else digest
    if source_hash is not None and source_hash != actual_source:
        _fail("hash mismatch: %s" % path)
    return digest, before, b"".join(chunks) if chunks is not None else None


def _inventory(root, marker):
    """Find files that can affect local weights, tokenization or processing."""
    if not root.is_dir() or root.is_symlink():
        _fail("model directory is missing or a symlink: %s" % root)
    names = []

    def walk_error(error):
        _fail("cannot inspect model directory: %s" % error.filename)

    for directory, subdirectories, files in os.walk(root, followlinks=False, onerror=walk_error):
        directory = Path(directory)
        for name in subdirectories:
            if (directory / name).is_symlink():
                _fail("symlink directory in model checkpoint: %s" % (directory / name))
        for name in files:
            path = directory / name
            if path == marker:
                continue
            relevant = path.suffix.lower() in _LOADABLE_SUFFIXES or name in {"merges.txt", "vocab.txt"}
            if relevant:
                _file_anchor(path)
                names.append(path.relative_to(root).as_posix())
    return tuple(sorted(names))


class CheckpointIntegrity:
    """Share one verification across roles and reject subsequent mutations."""

    def __init__(self):
        self._registered = {}

    @staticmethod
    def _check(record):
        for path, expected in record["anchors"].items():
            if _file_anchor(path) != expected:
                _fail("file changed after verification: %s" % path)
        if record.get("root") is not None:
            if _inventory(record["root"], record["marker"]) != record["inventory"]:
                _fail("model file inventory changed after verification")

    def assert_unchanged(self):
        for record in self._registered.values():
            self._check(record)

    def register(self, settings):
        if "verified_manifest" not in settings:
            return {"verification": "not_configured"}
        root = _absolute_path(settings.get("model"), "model")
        marker = _absolute_path(settings["verified_manifest"], "verified_manifest")
        revision = settings.get("revision")
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            _fail("revision must be an immutable 40-character commit")
        self.assert_unchanged()
        key = ("manifest", str(root), revision, str(marker))
        if key in self._registered:
            return copy.deepcopy(self._registered[key]["identity"])

        manifest_sha256, marker_anchor, marker_bytes = _read_file(marker, capture=True)
        manifest = _json(marker_bytes, "verification manifest")
        if manifest.get("repo_id") != _REPO or manifest.get("revision") != revision:
            _fail("verification manifest repository or revision mismatch")
        files = manifest.get("verified_files")
        if not isinstance(files, list) or not files:
            _fail("verified_files must be a nonempty list")
        if type(manifest.get("file_count")) is not int or manifest["file_count"] != len(files):
            _fail("verification manifest file_count mismatch")
        entries = {}
        for item in files:
            if not isinstance(item, dict):
                _fail("invalid verified_files entry")
            name = _relative_path(item.get("path")).as_posix()
            if name in entries:
                _fail("duplicate manifest file: %s" % name)
            if type(item.get("bytes")) is not int or item["bytes"] < 0:
                _fail("invalid manifest byte count: %s" % name)
            if not isinstance(item.get("hash"), str) or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", item["hash"]):
                _fail("invalid manifest hash: %s" % name)
            entries[name] = item
        if not {"config.json", "model.safetensors.index.json"} <= entries.keys():
            _fail("manifest must cover config.json and model.safetensors.index.json")
        total_bytes = sum(item["bytes"] for item in files)
        if "total_bytes" in manifest and (type(manifest["total_bytes"]) is not int or manifest["total_bytes"] != total_bytes):
            _fail("verification manifest total_bytes mismatch")
        inventory = _inventory(root, marker)
        missing = set(inventory) - entries.keys()
        if missing:
            _fail("unverified model files: %s" % ", ".join(sorted(missing)))

        anchors = {marker: marker_anchor}
        digests = []
        index = None
        for name, item in sorted(entries.items()):
            path = _model_file(root, name)
            digest, anchor, data = _read_file(
                path, item["bytes"], item["hash"], capture=name == "model.safetensors.index.json",
            )
            anchors[path] = anchor
            digests.append({"path": name, "bytes": item["bytes"], "sha256": digest})
            if data is not None:
                index = _json(data, "model.safetensors.index.json")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            _fail("model index requires a nonempty weight_map")
        shards = set()
        for tensor, name in weight_map.items():
            if not isinstance(tensor, str) or not tensor:
                _fail("invalid tensor name in model index")
            name = _relative_path(name).as_posix()
            if name not in entries or not name.endswith(".safetensors"):
                _fail("model index references an unverified weight shard: %s" % name)
            shards.add(name)
        for field, expected in (("indexed_tensors", len(weight_map)), ("shard_count", len(shards))):
            if field in manifest and (type(manifest[field]) is not int or manifest[field] != expected):
                _fail("verification manifest %s mismatch" % field)

        identity = {
            "verification": "verified", "repo_id": _REPO, "revision": revision,
            "manifest_sha256": manifest_sha256, "files": digests,
            "content_sha256": hashlib.sha256(json.dumps(digests, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
        }
        record = {"identity": identity, "anchors": anchors, "root": root, "marker": marker, "inventory": inventory}
        self._check(record)
        self._registered[key] = record
        return copy.deepcopy(identity)

    def register_file(self, path, expected_sha256, revision=None):
        """Verify a pinned standalone checkpoint, such as OpenCLIP weights."""
        path = _absolute_path(path, "checkpoint")
        if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            _fail("checkpoint sha256 must be a 64-character digest")
        if revision is not None and not isinstance(revision, str):
            _fail("checkpoint revision must be a string")
        self.assert_unchanged()
        key = ("file", str(path), expected_sha256, revision)
        if key in self._registered:
            return copy.deepcopy(self._registered[key]["identity"])
        digest, anchor, _ = _read_file(path, source_hash=expected_sha256)
        identity = {"verification": "verified", "sha256": digest, "content_sha256": digest, "bytes": anchor[2]}
        if revision is not None:
            identity["revision"] = revision
        record = {"identity": identity, "anchors": {path: anchor}}
        self._check(record)
        self._registered[key] = record
        return copy.deepcopy(identity)
