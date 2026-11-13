"""VideoTree's official query-time tree over an append-only streaming archive.

No questions enter observe(). Native VideoTree has no online eviction policy;
the entire feature/caption archive is charged and overflow is unsupported.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import tempfile

from baselines.streambench.accounting import measure_state
from baselines.streambench.method import StreamingMemoryMethod
from .egorag import jpeg_frame, native_module, validate_frames, wire_frame

UPSTREAM_REVISION = "605ec8e917f9a0f95d37a73a0b50bf0e8cfe076e"


class LocalArtifacts:
    """Content identity for local loader inputs, without loading any model.

    Hash once at construction; guard file inventory and stat anchors before
    reuse. This identifies actual bytes separately from the declared revision.
    """
    suffixes = {".json", ".bin", ".safetensors", ".pt", ".pth", ".h5", ".msgpack",
                ".gguf", ".model", ".txt", ".py", ".tiktoken", ".vocab", ".merges", ".jinja"}

    def __init__(self, checkpoint, revision, *, required=(), weights=False):
        self.root = Path(checkpoint).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Local checkpoint directory required: {self.root}")
        if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
            raise ValueError("Local artifacts require an immutable 40-character revision")
        inventory = self._inventory()
        if not set(required) <= set(inventory):
            raise FileNotFoundError(f"Missing required local artifacts: {sorted(set(required) - set(inventory))}")
        if weights and not any(name.endswith((".bin", ".safetensors")) for name in inventory):
            raise FileNotFoundError("No model weights in local checkpoint")
        files, self.anchors = {}, {}
        for name in inventory:
            path = self.root / name
            before = self._anchor(path)
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if before != self._anchor(path):
                raise ValueError("Local checkpoint changed while hashing")
            files[name] = {"bytes": before[2], "sha256": digest.hexdigest()}
            self.anchors[name] = before
        self.identity = {"checkpoint": str(self.root), "revision": revision, "files": files}
        self.assert_unchanged()

    def _inventory(self):
        return tuple(sorted(path.relative_to(self.root).as_posix() for path in self.root.rglob("*")
                            if path.is_file() and path.suffix.lower() in self.suffixes
                            and not {".git", ".cache", "__pycache__"}.intersection(path.relative_to(self.root).parts)))

    @staticmethod
    def _anchor(path):
        value = path.stat()
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns

    def assert_unchanged(self):
        if self._inventory() != tuple(self.anchors):
            raise ValueError("Local checkpoint artifact inventory changed after verification")
        for name, before in self.anchors.items():
            if before != self._anchor(self.root / name):
                raise ValueError("Local checkpoint changed after verification")


def build_tree(features, captions, score, *, initial_clusters=8, maximum_clusters=32,
               high_relevance_threshold=4, branch_width=4, seed=0):
    """Run official cosine kmeans, representative selection, and Ward expansion.

Repairs only ID alignment: scores correspond to temporal representatives, so
map them back to cluster labels before official depth expansion. Captions use
explicit zero-based IDs instead of upstream's erroneous index-minus-one.
"""
    import numpy as np
    import torch
    from baselines.streambench.vendor.videotree import kernels
    if features.ndim != 2 or len(features) != len(captions):
        raise ValueError("Expected one feature row per caption")
    if not len(features):
        return [], {"cluster_count": 0, "read_caption_indices": []}
    if any(type(x) is not int or x < 1 for x in (initial_clusters, maximum_clusters, high_relevance_threshold, branch_width)):
        raise ValueError("Tree parameters must be positive integers")
    if initial_clusters > maximum_clusters:
        raise ValueError("initial_clusters exceeds maximum_clusters")
    # No checkpoint-specific GPU placement; arithmetic remains native fp32.
    features = features.detach().cpu()
    count = min(initial_clusters, len(features))
    maximum = min(maximum_clusters, len(features))
    read = set()
    rng = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            while True:
                if count == 1:
                    # Upstream pairwise_cosine.squeeze() destroys the cluster
                    # axis for k=1. Its exact one-cluster solution is the mean.
                    labels = torch.zeros(len(features), dtype=torch.long)
                    centers = features.float().mean(dim=0, keepdim=True)
                else:
                    labels, centers = kernels.kmeans(features, count, distance="cosine", seed=seed,
                                                     device=torch.device("cpu"), tqdm_flag=False)
                groups = kernels.find_closest_points_per_cluster(features, labels, centers)
                ordered = sorted((indices[0], cluster) for cluster, indices in groups.items() if indices)
                read.update(index for index, _ in ordered)
                relevance = score([captions[index] for index, _ in ordered])
                if len(relevance) != len(ordered) or any(type(value) is not int or value not in (1, 2, 3) for value in relevance):
                    raise ValueError("VideoTree scorer must return one relevance integer 1..3 per representative")
                by_cluster = [1] * count
                for (_, cluster), value in zip(ordered, relevance):
                    by_cluster[cluster] = value
                if relevance.count(3) >= high_relevance_threshold or count >= maximum:
                    break
                count = min(count * 2, maximum)
            hierarchy = kernels.hierarchical_clustering_with_external_primary(
                features, labels.tolist(), by_cluster,
                num_subclusters=branch_width, num_subsubclusters=branch_width)
            # Upstream skips a high-relevance subcluster of size one. Such a
            # leaf is already resolved and must remain represented in the tree.
            for cluster, node in hierarchy.items():
                if by_cluster[cluster] == 3 and isinstance(node, dict):
                    represented = {index for child in node.values() for leaf in child.values() for index in leaf}
                    missing = [index for index in range(len(features)) if labels[index] == cluster and index not in represented]
                    next_id = max(node, default=-1) + 1
                    for index in missing:
                        node[next_id] = {0: [index]}
                        next_id += 1
            selected = sorted(set(kernels.find_closest_points_in_temporal_order_subsub(features, hierarchy, by_cluster)))
            read.update(selected)
            return selected, {"cluster_count": count, "cluster_labels": labels.tolist(),
                              "relevance_by_cluster": by_cluster, "hierarchy": hierarchy,
                              "read_caption_indices": sorted(read)}
    finally:
        np.random.set_state(rng)


class EVAEncoder:
    """Offline official EVA-CLIP encode_image, using its actual output dtype."""
    def __init__(self, checkpoint, processor_checkpoint, *, revision, processor_revision,
                 device="cuda:0", dtype="float16"):
        if dtype not in ("float16", "bfloat16", "float32"):
            raise ValueError("Unsupported encoder dtype")
        self.artifacts = LocalArtifacts(checkpoint, revision, weights=True,
                                       required=("config.json", "modeling_evaclip.py", "configuration_evaclip.py"))
        self.processor_artifacts = LocalArtifacts(processor_checkpoint, processor_revision,
                                                 required=("preprocessor_config.json",))
        if json.loads((self.artifacts.root / "config.json").read_text()).get("model_type") != "clip":
            raise ValueError("Encoder checkpoint must be the official EVA-CLIP implementation")
        self.settings = {"checkpoint": str(self.artifacts.root), "processor_checkpoint": str(self.processor_artifacts.root),
                         "revision": revision, "processor_revision": processor_revision,
                         "device": device, "dtype": dtype}
        self.identity = {"settings": self.settings, "encoder_artifacts": self.artifacts.identity,
                         "processor_artifacts": self.processor_artifacts.identity}
        self.model = self.processor = None

    def assert_unchanged(self):
        self.artifacts.assert_unchanged()
        self.processor_artifacts.assert_unchanged()

    def __call__(self, frames):
        import torch
        self.assert_unchanged()
        if self.model is None:
            from transformers import AutoModel, CLIPImageProcessor
            settings = self.settings
            self.processor = CLIPImageProcessor.from_pretrained(settings["processor_checkpoint"],
                                                                revision=settings["processor_revision"], local_files_only=True)
            self.model = AutoModel.from_pretrained(settings["checkpoint"], revision=settings["revision"],
                                                  torch_dtype=getattr(torch, settings["dtype"]),
                                                  trust_remote_code=True, local_files_only=True).to(settings["device"]).eval()
            if not hasattr(self.model, "encode_image") or self.model.config.model_type != "clip":
                raise ValueError("Encoder checkpoint must expose official EVA-CLIP encode_image")
        pixels = self.processor(images=frames, return_tensors="pt", padding=True).pixel_values
        with torch.inference_mode():
            features = self.model.encode_image(pixels.to(device=self.settings["device"], dtype=getattr(torch, self.settings["dtype"])))
        if features.ndim != 2 or features.shape[1] != self.model.config.projection_dim:
            raise ValueError("Unexpected official EVA feature shape")
        return features.detach().cpu().contiguous()


class VideoTreeLanguageBackend:
    """Real configured VLM/LLM calls through the existing audited backend.

The paper's LaViLa caption data is not shipped by VideoTree. A configured VLM
captions each observed sample; this substitution is always declared in output.
"""
    def __init__(self, config):
        if config.get("test_only"):
            raise ValueError("VideoTree production factory does not accept fixture inference")
        required = {"captioner", "relevance", "answerer"}
        if not required <= set(config.get("models", {})):
            raise ValueError("language_config needs captioner, relevance and answerer model roles")
        if "evidence" in config["models"]:
            raise ValueError("VideoTree reserves the native evidence role for relevance scoring")
        native_config = copy.deepcopy(config)
        native_config["models"]["evidence"] = native_config["models"].pop("relevance")
        self._temporary = tempfile.TemporaryDirectory(prefix="streambench-videotree-")
        backend = native_module("backends")
        self.models = backend.Models(native_config, Path(self._temporary.name) / "calls.jsonl")
        # Native verified manifests already hash all local loader inputs. If
        # one is absent, content-address the local checkpoint ourselves.
        local, self._artifacts, shared = {}, [], {}
        for role, settings in native_config["models"].items():
            if settings["backend"] != "hf_local":
                continue
            verified = self.models.cache_identity.get(role, {})
            if verified.get("verification") == "verified":
                local[role] = verified
            else:
                key = (settings["model"], settings["revision"])
                if key not in shared:
                    shared[key] = LocalArtifacts(*key, required=("config.json",), weights=True)
                    self._artifacts.append(shared[key])
                local[role] = shared[key].identity
        self.identity = {"config": copy.deepcopy(config), "native_role_map": {"relevance": "evidence"},
                         "native_cache_identity": copy.deepcopy(self.models.cache_identity), "local_artifacts": local,
                         "endpoint_sha256": {role: hashlib.sha256(endpoint.encode()).hexdigest()
                                             for role, (endpoint, _) in self.models._connections.items()},
                         "backend_code_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                                 for path in sorted(Path(backend.__file__).parent.glob("*.py"))}}

    def assert_unchanged(self):
        self.models.assert_checkpoints_unchanged()
        for artifacts in self._artifacts:
            artifacts.assert_unchanged()

    def __call__(self, frames, timestamps):
        self.assert_unchanged()
        return [self.models.generate("captioner", "Describe the observed first-person scene factually.",
                                      "Describe visible actions, objects, people and surroundings concisely. Do not infer unseen events.",
                                      frames=[wire_frame(jpeg_frame(frame, stamp))])
                for frame, stamp in zip(frames, timestamps)]

    def score(self, question, choices, captions):
        self.assert_unchanged()
        prompt = json.dumps({"question": question, "choices": choices, "captions": captions}, ensure_ascii=False)
        response = self.models.generate("evidence", "Score each caption's relevance to answering the question: 1 low, 2 medium, 3 high. Return only a JSON list of integers in caption order.", prompt)
        try:
            values = json.loads(response)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid VideoTree relevance JSON") from exc
        if not isinstance(values, list):
            raise ValueError("VideoTree relevance must be a JSON list")
        return values

    def answer(self, question, choices, captions):
        self.assert_unchanged()
        prompt = json.dumps({"question": question, "choices": choices, "captions_in_temporal_order": captions}, ensure_ascii=False)
        instruction = ("Answer using only the video captions. Return the single capital letter of the correct choice."
                       if choices else "Answer the question concisely using only the video captions.")
        return self.models.generate("answerer", instruction, prompt)


class VideoTreeMethod(StreamingMemoryMethod):
    def __init__(self, memory_budget_bytes, *, encoder, captioner, reasoner, test_only=False,
                 initial_clusters=8, maximum_clusters=32, high_relevance_threshold=4,
                 branch_width=4, seed=0, identity=None):
        super().__init__(memory_budget_bytes)
        self.encoder, self.captioner, self.reasoner = encoder, captioner, reasoner
        self.test_only = test_only
        self.parameters = {"initial_clusters": initial_clusters, "maximum_clusters": maximum_clusters,
                           "high_relevance_threshold": high_relevance_threshold, "branch_width": branch_width, "seed": seed}
        self.identity = copy.deepcopy(identity or {"test_only": test_only})
        if not test_only and not identity:
            raise ValueError("Production VideoTree requires explicit checkpoint/backend identity")
        self._features = None
        self._captions, self._timestamps = [], []

    def initialize(self):
        import torch
        self._check_backends()
        self._features = torch.empty((0, 0))
        self._captions, self._timestamps = [], []
        self.minimum_memory_bytes = self.memory_bytes()["persistent_total_bytes"]
        self.enforce_budget()

    def _check_backends(self):
        for backend in (self.encoder, self.captioner, self.reasoner):
            check = getattr(backend, "assert_unchanged", None)
            if check is not None:
                check()

    def observe(self, frames, timestamps):
        import torch
        self._check_backends()
        if self._features is None:
            raise RuntimeError("initialize or load_checkpoint before using VideoTree")
        timestamps = validate_frames(frames, timestamps, self._timestamps[-1] if self._timestamps else None)
        features = self.encoder(frames)
        if not isinstance(features, torch.Tensor) or features.ndim != 2 or len(features) != len(frames) or not torch.isfinite(features).all():
            raise ValueError("Encoder must return finite [frames, feature_dimension] tensors")
        captions = self.captioner(frames, timestamps)
        if len(captions) != len(frames) or any(not isinstance(caption, str) or not caption.strip() for caption in captions):
            raise ValueError("Captioner must return nonempty text for every observed frame")
        features = features.detach().cpu().contiguous()
        if len(self._features):
            if features.shape[1:] != self._features.shape[1:] or features.dtype != self._features.dtype:
                raise ValueError("Encoder feature shape/dtype changed within stream")
            features = torch.cat([self._features, features])
        self._features = features
        self._captions.extend(captions)
        self._timestamps.extend(copy.deepcopy(timestamps))

    def finalize_chunk(self):
        return self.enforce_budget()

    def _metadata(self):
        return {"parameters": self.parameters, "identity": self.identity, "timestamps": self._timestamps,
                "upstream_revision": UPSTREAM_REVISION, "test_only": self.test_only}

    def persistent_state(self):
        return {"visual_memory_bytes": self._features, "text_memory_bytes": self._captions,
                "metadata_bytes": self._metadata()}

    def state_dict(self):
        self._check_backends()
        return {"schema_version": 1, **copy.deepcopy(self._metadata()), "features": self._features,
                "captions": list(self._captions)}

    def load_state_dict(self, state):
        import torch
        self._check_backends()
        if state.get("schema_version") != 1 or state["parameters"] != self.parameters or state["identity"] != self.identity or state["upstream_revision"] != UPSTREAM_REVISION or state["test_only"] != self.test_only:
            raise ValueError("VideoTree checkpoint model/config/schema mismatch")
        features = state["features"]
        if not isinstance(features, torch.Tensor) or features.ndim != 2 or len(features) != len(state["captions"]) or len(features) != len(state["timestamps"]):
            raise ValueError("Malformed VideoTree checkpoint")
        self._features = features.detach().cpu().clone()
        self._captions, self._timestamps = copy.deepcopy(state["captions"]), copy.deepcopy(state["timestamps"])

    def answer(self, question, choices=None):
        self._check_backends()
        if self._features is None:
            raise RuntimeError("initialize or load_checkpoint before using VideoTree")
        if not len(self._features):
            return {"answer": "", "status": "no_observed_evidence", "query_read_bytes": 0, "num_retrieved_items": 0}
        selected, tree = build_tree(self._features, self._captions,
                                    lambda captions: self.reasoner.score(question, choices or [], captions), **self.parameters)
        answer = self.reasoner.answer(question, choices or [], [self._captions[index] for index in selected])
        read = measure_state({"visual_memory_bytes": self._features,
                              "text_memory_bytes": [self._captions[i] for i in tree["read_caption_indices"]],
                              "metadata_bytes": [self._timestamps[i] for i in selected]})["persistent_total_bytes"]
        return {"answer": answer, "status": "complete", "query_read_bytes": read,
                "num_retrieved_items": len(selected), "num_text_items_used": len(selected),
                "selected_frame_ids": [self._timestamps[i]["global_frame_id"] for i in selected],
                "retrieved_items": [self._timestamps[i] for i in selected], "tree": tree,
                "test_only": self.test_only, "captioner_deviation": self.identity.get("captioner_deviation")}


def create_method(config):
    encoder_config = config.get("encoder")
    language_config = config.get("language_config")
    if not isinstance(encoder_config, dict) or not isinstance(language_config, dict):
        raise ValueError("VideoTree requires encoder checkpoint config and language_config")
    encoder = EVAEncoder(**encoder_config)
    language = VideoTreeLanguageBackend(language_config)
    identity = {"encoder": encoder.identity, "language": language.identity,
                "captioner_deviation": "Configured per-sample VLM captions replace upstream precomputed LaViLa captions"}
    return VideoTreeMethod(config["memory_budget_bytes"], encoder=encoder, captioner=language, reasoner=language,
                           identity=identity, **{key: config[key] for key in ("initial_clusters", "maximum_clusters", "high_relevance_threshold", "branch_width", "seed") if key in config})
