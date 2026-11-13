"""AKS (Tang et al., CVPR 2025), official CLIP-scored variant.

Question-free ingestion retains image features and lossless sampled frames.
Native adaptive sampling controls QA reads, not storage: no eviction is invented.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
from pathlib import Path
import re

from baselines.streambench.accounting import measure_state
from baselines.streambench.method import StreamingMemoryMethod
from .egorag import validate_frames

UPSTREAM_REVISION = "b0b8a58fedf1d05d78151e2969cecde60e83721d"


def adaptive_select(scores, *, max_num_frames=64, ratio=1, t1=0.8, t2=-100., all_depth=5):
    """Independent implementation of upstream frame_select.py's arithmetic.

Normalize scores, recursively bisect low-contrast intervals, allocate
floor(N/2**depth) frames per leaf, and select highest-scoring frames there.
Native tie order and underfilled quotas are retained. Constant scores use zero
normalized values instead of the upstream undefined division by zero.
"""
    import numpy as np
    if type(max_num_frames) is not int or max_num_frames < 1 or type(ratio) is not int or ratio < 1:
        raise ValueError("max_num_frames and ratio must be positive integers")
    if type(all_depth) is not int or all_depth < 0 or all_depth > int(math.log2(max_num_frames)):
        raise ValueError("all_depth must not produce zero frame quotas at any depth")
    if not math.isfinite(t1) or not math.isfinite(t2):
        raise ValueError("Selection thresholds must be finite")
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("AKS scores must be a finite vector")
    # Match upstream's integer ratio truncation, including its final remainder.
    ids = np.arange(len(values) // ratio, dtype=np.int64) * ratio
    values = values[ids]
    if len(values) < max_num_frames:
        return ids.tolist()
    span = float(values.max() - values.min())
    values = (values - values.min()) / span if span else np.zeros_like(values)
    chosen = []

    def visit(begin, end, depth):
        section = values[begin:end]
        order = sorted(range(len(section)), key=lambda index: (-float(section[index]), index))
        contrast = float(np.mean(section[order[:max_num_frames]]) - np.mean(section))
        keep_leaf = contrast > t1 and float(np.std(section)) > t2
        if keep_leaf or depth >= all_depth:
            chosen.extend(int(ids[begin + index]) for index in order[:max_num_frames // 2**depth])
        else:
            middle = begin + len(section) // 2
            visit(begin, middle, depth + 1)
            visit(middle, end, depth + 1)

    visit(0, len(values), 0)
    return sorted(chosen)


def _checkpoint_identity(checkpoint, revision, expected_type):
    """Hash existing local inputs once; no download or model initialization."""
    root = Path(checkpoint).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Local checkpoint directory required: {root}")
    if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError("Local checkpoint requires an immutable 40-character revision")
    model_config = json.loads((root / "config.json").read_text())
    if model_config.get("model_type") != expected_type:
        raise ValueError(f"Expected {expected_type} checkpoint")
    files, anchors = {}, {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".bin", ".safetensors", ".model", ".txt", ".py", ".tiktoken", ".jinja"}:
            continue
        initial = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        final = path.stat()
        anchor = (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns, final.st_ctime_ns)
        if anchor != (initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns, initial.st_ctime_ns):
            raise ValueError("Checkpoint changed while hashing")
        name = str(path.relative_to(root))
        files[name] = {"bytes": final.st_size, "sha256": digest.hexdigest()}
        anchors[path] = anchor
    if not any(name.endswith((".bin", ".safetensors")) for name in files):
        raise FileNotFoundError("No model weights in local checkpoint")
    return {"checkpoint": str(root), "revision": revision, "model_type": expected_type, "files": files}, anchors


def _check_anchors(anchors):
    for path, before in anchors.items():
        stat = path.stat()
        if before != (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns):
            raise ValueError("Local checkpoint changed after verification")


class CLIPScorer:
    """Official AKS CLIP image/text feature calls and cosine score."""
    def __init__(self, checkpoint, *, revision, device="cuda:0", dtype="float32"):
        if dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("Unsupported CLIP dtype")
        self.identity, self._anchors = _checkpoint_identity(checkpoint, revision, "clip")
        self.identity.update(device=device, dtype=dtype, reference_model="openai/clip-vit-base-patch32")
        self.model = self.processor = None

    def _load(self):
        import torch
        _check_anchors(self._anchors)
        if self.model is None:
            from transformers import CLIPModel, CLIPProcessor
            common = {"revision": self.identity["revision"], "local_files_only": True}
            self.model = CLIPModel.from_pretrained(self.identity["checkpoint"],
                                                   torch_dtype=getattr(torch, self.identity["dtype"]), **common)
            self.model.to(self.identity["device"]).eval()
            self.processor = CLIPProcessor.from_pretrained(self.identity["checkpoint"], **common)

    def encode_images(self, frames):
        import torch
        self._load()
        inputs = self.processor(images=frames, return_tensors="pt", padding=True)
        pixels = inputs["pixel_values"].to(device=self.identity["device"], dtype=getattr(torch, self.identity["dtype"]))
        with torch.inference_mode():
            features = self.model.get_image_features(pixel_values=pixels)
        return features.detach().cpu().contiguous()

    def score(self, features, question):
        import torch
        self._load()
        # Native CLIP branch scores the question alone; choices are not encoded.
        inputs = self.processor(text=question, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(self.identity["device"]) for key, value in inputs.items()}
        with torch.inference_mode():
            text = self.model.get_text_features(**inputs).detach().cpu()
            result = torch.nn.functional.cosine_similarity(features, text, dim=-1)
        return result.tolist()


class Qwen2VLAnswerer:
    """Real local Qwen2-VL, an answer backbone supported by official AKS."""
    def __init__(self, checkpoint, *, revision, device="cuda:0", dtype="bfloat16",
                 max_new_tokens=128, max_pixels=12845056):
        if dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("Unsupported Qwen2-VL dtype")
        if type(max_new_tokens) is not int or max_new_tokens <= 0 or type(max_pixels) is not int or max_pixels <= 0:
            raise ValueError("Qwen2-VL max_new_tokens/max_pixels must be positive")
        self.identity, self._anchors = _checkpoint_identity(checkpoint, revision, "qwen2_vl")
        self.identity.update(device=device, dtype=dtype, max_new_tokens=max_new_tokens, max_pixels=max_pixels,
                             reference_model="Qwen/Qwen2-VL-7B-Instruct")
        self.model = self.processor = None

    def _load(self):
        import torch
        _check_anchors(self._anchors)
        if self.model is None:
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
            common = {"revision": self.identity["revision"], "local_files_only": True}
            self.processor = AutoProcessor.from_pretrained(self.identity["checkpoint"], max_pixels=self.identity["max_pixels"], **common)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.identity["checkpoint"], torch_dtype=getattr(torch, self.identity["dtype"]),
                device_map={"": self.identity["device"]}, **common).eval()

    def answer(self, frames, question, choices, timestamps):
        import numpy as np
        import torch
        self._load()
        prompt = question
        if choices:
            prompt += "\n" + "\n".join(f"{chr(65 + index)}. {choice}" for index, choice in enumerate(choices))
            prompt += "\nAnswer with only the capital letter of the correct choice."
        messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Frames have already been selected. The processor may resize/pad for
        # temporal patches, but must not decode or sample another source video.
        video = [np.asarray(frame.convert("RGB")) for frame in frames]
        inputs = self.processor(text=[text], videos=[video], padding=True, return_tensors="pt")
        inputs = inputs.to(self.identity["device"])
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, do_sample=False, num_beams=1,
                                           max_new_tokens=self.identity["max_new_tokens"])
        tail = outputs[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(tail, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


class AKSMethod(StreamingMemoryMethod):
    def __init__(self, memory_budget_bytes, *, scorer, answerer, test_only=False, identity=None,
                 max_num_frames=64, ratio=1, t1=0.8, t2=-100., all_depth=5):
        super().__init__(memory_budget_bytes)
        self.scorer, self.answerer = scorer, answerer
        self.test_only = test_only
        self.identity = copy.deepcopy(identity or {"test_only": test_only})
        if not test_only and not identity:
            raise ValueError("Production AKS requires explicit checkpoint identity")
        self.parameters = {"max_num_frames": max_num_frames, "ratio": ratio, "t1": t1, "t2": t2, "all_depth": all_depth}
        adaptive_select([], **self.parameters)
        self._features = None
        self._frames, self._timestamps = [], []

    def initialize(self):
        import torch
        self._features = torch.empty((0, 0))
        self._frames, self._timestamps = [], []
        self.minimum_memory_bytes = self.memory_bytes()["persistent_total_bytes"]
        self.enforce_budget()

    def observe(self, frames, timestamps):
        import torch
        from PIL import Image
        if self._features is None:
            raise RuntimeError("initialize or load_checkpoint before using AKS")
        timestamps = validate_frames(frames, timestamps, self._timestamps[-1] if self._timestamps else None)
        if any(not isinstance(frame, Image.Image) for frame in frames):
            raise TypeError("AKS requires common PIL frames")
        features = self.scorer.encode_images(frames)
        if not isinstance(features, torch.Tensor) or features.ndim != 2 or len(features) != len(frames) or not torch.isfinite(features).all():
            raise ValueError("AKS encoder must return finite [frames,feature_dimension] tensors")
        features = features.detach().cpu().contiguous()
        if len(self._features):
            if features.shape[1:] != self._features.shape[1:] or features.dtype != self._features.dtype:
                raise ValueError("AKS feature shape/dtype changed")
            features = torch.cat([self._features, features])
        retained = []
        for frame in frames:
            output = io.BytesIO()
            frame.convert("RGB").save(output, format="PNG")
            retained.append(output.getvalue())
        self._features = features
        self._frames.extend(retained)
        self._timestamps.extend(copy.deepcopy(timestamps))

    def finalize_chunk(self):
        return self.enforce_budget()

    def _metadata(self):
        return {"parameters": self.parameters, "identity": self.identity, "timestamps": self._timestamps,
                "upstream_revision": UPSTREAM_REVISION, "variant": "official_clip_scoring_qwen2_vl_answer",
                "test_only": self.test_only}

    def persistent_state(self):
        return {"visual_memory_bytes": {"clip_features": self._features, "sampled_pngs": self._frames},
                "metadata_bytes": self._metadata()}

    def state_dict(self):
        return {"schema_version": 1, **copy.deepcopy(self._metadata()), "features": self._features,
                "frames_png": list(self._frames)}

    def load_state_dict(self, state):
        import torch
        if state.get("schema_version") != 1 or state["parameters"] != self.parameters or state["identity"] != self.identity or state["upstream_revision"] != UPSTREAM_REVISION or state["test_only"] != self.test_only:
            raise ValueError("AKS checkpoint config/model/schema mismatch")
        features = state["features"]
        if not isinstance(features, torch.Tensor) or features.ndim != 2 or len(features) != len(state["frames_png"]) or len(features) != len(state["timestamps"]):
            raise ValueError("Malformed AKS checkpoint")
        self._features = features.detach().cpu().clone()
        self._frames, self._timestamps = copy.deepcopy(state["frames_png"]), copy.deepcopy(state["timestamps"])

    def answer(self, question, choices=None):
        from PIL import Image
        if self._features is None:
            raise RuntimeError("initialize or load_checkpoint before using AKS")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("A nonempty question is required")
        if not len(self._features):
            return {"answer": None, "status": "no_observed_evidence", "query_read_bytes": 0, "num_retrieved_items": 0}
        scores = self.scorer.score(self._features, question)
        if len(scores) != len(self._features):
            raise ValueError("AKS must score every retained frame")
        selected = adaptive_select(scores, **self.parameters)
        if not selected:
            raise ValueError("AKS selection has no frames under the configured ratio")
        frames = []
        for index in selected:
            with Image.open(io.BytesIO(self._frames[index])) as image:
                frames.append(image.convert("RGB"))
        answer = self.answerer.answer(frames, question, list(choices or []), [self._timestamps[i] for i in selected])
        reads = measure_state({"visual_memory_bytes": [self._features, [self._frames[i] for i in selected]],
                               "metadata_bytes": [self._timestamps[i] for i in selected]})["persistent_total_bytes"]
        return {"answer": answer, "status": "complete", "query_read_bytes": reads,
                "num_retrieved_items": len(selected), "num_visual_tokens_used": None,
                "retrieved_frame_ids": [self._timestamps[i]["global_frame_id"] for i in selected],
                "retrieved_items": [copy.deepcopy(self._timestamps[i]) for i in selected],
                "selected_scores": [float(scores[i]) for i in selected],
                "variant": "official_clip_scoring_qwen2_vl_answer", "test_only": self.test_only}


def create_method(config):
    if not isinstance(config.get("scorer"), dict) or not isinstance(config.get("answerer"), dict):
        raise ValueError("AKS requires explicit scorer and answerer local checkpoint configs")
    scorer = CLIPScorer(**config["scorer"])
    answerer = Qwen2VLAnswerer(**config["answerer"])
    return AKSMethod(config["memory_budget_bytes"], scorer=scorer, answerer=answerer,
                      identity={"scorer": scorer.identity, "answerer": answerer.identity},
                      **{name: config[name] for name in ("max_num_frames", "ratio", "t1", "t2", "all_depth") if name in config})
