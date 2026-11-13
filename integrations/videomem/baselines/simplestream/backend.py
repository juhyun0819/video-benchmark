"""Qwen2.5-VL recent-window inference with native image positions and decoding.

The paper's recent-N policy is retained. The released cached-generation bugs
are not reproduced: every image has its own native chat-template delimiters,
and generation receives actual input IDs and grids. A scoped prefill hook only
replaces image placeholder embeddings with verified post-merger features.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import threading
import time

import torch

from baselines.rekv_qwen25.rekv_qwen25.inputs import REPO_ID, REVISION, validate_checkpoint

UPSTREAM_REVISION = "be227ee8ce90a258b286a34161688c2a59eeeadb"
LAYOUT = "native_per_image_chat_mrope"


def question_prompt(question, choices=None):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string")
    choices = [] if choices is None else choices
    if not isinstance(choices, list) or len(choices) > 26 or any(
            not isinstance(choice, str) or not choice.strip() for choice in choices):
        raise ValueError("choices must be a list of up to 26 nonempty strings")
    if any("<|" in value or "|>" in value for value in [question, *choices]):
        raise ValueError("Question text cannot insert model control tokens")
    if choices:
        options = "; ".join(f"{chr(65+i)}. {value}" for i, value in enumerate(choices)) + ";"
        return f"{question}\nOptions: {options}\nOnly give the best option's letter directly."
    return question


def messages_for_question(frame_count, question, choices=None):
    if type(frame_count) is not int or frame_count < 0:
        raise ValueError("frame_count must be a nonnegative integer")
    return [{"role": "user", "content": [
        *[{"type": "image"} for _ in range(frame_count)],
        {"type": "text", "text": question_prompt(question, choices)},
    ]}]


def build_cached_prompt(processor, encoded_frames, question, choices=None, *, merge_size=2):
    """Match the native processor's per-image placeholder expansion exactly."""
    if type(merge_size) is not int or merge_size <= 0:
        raise ValueError("Invalid spatial merge size")
    text = processor.apply_chat_template(messages_for_question(len(encoded_frames), question, choices),
                                         tokenize=False, add_generation_prompt=True)
    token = getattr(processor, "image_token", "<|image_pad|>")
    pieces = text.split(token)
    if len(pieces) != len(encoded_frames)+1:
        raise ValueError("Chat template image slots do not match the frame window")
    grids, lengths = [], []
    previous_id = previous_time = None
    for frame in encoded_frames:
        grid = frame.grid_thw.detach().cpu().reshape(-1)
        if (grid.shape != (3,) or grid.dtype != torch.int64 or int(grid[0]) != 1
                or bool((grid <= 0).any()) or int(grid[1]) % merge_size or int(grid[2]) % merge_size):
            raise ValueError("Expected one native image grid per frame")
        length = int(grid.prod()) // merge_size**2
        if (frame.features.ndim != 2 or frame.features.shape[0] != length
                or not frame.features.is_floating_point() or not torch.isfinite(frame.features).all()):
            raise ValueError("Frame features do not match their native image grid")
        if (previous_id is not None and (frame.frame_id <= previous_id or frame.timestamp <= previous_time)):
            raise ValueError("Recent frames must retain strict chronological order")
        previous_id, previous_time = frame.frame_id, frame.timestamp
        grids.append(grid)
        lengths.append(length)
    expanded = pieces[0] + "".join(token*length + piece for length, piece in zip(lengths, pieces[1:]))
    inputs = dict(processor.tokenizer([expanded], padding=True, return_tensors="pt"))
    if grids:
        inputs["image_grid_thw"] = torch.stack(grids)
    image_id = processor.tokenizer.convert_tokens_to_ids(token)
    if int((inputs["input_ids"] == image_id).sum()) != sum(lengths):
        raise ValueError("Tokenized image placeholder count differs from the feature window")
    return inputs


@contextmanager
def cached_vision_prefill(model, features):
    """Replace only first-prefill image embeddings; leave native positions/KV intact."""
    expected = int(features.shape[0])
    called = 0
    multimodal = model.model

    def replace(module, args, kwargs):
        nonlocal called
        ids = kwargs.get("input_ids")
        if called or not expected:
            return args, kwargs
        if ids is None or kwargs.get("inputs_embeds") is not None:
            raise ValueError("Cached prefill requires native input_ids without prebuilt embeddings")
        cache_position = kwargs.get("cache_position")
        if cache_position is not None and (not cache_position.numel() or int(cache_position[0]) != 0):
            raise ValueError("Cached visual replacement is only valid at initial prefill")
        past = kwargs.get("past_key_values")
        if past is not None and past.get_seq_length() != 0:
            raise ValueError("Cached visual prefill cannot inherit an earlier answer's KV")
        if kwargs.get("pixel_values") is not None or kwargs.get("pixel_values_videos") is not None:
            raise ValueError("Cached prefill must not also provide raw pixels")
        mask = ids == model.config.image_token_id
        if int(mask.sum()) != expected:
            raise ValueError("Prefill image slots differ from cached feature rows")
        embeds = module.get_input_embeddings()(ids)
        if features.ndim != 2 or features.shape[1] != embeds.shape[-1]:
            raise ValueError("Cached feature hidden width does not match the checkpoint")
        replacement = features.to(device=embeds.device, dtype=embeds.dtype)
        kwargs = dict(kwargs)
        kwargs["inputs_embeds"] = embeds.masked_scatter(mask.unsqueeze(-1).expand_as(embeds), replacement)
        called += 1
        return args, kwargs

    handle = multimodal.register_forward_pre_hook(replace, with_kwargs=True)
    try:
        yield
        if expected and called != 1:
            raise RuntimeError("Cached visual features were not consumed exactly once")
    finally:
        handle.remove()


class Qwen25Backend:
    def __init__(self, model_path, ledger_path, device="cuda:2", max_new_tokens=128,
                 attn_implementation="sdpa"):
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        if attn_implementation not in {"sdpa", "eager", "flash_attention_2"}:
            raise ValueError("Unsupported attention implementation")
        torch.device(device)
        verified = validate_checkpoint(Path(model_path), Path(ledger_path))
        import transformers
        self.model_path, self.ledger_path = str(Path(model_path).resolve()), str(Path(ledger_path).resolve())
        self.device, self.max_new_tokens = device, max_new_tokens
        self.attn_implementation = attn_implementation
        self.identity = dict(model_id=REPO_ID, revision=REVISION, model_path=self.model_path,
            verification_manifest_sha256=verified["verification_manifest_sha256"],
            config_sha256=verified["config_sha256"], preprocessor_sha256=verified["preprocessor_sha256"],
            upstream_revision=UPSTREAM_REVISION, method="SimpleStream (Qwen2.5-VL-7B, native-mRoPE adaptation)",
            vision_layout=LAYOUT, observation="independent_image_post_merger", dtype="bfloat16",
            processor_use_fast=False,
            torch_version=str(torch.__version__), transformers_version=transformers.__version__,
            attn_implementation=attn_implementation, max_new_tokens=max_new_tokens,
            do_sample=False, num_beams=1, training=False, persistent_llm_kv=False)
        generation_path = Path(self.model_path)/"generation_config.json"
        if generation_path.exists():
            generation = json.loads(generation_path.read_text())
            self.identity.update(generation_config_sha256=hashlib.sha256(generation_path.read_bytes()).hexdigest(),
                repetition_penalty=generation.get("repetition_penalty", 1.0),
                eos_token_id=generation.get("eos_token_id"))
        self.model = self.processor = None
        self._lock = threading.RLock()

    def _load(self):
        if self.model is None:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            validate_checkpoint(Path(self.model_path), Path(self.ledger_path))
            self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True, use_fast=False)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path, local_files_only=True, torch_dtype=torch.bfloat16,
                attn_implementation=self.attn_implementation, device_map={"": self.device}).eval()
        return self.model, self.processor

    @torch.inference_mode()
    def encode_frames(self, frames):
        """Encode each sampled image separately, as in the existing 1-fps archive."""
        with self._lock:
            model, processor = self._load()
            results = []
            for image in frames:
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": image}, {"type": "text", "text": "."}]}]
                inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                    return_dict=True, return_tensors="pt")
                grid = inputs["image_grid_thw"].to(self.device)
                values = model.model.get_image_features(
                    inputs["pixel_values"].to(self.device, dtype=torch.bfloat16), grid)
                if isinstance(values, (tuple, list)):
                    values = torch.cat(list(values), dim=0)
                if grid.shape != (1, 3) or not torch.isfinite(values).all():
                    raise ValueError("Invalid native image features")
                results.append((values.detach().cpu().contiguous(), grid[0].detach().cpu().clone()))
            return results

    @torch.inference_mode()
    def answer(self, encoded_frames, question, choices=None):
        question_prompt(question, choices)
        if not isinstance(encoded_frames, (list, tuple)) or not encoded_frames:
            raise ValueError("No observed frames are available for a visual answer")
        with self._lock:
            model, processor = self._load()
            merge_size = model.config.vision_config.spatial_merge_size
            inputs = build_cached_prompt(processor, encoded_frames, question, choices, merge_size=merge_size)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            features = (torch.cat([frame.features for frame in encoded_frames], dim=0)
                        if encoded_frames else torch.empty((0, model.config.hidden_size)))
            prompt_tokens = inputs["input_ids"].shape[1]
            began = time.perf_counter()
            # Every answer starts a new native generation; questions never alter
            # the observation memory or seed the next answer's model state.
            model.model.rope_deltas = None
            try:
                with cached_vision_prefill(model, features):
                    output = model.generate(**inputs, do_sample=False, num_beams=1,
                        num_return_sequences=1, max_new_tokens=self.max_new_tokens, use_cache=True)
                tail = output[0, prompt_tokens:].detach().cpu()
            finally:
                model.model.rope_deltas = None
            eos = model.generation_config.eos_token_id
            eos = [] if eos is None else eos if isinstance(eos, list) else [eos]
            return dict(answer=processor.decode(tail, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                token_ids=tail.tolist(), used_frame_ids=[frame.frame_id for frame in encoded_frames],
                visual_tokens=int(features.shape[0]), prompt_tokens=int(prompt_tokens),
                elapsed_seconds=time.perf_counter()-began,
                completed_eos=bool(tail.numel() and int(tail[-1]) in eos), vision_layout=LAYOUT)
