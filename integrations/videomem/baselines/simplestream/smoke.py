"""Fail-closed real-GPU fidelity check for the native-mRoPE SimpleStream adaptation.

Run from the repository root, on an explicitly selected idle GPU:
    python -m baselines.simplestream.smoke --gpu 2 --output NEW_DIRECTORY

This exercises actual day1 images and committed features. It does not reproduce
the upstream cached-prefix bugs or claim the original paper's evaluation score.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time
import traceback
from types import SimpleNamespace

import torch

from .backend import (
    LAYOUT, Qwen25Backend, build_cached_prompt, cached_vision_prefill,
    messages_for_question,
)
from baselines.rekv_qwen25.rekv_qwen25.inputs import read_feature_frames
from baselines.rekv_qwen25.rekv_qwen25.smoke import (
    Capture, FEATURES, LEDGER, MODEL, golden_images, past_layers,
)

WINDOWS = (1, 4, 8, 16)
QUESTION = "Describe what you see in these recent frames."
MAX_NEW_TOKENS = 8


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _write_report(path, report):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tensor_hash(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _comparison(left, right):
    left, right = left.detach().cpu(), right.detach().cpu()
    shape_match, dtype_match = left.shape == right.shape, left.dtype == right.dtype
    result = dict(exact=False, shape_match=shape_match, dtype_match=dtype_match,
                  shape=list(left.shape), other_shape=list(right.shape),
                  dtype=str(left.dtype), other_dtype=str(right.dtype))
    if not (shape_match and dtype_match):
        return result
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    exact = finite and torch.equal(left, right)
    result.update(exact=exact, finite=finite)
    if finite:
        result["max_abs"] = (0.0 if exact or not left.numel()
                             else float((left.float() - right.float()).abs().max()))
    return result


def _require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def _gpu_preflight(physical_gpu):
    if physical_gpu < 0:
        raise ValueError("--gpu must be a nonnegative physical GPU index")
    output = subprocess.check_output([
        "nvidia-smi", f"--id={physical_gpu}",
        "--query-gpu=index,uuid,name,memory.used,memory.total", "--format=csv,noheader,nounits",
    ], text=True, timeout=15).strip()
    fields = [part.strip() for part in output.split(",")]
    if len(fields) != 5 or int(fields[0]) != physical_gpu:
        raise RuntimeError(f"Cannot establish selected GPU identity: {output!r}")
    used_mib = int(fields[3])
    if used_mib > 1024:
        raise RuntimeError(f"GPU {physical_gpu} is occupied ({used_mib} MiB > 1024 MiB); refusing model load")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    logical_gpu = physical_gpu
    if visible is not None:
        identifiers = [item.strip() for item in visible.split(",") if item.strip()]
        matches = [i for i, identifier in enumerate(identifiers)
                   if identifier in {str(physical_gpu), fields[1]}]
        if len(matches) != 1:
            raise RuntimeError("Selected physical GPU is not unambiguously visible in CUDA_VISIBLE_DEVICES")
        logical_gpu = matches[0]
    return dict(physical_index=physical_gpu, logical_index=logical_gpu, uuid=fields[1],
                name=fields[2], initial_used_mib=used_mib, total_mib=int(fields[4]),
                cuda_visible_devices=visible, checked_at=_utc())


def _to_device(inputs, device):
    return {key: value.to(device) if torch.is_tensor(value) else value
            for key, value in inputs.items()}


def _raw_inputs(processor, images, question):
    text = processor.apply_chat_template(messages_for_question(len(images), question),
                                         tokenize=False, add_generation_prompt=True)
    return dict(processor(text=[text], images=images, padding=True, return_tensors="pt"))


def _post_kv_cpu(output):
    return tuple(tuple(tensor.detach().cpu().clone() for tensor in layer)
                 for layer in past_layers(output.past_key_values))


@contextmanager
def _visual_capture(model):
    values = []

    def capture(module, args, output):
        if not torch.is_tensor(output):
            raise TypeError("Expected Qwen2.5 post-merger visual tensor")
        values.append(output.detach().cpu().clone())

    handle = model.model.visual.register_forward_hook(capture)
    try:
        yield values
    finally:
        handle.remove()


@contextmanager
def _generation_capture(model):
    calls = []

    def capture(module, args, kwargs):
        position = kwargs.get("position_ids")
        cursor = kwargs.get("cache_position")
        _require(torch.is_tensor(position) and torch.is_tensor(cursor),
                 "Native generation omitted position IDs or cache cursor")
        entry = dict(position_ids=position.detach().cpu().clone(),
                     cache_position=cursor.detach().cpu().clone())
        if not calls:
            _require(cursor.numel() > 0 and int(cursor[0]) == 0,
                     "Generation did not start with an empty cache at position zero")
            entry["prefill_embeddings_sha256"] = _tensor_hash(kwargs["inputs_embeds"])
        calls.append(entry)

    handle = model.model.language_model.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        yield calls
    finally:
        handle.remove()


def _compare_generation_calls(raw, cached):
    result = dict(exact=False, raw_calls=len(raw), cached_calls=len(cached), calls=[])
    if not raw or len(raw) != len(cached):
        return result
    for left, right in zip(raw, cached):
        item = dict(positions=_comparison(left["position_ids"], right["position_ids"]),
                    cache_position=_comparison(left["cache_position"], right["cache_position"]))
        if "prefill_embeddings_sha256" in left:
            item["prefill_embeddings_exact"] = (left["prefill_embeddings_sha256"] ==
                                                  right.get("prefill_embeddings_sha256"))
            item["prefill_embeddings_sha256"] = left["prefill_embeddings_sha256"]
        result["calls"].append(item)
    result["exact"] = all(item["positions"]["exact"] and item["cache_position"]["exact"]
                          and item.get("prefill_embeddings_exact", True) for item in result["calls"])
    return result


def _native_generate(model, inputs):
    model.model.rope_deltas = None
    try:
        output = model.generate(**inputs, do_sample=False, num_beams=1,
                                max_new_tokens=MAX_NEW_TOKENS, use_cache=True)
        return output[0, inputs["input_ids"].shape[1]:].detach().cpu().tolist()
    finally:
        model.model.rope_deltas = None


def _run_window(backend, frames, images, record):
    model, processor = backend._load()
    n = len(frames)
    began = time.perf_counter()
    record.update(frames=n, status="running", started_at=_utc(), comparisons={}, timings={})
    comparisons, timings = record["comparisons"], record["timings"]
    cached_cpu = build_cached_prompt(processor, frames, QUESTION,
                                    merge_size=model.config.vision_config.spatial_merge_size)
    raw_cpu = _raw_inputs(processor, images, QUESTION)
    raw_metadata = {key: value for key, value in raw_cpu.items() if key != "pixel_values"}
    comparisons["prompt_keys_exact"] = set(raw_metadata) == set(cached_cpu)
    _require(comparisons["prompt_keys_exact"], f"{n}f raw/cached prompt field mismatch")
    comparisons["prompt"] = {key: _comparison(raw_metadata[key], cached_cpu[key])
                             for key in cached_cpu}
    _require(all(value["exact"] for value in comparisons["prompt"].values()),
             f"{n}f raw/cached token IDs, masks or grids differ")
    record.update(prompt_tokens=int(cached_cpu["input_ids"].shape[1]),
                  visual_tokens=sum(frame.features.shape[0] for frame in frames),
                  frame_ids=[frame.frame_id for frame in frames],
                  timestamps=[frame.timestamp for frame in frames])
    raw = _to_device(raw_cpu, backend.device)
    cached = _to_device(cached_cpu, backend.device)
    features = torch.cat([frame.features for frame in frames], dim=0)
    torch.cuda.synchronize()
    start = time.perf_counter()
    model.model.rope_deltas = None
    with _visual_capture(model) as visual, Capture(model) as raw_capture:
        output = model(**raw, use_cache=True, logits_to_keep=1, return_dict=True)
        torch.cuda.synchronize()
        timings["raw_native_forward_seconds"] = time.perf_counter() - start
        raw_logits = output.logits.detach().cpu().clone()
        raw_post = _post_kv_cpu(output)
        raw_delta = model.model.rope_deltas.detach().cpu().clone()
    del output
    _require(len(visual) == 1, "Native raw prefill must call the visual tower exactly once")
    comparisons["raw_batched_visual_vs_archive"] = _comparison(visual[0], features)
    _require(comparisons["raw_batched_visual_vs_archive"]["exact"],
             f"{n}f raw image features differ from committed archive")
    del visual

    model.model.rope_deltas = None
    torch.cuda.synchronize()
    start = time.perf_counter()
    with cached_vision_prefill(model, features), Capture(model) as cached_capture:
        output = model(**cached, use_cache=True, logits_to_keep=1, return_dict=True)
        torch.cuda.synchronize()
        timings["cached_native_forward_seconds"] = time.perf_counter() - start
        cached_logits = output.logits.detach().cpu().clone()
        cached_post = _post_kv_cpu(output)
        cached_delta = model.model.rope_deltas.detach().cpu().clone()
    del output
    for key in ("inputs_embeds", "position_ids", "attention_mask"):
        comparisons[key] = _comparison(raw_capture.inputs[key], cached_capture.inputs[key])
    comparisons["rope_deltas"] = _comparison(raw_delta, cached_delta)
    comparisons["logits"] = _comparison(raw_logits, cached_logits)
    comparisons["attention_mask_exact"] = raw_capture.mask == cached_capture.mask
    names = {f"layer_{layer:02d}.{kind}" for layer in range(28) for kind in ("key", "value")}
    _require(set(raw_capture.kv) == set(cached_capture.kv) == names,
             "Passive capture did not contain all 28 layers' raw K/V")
    comparisons["pre_rope_kv"] = {name: _comparison(raw_capture.kv[name], cached_capture.kv[name])
                                   for name in sorted(names)}
    _require(len(raw_post) == len(cached_post) == 28, "Expected all 28 native cache layers")
    _require(all(len(left) == len(right) == 2 for left, right in zip(raw_post, cached_post)),
             "Unexpected native cache layer contents")
    comparisons["post_rope_kv"] = {
        f"layer_{layer:02d}.{kind}": _comparison(raw_post[layer][i], cached_post[layer][i])
        for layer in range(28) for i, kind in enumerate(("key", "value"))}
    _require(all(comparisons[key]["exact"] for key in
                 ("inputs_embeds", "position_ids", "attention_mask", "rope_deltas", "logits"))
             and comparisons["attention_mask_exact"]
             and all(value["exact"] for value in comparisons["pre_rope_kv"].values())
             and all(value["exact"] for value in comparisons["post_rope_kv"].values()),
             f"{n}f native raw/cached embeddings, positions, attention, KV or logits differ")
    del raw_post, cached_post, raw_capture, cached_capture, raw_logits, cached_logits
    model.model.rope_deltas = None

    torch.cuda.synchronize()
    start = time.perf_counter()
    with _generation_capture(model) as raw_calls:
        raw_tokens = _native_generate(model, raw)
    torch.cuda.synchronize()
    timings["raw_native_generate_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    with _generation_capture(model) as cached_calls:
        answer = backend.answer(frames, QUESTION)
    torch.cuda.synchronize()
    timings["cached_native_generate_seconds"] = time.perf_counter() - start
    comparisons["generated_tokens_exact"] = raw_tokens == answer["token_ids"]
    comparisons["generation_native_positions"] = _compare_generation_calls(raw_calls, cached_calls)
    comparisons["answer_state_cleared"] = model.model.rope_deltas is None
    record.update(raw_token_ids=raw_tokens, cached_token_ids=answer["token_ids"], answer=answer["answer"])
    _require(comparisons["generated_tokens_exact"]
             and comparisons["generation_native_positions"]["exact"]
             and comparisons["answer_state_cleared"], f"{n}f native generation fidelity failed")
    timings["window_total_seconds"] = time.perf_counter() - began
    record.update(status="passed", finished_at=_utc(),
                  peak_vram_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_vram_bytes=torch.cuda.max_memory_reserved())


@torch.inference_mode()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True, help="Physical nvidia-smi GPU index")
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing paths are refused")
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--features", type=Path, default=FEATURES)
    args = parser.parse_args(argv)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report_path = out / "report.json"
    started = time.perf_counter()
    report = dict(schema="simplestream_native_qwen25_fidelity_v1", status="running",
                  started_at=_utc(), started_time=time.time(), pid=os.getpid(), windows=[],
                  adaptation="Qwen2.5-VL-7B causal recent-window policy with native per-image mRoPE",
                  original_published_metric_reproduced=False, vision_layout=LAYOUT,
                  selected_windows=list(WINDOWS), max_new_tokens=MAX_NEW_TOKENS,
                  question=QUESTION, training=False, persistent_llm_kv=False,
                  feature_root=str(args.features.resolve()), timings={})
    backend, images = None, []
    _write_report(report_path, report)
    try:
        report["gpu"] = _gpu_preflight(args.gpu)
        import transformers
        _require(transformers.__version__ == "4.57.3", "This fidelity check is pinned to Transformers 4.57.3")
        report["runtime"] = dict(torch=torch.__version__, transformers=transformers.__version__,
                                 cuda=torch.version.cuda)
        report["source_sha256"] = {name: _sha256(Path(__file__).with_name(name))
                                   for name in ("backend.py", "smoke.py")}
        torch.set_num_threads(4)
        logical_gpu = report["gpu"]["logical_index"]
        torch.cuda.set_device(logical_gpu)
        torch.cuda.reset_peak_memory_stats(logical_gpu)
        start = time.perf_counter()
        features, rows = read_feature_frames(args.features, 0, max(WINDOWS))
        header = json.loads((args.features / "archive.json").read_text())
        report["archive"] = dict(archive_id=header["archive_id"],
                                 header_sha256=_sha256(args.features / "archive.json"),
                                 frame_ids=[row["frame_id"] for row in rows],
                                 feature_sha256=[row["shared_feature_sha256"] for row in rows])
        frames = [SimpleNamespace(frame_id=row["frame_id"], timestamp=row["timestamp"],
                  grid_thw=torch.tensor(row["encoder"]["grid_thw"], dtype=torch.int64), features=features[i])
                  for i, row in enumerate(rows)]
        report["timings"]["read_verified_features_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        images = golden_images(0, max(WINDOWS))
        _require(len(images) == max(WINDOWS), "Raw day1 decoder returned an incomplete frame range")
        report["timings"]["decode_raw_images_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        backend = Qwen25Backend(args.model, args.ledger, device=f"cuda:{logical_gpu}",
                                max_new_tokens=MAX_NEW_TOKENS, attn_implementation="sdpa")
        model, processor = backend._load()
        model.requires_grad_(False)
        _require("Fast" not in type(processor.image_processor).__name__,
                 "Expected use_fast=False image processor to match the verified archive")
        report["checkpoint"] = backend.identity
        report["processor"] = dict(image_processor_class=type(processor.image_processor).__name__,
                                   use_fast=False)
        report["timings"]["load_model_seconds"] = time.perf_counter() - start
        report.update(ready_at=_utc(), ready_time=time.time())
        _write_report(report_path, report)
        print(f"SimpleStream fidelity ready on physical GPU {args.gpu}", flush=True)

        start = time.perf_counter()
        encodings = backend.encode_frames(images)
        _require(len(encodings) == len(frames), "Per-frame encoder returned the wrong frame count")
        report["per_frame_encoding"] = [dict(frame_id=frame.frame_id,
            features=_comparison(encoded[0], frame.features), grid=_comparison(encoded[1], frame.grid_thw))
            for frame, encoded in zip(frames, encodings)]
        report["timings"]["per_frame_encode_seconds"] = time.perf_counter() - start
        _require(all(item["features"]["exact"] and item["grid"]["exact"]
                     for item in report["per_frame_encoding"]),
                 "Per-frame native image features differ from the verified archive")
        del encodings
        for count in WINDOWS:
            record = {}
            report["windows"].append(record)
            _write_report(report_path, report)
            _run_window(backend, frames[:count], images[:count], record)
            _write_report(report_path, report)
            print(f"SimpleStream fidelity passed: {count} frames, all native tensors and generated IDs exact", flush=True)

        start = time.perf_counter()
        isolation = report["answer_isolation"] = dict(status="running", order=["A(4f)", "B(1f)", "A(4f)"])
        hooks_before = len(model.model._forward_pre_hooks)
        answers, cleared = [], []
        for selected, question in ((frames[:4], QUESTION),
                                   (frames[:1], "What objects are visible in this image?"),
                                   (frames[:4], QUESTION)):
            answers.append(backend.answer(selected, question))
            cleared.append(model.model.rope_deltas is None)
        isolation.update(first_A_token_ids=answers[0]["token_ids"], B_token_ids=answers[1]["token_ids"],
                         second_A_token_ids=answers[2]["token_ids"],
                         repeated_A_exact=answers[0]["token_ids"] == answers[2]["token_ids"]
                                          and answers[0]["answer"] == answers[2]["answer"],
                         state_cleared_after_each=cleared,
                         prefill_hooks_restored=len(model.model._forward_pre_hooks) == hooks_before)
        _require(isolation["repeated_A_exact"] and all(cleared) and isolation["prefill_hooks_restored"],
                 "A-B-A independent-answer state isolation failed")
        isolation["status"] = "passed"
        report["timings"]["answer_isolation_seconds"] = time.perf_counter() - start
        report.update(status="passed", all_native_comparisons_exact=True,
                      peak_vram_bytes=torch.cuda.max_memory_allocated(logical_gpu),
                      peak_reserved_vram_bytes=torch.cuda.max_memory_reserved(logical_gpu))
    except BaseException as exc:
        report.update(status="failed", all_native_comparisons_exact=False,
                      error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        if backend is not None and backend.model is not None:
            backend.model.model.rope_deltas = None
        raise
    finally:
        for image in images:
            image.close()
        report.update(finished_at=_utc(), finished_time=time.time(),
                      total_seconds=time.perf_counter() - started,
                      cpu_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024)
        _write_report(report_path, report)
        print(f"SimpleStream fidelity {report['status']}: {report_path}", flush=True)


if __name__ == "__main__":
    main()
