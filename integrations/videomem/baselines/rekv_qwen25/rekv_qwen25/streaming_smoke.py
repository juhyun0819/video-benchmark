"""GPU0-only ReKV streaming fidelity, window crossing and atomic resume smoke.

All observations are query-independent. The native reference and adaptation
use the same fixed open prefix, absolute three-axis positions and one-frame
chunks. Larger chunks are measured only as experiments; any non-exact result
keeps the allowed production chunk at one frame. This is not a full archive run
or permission to independently reset temporal shards.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import threading
import time

import psutil
import torch
import transformers
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .archive import KVArchiveWriter, load_block
from .attention import ReKVSession
from .inputs import read_feature_frames, validate_checkpoint
from .smoke import (
    Capture, FEATURES, LEDGER, MODEL, ROOT, compare, forward_cached, metadata,
    monitor, observation_inputs, tensor_hash, write_json,
)
from .state_io import flatten_state, unflatten_state


TOKENS_PER_FRAME = 678


def _mask(start, end, dtype, device):
    query = torch.arange(start, end, device=device)
    keys = torch.arange(end, device=device)
    result = torch.zeros((1, 1, end-start, end), dtype=dtype, device=device)
    return result.masked_fill_(keys[None, :] > query[:, None], torch.finfo(dtype).min)


def _mask_record(mask):
    if mask is None:
        return None
    return dict(shape=list(mask.shape), dtype=str(mask.dtype), sha256=tensor_hash(mask))


@contextmanager
def _observe_actual_short_masks(session):
    """Observe the mask actually handed to native HF by the short-path wrapper."""
    original = session._originals[0]
    records = []

    def observed(*args, **kwargs):
        records.append(_mask_record(kwargs["attention_mask"]))
        return original(*args, **kwargs)

    session._originals[0] = observed
    try:
        yield records
    finally:
        session._originals[0] = original


def _comparison(left, right):
    result = compare(left, right)
    if result.get("exact"):
        return dict(exact=True, max_abs=0.0)
    return result


def _kv_comparison(actual, expected):
    if set(actual) != set(expected):
        return dict(exact=False, keys_match=False)
    mismatches, maximum = [], 0.0
    for key in sorted(expected):
        result = _comparison(actual[key], expected[key])
        if not result["exact"]:
            mismatches.append(dict(tensor=key, **result))
            maximum = max(maximum, result.get("max_abs", float("inf")))
    return dict(exact=not mismatches, max_abs=maximum, mismatched_tensors=mismatches)


def _step(model, session, embeddings, positions):
    session.set_positions(positions)
    return model.model.language_model(
        inputs_embeds=embeddings, position_ids=positions,
        attention_mask={"full_attention": None}, use_cache=False,
    ).last_hidden_state


def _cpu(value):
    return value.detach().to(device="cpu", copy=True).contiguous()


def _bounds(n_init, frame, count=1):
    return n_init + frame*TOKENS_PER_FRAME, n_init + (frame+count)*TOKENS_PER_FRAME


def _native_reference(model, inputs, embeddings, n_init, frames, output):
    """Native HF, separate fixed prefix followed by one image frame per call."""
    prefix_inputs = dict(position_ids=inputs["position_ids"][:, :, :n_init],
                         attention_mask=inputs["attention_mask"][:, :n_init])
    with Capture(model) as capture:
        prefix = forward_cached(model, prefix_inputs, embeddings[:, :n_init])
    cache = prefix.past_key_values
    prefix_raw = capture.kv
    prefix_logits = _cpu(prefix.logits)
    del prefix, capture
    observations, evidence = [], []
    for frame in range(frames):
        start, end = _bounds(n_init, frame)
        padding_mask = torch.ones((1, end), dtype=torch.int64, device=model.device)
        with Capture(model) as capture:
            begin = time.perf_counter()
            result = model.model.language_model(
                inputs_embeds=embeddings[:, start:end],
                position_ids=inputs["position_ids"][:, :, start:end],
                attention_mask=padding_mask, past_key_values=cache,
                cache_position=torch.arange(start, end, device=model.device), use_cache=True,
            )
            torch.cuda.synchronize()
            seconds = time.perf_counter()-begin
        observations.append(dict(kv=capture.kv, hidden=_cpu(result.last_hidden_state), mask=capture.mask))
        evidence.append(dict(frame=frame, cumulative_tokens=end, seconds=seconds, mask=capture.mask))
        del result, capture, padding_mask
    del cache
    torch.cuda.empty_cache()
    write_json(output/"native_reference.json", evidence)
    return prefix_raw, prefix_logits, observations


def _check_bound(session):
    expected = min(session.total_seen, session.n_local)
    for state in session.layer_states:
        assert state["key"].shape[-2] == expected
        assert state["value"].shape[-2] == expected
        assert state["positions"].shape[-1] == expected
        assert state["init_key"].shape[-2] == min(session.n_init, session.total_seen)
        assert state["key"].untyped_storage().nbytes() == state["key"].numel()*state["key"].element_size()
    if session.total_seen > session.n_local:
        assert session._native_cache is None


def _stream_and_resume(model, inputs, embeddings, rows, n_init, frames,
                       prefix_raw, prefix_logits, native, config, output):
    session = ReKVSession(model, n_init=n_init, n_local=15000)
    config = dict(config, attention_policy=session.policy, official_rekv=session.upstream)
    report = dict(frames=[], attention_policy=session.policy, official_rekv=session.upstream)
    try:
        with _observe_actual_short_masks(session) as observed_masks:
            prefix_hidden = _step(model, session, embeddings[:, :n_init], inputs["position_ids"][:, :, :n_init])
            report["prefix_raw"] = _kv_comparison(session.latest_kv, prefix_raw)
            report["prefix_logits"] = _comparison(_cpu(model.lm_head(prefix_hidden[:, -1:])), prefix_logits)
            if not report["prefix_raw"]["exact"] or not report["prefix_logits"]["exact"]:
                raise RuntimeError("ReKV prefix differs from original native HF")
            del prefix_hidden
            for frame in range(frames):
                start, end = _bounds(n_init, frame)
                torch.cuda.reset_peak_memory_stats()
                before_masks = len(observed_masks)
                begin = time.perf_counter()
                hidden = _step(model, session, embeddings[:, start:end], inputs["position_ids"][:, :, start:end])
                torch.cuda.synchronize()
                seconds = time.perf_counter()-begin
                _check_bound(session)
                item = dict(frame=frame, cumulative_tokens=end,
                            mode="native_short" if end <= 15000 else "rekv_local_plus_sink",
                            sink_exposed=min(n_init, max(end-15000, 0)),
                            seconds=seconds, fps=1/seconds,
                            raw_capture_cpu=True, mask_observation_in_timing=end <= 15000,
                            peak_vram_bytes=torch.cuda.max_memory_allocated(),
                            cpu_rss=psutil.Process().memory_info().rss,
                            all_raw_finite=all(bool(torch.isfinite(t).all()) for t in session.latest_kv.values()),
                            hidden_finite=bool(torch.isfinite(hidden).all()),
                            bounded_local_tokens=min(end, 15000))
                if frame < len(native):
                    item["raw_vs_native"] = _kv_comparison(session.latest_kv, native[frame]["kv"])
                    item["hidden_vs_native"] = _comparison(_cpu(hidden), native[frame]["hidden"])
                    item["actual_causal_mask_exact"] = (
                        len(observed_masks) == before_masks+1 and observed_masks[-1] == native[frame]["mask"]
                    )
                    if not (item["raw_vs_native"]["exact"] and item["hidden_vs_native"]["exact"]
                            and item["actual_causal_mask_exact"]):
                        report["frames"].append(item)
                        raise RuntimeError(f"One-frame short native parity failed at frame {frame}")
                if not item["all_raw_finite"] or not item["hidden_finite"]:
                    raise RuntimeError(f"Non-finite streaming output at frame {frame}")
                report["frames"].append(item)
                write_json(output/"streaming_progress.json", report)
                print(f"streaming frame {frame+1}/{frames}: {item['mode']} {seconds:.3f}s", flush=True)
                del hidden

        # One committed frame payload plus the full bounded continuous resume state.
        # Its availability is the last observed frame, never an earlier cutoff.
        begin = time.perf_counter()
        saved = session.export_state()
        flat, descriptor = flatten_state(saved)
        del saved
        state_export_seconds = time.perf_counter()-begin
        payload = dict(session.latest_kv, **flat)
        checkpoint_meta = metadata(rows[frames-1:frames])
        checkpoint_meta.update(resume_state=descriptor, resume_through_frame_id=frames-1,
                               cumulative_frames=frames, cumulative_tokens=session.total_seen)
        writer = KVArchiveWriter(output/"continuous_resume_checkpoint", config)
        begin = time.perf_counter()
        receipt = writer.commit(0, payload, checkpoint_meta)
        commit_seconds = time.perf_counter()-begin
        begin = time.perf_counter()
        loaded, loaded_meta = load_block(writer.root, 0, writer.config_id)
        reload_seconds = time.perf_counter()-begin
        all_tensors_exact = all(torch.equal(value, loaded[name]) for name, value in payload.items())
        state = unflatten_state(loaded, loaded_meta["resume_state"])
        if not all_tensors_exact:
            raise RuntimeError("Lossless atomic resume roundtrip failed")
        report["resume_io"] = dict(
            exported_state_seconds=state_export_seconds, committed_seconds=commit_seconds,
            reload_and_checksum_seconds=reload_seconds,
            serialized_bytes=receipt["serialized_bytes"],
            commit_MB_per_second=receipt["serialized_bytes"]/commit_seconds/1e6,
            reload_MB_per_second=receipt["serialized_bytes"]/reload_seconds/1e6,
            all_tensors_exact=all_tensors_exact, config_id=writer.config_id,
            through_frame_id=frames-1, contains_native_post_rope_cache=False,
        )
        del payload, flat
        expected = []
        for frame in range(frames, frames+2):
            start, end = _bounds(n_init, frame)
            hidden = _step(model, session, embeddings[:, start:end], inputs["position_ids"][:, :, start:end])
            expected.append(dict(kv=session.latest_kv, hidden=_cpu(hidden)))
        session.import_state(state)
        del state, loaded
        report["resume_continuations"] = []
        for offset, frame in enumerate(range(frames, frames+2)):
            start, end = _bounds(n_init, frame)
            hidden = _step(model, session, embeddings[:, start:end], inputs["position_ids"][:, :, start:end])
            item = dict(frame=frame, raw_kv=_kv_comparison(session.latest_kv, expected[offset]["kv"]),
                        hidden=_comparison(_cpu(hidden), expected[offset]["hidden"]))
            report["resume_continuations"].append(item)
            if not item["raw_kv"]["exact"] or not item["hidden"]["exact"]:
                raise RuntimeError("Continuous disk resume changed the next observation")
        report["bounded_state_verified"] = True
        report["window_crossed"] = any(item["mode"] == "rekv_local_plus_sink" for item in report["frames"])
        if not report["window_crossed"]:
            raise RuntimeError("Smoke did not cross the true 15000-token window")
        return report
    finally:
        write_json(output/"streaming_progress.json", report)
        session.restore()
        del session
        gc.collect()
        torch.cuda.empty_cache()


def _chunk_experiments(model, inputs, embeddings, n_init, native, output):
    """Same 16 frames and prefix, varied call size; no production policy change."""
    results = []
    for chunk_frames in (1, 4, 8, 16):
        session = ReKVSession(model, n_init=n_init, n_local=15000)
        try:
            _step(model, session, embeddings[:, :n_init], inputs["position_ids"][:, :, :n_init])
            torch.cuda.reset_peak_memory_stats()
            elapsed, frame_comparisons = 0.0, []
            for first in range(0, 16, chunk_frames):
                start, end = _bounds(n_init, first, chunk_frames)
                begin = time.perf_counter()
                hidden = _step(model, session, embeddings[:, start:end], inputs["position_ids"][:, :, start:end])
                torch.cuda.synchronize()
                elapsed += time.perf_counter()-begin
                hidden_cpu = _cpu(hidden)
                for offset in range(chunk_frames):
                    sl = slice(offset*TOKENS_PER_FRAME, (offset+1)*TOKENS_PER_FRAME)
                    actual = {name: tensor[:, :, sl] for name, tensor in session.latest_kv.items()}
                    item = dict(frame=first+offset,
                                raw_kv=_kv_comparison(actual, native[first+offset]["kv"]),
                                hidden=_comparison(hidden_cpu[:, sl], native[first+offset]["hidden"]))
                    frame_comparisons.append(item)
                del hidden, hidden_cpu
            exact = all(item["raw_kv"]["exact"] and item["hidden"]["exact"] for item in frame_comparisons)
            result = dict(chunk_frames=chunk_frames, total_frames=16, seconds=elapsed, fps=16/elapsed,
                          raw_capture_cpu_in_timing=True, mask_observation_in_timing=False,
                          exact_vs_native_one_frame=exact, comparisons=frame_comparisons,
                          peak_vram_bytes=torch.cuda.max_memory_allocated(),
                          cpu_rss=psutil.Process().memory_info().rss)
            results.append(result)
            write_json(output/"chunk_experiments.json", results)
            print(f"chunk={chunk_frames}: {16/elapsed:.3f} fps, exact={exact}", flush=True)
            if chunk_frames == 1 and not exact:
                raise RuntimeError("Repeated one-frame streaming is not exactly reproducible")
        finally:
            session.restore()
            del session
            gc.collect()
            torch.cuda.empty_cache()
    return results


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--frames", type=int, default=25)
    parser.add_argument("--output", default=str(ROOT/"benchmarks/gpu0_streaming_v1"))
    args = parser.parse_args()
    if args.gpu != 0:
        parser.error("This authorized smoke uses GPU0 only")
    if args.frames < 25:
        parser.error("At least 25 frames are needed to exercise the real window crossing")
    if transformers.__version__ != "4.57.3":
        raise RuntimeError("Use the pinned Transformers 4.57.3 environment")
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    telemetry = threading.Thread(target=monitor, args=(stop, output/"telemetry.jsonl", 0), daemon=True)
    telemetry.start()
    status = dict(status="running", full_run_started=False, qa_used=False,
                  temporal_policy="continuous state; independent resets not enabled",
                  production_chunk_frames=1, gpu=0)
    model = None
    try:
        identity = validate_checkpoint(MODEL, LEDGER)
        features, rows = read_feature_frames(FEATURES, 0, args.frames+2)
        print("Loading exact frozen checkpoint once for all streaming checks", flush=True)
        begin = time.perf_counter()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(MODEL), local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", device_map={"": "cuda:0"}, use_safetensors=True,
        ).eval().requires_grad_(False)
        processor = AutoProcessor.from_pretrained(str(MODEL), local_files_only=True, use_fast=False)
        inputs, embeddings, n_init, delta = observation_inputs(model, processor, features)
        torch.cuda.synchronize()
        status.update(checkpoint=identity, model_load_seconds=time.perf_counter()-begin,
                      n_init=n_init, tokens_per_frame=TOKENS_PER_FRAME,
                      n_local=15000, dtype="bfloat16", transformers=transformers.__version__,
                      torch=torch.__version__, source_frame_ids=[row.get("global_frame_id", row.get("frame_id")) for row in rows],
                      rope_delta=delta.tolist())
        config = dict(backbone=identity, sampling_fps=1, frame_feature_shape=[676, 3584],
                      dtype="bfloat16", key_representation="pre_rope", n_local=15000, n_init=n_init,
                      input_protocol="open fixed user prefix followed by independent image blocks",
                      temporal_policy=status["temporal_policy"],
                      purpose="GPU0 bounded continuous state resume fidelity smoke")
        prefix_raw, prefix_logits, native = _native_reference(
            model, inputs, embeddings, n_init, 16, output,
        )
        status["streaming"] = _stream_and_resume(
            model, inputs, embeddings, rows, n_init, args.frames,
            prefix_raw, prefix_logits, native, config, output,
        )
        write_json(output/"report.json", status)
        status["chunk_experiments"] = _chunk_experiments(model, inputs, embeddings, n_init, native, output)
        status.update(status="gpu0_streaming_smoke_complete",
                      short_prefix_native_exact=True, continuous_resume_exact=True,
                      real_window_crossed=True,
                      temporal_shard_equivalence="not yet tested on full model; no independent shard execution enabled")
        write_json(output/"report.json", status)
        print("GPU0 streaming smoke complete", flush=True)
    except Exception as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output/"report.json", status)
        raise
    finally:
        stop.set()
        telemetry.join(timeout=15)
        del model


if __name__ == "__main__":
    main()
