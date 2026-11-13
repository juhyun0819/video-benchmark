"""Use committed visual features as a causal recent-frame SimpleStream window."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                prefix=path.name+".", suffix=".partial", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Atomic publication with no overwrite, including concurrent callers.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inspect", "answer"])
    parser.add_argument("--archive", required=True)
    parser.add_argument("--cutoff-seconds", type=float, required=True)
    parser.add_argument("--recent-frames", type=int, default=4)
    parser.add_argument("--model")
    parser.add_argument("--ledger")
    parser.add_argument("--device", help="Explicit device, e.g. cuda:2; required for answer")
    parser.add_argument("--question")
    parser.add_argument("--choices", type=json.loads, default=[])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--output", help="New JSON output path; existing files are refused")
    args = parser.parse_args()
    if args.cpu_threads <= 0:
        parser.error("--cpu-threads must be positive")
    if args.output and Path(args.output).exists():
        parser.error("Output exists; choose a new path")
    if args.command == "answer" and not all((args.model, args.ledger, args.device, args.question)):
        parser.error("answer requires --model, --ledger, --device and --question")
    if args.command == "answer":
        from .backend import question_prompt
        question_prompt(args.question, args.choices)
    import torch
    torch.set_num_threads(args.cpu_threads)
    from .archive import QwenFeatureArchive
    archive = QwenFeatureArchive(args.archive)
    frames = archive.recent(args.cutoff_seconds, args.recent_frames)
    if args.command == "answer" and not frames:
        parser.error("No observed frames are available at this cutoff")
    result = dict(method="SimpleStream (Qwen2.5-VL-7B, native-mRoPE adaptation)",
        archive_id=archive.header["archive_id"], archive_root=str(Path(args.archive).resolve()),
        cutoff_seconds=args.cutoff_seconds, cutoff_rule="timestamp <= cutoff",
        recent_frames=args.recent_frames, frame_ids=[frame.frame_id for frame in frames],
        timestamps=[frame.timestamp for frame in frames],
        visual_tokens=sum(frame.features.shape[0] for frame in frames),
        feature_payload_bytes=sum(frame.features.numel()*frame.features.element_size() for frame in frames),
        training=False, retrieval=False, persistent_llm_kv=False)
    if args.command == "answer":
        from .backend import Qwen25Backend
        backend = Qwen25Backend(args.model, args.ledger, device=args.device, max_new_tokens=args.max_new_tokens)
        result.update(backend.answer(frames, args.question, args.choices))
        result.update(checkpoint=backend.identity, question=args.question, choices=args.choices)
    if args.output:
        write_new_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
