"""Cuts Ego4D narration.json down to what `npm run ingest -- ego4d` reads.

narration.json is over 1 GB, which Node cannot parse as one string. This keeps only narration_pass_1 of the
videos whose scenarios match an ego4d category in config/categories.json, so the output stays small.
Keyword matching and every other rule stay in the ingest adapter.

    python3 scripts/convert_ego4d_narrations.py \
        --metadata data/raw/ego4d/ego4d.json \
        --narrations data/raw/ego4d/v2/annotations/narration.json \
        --out data/raw/ego4d/narration_selected.json

Rerun it after changing the ego4d scenarios in config/categories.json.
"""
import argparse
import json
import os
import sys
from pathlib import Path


def ego4d_scenarios(categories):
    """Returns the scenarios used by ego4d categories."""
    return {s for c in categories["categories"] if c["dataset"] == "ego4d" for s in c["scenarios"]}


def select_narrations(meta, narrations, categories):
    """Returns {video_uid: {"narration_pass_1": ...}} for videos in an ego4d category scenario."""
    wanted = ego4d_scenarios(categories)
    selected = {}
    for video in meta["videos"]:
        uid = video["video_uid"]
        if not wanted.intersection(video.get("scenarios") or []):
            continue
        pass1 = (narrations.get(uid) or {}).get("narration_pass_1")
        if pass1 is not None:
            selected[uid] = {"narration_pass_1": pass1}
    return selected


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False)
    os.replace(tmp, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Keep narration_pass_1 of Ego4D videos in the ego4d categories.")
    parser.add_argument("--metadata", required=True, help="path to ego4d.json")
    parser.add_argument("--narrations", required=True, help="path to annotations/narration.json")
    parser.add_argument("--categories", default="config/categories.json", help="category file")
    parser.add_argument("--out", required=True, help="output .json path")
    args = parser.parse_args(argv)

    categories = read_json(args.categories)
    meta = read_json(args.metadata)
    narrations = read_json(args.narrations)
    selected = select_narrations(meta, narrations, categories)

    out = Path(args.out)
    write_json_atomic(out, selected)
    print(
        json.dumps({"videos_written": len(selected), "out": str(out), "bytes": out.stat().st_size}),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
