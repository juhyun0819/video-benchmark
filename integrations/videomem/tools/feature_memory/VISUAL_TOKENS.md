# Extract visual tokens from a directory of videos

[`extract_visual_tokens.py`](extract_visual_tokens.py) extracts **Qwen2.5-VL-7B
post-merger visual features at 1 fps**. It loads only the vision encoder, keeps
one model per selected GPU, and distributes whole source files across workers.
It does not run QA, a language-model decoder, KV-cache extraction, captioning,
or any bounded-memory policy.

Every selected MP4 is processed from beginning to end. `clips.csv` and timeline
metadata are copied for later reference; they do **not** select or shorten the
input intervals. Each source has its own timestamps and archive, so videos of
different resolutions can share a run. No concatenated timeline is invented.

## Environment

Run from the repository root. The tested environment is Python 3.9,
PyTorch 2.5.1, torchvision 0.20.1, and Transformers 4.57.3. Install `ffmpeg` and
`ffprobe` using your system package manager; PyAV performs video decoding.

```bash
python3.9 -m venv /path/to/visual-token-env
PY=/path/to/visual-token-env/bin/python
"$PY" -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
"$PY" -m pip install -r baselines/feature_memory/requirements.in
```

Use a local `Qwen/Qwen2.5-VL-7B-Instruct` checkpoint containing its config,
preprocessor config, safetensors index, and weight shards. The existing encoder
was exercised with revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`.
Loading is local-only and uses SDPA; FlashAttention is not required. Model
weights and input videos are not included in Git.

Expected input layout:

```text
dataset-root/
  videos/
    work/example.mp4
    home/another-example.mp4
    ego4d/third-example.mp4
  clips.csv                 # optional, mapping only
  summary.json              # optional
  timeline/manifest.json    # optional, mapping only
```

The category names under `videos/` are arbitrary. The default scans
`videos/**/*.mp4`, rejecting duplicate physical files. Paths are explicit;
a missing dataset raises an error and never falls back to another directory.

## Start with a short test

Choose an existing source path relative to the dataset root. `--max-frames-per-source`
is a test prefix and must be omitted for full extraction.

```bash
DATASET=/path/to/cook
MODEL=/path/to/Qwen2.5-VL-7B-Instruct

"$PY" tools/feature_memory/extract_visual_tokens.py \
  --dataset-root "$DATASET" --model "$MODEL" \
  --output /path/to/new/visual-token-smoke \
  --devices 0 --cpu-threads 2 \
  --source-relative videos/work/example.mp4 \
  --max-frames-per-source 300
```

Review `status.json`, each source's `archive.json` and `source_receipt.json`,
and the final root `complete.json`. Use a **different output directory** for
the full run; a prefix archive cannot be resumed as a full archive.

## Extract all sources on several GPUs

Only use GPUs available for this job. Device numbers are logical indices within
the inherited `CUDA_VISIBLE_DEVICES`; without that variable they refer to the
usual CUDA enumeration. For example, these commands select physical GPUs 2–7:

```bash
OUTPUT=/path/to/new/cook-visual-tokens
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
"$PY" tools/feature_memory/extract_visual_tokens.py \
  --dataset-root "$DATASET" --model "$MODEL" --output "$OUTPUT" \
  --devices 0 1 2 3 4 5 --cpu-threads 2 --frames-per-shard 16
```

The default is one worker on GPU 0. Each worker processes one frame at a time
and keeps its vision model loaded between files. It never starts another
experiment. Original files are read only; source hashes and file identities
are checked. Decoding, preprocessing, GPU encoding, transfers, and archive
writes are reported separately. NAS writes may limit throughput even when GPU
utilization is low.

```bash
watch -n 5 "cat '$OUTPUT/status.json'"
```

`status.json` includes phase, worker state, frame counts, payload bytes, timings,
and errors. A successful process launch is not evidence of completed extraction:
check committed `ready_*.json` shards and final `complete.json` receipts.

## Representation and output

Features are bf16, shaped `[visual_tokens, 3584]`, after the vision spatial
merger and before the language model. The checkpoint's native image processor
determines the token count; there is no external resize or token-budget cap.
Each sampled frame is encoded independently as an image, with temporal padding
duplicating only that frame. These are not native paired-video features.

Payload bytes per frame are `visual_tokens * 3584 * 2`. For example, a 640×360
input produces 299 tokens (2,143,232 bytes) under the tested processor; a
960×540 input produces 646 tokens (4,630,528 bytes). JSON indexes and receipts
add small serialized overhead. Raw decoded frames and audio are not saved.

```text
output/
  sources.json              # exact source list and model identity
  status.json
  completed_sources.json
  complete.json             # only after all selected sources complete
  dataset_mapping/          # copied metadata, never encoder input
  sources/<source-id>/
    archive.json
    features_00000000.bin
    index_00000000.json
    ready_00000000.json
    ...
    complete.json
    source_receipt.json
```

`ready_*.json` is the commit boundary: payload and index are written, hashed,
and flushed before the ready receipt appears. Read committed data using the
repository reader, which validates checksums and shapes:

```python
from baselines.feature_memory.archive import iter_archive

for batch in iter_archive('/path/to/output/sources/SOURCE_ID', wait=False):
    features = batch['features']  # CPU bf16 [frames_in_shard, tokens, 3584]
    for tensor, record in zip(features, batch['frames']):
        print(record['source_relative_path'], record['source_local_timestamp'], tensor.shape)
```

Sampling uses the first real PTS at or after each source-local integer-second
target. Recorded timestamps are actual video PTS, normalized by the video's
start time; they need not be exact integer seconds. The duration-based frame
count is checked against decoding. A source with missing targets, inconsistent
end timing, or an unexpected grid change fails explicitly; it is not padded or
silently marked complete. In particular, the current strict decoder can reject
a fractional-duration file whose final sampling target has no real frame.

## Reuse and limitations

Re-running the same command with `--resume` validates completed sources and
skips them. Add `--verify-existing` to rehash all feature and index bytes;
otherwise write-time hashes are trusted only while recorded file identities
remain unchanged. The source list, checkpoint identity, shard size, and test
scope must match. An incomplete source is preserved and causes an error:
continuing inside a partially written source is not implemented.

The advanced `--source-map` option accepts an explicitly supplied
`cook_source_map_v1` JSON mapping. It accounts for all original sources,
including omitted ones, records recovery provenance, and never substitutes
files automatically. Matching size, mtime, and video geometry is **not** proof
of historical byte equality; current content is hashed afresh. No machine-local
source maps or recovered source selections are included in this handoff.

## Validation

```bash
"$PY" -m pip install pytest
"$PY" -m pytest -q tools/feature_memory/tests/test_extract_visual_tokens.py
```

Tests exercise generated videos with different shapes, source-local timestamps,
archive readback and checksums, test prefixes, changed-source rejection,
duplicate detection, and explicit mapped-source scope. The model encoder also
has a separate real-GPU two-frame smoke verification from development; CPU
tests do not assert GPU throughput or completion of any large dataset.
