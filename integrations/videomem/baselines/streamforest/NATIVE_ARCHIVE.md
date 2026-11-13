# Native StreamForest encoder archive

This archive is an encoder cache, independent of questions, query timestamps,
memory budgets, and forest state. Extraction never runs an LLM or constructs a
Persistent Event Memory Forest. The Qwen2.5 archive remains separate.

## Official source and exact cache boundary

- Repository: https://github.com/MCG-NJU/StreamForest
- Source revision: `f9d14670ff451ff5865638e17c9958e99bf2fec5`.
- Checkpoint: `MCG-NJU/StreamForest-Qwen2-7B`.
- Checkpoint revision: `6840f91e17346b4c406ba03b57d1a63582765c91`.
- Vision: checkpoint-embedded SigLIP SO400M, patch14, 384 pixels, 1152 channels.
  Native loading removes the final layer from the 27-layer architecture and
  replaces its pooling head with Identity. Exactly 421 checkpoint tensors load
  strictly into this 26-layer vision tower. `SigLipVisionTower.forward` returns
  `hidden_states[-1]`; do not substitute a different hidden state or normalized
  output. No external unfinetuned SigLIP weights are substituted.
- Connector: `tome729_fstw_pemf`, including an MLP 1152 -> 3584 -> 3584.

The released `llava/model/language_model/llava_qwen.py:generate_online` performs:

```text
official SigLipImageProcessor
  -> official SigLipVisionTower.forward
  -> CACHE: [729, 1152] native visual features
  -> MemoryManager.update(frame_features)
  -> MemoryManager.get_memory_tokens()
  -> mm_projector.mlp(memory_tokens)
  -> language model
```

The MLP is **after memory compression**, so caching projected tokens and feeding
them into the manager would change native operation order and feature space.
Native raw tower output is the last query-independent tensor before this online
manager. No token merging or reduction occurs during extraction.

There are two different upstream routes. The general batch projector additionally
adds spatial and temporal sinusoidal positions **before** constructing memory,
and only then applies its MLP. The online method above does not add those
projector positions. This archive implements the online route used by this
repository's streaming baseline; it does not claim the two routes are equal.
The cache retains the raw common vision representation, patch grid, global frame
position, source PTS, and optional batch temporal index. A future batch-route
adapter must use the official positional addition at replay time, with the
chosen stream's frame origin. It must not silently use raw cached features as
position-enriched batch-projector memory input.

The official builder casts weights to **float16**, and the official evaluation
script preprocesses video frames and calls `.half()`. Extraction uses fp16,
including input pixels, even though the checkpoint config records bf16 storage.
Vision attention is the unchanged native implementation. TF32 is disabled.
The processor resizes RGB to 384x384 with bicubic interpolation, rescales by
1/255 and normalizes with mean/std 0.5 per channel. Learned patch positions
remain part of the native encoder; no temporal embedding is added to this cache.

Only the two official checkpoint shards containing embedded vision tensors,
the index and the config are needed. Downloads are pinned and each full shard
is checked against its remote LFS SHA-256. This model directory is intentionally
a **vision extraction subset**, not a complete checkpoint for answering.

## Timeline and inventory

`inventory.json` is derived from the existing Qwen2.5 inventory, then source
SHA-256 values, actual FFprobe durations, and source file states are checked
again. Extraction checks source states before and after each decoded range.
No Qwen feature values, QA files, answers, or query times are inputs.

At 1 fps, take the first actual source PTS at or after each source-local integer
second in `[0, duration)`. Preserve the Qwen global timeline using **cumulative
exact video durations**, rather than rounding each Day's duration to its sampled
frame count. Global frame IDs are continuous; global timestamps have fractional
Day boundaries. Missing expected sampling targets fail extraction.

| Source | Seconds | Frames |
| --- | ---: | ---: |
| Day1 | 39019.76 | 39020 |
| Day2 | 29440.24 | 29441 |
| Day3 | 34641.16 | 34642 |
| Day4 | 35013.88 | 35014 |
| Day5 | 31581.72 | 31582 |
| Day6 | 19372.20 | 19373 |
| Day7 | 32655.96 | 32656 |
| Total | 221724.92 | 221728 |

One frame is 729 * 1152 * 2 = **1,679,616 bytes**. Total uncompressed tensor
payload is **372,417,896,448 bytes** (372.418 GB / about 346.841 GiB), plus
indexes, receipts, configuration and logs. A 100-frame shard is 167.962 MB.

## Storage and integrity

```text
streamforest_native_day1_7_v1/
  config.json
  manifest.json
  shards/shard_000000/tokens.bin
  shards/shard_000000/index.json
  receipts/000000.ready.json
  logs/gpu-utilization.jsonl
  logs/progress.json
  logs/extraction-summary.json
```

Tensor payload is contiguous little-endian IEEE fp16 in frame/token/channel
order, without compression. The storage implementation views fp16 bits as
signed int16 for NumPy transport; it does not numerically convert features to
integers. Reload returns the exact fp16 values.

Each frame index stores global ID/time (also exact fractions), Day/source ID,
source-local PTS and time base, source hash, decoded RGB hash, preprocessed fp32
input hash, feature hash, resolution, shape, dtype, patch grid and temporal/frame
position metadata. Immutable model/encoder/connector identifiers and official
revisions are inherited through the index's `config_id`. The config also hashes
the extraction implementation, upstream source manifest, and checkpoint config.

Payload and index use same-directory temp files, fsync, atomic rename and parent
directory fsync. A ready receipt is written last as the commit point, with both
file hashes, count, frame range, timestamp range and payload bytes. A writer lock
prevents concurrent writes to one archive. Resume revalidates every committed
shard and starts after the valid contiguous prefix. Uncommitted orphan files do
not count as completed shards. Corrupt or noncontiguous committed shards cause
an error, rather than silently re-encoding valid committed frames. Restart uses
the same command, output directory, frame count and immutable config.

The reader validates config identity, readiness, both hashes, shape/dtype,
expected frame count, source-local and global timestamps, frame continuity and
per-frame hashes before yielding chronological tensors. It rejects incomplete
archives by default. It never yields partial-shard data.

```python
from streamforest_videomem.native_archive import Archive

for frame in Archive(archive_path, expected_config_id=reviewed_config_id):
    # In the future evaluator: stop before the current query cutoff, answer,
    # then continue replay. Use a persistent iterator with lookahead.
    # The official manager takes the tensor directly; timestamp is audit state.
    memory.update(frame.visual_tokens.to(memory_device))
```

Enforce the evaluator's reviewed cutoff convention when feeding the iterator;
never build the final full-stream forest first and reuse it for earlier queries.
This module does not implement evaluation or choose any memory budget.

## Parallel extraction and preflight

Eight GPU processes partition disjoint 25-frame tasks. Each process has a CPU
decoder/preprocessor thread with a queue bounded to two prepared batches.
The coordinator bounds the entire in-flight/reorder window to 16 tasks, reorders
results by global frame ID, and writes 100-frame shards in order. No method or
memory-budget partitioning is involved. The coordinator handles NAS writing;
GPU processes can encode later bounded tasks while writes proceed.

Logs include synchronized encoder/H2D/D2H latency, batch latency, peak allocated
VRAM, decode/preprocess compute time, input/reorder backlog, wall throughput,
serialization time and NAS fsync write MB/s. NVIDIA-SMI utilization and device
memory use are sampled every second. CPU-only decoding uses one codec thread
per worker; no audio is decoded and raw images are not saved.

Use the dedicated Python 3.11 environment, with this directory as the cwd:

```bash
export USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD"
/convei_nas2/zzuhyun/hyunjun/envs/streamforest-native/bin/python \
  -m streamforest_videomem.native_archive_smoke \
  --inventory /home/zzuhyun/현준/streamforest_work/inventory.json \
  --model-path /convei_nas2/zzuhyun/hyunjun/models/StreamForest-Qwen2-7B-native-vision \
  --output /convei_nas2/zzuhyun/hyunjun/feature_archives/streamforest_native_day1_7_v1/preflight
```

The preflight builds a 300-frame sequential reference, tests batch candidates
1/4/8/16/32 on a 32-frame validation prefix, and uses only an exactly matching
candidate. It extracts exactly 300 distinct production frames with all 8 GPUs,
then compares all tensors and input/timestamp metadata against the sequential
reference, including storage reload. A batch-dependent mismatch falls back to
batch 1 and must then pass exact comparison. Validation/reference encodings are
reported separately from production coverage, not counted as additional unique
production frames. One actual native `generate_online` call is intercepted
immediately before its first manager update to verify the raw-frame boundary
without constructing a forest or running QA.

`preflight-report.json` supplies the requested checkpoint/boundary/shape/dtype,
bytes/frame, full size, measured 300-frame throughput, projected full runtime,
batch checks, native-boundary check and frame coverage. The projection excludes
model-loading time and has uncertainty from the short test. The smoke command
never launches full extraction.

After reviewing that report, use the approved batch size for full extraction:

```bash
/convei_nas2/zzuhyun/hyunjun/envs/streamforest-native/bin/python \
  -m streamforest_videomem.native_archive extract \
  --inventory /home/zzuhyun/현준/streamforest_work/inventory.json \
  --model-path /convei_nas2/zzuhyun/hyunjun/models/StreamForest-Qwen2-7B-native-vision \
  --output /convei_nas2/zzuhyun/hyunjun/feature_archives/streamforest_native_day1_7_v1 \
  --gpus 0,1,2,3,4,5,6,7 --batch-size APPROVED_BATCH_SIZE
```

Validate a completed archive using the same module's `validate --output PATH`.
Run the integrity tests with `python -m pytest -q tests/test_native_archive.py`.

## Measured preflight (2026-11-11)

The 300-frame test passed with all eight RTX A6000 GPUs. Production coverage was
exactly 300 unique IDs, without missing or duplicate frames. All saved/reloaded
tensors, input hashes, frame ordering and timestamps matched the sequential
reference bitwise; maximum absolute difference was zero. The actual official
online memory input capture also matched bitwise. Nine integrity/decoder/resume
tests passed. No QA was read and no real memory manager was updated.

Batch sizes 4/8/16/32 did not match the sequential tensor values, so the selected
batch size is **1**. Those alternatives were rejected rather than accepting a
looser tolerance. GPU data parallelism still uses all eight GPUs.

The 8-GPU test measured **22.506 frames/s** over 13.330 seconds after workers
became ready; total worker startup plus extraction was 23.148 seconds. Linear
projection for 221,728 frames is **9,851.9 seconds (2h 44m 12s)**, excluding
startup, final full integrity validation and workload variation. A 300-frame
test is a short throughput sample, not a guaranteed completion time.

The three atomic NAS payload writes measured 108.25, 49.54 and 63.10 MB/s,
with serialization taking 0.17, 0.25 and 0.15 seconds. Durable payload writes
consumed 7.60 seconds in total, showing NAS writes are a substantial throughput
limit. The bounded in-flight window allows workers to prepare later tasks while
the coordinator writes, without an unbounded RAM backlog. Maximum observed
reorder backlog was four 25-frame tasks; decoder/preprocessor input backlog was
at most one batch. GPU utilization, device memory, synchronized batch timing,
decode timings and per-shard write rates remain in the preflight logs.

Detailed measured values and dependency versions are in
`/convei_nas2/zzuhyun/hyunjun/feature_archives/streamforest_native_day1_7_v1/preflight/preflight-report.json`.
The full run has **not** started: the requested preflight review comes first.
