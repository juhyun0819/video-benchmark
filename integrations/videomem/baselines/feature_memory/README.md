# Visual feature archive, short KV probe, and optional bounded memories

The current experiment keeps the **visual feature archive** running. Bounded
memory construction was stopped at the user's request; its implementations and
tests remain available for later experiments. The separate short KV probe
measures decoder cache storage for 1/4/8 archived frames. It does not ask or
answer questions. Running `archive` never starts the CPU memory consumer.

## Runtime and local inputs

Run commands from the repository root. The exercised Linux environment uses
Python 3.9.21, PyTorch 2.5.1+cu121, torchvision 0.20.1, and Transformers 4.57.3.
[requirements.in](requirements.in) records the other tested direct runtime
dependencies. It is not a complete transitive lockfile. A fresh environment can
be prepared with:

```bash
python3.9 -m venv /path/to/feature-env
PY=/path/to/feature-env/bin/python
$PY -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
$PY -m pip install -r baselines/feature_memory/requirements.in
```

`ffprobe` must be on PATH for video inventory; `ffmpeg` is also needed by the
small-video tests. CUDA is required for the pretrained GPU runs. `nvidia-smi`
adds device telemetry. PyAV performs the actual chronological decoding.
Model loading is local-only and uses SDPA; FlashAttention is not required.

```bash
PY=/path/to/feature-env/bin/python
VIDEOS=/path/to/day1_7_videos
MODEL=/path/to/Qwen2.5-VL-7B-Instruct
ARCHIVE=/path/to/new/feature-archive
KV_OUTPUT=/path/to/new/kv-probe
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

The model directory must contain the Qwen2.5-VL-7B-Instruct configuration,
processor/tokenizer files, safetensors index and weight shards. The measured
checkpoint is `Qwen/Qwen2.5-VL-7B-Instruct` revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`. Existing download verification
manifests are reused only while file identities still match. Videos, model
weights, feature shards, cache tensors and live job files stay outside Git.
Output directories must be new; these commands do not resume an existing run.

## Visual archive only

Start with a five-minute prefix, review its shapes and byte counts, then use a
different output directory for the full stream:

```bash
$PY -m baselines.feature_memory archive \
  --videos "$VIDEOS"/day{1..7}.mp4 --model "$MODEL" \
  --mode sequential --devices 0 --max-seconds 300 \
  --archive-frames-per-shard 100 --output /path/to/new/archive-smoke

$PY -m baselines.feature_memory archive \
  --videos "$VIDEOS"/day{1..7}.mp4 --model "$MODEL" \
  --mode parallel --devices 0 1 2 3 4 5 6 7 \
  --archive-frames-per-shard 100 --output "$ARCHIVE"
```

Device IDs refer to the GPUs visible to the process; select only available
devices. `--dataset-status /path/to/download_status.json` can reuse verified
download hashes, and `--timeline /path/to/timeline.json` can supply provisional
durations for missing future sources. Both are optional. Without download
status, source files are hashed by the existing manifest builder. A full run
requires all videos to be present and verified.

Each frame becomes a `[676,3584]` bf16 post-merger image feature in the measured
720×720 dataset: 4,845,568 bytes/frame. A 100-frame shard has 484,556,800 payload
bytes. Other source resolutions can change token counts; the fixed-shape archive
checks its planned representation. Only encoded features and provenance are
saved, never raw images or audio. The Day1–7 planned payload is
1,074,398,101,504 bytes for 221,728 frames. This is the full-run projection;
`complete.json` is written only after all expected frames have been committed.

`archive.json` records source/model identity and geometry. Each
`features_N.bin`/`index_N.json` pair is completed and hashed before its
`ready_N.json` is atomically published. Readers use only ready shards and reject
partial or changed files. `status.json` exposes progress and failures. A failed
producer is not treated as a completed partial archive.

`--seed-run` is optional and intended only for a preserved stopped online run:
reused frames must match committed contiguous journal entries, source inventory,
encoder implementation and feature hashes. Uncommitted tails are excluded.
Reused frames and new encoder calls are reported separately. Full automatic
crash resume is not implemented.

## Short decoder KV measurement

The probe reuses the first committed features without decoding or encoding
frames again. It checks archive/checkpoint identity, constructs the native
image placeholders and MRoPE positions, and runs independent neutral prefills
with `use_cache=True` and `logits_to_keep=1`. No generated answer or memory policy
is involved. Use a separate available GPU while the archive runs:

```bash
CUDA_VISIBLE_DEVICES='' $PY tools/feature_memory/measure_qwen25_kv.py \
  --archive "$ARCHIVE" --model "$MODEL" --cpu-check \
  --output /path/to/new/kv-cpu-check

CUDA_VISIBLE_DEVICES=7 $PY tools/feature_memory/measure_qwen25_kv.py \
  --archive "$ARCHIVE" --model "$MODEL" --device cuda:0 --frames 1 4 8 \
  --output "$KV_OUTPUT"
```

The CPU check compares cached-feature insertion with the ordinary native image
placeholder path on a tiny model, including bit-exact K/V and logits, and tests
safetensors roundtrips. It does not load the pretrained 7B model. The GPU probe
records every layer's actual K/V shape, dtype, payload and deduplicated backing
storage bytes; file lengths, exact reload checks, timings and GPU peaks are
stored in `kv_N_frames.json`, `input_plan.json` and `summary.json`. The safetensors
files contain K/V tensors only, not complete resumable generation checkpoints.

The completed local 1/4/8-frame probe measured:

| Frames | Sequence positions | K/V payload bytes | Safetensors file bytes |
| ---: | ---: | ---: | ---: |
| 1 | 697 | 39,968,768 | 39,973,896 |
| 4 | 2,731 | 156,606,464 | 156,611,712 |
| 8 | 5,443 | 312,123,392 | 312,128,680 |

All files reloaded exactly. Native bf16 cache geometry is 28 layers × K/V ×
4 KV heads × 128 dimensions: **57,344 bytes per sequence position**. The prompt
adds 19 fixed text positions and 2 image-boundary positions per frame. Visual
positions alone use 38,764,544 cache bytes/frame, eight times the archived visual
feature payload. The 61.59-hour visual-only KV projection is 8,595,184,812,032
bytes; this is not a full-history LLM run or a claim that the history fits the
128,000-position context limit.

Both archive and probe use independently encoded images at 1fps. The native
video path groups two source frames into temporal patches and uses video-time
MRoPE; its features and positions are different. Original timestamps remain
provenance and are not inserted into this neutral probe prompt.

## Optional CPU memory construction

`consume` is a separate explicit command, currently stopped in the live
experiment. It can read completed shards while encoding continues, or reuse an
archive later without encoder calls. First run a 300-frame smoke:

```bash
$PY -m baselines.feature_memory consume --archive "$ARCHIVE" \
  --max-frames 300 --snapshot-seconds 300 --batch-frames 100 \
  --output /path/to/new/memory-smoke

# Optional full construction after reviewing the smoke.
$PY -m baselines.feature_memory consume --archive "$ARCHIVE" \
  --batch-frames 100 --output /path/to/new/memory-run
```

One reader validates each shard and copies a bounded batch to one shared RAM
buffer. Four spawned CPU processes each own one FluxMem budget; a fifth owns
all twelve FIFO/Uniform/Reservoir states. They map that buffer read-only and
update in timestamp order. All acknowledge before reuse, and batches split at
snapshot cutoffs. Workers cannot read the archive themselves. The 100-frame
buffer needs at least 484,556,800 bytes available in Linux `/dev/shm`; provision
more than this for process/system overhead. Retained feature banks reside on
disk and require capacity for the configured budgets and snapshot copies.

The original `build` and `run` commands remain available as references below.

The read-only monitor is also included in the repository:

```bash
$PY tools/feature_memory/watch_feature_memory.py \
  --job-status /path/to/feature_memory_staged_job.json --gpu
```

It reads a supervisor job manifest, not the archive's `status.json` directly.
That manifest points to the archive, optional consumer and KV-probe status.
The optional `tools/feature_memory/start_feature_memory_staged.py` supervisor
starts both archive and CPU stages and includes a CPU smoke gate; use the
explicit module commands above for the current archive-only workflow.

## Online reference

The optional `build` pipeline constructs **FIFO / Uniform / Reservoir / FluxMem
× 5/10/25/50%**, sharing each frame's bf16 Qwen2.5-VL visual output exactly once.
It uses the existing decoder/encoder and never creates a full-history feature
archive. The original token-budget pilot remains available as `run` for prior
measurement reproducibility; its archive settings do not apply to `build`.

## Fixed full-stream byte budgets

Budgets use the **entire Day1–7 inventory**, even in a 300-second smoke or at an
early snapshot. They are never recomputed from the observed prefix.

| Ratio | Fixed hard bytes | GB |
| --- | ---: | ---: |
| 5% | 53,719,905,075 | 53.719905075 |
| 10% | 107,439,810,150 | 107.439810150 |
| 25% | 268,599,525,376 | 268.599525376 |
| 50% | 537,199,050,752 | 537.199050752 |

Reference: 221,728 samples across 221,724.92 playback seconds; 676×3584 bf16
features/sample, totaling 1,074,398,101,504 payload bytes. Fixed ratios are
converted using decimal arithmetic and floor rounding. Capacity further
deducts metadata/control and FluxMem tier/provenance bytes.

At every update and snapshot, persistent bytes must remain within that state's
fixed cap. Each snapshot separately reports `effective_retention_ratio = actual
persistent bytes / observed cumulative feature bytes`,
`effective_visual_retention_ratio = retained feature bytes / observed feature
bytes`, and `budget_utilization = actual persistent bytes / fixed cap`.
The first ratio can exceed 100% before a bank fills because it includes metadata.

The actual host has A5000 24GB×8 and approximately 270GB RAM. The 16 state caps
sum to **3.868TB**, so features use bounded disk record stores on NAS. Disk
records and live control bytes are charged to each policy; FluxMem short/middle
tier tensors are also charged. This is a persistent-storage comparison, not a
claim that TB-scale states fit in CPU RAM. The store grows only for retained
rows, overwrites evicted rows, and truncates uniform compactions. There is no
hidden full-history store. Read/write buffers are bounded to 1,024 records and
file handles have a bounded LRU. Python/kernel/filesystem allocator overhead is
working-system overhead, separate from functional persistent payload.

## Build and parallelism

From the repository root, using the environment and local paths defined above:

```bash
# Reference A: one encoder, ordered updates. All seven source paths are needed
# even for a prefix smoke, because they define the fixed budget reference.
$PY -m baselines.feature_memory build \
  --videos "$VIDEOS"/day{1..7}.mp4 \
  --model "$MODEL" --mode sequential --devices 0 \
  --max-seconds 300 --snapshot-seconds 300 --output /path/to/new/smoke-A

# Mode B: GPU workers encode different chunks, parent commits in time order.
$PY -m baselines.feature_memory build \
  --videos "$VIDEOS"/day{1..7}.mp4 \
  --model "$MODEL" --mode parallel --devices 0 1 2 3 4 5 6 7 \
  --chunk-frames 8 --max-inflight-chunks 8 --max-queue-bytes 2147483648 --flux-workers 4 \
  --output /path/to/new/full-byte-run
```

Default snapshots are 5h/10h/20h/40h/60h plus the actual final endpoint. The
entire Day1–7 input is one continuous playback stream. It is never partitioned
into independent memories. Truly independent subjects can use separate build
invocations, stream IDs, GPU selections and output roots.

Chunks are configurable. The small default keeps raw pixels, features and IPC
copies bounded; requested larger chunks are constrained by the queue byte cap.
No GPU is assigned to a method/budget. Each frame has exactly one encoder call;
one ordered tensor is consumed by all 16 states. The runner hashes features
before and after fanout to reject accidental mutation. Prefetched future
features stay temporary and cannot enter an earlier snapshot. Queue bytes,
backpressure and worker memory peaks are logged; long snapshot writes do not
trigger spurious encoder timeouts.

`--flux-workers 4` additionally updates the four independent FluxMem states on
CPU threads while the caller updates the twelve lightweight states. Every
consumer receives the same immutable tensor, and a barrier completes all
sixteen updates before advancing to the next frame or taking a snapshot.
There is no additional feature queue or encoder call for this fanout. The
default `--flux-workers 1` is the serial reference; summed per-state update
time and the actual fanout wall time are reported separately.

## FluxMem identity and pressure handling

The official TAS and SDC kernel functions are reused without changing their
feature math, thresholds or anchors. The repository's existing persistent
streaming adapter supplies causal short/middle/long transitions. Prefix tests
compare output features and positions exactly with the released kernel.

Adaptations are explicit: shared single-image grids replace native two-frame
video patches; persistent tiers may shrink only when an absolute cap cannot fit
native 4-short/128-middle grids (all current full-stream caps fit them). On byte
pressure, additional native SDC passes run until no more tokens can merge, and
the retained bank is recompressed with bounded working memory. Incoming grids
then follow the same fixed-point rule. Additional oldest-token eviction is a
fallback only when native fixed-point state still cannot fit. The snapshot
records native recompression events/bytes saved, fallback events/tokens/bytes
and the cause. This budgeted streaming adaptation is not labeled a reproduction
of native FluxMem accuracy.

Merged records contain exact represented-token counts, source frame IDs and
time spans. Full membership lists are not stored. TAS removals are reported as
removals, not invented merges. Repeated SDC retains the native unweighted means
of current centroids; provenance counts track original membership separately.

## Snapshots and accounting

`snapshots/<method>/<ratio>/<time>/` contains immutable record shards, control,
hash manifests and `benchmark_metadata.json`. FluxMem adds its bounded native
tiers. Snapshot copies are archival experiment artifacts and are unavailable to
live policies; they are reported separately from each live hard cap. Every
snapshot is read back and all feature/provenance shard bytes are SHA256-verified.

Snapshot copying prefers the kernel `copy_file_range` operation, confirmed to
issue NFSv4.2 server-side COPY on this NAS. Unsupported filesystems fall back to
the original bounded streaming copy. Source and destination hashes are still
checked independently; this changes transport, not memory contents or budgets.
NFS destination size is validated after writer close because cached attributes
can remain stale immediately after a successful server COPY and fsync.

Use `byte_memory.SnapshotReader` or `fluxmem.FluxMemSnapshotReader` to load the
immutable state as bounded tensor batches without copying an entire bank.
`ByteBudgetMemory.load` / `fluxmem.load_snapshot` additionally restore mutable
state and counters for continuation tests. Full pipeline decoder/queue crash
resume is not implemented; an interrupted full build should not be silently
replayed into an existing state.

`budget_plan.json` and `planned_summary.txt` describe the fixed pre-run settings.
`frames.jsonl` records exactly-once shared feature digests and timing;
`progress.jsonl` records resources, queues and each state's actual/cap bytes.
`summary.json` and `summary.txt` give final measured cumulative uncompressed
bytes without storing that full representation, all 16 states, effective ratios,
encoder/decode/update times, GPU utilization and storage accounting.

The five-minute smoke cannot fill a 53GB+ fixed cap. Small synthetic-budget
tests separately exercise eviction, hard-cap enforcement, native FluxMem
recompression and fallback accounting; the smoke checks the full fixed caps,
shared pretrained encoder and actual snapshot serialization.

## Legacy token-footprint pilot

Measure retained representations before any QA experiment. One sequential 1 Hz
video iterator supplies every policy/budget, with no questions, audio decoding,
saved raw frames, LLM inference, future-frame encoding, or snapshot replay.

## Representation and reuse

- Reuses `streamforest_videomem.video.iter_frames`, the existing video manifest
  validation, and `streambench.accounting.measure_state`.
- Loads only the official Qwen2.5-VL `visual.*` parameters, including the spatial
  merger. The output is the image embedding before the LLM, **not KV state**.
- Each frame uses the native **image** processor. Its temporal patch padding
  repeats the current image. This is an explicit independent-frame baseline;
  features differ from the native two-frame video path.
- Native 720×720 input becomes 728×728, grid `[1,52,52]`, then `[676,3584]`
  post-merger features. No budget-dependent resize or token cap is applied.
- bf16/fp16: 7,168 bytes/token and 4,845,568 bytes/frame; fp32 doubles both.
  These shapes are also checked against actual GPU outputs at runtime.

## Policies and byte accounting

`TokenMemory(method, budget, seed).update(features, frame_id, timestamp)` is the
shared interface; `stats()`, `persistent_state()` and `snapshot()` expose state.
Features are CPU tensors with shape `[tokens, hidden]` and owned storage.

| Method | Retention policy |
| --- | --- |
| `fifo` | Latest B visual token rows, including partial frames |
| `uniform` | Deterministic online dyadic uniform token-index stride |
| `reservoir` | Seeded Algorithm R; each observed token has equal inclusion probability |

All use B ∈ {512,1024,2048,4096} as a **token upper bound**, not a frame count.
`retained_items` counts distinct frames represented by at least one token, not
complete frames. At 676 tokens/frame, B=512 cannot retain a complete frame.

The uniform stride doubles when its bank would overflow. Retained indices are
equally spaced over the observed token history, but occupancy may be below B.
This is not exact moving `linspace(...,B)` or uniform whole-frame sampling:
discarded features cannot be recovered in a bounded single pass. The token
stride can also prefer particular spatial positions; this matters in later
accuracy experiments. Reservoir is the unbiased random token alternative.

Actual bank bytes include every allocated B-capacity feature/index buffer,
including unused uniform slots, plus deterministic metadata/counter/RNG payload.
The existing accountant charges complete backing allocations, not view sizes.
Python allocator headers, decoder buffers, encoder weights, update/snapshot
workspace and CUDA allocator reserves are separate runtime costs. Peak process
RSS, CUDA allocated bytes and frozen model bytes are reported separately.

The table uses `actual_memory_compression_ratio = all seen feature bytes / actual
bank bytes`. Snapshots also retain `compression_ratio = all seen feature bytes /
retained feature payload bytes`; its definition is explicit. Report both when
comparing a partly occupied bank. MB/GB are decimal; MiB/GiB use powers of 1024.

## Legacy pilot commands

Use the repository root and the environment/local paths defined above:

```bash
# First, a five-minute run. Every output directory must be new.
$PY -m baselines.feature_memory run \
  --videos "$VIDEOS"/day{1..7}.mp4 \
  --model "$MODEL" --dtype bfloat16 --device cuda:0 \
  --max-seconds 300 --snapshot-seconds 60 300 \
  --output /path/to/new/smoke

# After reviewing the smoke: all available Day1–7 data, in one pass.
# Default snapshots: 1h,5h,10h,20h,40h,60h, plus the final endpoint.
$PY -m baselines.feature_memory run \
  --videos "$VIDEOS"/day{1..7}.mp4 \
  --model "$MODEL" --dtype bfloat16 --device cuda:0 \
  --output /path/to/new/full
```

The data's actual duration is approximately 61.59 hours, so the final snapshot
can be later than 60 hours. Use `--max-seconds 216000` for an exact 60-hour prefix.
Each source resets its integer-second sampling grid; counts sum `ceil(duration)`
per source. Snapshot cutoffs are exclusive, on cumulative **rendered playback
time**, not the original mixed shooting dates. The Day1 render's +2.96 seconds
are included without rescaling. Future sources with incomplete downloads may
have provisional inventory durations; a full run refuses unverified sources.

With `--dataset-status`, completed download SHA256 records are trusted by path
and size and current file identity is bound/checked throughout decoding; source
bytes are not rehashed in this run. Without it, the existing local manifest
builder computes source fingerprints. This distinction is recorded in provenance.

## Artifacts

- `inventory.json`, `encoder.json`: durations, source provenance, model revision,
  native preprocessing, representation and frozen parameter bytes.
- `frames.jsonl`: every observed timestamp, tensor shape/dtype/token bytes, and
  separate decoding, preprocessing, H2D, synchronized GPU encoding, D2H, memory
  update and serialization timings. Decoding includes RGB conversion and the
  reused iterator's timestamp/pixel-hash checks.
- `history/features.bin`, `index.jsonl`, `format.json`: actual uncompressed
  feature bytes and seek metadata. Data are contiguous little-endian tensors;
  the index gives offset, dtype and shape. No history-sized RAM buffer is used.
  **This unbounded measurement archive is never read by bounded policies.**
- `dtype_probes/`: the first real feature serialized as bf16/fp16/fp32
  safetensors. The additional dtypes are storage casts, not additional encoder
  inference runs; numerical equivalence across inference dtypes is not claimed.
- `snapshots/<time>/<method>_B<budget>/`: retained feature safetensors and JSON
  metadata, including token/frame IDs, timestamps, counters and RNG state.
- `summary.json`, `summary.txt`, `status.json`: actual payload/file lengths,
  all 12 configurations, full-history projection, timing, progress and failures.

The full history is saved by default to obtain **measured serialized sizes**.
At native resolution, 60h in bf16 is 1,046,642,688,000 feature bytes before index
overhead. Allocate output storage accordingly. `--no-history-archive` skips this
archive but still writes bounded snapshots; it then cannot report a measured
full-history serialized size. The sample-derived full-data projections are
explicitly distinguished from measured processed-history bytes.

Read a stored bf16 frame by reading its indexed byte range and using
`torch.frombuffer(bytearray(data), dtype=torch.bfloat16).reshape(shape)`.
Snapshots are read with `safetensors.torch.load_file`. They export state for
inspection; crash-resume of the streaming run is not currently implemented.

## Validation

```bash
$PY -m pip install pytest==8.4.2
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest -q baselines/feature_memory/tests
```

Tests cover actual small-video decoding, one-pass/causal boundaries, exact FIFO,
dyadic spacing, reservoir equivalence to sequential Algorithm R and inclusion
frequencies, bounded allocation ownership, native image preprocessing, selective
vision checkpoint loading, snapshot bytes and binary feature roundtrips.
Actual pretrained GPU smoke results live outside the source repository.
