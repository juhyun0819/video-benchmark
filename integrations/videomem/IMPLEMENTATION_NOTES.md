# Continuous VideoMem memory benchmark

`baselines.streambench` adds a common observation/evaluation layer around method-specific
representations. Legacy runners and vendored memory kernels are retained. Read
this document and the linked method notes before interpreting any result as a
reproduction of a paper. CPU structural validation is distinct from pretrained
model execution and scientific benchmark accuracy.

Historical planning notes and verbatim task requests are not included in this
source handoff. The current interface and operational contracts are documented
in the [streambench guide](baselines/streambench/README.md).

## Existing architecture and new abstraction

The previous `benchmark/` directory builds video/QA datasets and has a Streamlit
UI. It is not a unified memory evaluator. Baseline-specific `run.py` entrypoints
select model/checkpoint, sampling, bank and QA bridge independently. FIFO and
Reservoir reuse the Qwen3 streaming sampler/bridge; FluxMem variants use separate
packages with colliding `streaming` names. Most previous runs stop at the last
question's observation horizon and save only one final bank.

The package, configs, tests and method documentation live together under
[`baselines/streambench`](baselines/streambench/README.md). The entrypoint is
`python -m baselines.streambench`. `manifest.py` owns common sampled
PTS; `runner.py` owns question-free observation and cumulative day boundaries;
`method.py` defines lifecycle/state/budget contracts; `checkpoint.py` persists
state; `accounting.py` measures the actual exposed live payload. Adapters own
native models, state and selection/compression. Optional native dependencies load
only when needed. Execute each production method in its compatible environment
and a separate process; the budget sweep does this automatically.

| Method | Encoder | Stored representation / persistent state | Budget controllable? | Minimum memory | Query read mechanism | Retraining needed? |
|---|---|---|---|---|---|---|---|
| FIFO | Native Qwen3-VL | Main + 3 DeepStack tensors, spatial/temporal positions, record metadata, pending temporal frame, counters | Whole oldest grids; auto fixed capacity from actual first grid and B | Measured first grid + pending/protocol reserve | Native Qwen3 bridge reads retained grids | No |
| Reservoir | Native Qwen3-VL | Same features + Algorithm R slots, seen count, local RNG | Fixed K inferred from actual first grid and B; original Algorithm R | Measured first grid + pending/protocol reserve | Retained slots sorted chronologically for QA | No |
| FluxMem streaming / Qwen3 variant | Released Qwen2.5-VL / native Qwen3-VL | Short/mid/long tensors, normalized/backward caches, RoPE positions, trigger state, DeepStack when applicable, pending frame | Native tier compression; default long archive unbounded and rejects B overflow. Existing finite-long FIFO extension must be explicit | Measured dtype/grid/tier state | Existing native stored-feature bridge | No |
| StreamForest | Native SigLIP + Qwen2 | now/short/long event tensors, previous raw frame, event buffers, merge counts/times/similarities, manager bookkeeping | Native PEMF long quota derived from B; native event/merge policy | Actual raw/recent/event floor + metadata reserve | Native projector + Qwen2, fresh QA KV | No |
| EgoRAG | Configured native caption VLM + embedding model | Native SQLite caption/summary/embedding/cache pages, public timestamps, retained sampled JPEG evidence | Retain-all native policy; overflow explicitly unsupported | Empty DB/protocol measured; grows with observations | Native hierarchy/retrieval and VLM evidence; no original-video reread | No |
| VideoTree | Official EVA-CLIP + explicitly configured captioner/reasoner | Frame features, captions, frame metadata; trees are query-local | No original online eviction; retain-all observation overflows explicitly | Measured initial state; grows with observations | Official clustering/depth kernels and query-guided tree | No new training; pretrained dependencies required |
| StreamTTT | Released StreamTTT-4B/Qwen3-VL | Recent KV, 12 recurrent tensors per TTT layer, fast weights/momentum, convolution prefix, partial chunks, temporal grouping/position counters | Native recent-KV knob; never shrink trained architecture | Analytical fixed tensor floor about 676.6 MiB; runtime adds KV/partial buffers/metadata | Released multiple-choice logits path; streaming free-form unavailable | No dataset retraining; original online TTT updates preserved |
| Cambrian-S | Released Cambrian-S-7B-LFP | Sensory/long-term unrotated KV, predicted features/surprise state, positions, counters and Recall Memory metadata | Original sensory/pooling/drop/drop-merge policy with explicit native knobs; actual B checked | Measured after actual LFP grid and prefix initialization | Official patched attention + surprise/recency recall; QA flush on a snapshot | No |
| ReKV | See dedicated method notes | Native layer KV, CPU archives, retrieval keys, GPU LRU/cache metadata | Native archive retention; no invented archive eviction | Native allocations and state measured | Native KV retrieval | No |
| AKS | See dedicated method notes | Query-independent visual features + retained sampled evidence and timestamps | Query-adaptive selection is a read policy; observation retain-all has no native B eviction | Actual retained evidence/features measured | Official adaptive keyframe selection | No |

Detailed official repositories/checkpoints, pinned revisions, native configuration,
state schemas and every intentional deviation are in:

- [Native visual memory adapters](baselines/streambench/docs/streambench_native.md)
- [EgoRAG and VideoTree](baselines/streambench/docs/streambench_text_tree.md)
- [StreamTTT and Cambrian-S](baselines/streambench/docs/streambench_recurrent.md)
- [ReKV](baselines/streambench/docs/streambench_rekv.md)
- [AKS](baselines/streambench/docs/streambench_aks.md)

## Accounting convention

`B` is persistent *payload* bytes at stable boundaries. All retrievable state is
included regardless of CPU/GPU/disk placement. Frozen parameters and temporary
forward/QA snapshots are excluded. Live tensor allocations are deduplicated by
physical address range and device across all categories. A small view retains
and is charged for its entire underlying storage. Separate CPU/GPU copies count
separately. Tensor and NumPy views over the same CPU allocation are deduplicated.

Text/field names use UTF-8 bytes, floats and ordinary integers use 8 bytes, bools
use 1 byte, larger integers expand as needed; container contents are recursive.
Non-owning NumPy views are measured against their entire backing allocation;
checkpointing them fails closed because NumPy pickle does not preserve aliases.
Adapters must expose torch storage/views or an explicit owned representation.
`DiskState` is accounting-only: checkpoint state must embed complete file/index
contents, not just external paths. Current adapters use self-contained tensors,
SQLite bytes and image bytes.

Python container/allocator headers and unused process heap are runtime RAM,
reported separately. SQLite uses the complete serialized database-page size,
including indices/page slack; caption/embedding data inside these pages is not
counted again. An explicit disk-state reference charges file sizes, deduplicated
by inode. Adapters cannot silently hide opaque state: unknown types raise errors.

Global RNG state and the benchmark's continuation cursor are charged in
`metadata_bytes`. Model-specific RNG state is also charged. Model paths/config
identity and native control counters that affect future updates remain included.
Archived experiment logs/checkpoints are excluded only because adapters cannot
retrieve them to answer questions or update memory. Runtime RSS is the Linux
process high-water mark, and GPU metrics distinguish allocated/reserved memory.
Chunk latency currently measures encoding/update/finalization and accounting;
manifest scanning/decoding and model load are additional wall time.

`memory_bytes()` returns visual, KV, text, fast-weight and metadata components
plus `persistent_total_bytes`. Initial/minimum, chunk average/maximum and day
sizes are logged. Tensor totals must not be compared to checkpoint file sizes:
serialization has its own metadata and storage overhead.

Unsupported budgets raise `UnsupportedBudget` and produce
`status: unsupported_under_budget`, never successful truncated runs. A minimum
based only on fixed tensors is labeled analytical, not a loaded-model measurement.
When the native method has no bounded retention policy (e.g. VideoTree/AKS), B
is a feasibility limit, not permission to invent a new memory algorithm.

## Continuous input and checkpoints

`prepare` scans each source sequentially and writes a JSONL frame manifest with
actual PTS, source frame indices, source SHA256/stat, consecutive global IDs and
concatenated experience timestamps. It holds no full-day pixel/feature arrays.
All methods decode the identical manifest, keeping at most one chunk of PIL
frames (plus decoder buffers). Model-specific preprocessing is inside adapters.
Source changes and manifest tampering fail closed. One video per day is supported.

The runner never receives questions during observation. It initializes once,
never resets at day boundaries, and writes `checkpoint_D1`,
`checkpoint_D1_D2`, ..., `checkpoint_D1_D7`. Each directory contains an atomic
`state.pt`, human-readable metadata and SHA256 integrity records. Full native
state, pending temporal buffers, RNG, frame cursor, accumulated days, method/
configuration/budget identity and manifest digest are restored. Resume skips
old manifest entries without opening/decoding their video files. A mismatched
stream/manifest/config/budget is rejected. Checkpoints use PyTorch object
serialization and must come from a trusted local producer.

For temporal-pair models, an unmatched sampled frame is retained and charged
across chunks/days and paired with the next chronological sample; this differs
from legacy source-local orphan dropping and is documented in native notes.
The pending frame has not yet passed the visual encoder at an odd-frame boundary.
It is continuation state, not fabricated model evidence for checkpoint QA.

## QA, read budget and scoring

`evaluate` loads any cumulative checkpoint independently of the event's day.
Every question starts from the same checkpoint and RNG; exceptions restore the
caller's original state. Only question text/choices reach model answer methods.
Labels, source-day metadata and scoring stay outside observation/answer prompts.
A public question must itself resolve any required Day/source/time alias; the
wrapper does not invent event identities.

Each row includes checkpoint, source day, days since event, B, actual memory,
read bytes R, retrieved items/frame IDs and available visual/KV/text counts.
Reading all features to rank them counts toward R even if only a few are selected
for decoding. Future R constraints can wrap this interface; first experiments
control B and observe R. Unknown read metrics are null, never fabricated zero.

`correct_idx` uses strict multiple-choice letter or exact choice-text matching;
free-form labels use explicitly labeled exact-match scoring, not a semantic
judge. Unlabeled questions have `correct: null`.

The uploaded `Day*/qa.json` files are **not automatically relabeled as validated
checkpoint questions**: they contain original input aliases/time coordinates,
`day_time_view`, and explicit unverified-render/evidence flags. Before real
accuracy evaluation, resolve their public aliases/time mapping, preserve their
provenance and set `question_id`/`source_day`. The format example is schema-only,
not a generated research question or verified ground truth.

## Commands

Run from the repository root inside the chosen method's environment. Example
paths below must be configured; no full seven-day run is started automatically.

```bash
python -m baselines.streambench list-methods
python -m baselines.streambench prepare \
  --sources baselines/streambench/configs/sources.local.json \
  --output /path/run-input/frames.jsonl --fps 1

# Question-free pilot, through Day2 only.
python -m baselines.streambench observe \
  --config baselines/streambench/configs/streamforest.local.json \
  --manifest /path/run-input/frames.jsonl \
  --output /path/runs/sf-256MiB --budget-mib 256 --chunk-size 8 --stop-after-day 2

# Continue the SAME experience; no Day1/Day2 replay.
python -m baselines.streambench observe \
  --config baselines/streambench/configs/streamforest.local.json \
  --manifest /path/run-input/frames.jsonl \
  --output /path/runs/sf-resumed --budget-mib 256 \
  --resume /path/runs/sf-256MiB/checkpoint_D1_D2

python -m baselines.streambench evaluate \
  --config baselines/streambench/configs/streamforest.local.json \
  --checkpoint /path/runs/sf-resumed/checkpoint_D1_D7 \
  --questions /path/reviewed/questions_day1.json \
  --output /path/evaluation/day1-at-day7.jsonl

python -m baselines.streambench sweep \
  --config baselines/streambench/configs/streamforest.local.json \
  --manifest /path/run-input/frames.jsonl --output /path/pilots/sf \
  --budgets-mib 64 128 256 512 1024 --stop-after-day 2
```

The evaluation config's budget must equal the checkpoint budget; set it in the
config when evaluating a sweep member. Each sweep starts an independent stream
per budget, while each individual member continues the same memory across days.
Use a fresh output directory for every independent run/evaluation.

## Validation

See [validation results](baselines/streambench/docs/streambench_validation.md) for executed commands,
results and limitations. Small features/tests exercise native memory kernels,
including real compression/eviction, separate-process resume and identical
sampling. They do not establish full pretrained inference equivalence or QA
accuracy. Full seven-day experiments remain gated on method/environment-specific
pretrained smoke validation and reviewed question mapping.
