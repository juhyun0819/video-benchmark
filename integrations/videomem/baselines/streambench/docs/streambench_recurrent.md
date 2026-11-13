# StreamTTT and Cambrian-S recurrent adapters

These adapters run official pretrained architectures. The implementation was
validated with CPU state/policy fixtures and the official StreamTTT fast-weight
kernel at small dimensions. No pretrained weights were downloaded, no training
was performed, and full 4B/7B inference or accuracy was **not validated** in this
workspace. The isolated test environment imports Qwen vision models with a
matching torchvision installation; use isolated native environments for experiments.

## Audited sources and checkpoints

| Method | Official source revision | Official pretrained checkpoint revision |
|---|---|---|
| StreamTTT | [`zeyun-zhong/StreamTTT`](https://github.com/zeyun-zhong/StreamTTT/tree/96aed21cc4d78dc7e1787a53a0a95691f3cfd07a), `96aed21cc4d78dc7e1787a53a0a95691f3cfd07a` | [`zeyun-zhong/StreamTTT-4B`](https://huggingface.co/zeyun-zhong/StreamTTT-4B/tree/58f05be5c71549d6c4826325e6d454d2b0b59228), `58f05be5c71549d6c4826325e6d454d2b0b59228` |
| Cambrian-S | [`cambrian-mllm/cambrian-s`](https://github.com/cambrian-mllm/cambrian-s/tree/058b1b4c96846a229d1c9bb69da6ecf5ceee4a6b), `058b1b4c96846a229d1c9bb69da6ecf5ceee4a6b` | [`nyu-visionx/Cambrian-S-7B-LFP`](https://huggingface.co/nyu-visionx/Cambrian-S-7B-LFP/tree/d04f69849c1ddbf7e62a7790f2461ef549d4df8e), `d04f69849c1ddbf7e62a7790f2461ef549d4df8e` |

Source checkouts must match these revisions and have no modified tracked files.
Model loading defaults to `local_files_only: true`. Set `model_path` to a cached
checkpoint directory or prepopulate the Hugging Face cache at the pinned revision.
A supplied local directory is identified by its path and declared revision; the
adapter checks its architecture but does not hash all frozen weight files.

The two environments are incompatible: StreamTTT's requirements pin torch 2.8.0,
transformers 4.57.1, qwen-vl-utils 0.0.14 and related packages; Cambrian-S's
`pyproject.toml` pins transformers 4.37.0. Run the methods in separate processes
with their respective official dependencies. Cambrian's checkpoint config says
4.39.2, but its executable repository pins 4.37.0 and its attention patch uses
that older Qwen2 API. Do not replace the patch with modern normal Qwen2 caching.

Both source repositories are Apache-2.0. `RecallMemory` adapts the memory-policy
code from the linked Cambrian revision; its attribution is retained in the module
docstring and file header. The Cambrian root and evaluation license texts are
included in `baselines/streambench/vendor/cambrian_s/`; no pretrained weights are vendored.

## StreamTTT

Native entry points are `StreamTTTQwen3VLForConditionalGeneration` in
`streamttt/models/modeling_streamttt_qwen3vl.py`, `FastWeightBlock` in
`streamttt/models/fast_weight_block.py`, and position/prompt helpers in
`streamttt/streaming/windowing.py`. Observation follows
`StreamTTTTrainer.prediction_step_multiforward` in `streamttt/train/trainer.py`.
The native encoder is Qwen3-VL-4B's vision tower; its released processor must be
used rather than the base Qwen processor.

Every one of the 36 decoder layers has four 640-dimensional fast-weight heads,
three SwiGLU matrices, and three momentum matrices. The checkpoint uses BF16,
`inter_multi=1`, `lact_chunk_size=1024`, headwise momentum/decay and a 4096-token
recent attention window. No architecture or update schedule is reduced to fit B.

Each layer's native recurrent tuple has **12** fields, in exact order:

1. `conv_prefix`
2. `w0`, `w1`, `w2`
3. `buf_k`, `buf_v`, `buf_lr`, `buf_log_decay`, `buf_log_momentum`
4. `m_next_w0`, `m_next_w1`, `m_next_w2`

The incomplete chunk is deliberately not committed early. `conv_prefix` keeps
the last three preconvolution QKV positions. Momentum matrices and weights use
the loaded matrix dtype; scalar learning-rate/decay/momentum buffers are FP32.
Accounting also includes recent per-layer KV, DynamicCache layer bookkeeping
(especially `cumulative_length`, which differs from physical cache length),
global token offset, M-RoPE maximum and deltas, pending temporal-patch frames and
timestamps, counters and configuration identity. Hook-owned references are
cleared after each forward so obsolete recurrent states cannot remain hidden.

Analytical tensor sizes for the pinned BF16, batch-one checkpoint:

| Component | Bytes |
|---|---:|
| Fast weights + momentum | 707,788,800 (675 MiB) |
| Full convolution prefixes | 1,658,880 |
| Fixed tensor floor | 709,447,680 (about 676.58 MiB) |
| Maximum incomplete 1023-token update buffers | 378,886,464 |
| KV per retained position across all layers | 147,456 |
| Nominal 4096-position KV | 603,979,776 (576 MiB) |

These are configuration-derived sizes, **not measured pretrained runtime
footprints**. Actual live tensor allocations, including any retained view backing
storage, are measured by the common accounting implementation. Scalar metadata,
protocol/RNG state and pending frames add to these values. Native sliding cache
layers normally retain `L-1` physical positions; the table uses L as an upper bound.

`recent_tokens` is the original configurable recent-attention capacity. It does
not change head dimensions, number of layers or the TTT chunk size. Budgets below
the fixed tensor floor fail before checkpoint loading. Larger budgets are checked
against actual persistent state; a partial TTT update can make a configuration
unsupported even when its fixed state fits. The 1024 MiB pilot is not guaranteed
to fit arbitrary partial-chunk boundaries, even with a tiny recent window. A
budget sweep must retain those unsupported outcomes, not silently remove state.

Streaming adaptations:

- The common manifest supplies already sampled frames. The released processor
  retains its resize/normalization behavior but does not resample the manifest.
  Global seconds are expressed as indices in a virtual 1 Hz metadata timebase.
- Observation contains only the fixed system/user prefix and video. QA text is
  appended after restoring a checkpoint snapshot.
- An odd temporal-patch frame survives chunk/day boundaries as an accounted CPU
  tensor. QA on that checkpoint processes the pending frame on its private branch;
  the native processor pads its final temporal patch, reported in QA metrics.
- An all-valid stream uses `attention_mask=None`, letting the native causal mask
  use global positions and cache offsets without allocating a seven-day mask.
- KV and incomplete-buffer views are cloned to compact allocations at stable
  boundaries. This changes storage ownership, not values or recurrence.
- All model state is restored after each QA, including exceptional paths.

The official release evaluates multiple-choice answer logits and explicitly
does not release streaming free-form generation. `answer(question, choices)`
scores the native option-token logits; missing choices raises `NotImplementedError`.
No replacement text-generation algorithm is claimed. The QA read metric reports
an explicit upper bound for tensors supplied to the native recurrent forward;
some momentum tensors need no arithmetic read when no update chunk completes.

## Cambrian-S-7B-LFP + Recall Memory

Native code is `CambrianS_VSR.generate_until` in
`lmms-eval/lmms_eval/models/simple/cambrians_vsr.py`, model loading is
`cambrian.model.builder.load_pretrained_model`, and the attention/model patch is
`lmms-eval/qwen2_monkey_patch.py::{Qwen2SdpaAttention,cambrian_qwen2_forward}`.

The public checkpoint uses Qwen2.5-7B-Instruct (28 layers, 4 KV heads, head dim
128) plus `google/siglip2-so400m-patch14-384` (1152-dimensional visual features),
an MLP projector, and its trained next-frame prediction head. No retraining is
needed. Frames are pooled to an 8x8 grid, unpadded for aspect ratio, and receive
one newline token per row. A square frame therefore has **72** LLM positions,
not merely 64. The official builder loads FP16 even though stored weights and
checkpoint config are FP32.

Saved state includes both sensory (`runtime`) and long-term lists for every
layer: `key_states`, `value_states`, `modalities`, `lengths`, `surprising_scores`;
benchmark `frame_ids` records exact merged provenance. The saved next-frame
`prediction`, unpadded grid dimensions, frame count, timestamps, native policy
parameters and counters are also necessary. The prefix KV is shared between the
two lists and is charged once; device restoration preserves those aliases.

The attention patch stores **unrotated** K/V. Each forward reassigns rotary
positions across the retained sequence after compression or recall. Physical
cache lengths and entry ordering are the native position state; introducing
globally increasing Qwen2 cache positions would change this method.

Preserved policy:

- Compare the preceding LFP output against the current unprojected SigLIP grid
  with mean cosine distance. The first observed frame has surprise 1.
- Keep the newest configured number of frames in sensory KV. On overflow,
  promote the oldest frame; spatially average-pool its KV only when its surprise
  is below the configured threshold.
- Apply the original `drop` or `drop_merge` rule to long-term memory. Least
  surprise wins eviction, with the earliest frame breaking ties. Adjacent
  surprising entries are averaged under `drop_merge`.
- Preserve upstream's exact consolidation trigger and strict `< capacity`
  termination, including its unconditional first drop after a triggered merge
  pass. This sometimes drops an entry even when merging alone would fit.
- On QA, flush sensory entries into a **private** memory branch, compress and
  consolidate there, and invoke native per-layer cosine/top-k recall and greedy
  decoding. Native `retrieval_topk <= 1` means no retrieval pruning; it is not a
  one-frame retrieval mode.

The official recipe varies by video duration (`lmms-eval/scripts/vsr.sh`):
sensory 32/64 frames, downsample ratio 2, surprise thresholds .25/.30/.35,
`drop` or `drop_merge`, long-term token budgets 16384/32768 and retrieval top-k
32/128/256/512. Adapter defaults use sensory 32, ratio 2, threshold .35, `drop`,
16384 long-term positions and retrieval top-k 32, matching the 240-minute recipe.

Byte enforcement first honors the configured native position cap, then repeats
the same long-term consolidation policy until the full common byte budget fits.
It never evicts mandatory sensory/prediction state. Sensory size and native
long-term capacity are explicit configuration knobs. FP16 KV costs 57,344 bytes
per position across 28 layers. A square frame costs 4,128,768 bytes; a 32-frame
sensory window costs 132,120,576 bytes before prefix, prediction and metadata.
The configured sensory tensor floor is computed after the first actual encoded
frame reveals the unpadded grid; it is reported separately from measured totals.

Streaming adaptations and limitations:

- The native evaluator preloads full videos and all features and reconstructs
  memory per question. This adapter encodes one chronological frame at a time
  and externalizes its native local variables. No disk visual-feature cache
  remains available for later queries.
- Its end-of-video sensory flush now happens only on a QA snapshot, never on a
  day boundary. The next day retains the exact predictor and sensory state.
- The model-forward monkey patch is scoped to this model instance rather than
  replacing a class method globally. Its executed math is the official code.
- Only a stable unpadded grid is supported, as required by the native pooling
  and merging implementation. A changed aspect ratio/grid fails explicitly.
- Surprise, sensory decisions, pooling/merging, evictions and merged frame IDs
  are logged through `drain_events()`. Those audit-only records do not influence
  later inference and are not counted or serialized as retrievable memory.
- Native QA concatenates every retained K and V before ranking. Consequently
  `query_read_bytes` counts full stored KV, while `retrieved_attention_bytes`
  records the smaller payload retained after ranking. `retrieved_frame_ids`
  contains the union of the exact selected provenance across layers.

## Execution

Create separate JSON configurations in the repository, supplying your cached
source/checkpoint paths and native environment. Example StreamTTT configuration:

```json
{"method":"streamttt","memory_budget_bytes":2147483648,
 "upstream_source":"/path/to/StreamTTT","model_path":"zeyun-zhong/StreamTTT-4B",
 "local_files_only":true,"recent_tokens":4096,"window_frames":16,"device":"cuda"}
```

Example Cambrian configuration:

```json
{"method":"cambrian_s","memory_budget_bytes":1073741824,
 "upstream_source":"/path/to/cambrian-s","model_path":"nyu-visionx/Cambrian-S-7B-LFP",
 "local_files_only":true,"sensory_window_size":32,"surprise_threshold":0.35,
 "compression_downsample_ratio":2,"consolidation_method":"drop",
 "consolidation_mem_budget":16384,"retrieval_topk":32,"device":"cuda:0"}
```

From the repository root, inside the appropriate method's environment:

```bash
python -m baselines.streambench observe --config baselines/streambench/configs/streamttt.json --manifest frames.jsonl --output runs/streamttt --chunk-size 8
python -m baselines.streambench observe --config baselines/streambench/configs/cambrian_s.json --manifest frames.jsonl --output runs/cambrian --chunk-size 8
python -m baselines.streambench evaluate --config baselines/streambench/configs/cambrian_s.json --checkpoint runs/cambrian/checkpoint_D1_D4 --questions questions_day1.json --output qa_day1_at_day4.jsonl
python -m baselines.streambench sweep --config baselines/streambench/configs/streamttt.json --manifest frames.jsonl --output runs/streamttt-pilot --budgets-mib 64 128 256 512 1024
```

Native recurrence checks, without pretrained weights:

```bash
python -m unittest discover -s baselines/streambench/tests -p test_streambench_recurrent.py
STREAMBENCH_STREAMTTT_SOURCE=/path/to/StreamTTT python -m unittest discover -s baselines/streambench/tests -p test_streambench_recurrent.py
```

The optional tests execute the official FastWeightBlock and official norm
definitions at small synthetic dimensions, including convolution state, a
partially filled TTT chunk, an update spanning serialization, and all 12 returned
state fields. A second test instantiates the real native StreamTTT multimodal
class at small synthetic dimensions and compares video logits, recent cache,
and TTT state across serialization in the same Python process. Cambrian's
Recall Memory has a separate fresh-process state test; full native StreamTTT
fresh-process inference remains unverified. This is structural recurrence evidence,
not a pretrained 4B model experiment. Use a compatible torch/torchvision pair.

Validation on 2026-11-10 from this repository root:

```bash
STREAMBENCH_STREAMTTT_SOURCE=/tmp/videomem-audit-streamttt PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ../streambench-test-env/bin/python -m pytest -q baselines/streambench/tests/test_streambench_recurrent.py
```

Historical pre-relocation result: **15 passed in 11.27 seconds**. The optional
upstream checkout was the pinned revision above. This includes same-process
native tiny-model tests and Cambrian fresh-process state tests; pretrained GPU
inference, native Cambrian 7B execution, and seven-day
accuracy/latency experiments remain unvalidated.

## Local-only Cambrian loading

Start a fresh process with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` when
`local_files_only=true` (the default). This is required because the official
builder separately loads the SigLIP vision tower and does not propagate its
main-checkpoint local-only flag. The adapter rejects an unsafe configuration
before importing/loading the native builder. Cache both the main checkpoint
and its required vision tower ahead of time.
