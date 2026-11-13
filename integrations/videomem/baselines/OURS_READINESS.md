# Ours baseline preparation

The target benchmark is **ours** (`datasets/videomem/day1.mp4` through
`day7.mp4`). Its reviewed QA has not arrived. These packages prepare source
inputs now and evaluate the shared canonical QA when it is supplied.
**EgoLife/JAKE is used only for compatibility tests.** The older annotation/QA
package in `data/` is not automatically promoted to the new benchmark.

## Implemented entry points

| Requested method | Module / instructions | Remaining runtime inputs |
| --- | --- | --- |
| GT Evidence Clip | [`ours_evidence`](ours_evidence/README.md) | Reviewed QA plus evidence intervals/points/source clips, verified answer backbone |
| Qwen3-VL-8B | [`ours_offline`](ours_offline/README.md), `--backbone qwen3_vl` | Reviewed QA, explicit frame budget, verified local checkpoint/environment |
| LLaVA-OneVision-7B | [`ours_offline`](ours_offline/README.md), `--backbone llava_onevision` | Reviewed QA, explicit frame budget, verified local checkpoint/environment |
| + AKS | [`ours_selection`](ours_selection/README.md), AKS config | Reviewed QA, local CLIP encoder/archive and answer backbone |
| + VideoTree | [`ours_selection`](ours_selection/README.md), VideoTree config | Reviewed QA, local EVA encoder/archive, independent captions, local text relevance model and answer backbone |
| Cambrian-S | [`ours_cambrian`](ours_cambrian/README.md) | Reviewed QA, native source/checkpoint/SigLIP and isolated official environment |

HERMES is outside this implementation. Existing methods and live EgoLife runs
keep their existing entry points. In particular, `jake_online` remains the
EgoLife test harness; the new target modules are all named `ours_*`.

The AKS path is the released CLIP-selector variant. VideoTree uses the existing
native tree kernels with selected raw frames sent to the target VLM and external
question-independent captions. These variants are explicitly documented;
neither is presented as an exact reproduction of every paper setting. The initial
source implementation launched no pretrained GPU evaluation or training.
Subsequent EgoLife smoke tests passed for the native Qwen3 and LLaVA offline
backbones with 32 causal frames on RTX 3090. Other method/environment combinations
still require their own model smoke check; implementation tests alone do not
establish those results. See the [offline supervisor](ours_offline/PARALLEL.md)
for disjoint GPU allocation, parallel decoding and audited QA-shard aggregation.

## Prepared local source inventory

- Video root: `/convei_nas2/zzuhyun/hyunjun/datasets/videomem`
- Prepared input: `/home/zzuhyun/hyunjun/ours_setup/input_v1`
- `sources.json`: seven videos, 5,543,123 frames at 25 fps, 221,724.92 seconds
  (61.59 recorded hours), exact fractional metadata and local file stat receipts.
- `questions.pending.json`: empty QA array and source/scorer hashes.
- `status.json`: `sources_ready_awaiting_QA`, zero evaluated questions.

Only video metadata was probed for ours. Video bytes, local inventories, pending
runtime files, feature archives and model artifacts are not committed. The
portable source list and QA schema are in `ours_offline/templates/`.

The prepared timeline concatenates actual video stream durations. A day number
is a grouping label, **not** `day * 86400`. When QA arrives, map day-local video
seconds to `sources[day].start_time + local_seconds`. If the delivered annotations
use a different clock, explicitly rebuild the source timeline and QA together;
do not silently reinterpret timestamps or derive query cutoffs from GT evidence.

Check readiness without a model:

```bash
python -m baselines.ours_offline validate \
  --root /convei_nas2/zzuhyun/hyunjun/datasets/videomem \
  --sources /home/zzuhyun/hyunjun/ours_setup/input_v1/sources.json \
  --canonical /home/zzuhyun/hyunjun/ours_setup/input_v1/questions.pending.json
```

This reports `awaiting_QA`. To regenerate metadata on another host, run
`ours_offline prepare` with the portable seven-day source list. Source receipts
are local to the prepared files and are not a full-file SHA256 verification.
AKS/EVA feature extraction and independent VideoTree captions can be prepared
before QA; their documented commands are explicit and have not been launched
as part of this source implementation.

## When reviewed QA arrives

1. Create `questions.json` from the pending template, retaining the matching
   source/scorer hashes and setting `status` to `ready`. Populate only reviewed
   questions according to the [QA schema](ours_offline/templates/qa.schema.json).
2. Supply each `qa_id`, global `query_time`, public `question`, `choices` and
   uppercase `answer` or `correct_idx`. Optional `day` only groups questions.
   Evidence access additionally needs the explicit evidence contract. For the
   day-long ours videos, bounded evidence intervals avoid reading a whole day
   merely because an event point lies in that file.
3. Run the same `ours_offline validate` command with `questions.json`, then
   `ours_offline plan` / `ours_evidence plan` and `ours_cambrian --plan-only` as
   appropriate. Selection validation checks its archive/captions and QA together.
4. Configure verified local model paths, immutable revisions, environments and
   explicit read/memory budgets in the linked method instructions. Run a small
   EgoLife model smoke check before the actual ours evaluation.
5. Execute each method against the **same** source/QA/scorer identities. Offline
   and selection methods support independent QA shards. Cambrian consumes one
   continuous history and never resets or zero-state shards the temporal stream.

All methods reject future input and preserve actual source timestamps in results.
Errors remain in the selected-QA denominator. Per-question output includes
method, ID, query time, actual frame times, prediction, raw output, GT, correctness,
latency, stream/update time and status/error. Native generation metadata varies
by backbone and is never invented. Full benchmark aggregation must check unique
QA IDs and coverage against the delivered manifest.

## Validation record

CPU tests cover native decoder frame identities, generic/rational clocks,
source/QA integrity, causal selections, evidence access, native processor inputs,
checkpoint isolation and resume. EgoLife planning checks use its own canonical
manifest and do not import those questions into ours.

On 2026-11-12, 58 CPU tests passed: offline 24, evidence 11, selection 12 and
Cambrian orchestration 11. The complete offline suite ran in the pinned
Transformers 4.57.6 environment; the complete selection suite ran with SciPy
1.15.3 and exercised the actual cosine-k-means/Ward VideoTree kernel. Optional
processor/tree tests can skip in environments missing their dependencies. Native
Qwen and LLaVA processor checks used no model weights or GPU inference.

All 44 files changed by this implementation pass the repository's source-export
rules and `git diff --check`. The whole-index export check still flags 31
pre-existing paths under `data/daywise_annotations_qa_v2_20261108/` for JSONL or
size rules; that previously tracked annotation package is unchanged here.

The real EgoLife fixture contains 500 QA. Offline planning produces 16,000 causal
frame requests at a 32-frame budget. Evidence planning at a 64-frame budget
accepts 498 QA and explicitly rejects QA140 and QA484 because annotated points
lie in source gaps; it does not substitute neighboring clips or drop questions.
No accuracy is implied by a successful plan.

Run the CPU checks from the repository root:

```bash
python -m unittest discover -s baselines/ours_offline/tests -v
python -m unittest discover -s baselines/ours_evidence/tests -v
python -m unittest discover -s baselines/ours_selection/tests -v
python -m unittest baselines.ours_cambrian.test_runner -v
```
