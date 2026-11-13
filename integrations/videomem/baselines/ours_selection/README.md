# ours offline AKS / VideoTree + target VLM

This package connects the existing actual AKS selector and VideoTree kernels to
**Qwen3-VL-8B** or **LLaVA-OneVision-7B** through the shared native raw-video backend
in `baselines.ours_offline`. No GPU evaluation was run when adding this package;
CPU tests cover the integration contracts. It is an offline evidence-selection
implementation, not a bounded online-memory method or a pretrained accuracy claim.

The target is **our dataset**, with QA not yet delivered. Both methods use its
eventual common QA file, source index and scorer. EgoLife is only a separate
validation dataset; this code imposes no EgoLife wall-clock, 20 fps or seven-day
assumptions. Sources use an explicit global timeline, arbitrary positive CFR fps,
and an optional positive day label.
Only question text/options reach model code; GT/reason/target_time do not. All
answers reread selected original RGB frames and use the answer model's native
video processor, without reusing its visual embeddings.

## Declared method variants

- **AKS:** reuses `streambench.adapters.aks.adaptive_select` and the released
  CLIP scoring branch (`openai/clip-vit-base-patch32`). Each QA scores **only the
  causal prefix** of image features with the question alone, then selects raw
  frames for either target VLM. This does not implement BLIP/SeViLA or reproduce
  the paper's default BLIP + LLaVA setting.
- **VideoTree:** reuses `streambench.adapters.videotree.build_tree`, including
  native cosine k-means, Ward expansion and the existing documented ID repairs.
  Features come from the actual local EVA-CLIP encoder. A separately configured
  **local text model** supplies relevance scores. The resulting representative
  **raw frames**, not tree-generated answers, go to the target VLM. This is an
  explicitly labeled selected-frame VLM variant, not exact GPT-4/LaViLa paper
  reproduction. No paid API is called or implicitly configured.

The VideoTree path requires externally generated, verified, question-independent
captions. It does not pretend that the released VideoTree code includes LaViLa
inference, and it never uses QA `reason`, GT, or annotation targets as captions.

Runtime requires the shared target-backbone environment and `requirements.inference.txt`;
VideoTree clustering additionally requires SciPy. Install matching torch/torchvision
and the verified EVA checkpoint's own dependencies for the target host. No dependency
or model download is performed by these commands.

## Work before QA arrives

1. Prepare the source index using `baselines.ours_offline` (see that package's
   README). Provide actual source start times, durations and native fps; gaps and
   overlaps are explicit. The index carries verified file stat anchors.
2. Run `prepare` below to encode the query-independent CLIP/EVA features. Neither
   a QA file nor the answer model is required by this command.
3. For VideoTree, prepare external question-independent captions with the evidence
   availability contract below. Do not derive them from QA or ground-truth answers.
4. Run the data-only readiness check. It reports `awaiting_QA` and loads no models:

```bash
python -m baselines.ours_selection validate \
  --root /path/to/ours-dataset --sources /path/input/sources.json \
  --archive /path/aks-archive
```

When QA arrives, add `--canonical /path/input/questions.json`. VideoTree also needs
`--captions /path/aligned-captions.json`. An empty `questions` array is accepted by
`validate` as QA-pending, but cannot start a fabricated or empty evaluation. Model
compatibility is checked by `evaluate`; `validate` checks only data readiness.

## Prepare selection features

Copy and edit a config under `configs/`. These are templates: all paths and
placeholder revisions must be replaced with verified local artifacts. Model
weights are not downloaded. The revision must be an immutable commit; the loader
hashes local encoder files. These hashes identify the local bytes; the user must
also establish their official provenance when acquiring them. The answer VLM
requires the shared verified ledger contract in `ours_offline`.

```bash
python -m baselines.ours_selection prepare \
  --config /path/aks.json \
  --root /path/to/ours-dataset \
  --sources /path/input/sources.json \
  --candidate-fps 1 --batch-size 8 --output /path/new-aks-archive
```

For VideoTree, use its config and a separate archive. The archive contains feature
rows and actual decoded source timestamps, never answer-model visual embeddings.
Its feature extraction is independent of questions. Candidate sampling uses a
source-local grid and clips overlapping a subsequent source are trimmed. Sparse
sampling, native resize and target-backbone settings are recorded; they are not
claimed to match every original paper preprocessing setting.

Source receipt anchors, archive file hashes, feature shape and chronology are
checked. A changed source or archive fails evaluation. Archive completion is
explicit, so a partial extraction cannot be mistaken for a complete index.
Feature archives are **unbounded offline storage**, not a 16 GiB memory baseline.

## VideoTree caption contract

Provide a JSON object with one caption entry per archive row:

```json
{
  "schema": "ours_videotree_captions_v1",
  "archive_sha256": "SHA256_OF_ARCHIVE_METADATA_JSON",
  "producer": {
    "model": "explicit-caption-model-id",
    "revision": "verified-caption-model-revision",
    "question_conditioned": false
  },
  "captions": [
    {
      "global_frame_id": 0,
      "global_timestamp": 0.0,
      "observed_until": 0.0,
      "available_at": 0.0,
      "text": "Visible scene description."
    }
  ]
}
```

For a caption based on a longer video interval, `observed_until` must be that
interval's **end**, and `available_at` must be no earlier. A caption enters a QA's
tree only when both its source frame and `available_at` are at or before the query
cutoff. Future interval descriptions are excluded before clustering/relevance.
The exporter is responsible for truthfully declaring the observed evidence range;
text provenance cannot be inferred merely from its wording. Missing, duplicate,
misaligned or question-conditioned caption entries fail validation.

## Evaluate with either answer backbone

```bash
python -m baselines.ours_selection evaluate \
  --config /path/aks.json \
  --root /path/to/ours-dataset \
  --sources /path/input/sources.json \
  --canonical /path/input/questions.json \
  --archive /path/aks-archive --output /path/new-aks-results \
  --shard 0 --shards 1
```

For VideoTree use its config/archive and add `--captions /path/aligned-captions.json`.
To change answer backbone, change `answerer.backbone`, model path/revision and
ledger together. Supported backbone values are `qwen3_vl` and `llava_onevision`.
`device` refers to the CUDA index visible to the process; no GPUs are auto-selected
and no other process is stopped. Distinct shards need distinct output directories
and appropriate device placement. CPU-only inspection of `--help` loads no models.

VideoTree can select more than the configured `max_answer_frames`. That QA returns
an explicit error instead of silently truncating representatives. Increase the
explicit read budget or deliberately adjust tree parameters; both are recorded.
AKS may underfill its native quotas; the runner does not invent fill frames.
The native video answer path requires at least two selected frames. A smaller
budget is rejected, and a selector returning one frame produces an explicit QA
error without duplicating it.

Outputs contain per-QA method/ID/cutoff, actual timestamps, chosen candidate IDs,
prediction/raw output/GT/correct, selection/decode/answer latency, status/errors,
and provenance. Incomplete generation and errors count as incorrect in the fixed
selected-QA denominator. A full-benchmark claim requires merging unique QA IDs and
checking coverage against the delivered manifest; no fixed QA count is assumed.

```bash
python -m unittest discover -s baselines/ours_selection/tests -q
```

Readiness distinguishes the missing pieces: `qa_status=awaiting_QA`,
`selection_features=not_prepared`, and (for VideoTree) `captions=missing` are
separate fields. `model_artifacts=not_checked` is deliberate in data-only validation.
Even `status=ready_inputs` means **ready for model preflight**, not tested model
weights or a completed inference run; `ready_to_evaluate` remains false until that
runtime work is performed by `evaluate`.

The actual vendored VideoTree cosine-k-means/Ward kernels were exercised through
this new selected-raw-frame VLM bridge on CPU: 12 tests passed, zero skipped, with
Python from `/home/common/miniconda3/envs/laser_env`, torch 2.5.1+cu121, SciPy 1.15.3
and NumPy 2.2.6. The relevance responses and answer model were controlled fixtures;
the tree kernels were not mocked. This confirms causal filtering, real hierarchy
construction, choice mapping and JSON output integration, not pretrained accuracy.
On environments lacking SciPy, the native-kernel test explicitly skips; the other
11 contract tests still run. Answer and selection/relevance model initialization
both happen before per-query latency measurement.
