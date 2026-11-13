# AKS streaming adapter

AKS here means Tang et al., **Adaptive Keyframe Sampling for Long Video
Understanding**, CVPR 2025. The
[primary paper page](https://openaccess.thecvf.com/content/CVPR2025/html/Tang_Adaptive_Keyframe_Sampling_for_Long_Video_Understanding_CVPR_2025_paper.html)
links its official [repository](https://github.com/ncTimTang/AKS), inspected at
`b0b8a58fedf1d05d78151e2969cecde60e83721d`.

| Encoder | Representation | Persistent state | Budget control | Minimum | QA read | Training |
|---|---|---|---|---|---|---|
| Official AKS CLIP branch, `openai/clip-vit-base-patch32` | Unnormalized projected image features plus lossless sampled PNGs | Features, images, timestamps, selector parameters, model identities | Native retain-all; reject overflow | Measured metadata, then grows with every sample | All CLIP feature rows; selected PNGs into Qwen2-VL | None |

## What upstream actually implements

The official `feature_extract.py` supports BLIP-large image-text matching,
CLIP-B/32 cosine similarity, and SeViLA scoring. Its
`feature_extract_store.py` can store CLIP features, but the exported BLIP
`itm-e` value is a question-conditioned cross-attention output. That BLIP output
cannot be constructed during a question-free observation stage.

This adapter explicitly implements the released **CLIP scoring variant**. It
calls the same `CLIPModel.get_image_features`,
`CLIPModel.get_text_features`, and cosine similarity operation. The question
alone is scored; answer labels and choices do not enter the CLIP text encoder.
Image encoding runs during chronological ingestion, while text encoding and
selection run only during QA. No BLIP or SeViLA substitution is silently made.

The official evaluation includes LLaVA-Video, LLaVA-OneVision and Qwen2-VL
backbones. This factory implements the supported
`Qwen/Qwen2-VL-7B-Instruct` path using the same Qwen2-VL model/processor/generate
APIs as the official `evaluation/qwen2_vl.py`. It does not claim to reproduce
the paper's default BLIP + LLaVA configuration. Both checkpoint directories and
immutable revisions must be supplied explicitly. No model is downloaded, and
no architecture is reduced to meet B.

## Native selection, storage and read budgets

AKS optimizes **which frames a question reads** from an already accessible
video. It provides no question-independent streaming eviction policy. A
faithful wrapper must retain the input candidates until the question arrives:

1. Encode every common sampled image without a question; retain its actual
   feature dtype and dimensions.
2. Store the same sampled RGB pixels losslessly as PNG, with the complete common
   timestamp record. Original video files are never reopened at QA.
3. At QA, compute cosine relevance to the question, normalize the score vector,
   and recursively split intervals whose high-score mean does not sufficiently
   exceed the interval mean.
4. Allocate `floor(max_num_frames / 2**depth)` selections per leaf; take its
   highest-scoring frames and return their chronological IDs.
5. Decode only these retained PNGs and pass them to the configured Qwen2-VL
   processor/answer model.

`max_num_frames` is a read limit R. It does not lower storage B. If retained
features, PNGs and metadata exceed B, `finalize_chunk` reports
`unsupported_under_budget`. No FIFO, reservoir, clustering or importance
eviction is invented under the AKS name.

The default selector settings are native `max_num_frames=64`, `ratio=1`,
`t1=0.8`, `t2=-100`, `all_depth=5`. Integer ratio truncation, stable tie order,
and floor quotas match the reference. Non-power-of-two frame limits can yield
fewer frames than requested; no fill heuristic changes this behavior.

Because the inspected AKS root has no LICENSE file, its source was not vendored.
The short numerical algorithm is independently implemented from its described
operations. A read-only parity probe executed the original `meanstd` function
in isolation and compared 2,484 nondegenerate cases over sequence lengths,
capacities, depths, thresholds and sampling ratios: selected IDs matched in
every case. Audited source identities:

| Source | SHA256 |
|---|---|
| `frame_select.py` | `8771504403f923bd8efce7dcb811f3c5cda0a296b7cf731a4039e6edd79e68d5` |
| Exact original `meanstd` function text | `7d207003b47eecee332d871e6c3f23506fc8795f2abba8206f62b005b93d6298` |

Explicit correctness adaptations: constant score vectors normalize to zeros
instead of NaNs; nonfinite scores are rejected; `all_depth` cannot exceed
`floor(log2(max_num_frames))`, avoiding the upstream zero-quota failure; float
thresholds are accepted instead of the upstream CLI's erroneous integer type.
Common manifest sampling replaces upstream `int(fps)` indexing and independent
video decoding. Qwen2-VL still applies its required resizing/temporal padding to
the already selected images. These preprocessing differences must be recorded
when comparing paper results.

## Checkpoint and accounting schema

Schema 1 stores `features`, `frames_png`, all timestamps, selector parameters,
upstream revision, method variant, full local checkpoint identities and the
test-only flag. Features and PNGs are counted in `visual_memory_bytes`; the
remaining continuation data is counted in `metadata_bytes`. Both live CPU/GPU
copies would count; this wrapper keeps one persistent CPU feature archive.
Frozen model weights and query-time tensors are excluded. There are no partial
frame buffers, retained score vectors, query caches, or hidden disk video access.

Local checkpoint identities include content hashes of all relevant existing
weight/configuration/tokenizer files, their supplied immutable revisions and
inference settings. Later calls check file identities/stat changes; changing a
model/config identity prevents checkpoint restoration. Remote model provenance
must still be verified when acquiring a checkpoint: a declared revision alone
is not evidence that arbitrary local bytes came from that upstream revision.

`query_read_bytes` reports the complete stored feature tensor used to score all
candidates, plus selected PNG bytes and selected timestamp payloads. Responses
include `retrieved_frame_ids`, `retrieved_items`, selected relevance scores and
the actual selected count. Visual-token counts are `null` because the adaptive
processor determines them; they are not guessed from frame counts. Query trees,
scores and generation KV are temporary. The common runner restores identical
checkpoint/RNG state for every question.

## Local configuration and commands

Create an `aks-stream.json` config with the following structure, replacing paths
and revision placeholders with verified local snapshots:

```json
{
  "method": "aks",
  "memory_budget_bytes": 268435456,
  "max_num_frames": 64,
  "ratio": 1,
  "t1": 0.8,
  "t2": -100,
  "all_depth": 5,
  "scorer": {
    "checkpoint": "/absolute/local/clip-vit-base-patch32",
    "revision": "REPLACE_WITH_VERIFIED_40_CHARACTER_REVISION",
    "device": "cuda:0",
    "dtype": "float32"
  },
  "answerer": {
    "checkpoint": "/absolute/local/Qwen2-VL-7B-Instruct",
    "revision": "REPLACE_WITH_VERIFIED_40_CHARACTER_REVISION",
    "device": "cuda:0",
    "dtype": "bfloat16",
    "max_new_tokens": 128
  }
}
```

```bash
python -m baselines.streambench observe --config aks-stream.json --manifest frames.jsonl --output runs/aks --chunk-size 30 --stop-after-day 1
python -m baselines.streambench evaluate --config aks-stream.json --checkpoint runs/aks/checkpoint_D1 --questions questions_day1.json --output answers-aks.jsonl
python -m baselines.streambench sweep --config aks-stream.json --manifest frames.jsonl --output pilot-aks --budgets-mib 64 128 256 512 1024 --stop-after-day 1
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q baselines/streambench/tests/test_streambench_aks.py
```

The adapter imports model libraries lazily and uses `local_files_only=True`.
The focused tests cover selection arithmetic/coverage, native underfilled
quotas, constant-score handling, stored pixels/features, overflow rejection,
query isolation, configuration mismatch, and fresh-process Day1→Day2 resume.
Their small injected encoder/answerer are explicitly test-only. No pretrained
weights were downloaded and no real-model QA quality or seven-day run has been
validated in this session.
