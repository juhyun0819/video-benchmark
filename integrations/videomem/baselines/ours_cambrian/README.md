# Cambrian-S on the ours continuous stream

This runner connects the existing native `CambrianSMethod` adapter to **ours**, the
target dataset. EgoLife/JAKE is only a compatibility test fixture. Dataset inputs
use generic global timestamps and per-file frame rates; day labels are optional.
The runner supplies one chronological raw-video stream and independent QA branches,
with no extra training, saved visual-feature cache or day-boundary state reset.

**Implementation and CPU orchestration tests are complete; pretrained Cambrian-S
inference and ours accuracy/performance have not been run or validated.**
See [native adapter notes](../streambench/docs/streambench_recurrent.md) for the
source audit, native policy and environment requirements.

## Inputs and QA-pending preparation

Prepare the verified source index with the shared [ours input tools](../ours_offline/README.md).
Each source has a relative `path`, global `start_time` in seconds, positive `duration`,
`fps`, `frames`, and verified `file_state`; `day` is optional. Each labeled QA needs
`qa_id`, global `query_time`, `question`, `choices` and `correct_idx` or `answer`.
The manifest is bound to the common `baselines.streambench.scoring.score_answer`
by `scorer_sha256`. The same manifest and scorer serve all ours methods.

QA is not available yet. A preflight can validate source files, plan the sampling
and report local model-directory availability without loading any model or GPU:

```bash
python -m baselines.ours_cambrian \
  --root /absolute/ours/videos \
  --sources /absolute/ours/prepared/sources.json \
  --config /absolute/cambrian-ours-config.json \
  --output /absolute/new-cambrian-plan --fps 1
```

Without `--manifest`, this writes `status=awaiting_qa`, `plan.json` and `status.json`.
`--plan-only` requests the same CPU-only preflight when a manifest is supplied.
Preflight checks paths and input metadata, not model architecture or checkpoint
integrity. **It does not consume observations or create a memory checkpoint.**
Supply the final QA manifest before starting inference: a future memory checkpoint
must never be used to answer retrospectively added earlier QA.

## Native model and environment

- Checkpoint: `nyu-visionx/Cambrian-S-7B-LFP`, revision
  `d04f69849c1ddbf7e62a7790f2461ef549d4df8e`.
- Source checkout: `cambrian-mllm/cambrian-s`, revision
  `058b1b4c96846a229d1c9bb69da6ecf5ceee4a6b`.
- Use the official isolated environment with `transformers==4.37.0`, PyTorch,
  PyAV and native project dependencies. The separate SigLIP vision tower must
  also be cached locally. The CLI sets Hugging Face offline mode before native
  imports and never downloads dependencies or models.
- Copy `config.example.json` and supply local source/model directories. During an
  actual run, the adapter verifies source revision and pretrained architecture.
  Verify weight-file provenance externally; the runner does not claim a full
  model-weight content audit from a directory name.

The explicit default is a **1 fps source-local sampling grid**, quantized to original
CFR frame positions. It is a dataset sampling setting, not a claim about every
native evaluation recipe. Native frame rates may differ by source; sampling fps
must not exceed any source's native rate. Exact fractional probe metadata is
supported. Overlapping clip tails belong to the next clip; gaps and supplied
global offsets are retained. Decoded PTS must match the selected indexed frame
and remain causal. VFR inputs require conversion/indexing to the documented CFR
source contract; the runner rejects mismatched decoded timestamps.

The example `16 GiB` budget applies to retained observation state, not total VRAM.
Native sensory/long-term memory and continuation state keep the adapter's
accounting and eviction policy. Audio is not supplied.

## Run and resume after QA arrives

```bash
python -m baselines.ours_cambrian \
  --root /absolute/ours/videos \
  --manifest /absolute/ours/prepared/questions.json \
  --sources /absolute/ours/prepared/sources.json \
  --config /absolute/cambrian-ours-config.json \
  --output /absolute/new-cambrian-run --gpu 2 \
  --fps 1 --chunk-frames 8 --checkpoint-every 10
```

`--model`, `--upstream` and `--budget-gib` override corresponding JSON config fields.
`--max-questions 2` stops after two new answers and writes a continuation checkpoint.
The inference CLI refuses occupied GPUs and existing output directories.

Resume with the **same manifest, source index, effective config, sample rate and
optional QA day selection**, adding `--resume /previous-run/checkpoint` and a new
output path. Completed QA rows are preserved. Checks include native frame count,
last timestamp, input/config identity, prediction coverage, common scores and
native checkpoint integrity. Changed identities, including newly added earlier QA,
are rejected. A resume checkpoint must be trusted local Torch serialization.

`--days` only filters QA with optional day labels; it never drops earlier source
observations or resets memory. All indexed sources are observed from the beginning,
and the source tail is processed after the final QA. No fixed day count or
24-hour/day timestamp convention is imposed.

## QA and artifacts

Only public questions and choices reach the model. Labels, evidence targets and
annotation diagnostics do not enter prompts. Each QA uses an explicit native
snapshot/restore with tensor/metadata hashes verifying the restored continuation
state. Native recall and greedy decoding remain unchanged.

Outputs include:

- `predictions.jsonl`: method, QA ID, cutoff, actual observation timestamps,
  prediction/raw output, GT, common score, errors, retrieval provenance,
  native answer latency and total latency including state snapshot/audits/restore.
- `updates.jsonl`: decode/update timing, actual timestamps, memory accounting and
  native compression/eviction events.
- `status.json`, `provenance.json`, `resume-audit.json` and native checkpoints.

The native adapter returns decoded text rather than token/EOS metadata; this
runner does not invent those fields. `status=ok` means the native call and
causal/state checks succeeded, not that the answer was correct.

```bash
python -m unittest baselines.ours_cambrian.test_runner -v
```

Tests cover causal cutoffs, optional-day and mixed-fps timelines, full prior-source
observation, QA mutation/error isolation, exact checkpoint continuation, source and
score integrity, and QA-pending preflight without GPU access. EgoLife-style day
transitions are test fixtures, not assumptions about ours.
