# VideoMem EgoRAG baseline adapter

This is the **official EgoRAG retrieval pipeline adapted to causal Day-video inputs**,
with explicitly documented model and video-answer differences. It is not a claim
of numerical reproduction of the EgoLife paper. Upstream is pinned to
`7a97157908757cc898c26835b718653055ecc5f5` from
https://github.com/EvolvingLMMs-Lab/EgoLife/tree/7a97157908757cc898c26835b718653055ecc5f5/EgoRAG .
Paper: https://arxiv.org/abs/2503.03803 . The original Python files are unmodified.
The export retains the seven upstream files required for method loading and provenance, plus the upstream license. Use the adapter CLI below; the upstream entrypoint files are retained as reference sources.

## Implemented path and fidelity

1. Probe local MP4s and stream content hashes. Describe 30-second **virtual clips**
   in a manifest; do not cut/re-encode whole videos. Thirty seconds is the paper's
   segment setting; the public create code actually trusts consecutive filename
   times and does not enforce this length. The adapter makes it configurable.
2. Caption actual sampled video frames using the original first-person caption
   prompt. Split caption sentences and embed text with configurable OpenCLIP
   (official default: `ViT-B-32`, `laion2b_s34b_b79k`). Persist captions, embeddings,
   summary versions and the index. No FIFO, eviction, or total-memory cap.
3. Build **10-minute -> hourly -> daily** summaries using the official prompts.
   Question-time snapshots are built only from fully observable clips. Windows
   use deterministic Day playback boundaries, while public code anchors some
   windows to the first observed item. Metadata records actual maximum source
   end, not a nominal boundary that might hide future content.
4. For ordinary questions, call the actual upstream `RagAgent.get_range` with a
   safe filtered summary interface. It selects one Day and one time range. Then
   call actual upstream `Chroma.custom_query(filter_first=True)` for cosine-ranked
   **sentence** retrieval (default top-3, possibly from fewer than 3 unique clips).
5. Preserve the English `last time` / `the last` branch: question-derived keyword
   and upper-time rewrite, top-50 sentence retrieval, chronological same-clip
   sentence context, official `get_ids` prompt and recency selection. Repair only
   its single-digit sentence-ID parser and validate offered IDs. This is a single
   retrieval plus reranking, not an iterative search loop. No additional loop is
   present in the pinned public code.
6. Revisit actual MP4 intervals using the official **video evidence** prompt.
   Feed evidence cards plus those sampled frames to a configurable visual
   answerer. The public CLI hardcodes text evidence and a text-only final helper;
   the paper describes video revisiting. Our final answerer receiving images is
   an explicit visual-answer adaptation, not a literal match to the shipped CLI.

`upstream.py` compiles unchanged named method ASTs from the pinned checkout and
loads literal upstream prompt definitions, avoiding imports of every optional
model SDK. It does not rewrite those function bodies. Per-run source/prompt/code
hashes make this reuse inspectable. `store.py` offers Chroma as the production
index engine and SQLite/NumPy as a portable alternative. SQLite is used for the
offline smoke; its exhaustive squared-L2 recency ranking is not Chroma ANN.

See [upstream_audit.md](upstream_audit.md) for exact source calls and public-code
bugs. Paper/code differences and adapter changes are intentionally visible.

## Local inputs and temporal meaning

Provide your own videos using `configs/videomem_sources.example.json` and reviewed
questions using `configs/videomem_questions.example.json`. This repository contains
no research videos, private labels, or evaluated benchmark questions. Day order and
playback offsets are explicit; wall-clock or original recording times are not
inferred from filenames. Query cutoffs must be supplied independently of labels.

Model names in `configs/videomem.json` are unset. Choose model IDs and immutable
revisions before real inference. The optional local Qwen template is described in
[local_qwen3vl.md](local_qwen3vl.md).

## Environment and credentials

Python 3.11, FFmpeg/FFprobe, NumPy, Pillow and pytest suffice for the offline smoke.
Use an isolated environment for production dependencies:

```bash
cd baselines/egorag
python -m venv --system-site-packages .venv-videomem
.venv-videomem/bin/python -m pip install 'chromadb==0.5.20' 'posthog<4' 'open_clip_torch==2.29.0'
```

Do not run an uncontrolled package upgrade in an existing training environment.
OpenCLIP weights will be needed on its first real embedding call. The smoke uses
explicit hash embeddings and a decoded-color test double instead, and is labeled
`test_only: true`. This is never an automatic fallback for unavailable models.

Copy `configs/videomem.json` to a local config and explicitly set model IDs and
immutable revisions/deployment versions for every role. Captioner, evidence,
answerer must support image inputs. Auxiliary roles can share a text endpoint.
No training of EgoGPT is required. An existing compatible local serving endpoint
can serve Qwen-VL; the API-compatible backend does not launch a model server. The optional `hf_local` backend loads models on the explicitly permitted GPUs when invoked.

Credentials and endpoint URLs are read from the environment only:

```bash
# Supply secrets through the execution environment/secret manager, not this file.
# Required variable names for the provided configuration:
# EGORAG_API_KEY, EGORAG_BASE_URL               (text model endpoint ending /v1)
# EGORAG_VISION_API_KEY, EGORAG_VISION_BASE_URL (vision endpoint ending /v1)
```

The endpoint uses the chat-completions wire format. HTTP bodies, truncations and
empty outputs are treated as execution failures; no random-answer fallback or
provider change. API retries are not automatic. Rerun identical failed items to
resume; only successful artifacts are reused. No API inference is run merely by
preparing a manifest. Unset model IDs fail clearly before preprocessing.

## Inputs and commands

Run from `EgoRAG/`; no editable installation is required. Use the selected isolated
Python for production commands. Example below uses `python` for readability.

```bash
# Source paths and offsets are explicit; all media stays in its existing location.
python -m videomem_adapter.cli prepare \
  --sources configs/videomem_sources.local.json \
  --clip-seconds 30 --fingerprint-cache videomem_runs/fingerprints \
  --output videomem_runs/manifest.json

# First use a few clips; model IDs/endpoints must already be configured.
python -m videomem_adapter.cli create \
  --config configs/my_models.local.json --manifest videomem_runs/manifest.json \
  --work videomem_runs/baseline --limit 3

# Sanitization requires a caller-supplied query_time; it never derives it from GT.
python -m videomem_adapter.cli sanitize \
  --questions /path/questions.json --output videomem_runs/questions.public.json

python -m videomem_adapter.cli summarize \
  --config configs/my_models.local.json --manifest videomem_runs/manifest.json \
  --work videomem_runs/baseline --questions videomem_runs/questions.public.json

python -m videomem_adapter.cli query \
  --config configs/my_models.local.json --manifest videomem_runs/manifest.json \
  --work videomem_runs/baseline --questions videomem_runs/questions.public.json

python -m videomem_adapter.cli answer \
  --config configs/my_models.local.json --manifest videomem_runs/manifest.json \
  --work videomem_runs/baseline --retrieval videomem_runs/baseline/query.json

# Labels are opened only here, after prediction.
python -m videomem_adapter.cli evaluate \
  --predictions videomem_runs/baseline/answer.json \
  --labels /path/labels.private.json --output videomem_runs/baseline/evaluation.json

python -m videomem_adapter.cli storage \
  --config configs/my_models.local.json --manifest videomem_runs/manifest.json \
  --work videomem_runs/baseline
```

For the **later, separately initiated full run**, replace `--limit 3` with
`--allow-full`. The query command also creates/reuses the required causal summary
snapshot, so `summarize` is useful for separate inspection but is not mandatory.
After changing processing/model settings, rerun `create` before query. Current
cache invalidation is conservative: changing even a later-stage setting can
invalidate earlier-stage caches. This costs work but avoids silently reusing a
different protocol. Appending data creates a new search index and reuses unchanged
caption/embedding artifacts, reindexing them when `create` is rerun.

Public question schema:

```json
{
  "question_id": "q001",
  "question": "Which bag did I take?",
  "choices": ["Red bag", "Blue bag"],
  "query_time": {"day_id": "Day1", "day_order": 1, "seconds": 60}
}
```

`choices` can be empty for free-text answers. MCQ evaluation accepts an exact
option letter; free-text evaluation is exact-match only, not an LLM judge.
Private labels: `[{"question_id":"q001","answer":"A"}]`.
Fields such as answer, reason, target_time, correct_idx, supplied keywords and
nested original QA metadata are dropped by the public whitelist. Five-choice
questions are supported. Duplicate IDs or mismatched prediction/label sets fail.

## Causality, resume and logs

- Entire clips crossing the cutoff are excluded, including their complete
  captions. A clip ending exactly at the cutoff is eligible; sampled frames are
  in the half-open source interval. No post-cutoff video frames are decoded for
  answer. Training/evaluation labels never determine visibility.
- Every summary has exact child clip IDs and maximum source end. Early questions
  never reuse a complete future-bearing daily summary. No arbitrary memory cap.
- Range-selector output is validated and intersected with the actual cutoff;
  unknown/future days and malformed selections are errors, not favorable retries.
- Source byte hash, source file state, clip bounds, config, effective endpoint
  hash, prompt/source code hashes and model revisions participate in identity.
  Mutable provider aliases still require an explicit revision/config epoch change
  by the operator; the adapter cannot discover an undocumented provider update.
- Missing preprocessing is `MemoryNotReady`, not “no relevant memory”. Technical
  failures remain in results/evaluation; they are not dropped from denominators.
- `model_calls.jsonl`: protocol namespace, role, model/revision, text prompts, image hashes/times,
  response, usage when available, timing, status. No credential or image-base64
  logging. A retrieved frame's bytes and explicit Day/clip/playback-time labels are sent to the configured endpoint.
- `query.json`: public question, cutoff, generated keywords, summary IDs,
  selected range, retrieved sentence/clip IDs, exact file interval and distance.
- `answer.json`: evidence, prediction, actual video intervals and decoded frame
  timestamps/hashes, errors and latency. Labels appear only in evaluation output.
- `storage.json`: original video bytes separately from logical captions,
  summaries, embedding records and physical SQLite/Chroma sizes. Logical record
  byte sizes are not additive physical disk measurements.

## Offline validation

```bash
python -m pytest -q tests/test_videomem_*.py
python -m videomem_adapter.cli smoke --work videomem_runs/smoke_v1
.venv-videomem/bin/python -m videomem_adapter.cli smoke \
  --work videomem_runs/smoke_chroma_v1 --index-engine chroma
```

Smoke uses a real four-second red/blue MP4 generated for the test. It captions both
segments, builds a full-video summary, asks a question at the halfway point, and
checks future summary exclusion and actual visual evidence/answer frame delivery.
A second pass must make zero new model calls. It is a **contract test**, not a
benchmark result or real LLM reasoning evaluation. Private real-media validation reports are excluded from this export.

Not yet validated: real caption/summary/answer model quality, OpenCLIP retrieval
quality, full multi-day runtime/cost, final Day render alignment, real memory QA
cutoffs, correctness of original answers after 1 Hz conversion, audio-dependent
questions (this first adapter is visual-only), or numerical reproduction of paper
scores. These are not implied by passing tests.
