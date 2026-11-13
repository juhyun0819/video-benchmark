# EgoGPT audiovisual caption archive

This adapter prepares resumable, query-independent captions with the released
EgoGPT model and the caption call from
[EgoLife's EgoRAG implementation](https://github.com/EvolvingLMMs-Lab/EgoLife/blob/7a97157908757cc898c26835b718653055ecc5f5/EgoRAG/egorag/models/egogpt.py).
It produces captions and their provenance; it does not run retrieval or answer
benchmark questions. The commands below describe how to run it, not evidence
that generation has finished.

## Protocol and scope

- **Model:** `lmms-lab/EgoGPT-7b-EgoIT-EgoLife`, revision
  `6103a335329cca604a1cacc72c88e52fb4f23738`, with its Whisper large-v3 speech
  encoder and SigLIP vision tower. The prepared runtime pins the SigLIP revision
  to `9fdffc58afc957d1a03a25b10dba0329ab15c2a3` and upstream source revision to
  `7a97157908757cc898c26835b718653055ecc5f5`. `job.json` records the runtime
  actually supplied to the command; worker provenance records loaded package
  versions, upstream revision, GPU, attention implementation, and generation
  configuration.
- **Windows:** continuous, non-overlapping 30-second intervals within each day.
  Each day's final shorter interval is retained. A clip never crosses a day
  boundary. Source and concatenated global times use exact fraction strings.
- **Video:** source-local samples at `clip_start + k` seconds, strictly before
  `clip_end`, decoded with the native Decord CPU reader. The current inventory
  is 25 fps; the adapter requires a validated integer frame rate. Frames retain
  their original dimensions until the model's own image processor. One whole
  clip is passed to the model. A full 30-second clip has 30 frames: there is
  **no 16-frame cap** from the separate demonstration script.
- **Audio:** native `librosa.load(original_source, sr=16000)` produces mono
  float32 audio. The source-local clip interval is then sliced, followed by
  native Whisper `pad_or_trim` and a 128-bin log-mel spectrogram. Video and audio
  are both supplied to the model. Missing audio, changed media, or incompatible
  stream origins cause an error rather than an audio-free substitution.
  Full-source resampling is retained by explicit user choice for fidelity to
  the original preprocessing; this mode has `strict_causal: false`.
- **Prompt:** the exact caption instruction below, prefixed with the official
  `<image>` and `<speech>` markers, using the actual default `qwen_1_5`
  conversation system. The upstream adapter accepts a `system_message`
  argument but never applies it; this adapter preserves that behavior.
- **Generation:** native FP16 loader, evaluation mode, `do_sample=False`,
  `temperature=0.5`, `max_new_tokens=4096`, and the tokenizer EOS token.
  Temperature is inactive under greedy decoding. Other generation settings
  come from the checkpoint. No caption-length or frame-count optimization is
  substituted.
- **Memory:** with the validated `offload_idle_encoders` policy, the original
  encoders run on CUDA in FP16 for the complete clip. After native multimodal
  input construction finishes, only the now-idle speech/vision weights and
  buffers move to CPU. Prepared inputs, projectors, language decoder, attention,
  logits computation, and generation settings are unchanged. Encoders return to
  CUDA before the next clip. This releases about 1.93 GiB and avoids the native
  Transformers full-logits clone exceeding the A5000's memory. The helper,
  unchanged native call, validation observations, and runtime identity are
  hash-bound to the job.
- **GPU libraries:** worker entry points restart with inherited `LD_LIBRARY_PATH`
  and `LD_PRELOAD` removed before importing Torch, so the pinned wheel resolves
  its own CUDA/cuDNN libraries. This prevents the host's CUDA11 cuDNN8.0.5 from
  mixing with the Torch2.1.2 CUDA12.1/cuDNN8.9.2 environment. The loaded cuDNN
  version and remaining library overrides are recorded in worker provenance.

```text
Imagine you are the character in the video, describing from a first-person perspective what you saw, and everything that happened over time. Use I as the subject.
```

Two dataset adaptations are explicit. First, the released adapter uses its
raw `HHMMSSFF` timestamp arguments directly as seconds when slicing audio;
this implementation fixes that bug by using source-local seconds. Second,
the manifest partitions these continuous daily recordings into exact 30-second
windows and preserves each final partial window. Therefore this is a pinned
implementation of the released caption protocol with documented adaptations,
not a claim of byte-identical captions to historical paper runs.

The optional waveform cache saves the result of the same full-source librosa
load as an uncompressed float32 `.npy` file. Clips read slices through a memory
map; values are preserved without repeated decoding/resampling. This cache is
audio preprocessing, not a visual feature archive or model KV cache. Receipts
record source state, preprocessing version, sample count, and payload identity.
Full-source resampling can let samples just after a clip boundary affect
resampled samples just before it. Slicing the resulting waveform at the clip
end does not remove that dependence. The archive therefore records the nominal
clip end separately from conservative availability at the full source's end.

## Commands

Run from the repository root. Set these paths to the prepared environment,
runtime receipt, and writable output locations. The launcher can start with
an existing project Python while the native runtime is still being prepared;
it uses the interpreter recorded in `runtime.json` for subsequent work.

```bash
cd /home/zzuhyun/videomem-baselines
CAPTION_LAUNCHER_PYTHON=/path/to/available/project/python
CAPTION_PYTHON=/path/to/egogpt-runtime/bin/python
CAPTION_RUNTIME=/path/to/verified/runtime.json
CAPTION_INVENTORY=/home/zzuhyun/qwen3-offline-local/inventory.json
CAPTION_OUTPUT=/path/to/caption-archive
CAPTION_AUDIO=/path/to/native-audio-cache

"$CAPTION_LAUNCHER_PYTHON" -u -m baselines.egogpt_caption.launch \
  --inventory "$CAPTION_INVENTORY" --runtime "$CAPTION_RUNTIME" \
  --output "$CAPTION_OUTPUT" --audio-cache "$CAPTION_AUDIO" \
  --gpus 0 1
```

Keep this long-running launcher under a process supervisor or a detached
session if the terminal may disconnect. It waits for media dependencies,
prepares native audio, waits for the verified model, runs a four-clip smoke
test, audits it, launches one full worker per requested physical GPU, and
audits complete archive coverage. The smoke selection contains the first
two clips and final clip of the first day, plus the final clip of the last
day, with duplicates removed for a smaller inventory. Smoke results use a
separate sibling directory named `${CAPTION_OUTPUT}_smoke`.

For individual stages, use the commands below. `prepare` requires the runtime
receipt's status to be `ready`. Do not launch manual workers concurrently with
the pipeline for the same archive.

```bash
"$CAPTION_PYTHON" -m baselines.egogpt_caption prepare \
  --inventory "$CAPTION_INVENTORY" --runtime "$CAPTION_RUNTIME" \
  --output "$CAPTION_OUTPUT" --workers 1

"$CAPTION_PYTHON" -m baselines.egogpt_caption audio-cache \
  --inventory "$CAPTION_INVENTORY" --output "$CAPTION_AUDIO"

"$CAPTION_PYTHON" -m baselines.egogpt_caption worker \
  --output "$CAPTION_OUTPUT" --index 0 --gpu 0 \
  --audio-cache "$CAPTION_AUDIO" --port 29580
```

For a small initial run, append `--limit 1` to the worker command. Remove it
for the complete assigned shard. `--clip-ids day1_000000 ...` selects named
clips belonging to that worker. These options restrict work; they do not
change the prepared protocol. Re-running a worker skips verified complete
records and retries unfinished clips.

For multiple GPUs, prepare a new output directory with the intended
`--workers N`, then launch indices `0` through `N-1`, each with its own GPU
and unused `--port`. Assignment is manifest position modulo worker count.
Do not change worker count, code, runtime, or inventory inside an existing
prepared job: resume verifies their identity.

Audit all manifest clips before reporting archive completion:

```bash
"$CAPTION_PYTHON" -m baselines.egogpt_caption.audit \
  --output "$CAPTION_OUTPUT"
```

For a selected smoke subset, add `--smoke` and its exact clip IDs:

```bash
"$CAPTION_PYTHON" -m baselines.egogpt_caption.audit \
  --output "${CAPTION_OUTPUT}_smoke" --smoke \
  --clip-ids day1_000000 day1_000001 day1_001300 day7_001088
```

These four IDs are for the current seven-day inventory. The audit verifies
manifest identity, complete records, receipts, and referenced provenance;
`--smoke` additionally checks recorded input shapes, dtypes, audio presence,
and preprocessing policy. It returns exit code 0 only when the requested
scope is complete, and 2 for missing or incomplete records. A `complete`
result with `--clip-ids` covers only that subset. Only an audit without
`--clip-ids` establishes full manifest coverage.

## Pipeline state and current run

`<output>/pipeline.json` records the live `phase`, launcher PID, heartbeat,
runtime and output paths, and active child or worker PIDs when applicable.
These phases distinguish preparation from actual caption inference:

| Phase | Meaning |
| --- | --- |
| `waiting_for_audio_runtime` | Waiting for `media_ready` or a fully ready runtime; no caption inference |
| `preparing_native_audio` | Loading and caching native full-source waveforms |
| `waiting_for_verified_model` | Audio preparation finished; waiting for runtime status `ready` |
| `preparing_smoke` | Preparing the separate smoke manifest and job |
| `smoke_inference` | Running the selected real-media caption clips |
| `verifying_smoke` | Auditing smoke completion and recorded protocol metadata |
| `preparing_full_job` | Preparing the full archive and worker assignments |
| `captioning` | Full workers are running |
| `auditing_full_job` | Verifying every manifest clip after workers exit |
| `complete` | Full audit passed and every worker exited successfully |
| `error` | Pipeline stopped; inspect `failed_phase`, error details, and logs |

The initial `original_v1` smoke stopped on full 30-frame CUDA memory errors;
its failed outputs are retained for inspection. The replacement `original_v2`
uses the validated idle-encoder offload policy. Its four-clip production smoke
passed (three EOS, one native length stop), and all eight full-job workers were
observed running on 2026-11-11. The 7,394-clip archive was still in progress at
that observation. Consult its live status file
for actual progress; these paths alone do not establish that workers are active:


| Item | Path |
| --- | --- |
| Full output | `/convei_nas2/zzuhyun/hyunjun/caption_archives/egogpt_egolife_day1_7_original_v2` |
| Pipeline state | `/convei_nas2/zzuhyun/hyunjun/caption_archives/egogpt_egolife_day1_7_original_v2/pipeline.json` |
| Native audio cache | `/convei_nas2/zzuhyun/hyunjun/audio_caches/egogpt_day1_7_librosa_v1` |
| Runtime receipt | `/home/zzuhyun/egogpt-caption-work/runtime.json` |
| Initial launcher process receipt | `/home/zzuhyun/egogpt-caption-work/pipeline-v2-process.json` |
| Launcher log | `/home/zzuhyun/egogpt-caption-work/pipeline-v2-launch.log` |

Stage logs live under the full output's `logs/` directory, including
`audio-cache.log`, `smoke-worker.log`, `smoke-audit.log`, `worker_XX.log`,
and `full-audit.log`. The initial process receipt records the launch command;
`pipeline.json` and current processes determine whether it is still running.

## Outputs and completeness

| Artifact | Contents |
| --- | --- |
| `job.json` | Inventory, runtime, protocol, source hashes, worker count, job identity |
| `manifest.json` | Every clip's exact source/global interval, sampled raw frame ordinals, counts |
| `captions/<clip_id>.json` | Caption text, generated/input token IDs, rendered prompt, media metadata, clip provenance, nominal clip end, conservative source-end availability, completion status |
| `receipts/<clip_id>.json` | Caption record hash and matching job identity |
| `provenance/<hash>.json` | Actual runtime and loaded model generation configuration, referenced by each caption's `provenance_hash` |
| `status/worker_XX.json` | Worker progress or failure state |
| `errors/<clip_id>.json` | Failure details when a clip raises an exception |
| `pipeline.json`, `logs/` | Orchestrator phase, heartbeat, and stage logs when using `launch` |
| Audio cache `<day>.npy`, `<day>.json` | Native waveform and preprocessing receipt, stored at `--audio-cache` |

A completed native generation contains nonempty text and either ends in EOS or
returns normally at the exact official 4096-token limit. Records distinguish
`finish_reason: eos` from `finish_reason: length`; the latter has `truncated: true`
and retains the original text and tokens, including any repetition. This matches
the released adapter's bounded generation rather than retrying the same capped
output indefinitely. Short non-EOS outputs, exceptions, and inconsistent stop
metadata remain incomplete. Audits report EOS and length-stop counts separately.
Workers stop after three consecutive failed or incomplete clips, preserving
records for inspection.
Caption files and receipts are written atomically; a record is reusable only
when its receipt, hash, completion status, and job identity match. A worker
restricted by `--limit` or `--clip-ids` reports `selection_complete` when that
selection is smaller than its full shard. Even a worker's `complete` status
only establishes its own shard. Use the full audit to establish whole-archive
coverage; receipt counts or a subset audit do not establish that result.

## Memory fidelity validation

On the pinned A5000 runtime, a 20-frame clip matched the raw path exactly in
prepared multimodal inputs, actual LLM inputs and positional metadata, full
prefill logits, every prefill KV tensor, all generated token IDs, final KV, and
final logits. Two 30-frame clips matched all first-prefill comparisons exactly.
Both offloaded 30-frame native generations returned without OOM; measured peak
allocated memory was 21.245 GiB. One ended in EOS; the other returned at the
unaltered 4096-token limit with repetitive text, which is retained and flagged.

The raw 30-frame references were intentionally stopped after prefill, before
the original full-logits clone would run out of memory. Full raw 30-frame decode
equality was therefore not empirically tested. These are checks on three clips,
not an exhaustive proof or a claim of byte-identical historical paper outputs.
The source audit confirms that offload moves only idle encoder storage after
its last use; all language-model computation remains the original path.

Evidence: `/home/zzuhyun/egogpt-caption-work/memory-validation.json`,
`offload-probe-report.json`, and `offload-probe-summary.json` in the same directory.
The original probe's coarse `failed` status came from its former EOS-only policy;
all tensor equality checks passed. The validation receipt explicitly records
this distinction and preserves the original observations unchanged.

## Preprocessing availability and later EgoRAG stages

`nominal_clip_end_global_seconds_fraction` is the actual end of the 30-second
or final partial clip; video sampling excludes that end frame. However, the selected audio
preprocessing reads and resamples the full original source before slicing.
The archive is marked `strict_causal: false`: finishing a clip does not make
its caption strictly causal at that clip's end. This is the user-selected
tradeoff in favor of the original full-source preprocessing.

For a query at global time `t`, nominal clip-end filtering alone cannot
establish strict sample causality. Conservative availability uses the full
source's end, recorded as
`conservative_available_at_global_seconds_fraction`. This time must also be
at or before `t` when enforcing that conservative rule. Do not silently treat the nominal end
as a guarantee that no later audio influenced the caption. Compare recorded
times as fractions, not rounded display timestamps. Generation itself does
not use questions, answers, or query-time labels.

Captions and provenance are sufficient for this preparation stage. A complete
EgoRAG baseline later also needs its caption text index (OpenCLIP embeddings
and Chroma) and hierarchical summaries derived from captions. Those are text
processing stages; they do not require another video caption pass. Retrieval
and summaries must propagate all contributing preprocessing metadata,
including `strict_causal: false` and conservative availability. A summary's
conservative availability cannot precede the latest such availability among
its inputs. A full-day summary cannot serve a query from earlier in that day;
completed constituent clips do not remove their full-source audio dependence.

Separate ASR transcripts and a visual feature archive are not required for
this caption pathway. The native speech encoder already consumes the audio
input. Question-specific retrieval and answer generation should be added only
with the finalized benchmark query times and evaluation protocol.
