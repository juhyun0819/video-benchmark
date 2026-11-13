# Ours offline raw-video baselines

The target is **our dataset**, with QA supplied later. This package implements
independent `Qwen/Qwen3-VL-8B-Instruct` and
`llava-hf/llava-onevision-qwen2-7b-ov-hf` inference using a shared canonical QA/scorer.
LLaVA uses ordinary `LlavaOnevisionForConditionalGeneration`, without ReKV patches.
No QA is fabricated or borrowed from the older automatically generated QA package.

**Ours QA remains pending; no ours accuracy result exists.** Unit and CPU processor
checks establish code-path behavior. Subsequent EgoLife GPU smoke tests passed for
both native backbones on RTX 3090 with 32 causal raw frames, BF16 and SDPA. EgoLife
is the validation dataset; these checks are not results for ours. Source metadata
preparation is useful now, before QA exists.

## Prepare sources now; attach QA later

A source list declares `timeline: "concatenate"` or `timeline: "explicit"`.
The portable [ours seven-day list](templates/ours_day1_day7.sources.json) contains
only relative `day1.mp4` through `day7.mp4` paths. Concatenation derives starts from
video stream `duration_ts * time_base` when available, otherwise reported video
stream duration. Audio/container duration is not used. No EgoLife day-times-86400
clock or fixed 20-fps interpretation is applied. With `explicit`, each source
instead supplies nonnegative global `start_time`, preserving declared gaps.

```bash
# This only probes metadata; it does not evaluate video or load models.
python -m baselines.ours_offline prepare \
  --root /convei_nas2/zzuhyun/hyunjun/datasets/videomem \
  --source-list baselines/ours_offline/templates/ours_day1_day7.sources.json \
  --output /path/to/new-prepared-input

python -m baselines.ours_offline validate \
  --root /convei_nas2/zzuhyun/hyunjun/datasets/videomem \
  --sources /path/to/new-prepared-input/sources.json \
  --canonical /path/to/new-prepared-input/questions.pending.json
```

Preparation writes:

- `sources.json`: source clocks, exact fractional fps/durations, frame counts,
  dimensions and local file size/inode/mtime/ctime receipts.
- `questions.pending.json`: **empty questions**, `awaiting_QA`, the common scorer
  hash and source-index hash. It cannot be evaluated until real QA is supplied.
- `status.json`: source/day counts and `sources_ready_awaiting_QA`.

The existing local prepared input is `/home/zzuhyun/hyunjun/ours_setup/input_v1`:
7 source videos, 5,543,123 frames at 25 fps, 221,724.92 video seconds. This is a
metadata inventory, not an accuracy result. No full-file SHA256 receipt is claimed.
Preparation verifies CFR metadata; a missing exact frame-count field is an error,
not a duration-based guess. Existing output directories are never overwritten.

When reviewed QA arrives, populate a new file following
[qa.schema.json](templates/qa.schema.json). Each question has `qa_id`, public
`question`, `choices`, `query_time` in **global seconds on the prepared timeline**,
and `answer` (uppercase letter/exact choice text) or `correct_idx`. `day` is an
optional positive grouping label. Do not derive the observation cutoff from GT
answer/evidence times. The builder stamps the common scorer hash into the pending
file; retain that identity and the source-index hash when adding QA.

## Causal raw sampling and native model input

The loader verifies common scorer identity, unique QA IDs, option/label consistency,
safe source paths and source stat receipts. Model runs rehash a trusted local model
verification ledger, including loader artifacts and weight-index shards. This
checks correspondence to the supplied ledger, not independent remote authenticity.

Each question reads only original frames with actual global PTS `<= query_time`.
`--max-frames` is mandatory: selection is uniform over **available raw CFR frame
positions**, preserving earliest/latest positions and avoiding duplicates. The
source's actual fps is used; optional closed global intervals can restrict evidence
access. Overlapping source tails belong to the later source. Prior-source history
remains available when evaluating a later-day QA subset.

This is an **explicit read-budget adaptation**, not a claim to reproduce each
model's untouched default sampling policy. After external selection, the native
HF video processor handles spatial resizing, temporal patches and normalization;
`do_sample_frames=False` prevents a second sampling pass. Qwen receives exact global
frame timestamps through a virtual 1-Hz metadata timebase, retaining native
DeepStack and position handling. Its temporal-pair formatting may average timestamps
including pairs across gaps. LLaVA's video path has no corresponding absolute-time
metadata channel: it sees selected frame order, while exact PTS remain in records.

Frames within one native video input must share dimensions. There is no hidden
pre-resize or visual-feature cache. Each question starts a fresh prefill/KV state.
Audio is not supplied. Questions marked audio/name-required remain in the denominator.
Only nonempty EOS-terminated outputs count as successful answers; truncated outputs
are marked incomplete and count as incorrect.

## Plan and run once real QA exists

Run modules from the repository root. Canonical/source arguments are JSON **files**.
Use a new output directory for each invocation. `plan` never loads a processor,
model or GPU; an empty pending-QA file gives a clear error rather than fake work.

```bash
DATA=/convei_nas2/zzuhyun/hyunjun/datasets/videomem
INPUT=/home/zzuhyun/hyunjun/ours_setup/input_v1

python -m baselines.ours_offline plan \
  --root "$DATA" --canonical /path/to/reviewed-ours-qa.json \
  --sources "$INPUT/sources.json" --max-frames 32 \
  --output /path/to/new-offline-plan

# Use the inference environment and a free physical GPU.
python -m baselines.ours_offline run \
  --root "$DATA" --canonical /path/to/reviewed-ours-qa.json \
  --sources "$INPUT/sources.json" --max-frames 32 \
  --output /path/to/new-qwen-run --backbone qwen3_vl \
  --model /convei_nas2/zzuhyun/hyunjun/models/Qwen3-VL-8B-Instruct \
  --model-revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --ledger /home/zzuhyun/hyunjun/jake_setup/Qwen3-VL-8B-Instruct.verification.json \
  --gpu 2 --shard 0 --shards 2 --channels-last-3d
```

The example reuses the already verified local model ledger; its directory name
only identifies where that model was downloaded. It does not select EgoLife inputs.
For LLaVA use `--backbone llava_onevision`, its local model directory, immutable
40-character revision and matching ledger. Do not supply Qwen-only
`--channels-last-3d` or `--max-pixels` options to LLaVA.

Default generation is BF16, SDPA, greedy, one beam and at most 256 new tokens.
`--attention`, `--dtype`, `--max-new-tokens` and `--transformers-version` are explicit
settings, persisted in provenance. The API targets Transformers 4.57.6. Flash
Attention requires a compatible separate installation. `--max-pixels` optionally
declares a Qwen **whole-video** resize pixel budget (`T*H*W`, via native
`videos_kwargs.size.longest_edge`); the native minimum pixel budget is clamped to
this maximum if necessary. Values below the padded frame count times one merged
patch's area are rejected. This is not a per-frame pixel budget. By default the
processor's native setting is kept. GPU memory feasibility still requires a separate smoke run. There is no
automatic fallback or frame-count reduction on OOM.

Use `--days`, `--qa-ids`, `--shard/--shards` or `--limit` for explicit subsets.
Missing IDs and empty shards are errors. Give each shard a distinct GPU/output and
merge by unique QA ID against the expected manifest set.

`--decode-workers 4` reads independent selected raw frames concurrently on CPU.
Each decoder keeps its own container and a single codec thread; frames in the
same source are decoded in one pass. Selected frame order, actual PTS checks and
the model input are unchanged. The default is one
worker. Account for all simultaneous QA shards when choosing the CPU thread budget.

For several GPUs or both backbones together, the [parallel supervisor](PARALLEL.md)
launches independent QA shards, tracks workers and GPU memory, and audits exact
canonical coverage when merging results. It never splits temporal state because
these offline answers each start a fresh model context.

### Published frame-budget comparison and audio subsets

WorldMM's [baseline setup](https://arxiv.org/html/2512.02425v2#A2.SS1)
reports 0.5 fps, capped at 768 frames, without captions or transcripts, for its
Qwen3-VL-8B result. The earlier 32-frame run is a separate budget ablation; 32 is
not a model limit. Use `--sampling-fps 0.5 --max-frames 768` to match that rate/cap.
Our implementation anchors the candidate grid at each source start, floors to
original CFR indices and uniformly caps the complete causal prefix. It keeps
`actual_timestamp <= query_time`, the native checkpoint processor's spatial budget,
and the documented prompt. The paper does not establish that all these choices
are identical to its baseline, so this is **published-budget alignment**, not a
claim to reproduce 38.6% exactly. More frames also change the native spatial resize
under Qwen's fixed whole-video pixel budget; processed grids are saved per QA.

New worker status and merged summaries include `performance.no_audio_required`.
This group uses exactly `need_audio == false`; `need_name` remains unrestricted,
unknown audio flags are reported separately, and failed/incomplete answers remain
in the completed denominator. `accuracy_over_completed` is the interim accuracy;
`accuracy_over_expected` includes pending QA as unearned credit and must not be
presented as a finished benchmark until `coverage_complete` is true.

```bash
python -m baselines.ours_offline.report --run /path/to/parallel-run \
  --output /path/to/audio-subsets.json
```

The read-only reporter also works on older runs: it obtains audio flags from the
bound canonical manifest, rescores answers, and ignores only an unfinished trailing
JSONL record. Changing settings always requires new output directories.

Results record method, QA ID, cutoff, actual PTS, prediction/raw output, GT/correct,
completion, latency and errors. Offline `stream_update_time` is zero. Plan and
provenance files preserve frame budget, model/manifest/scorer identities and code
hashes. No pretrained accuracy or throughput result is claimed here.

## Reusable interfaces

```python
from baselines.ours_offline.inputs import load_inputs, uniform_samples, decode_samples
from baselines.ours_offline.backend import NativeVideoBackend, verify_model_ledger

manifest, sources, labeled_questions = load_inputs(root, qa_file, source_file)
samples = uniform_samples(sources, cutoff, max_frames=32, intervals=None)
frames, actual_records = decode_samples(root, samples, cutoff)
# Verify the trusted model ledger before loading a deployed backend.
backend = NativeVideoBackend(config)
backend.initialize()
response = backend.answer(frames, actual_records, question_text, choice_strings)
```

`answer` takes only frames/timestamps and public question/options. It returns
`prediction`, `raw_output`, `complete`, token IDs/counts and processor metadata.
The evaluator alone calls the common scorer. `run.py` also exports
`add_model_arguments`, `backend_config`, `bind_gpu`, `select_questions` and
`evaluate_question`. Evidence-access references must label and select privileged
evidence separately from this backbone.

```bash
python -m unittest discover -s baselines/ours_offline/tests -v
```
