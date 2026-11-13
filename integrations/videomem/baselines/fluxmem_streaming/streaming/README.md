> Export note: commands below document the original integration. Prepared pilot inputs, environments, raw results, and checkpoints are excluded. Supply your own inputs and paths. See the variant README for export setup and limitations.

**Persistent FluxMem streaming inference — local extension**

This directory adds persistent projected-token memory and causal query scheduling to the unchanged official FluxMem model. It performs inference only: no training, fine-tuning, optimizer updates, or newly trained trigger. The reference implementation is the [official FluxMem repository](https://github.com/ShareLab-SII/FluxMem), pinned here to commit `6fdbf65b6f2af1a94ca41d542be5be86a565d338`. The algorithmic reference is [FluxMem, Algorithm 1 and §3](https://arxiv.org/html/2603.02096v1#S3).

The paper describes persistent short, medium, and long memory, responses before cascading compression, and proactive activation from backward novelty. The released code reconstructs its queues inside each `process_memory_streaming()` call and invokes that reduction during model prefill. This directory supplies the persistent queues, arrival scheduler, trigger state, finite-capacity eviction, and stored-embedding answer path locally. It is a paper-guided extension, not an official persistent-streaming release or a claim to reproduce the paper's results.

| Module | Responsibility |
| --- | --- |
| [video.py](video.py) | Sequential CPU decoding, exact PTS checks, source-local two-frame sampling, and image lifetime management. |
| [memory.py](memory.py) | `PersistentMemory`: incremental ingestion, TAS, SDC, ACT, snapshots, eviction, and state serialization. |
| [model_bridge.py](model_bridge.py) | Official two-frame visual encoding and explicit greedy decoding from stored projected embeddings. |
| [session.py](session.py) | Query cutoffs, response-before-cascade ordering, public prompts, and proactive task lifecycle. |
| [run.py](run.py) | Verified local weights, GPU isolation, provenance, event logs, predictions, and final memory checkpoint. |
| [../local_videomem/data.py](../local_videomem/data.py) | Existing manifest validation, chronological prefix selection, and public-question filtering. |

Run commands from `<original-workspace-path>`. Its `.venv` is a symlink to `<original-workspace-path>`, so package installation or upgrades through it affect the existing environment. The runtime uses Torch 2.9 and Transformers 4.51.3. `streaming/run.py` explicitly prioritizes this clone's official model and vision-utility source directories over the old editable installations. For direct Python or test invocation, set `PYTHONPATH` as shown below to obtain the same import resolution.

The prepared pilot inputs are `local_runs/stream_pilot_inputs_v1/manifest.json`, `questions.json`, and the predeclared `plan.json`. They use one real 90-second clip and synthetic questions at 10, 30, 30, and 90 seconds; repeated questions at 30 seconds exercise visual-memory reuse. This is an execution pilot, not a benchmark accuracy evaluation. The following command specifies GPU 4, 448-pixel sampling, 1 fps, short/medium/long capacities of 8/64/8 sampled frames, and gamma 0.5:

```bash
cd <original-workspace-path>
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python streaming/run.py \
  --manifest local_runs/stream_pilot_inputs_v1/manifest.json \
  --questions local_runs/stream_pilot_inputs_v1/questions.json \
  --output local_runs/stream_pilot_448_gpu4 \
  --model <local-model-or-verification-path> \
  --verification <local-model-or-verification-path> \
  --gpu 4 --fps 1 --max-side 448 --max-pixels 200704 \
  --short-frames 8 --medium-frames 64 --long-frames 8 \
  --trigger-threshold 0.5 --tas-policy upstream \
  --max-new-tokens 192 --proactive on
```

The output directory must not already exist. Choose a new output path for another run. The runner accepts physical GPU 4 or 5 and exposes only the selected device to Torch. Weights are verified before loading; inference uses bfloat16 and SDPA. `--trigger-threshold` is required rather than silently choosing a paper-recommended value: none is supplied by the paper. The pilot's gamma and small finite long capacity are explicit experiment settings. Use `--proactive off` for explicit questions only, or `--tas-policy paper` for the strict local TAS variant described below. Neither change should be silently mixed into the predeclared pilot.

Every question must supply a reviewed observation cutoff:

```json
{
  "question_id": "example",
  "question": "Describe the visible setting and activities.",
  "choices": [],
  "query_time": {"day_id": "SmokeClip", "day_order": 1, "seconds": 30}
}
```

Only `question_id`, `question`, `choices`, and `query_time` survive public-question filtering. The model receives the question and choices through the observed-video prompt; filenames, source descriptions, ground-truth answers, reasons, target times, and evidence intervals remain outside its prompt. The prepared private questions contain forbidden-field sentinels to exercise that separation. Empty choices request a short free-text answer; nonempty choices request an option letter. Manifest preparation and source fingerprinting are provided by [the existing data adapter](../local_videomem/README.md).

At 1 fps, two sampled frames form one Qwen temporal grid. The official image processor and visual encoder run once for each available pair, without passing text to the encoder. The iterator owns at most two sampled RGB PIL images, apart from codec and preprocessing buffers, and closes them when advanced or closed. `encode_pair()` borrows those RGB images without copying or closing them. Sampling preserves aspect ratio and only downsizes to `max_side`; the official processor then applies its patch-compatible sizing and pixel budget. Square 224- and 448-pixel inputs remain at those resolutions under the default budget.

Observation intervals are half-open: every included frame must precede the query cutoff. A pair whose second frame equals the cutoff is excluded. If only the first member is observable, the pair contributes no tokens to that answer; it is never padded with a future frame. Unpaired frames at source boundaries or the final cutoff are discarded, and pairs never span source files. CPU lookahead may decode the next pair, but the scheduler answers earlier queries before encoding or inserting it. The scheduler stops at the last explicit question's horizon.

The arrival protocol is `ingest() → snapshot()/answer if needed → advance()`. Ingestion appends the complete grid and computes its backward statistics. A response associated with that arrival sees chronological long, medium, and short memory before overflow compression or eviction. A query between arrivals sees the previously completed bank. `advance()` moves short overflow through TAS, medium overflow through SDC, and then evicts oldest long records if a finite limit was configured. Returned snapshots own detached copies and are not changed by later arrivals or answers.

All capacity arguments count **sampled raw frames**, converted into temporal-grid slots by integer division by two. Thus 8/64/8 means 4/32/4 grid slots; odd capacities round down. Short memory requires at least two sampled frames; medium and long may be zero. Even a grid with no retained tokens occupies a queue slot. A response before `advance()` can include one extra newly arrived grid.

**A finite `--long-frames` value is required to claim bounded retained visual memory.** Its default is `None`, preserving unbounded long-memory compatibility with the released reduction. With finite capacities and fixed spatial dimensions, the bank and an individual response snapshot have a fixed upper token bound; finite eviction intentionally loses older observations. This does not claim constant total process memory: questions, accumulated answer records, and audit output grow with the run, while model weights, decoder buffers, and per-answer KV caches consume additional memory.

TAS behavior is an explicit setting:

| Policy | Selection behavior |
| --- | --- |
| `upstream` (default) | Delegates to the released kernel: inclusive `distance >= Otsu threshold`, followed by its additional removal when similarity to the next grid exceeds 0.8. |
| `paper` | Local implementation of the paper's union: `(backward > backward_threshold) OR (forward > forward_threshold)`, with separate thresholds and no extra 0.8 similarity removal. |

The strict union follows [the paper's TAS definition](https://arxiv.org/html/2603.02096v1). Both local policies use the released 128-bin Otsu implementation over `[0, 2]`, selecting the first maximizing bin center, with a median fallback for a degenerate distribution. The first grid has no backward comparison. SDC always delegates to the released kernel: it operates on retained spatial neighbors, emits component means at representative coordinates, and leaves degenerate Otsu cases unmerged.

ACT uses the **Boolean fraction** `r = mean(backward_distance > Otsu(backward_distance))`, not a mean distance or a compression ratio. It fires only when `r > gamma` and a previous successful response exists, following [the paper's trigger and Algorithm 1](https://arxiv.org/html/2603.02096v1). Equality does not trigger. In this implementation, trigger distances at or below `1e-6` are zeroed to suppress normalization roundoff; this adjustment does not change stored TAS distances. Constant distances yield no strictly above-threshold tokens, even when the constant is large, and the first grid's trigger score is zero.

The session keeps the **last successfully answered explicit question** as the active proactive task. That lifecycle is a local policy; the paper and released API do not define a task-management interface. Explicit queries take priority over a proactive trigger on the same arrival. Successful completion means EOS termination with nonempty decoded text. An empty memory produces `insufficient_observation`; a length-limited or empty answer produces `incomplete_generation` and does not arm or replace the active task.

Each answer builds a standard video-placeholder chat prompt and substitutes the stored projected embeddings. It calls the model with `use_fluxmem=False`, no pixels, and explicit three-axis positions. It does not re-run vision or recompress memory. The explicit cached greedy loop is necessary because the official `prepare_inputs_for_generation()` overwrites supplied position IDs. Memory coordinates retain their gaps; all three axes receive the text-prefix offset, suffix text begins after their maximum, and generated text continues after the maximum prefill coordinate. KV-cache offsets separately count the actual retained token sequence, and each question starts a fresh cache. This interface supplies argmax decoding and EOS stopping, not all Hugging Face sampling or logits-processor features; it performs no automatic context truncation.

The CLI uses compact native video coordinates: `time_id = int(pair_index * 2 / fps * tokens_per_second)`. With Qwen's `tokens_per_second=2` and 1 fps, IDs are 0, 4, 8, and so on. Original Day/file PTS remains in audit metadata; source gaps and Day transitions are not represented as wall-clock gaps in RoPE. One bank requires fixed merged height, width, hidden dimension, dtype, and device. A spatial-shape change fails explicitly; the iterator does not silently crop or pad different aspect ratios into one grid.

| Output | Meaning |
| --- | --- |
| `provenance.json` | Upstream commit, local source hashes, verified model revision, runtime settings, GPU identity, and local policy declarations. |
| `manifest.json`, `public_questions.json` | The audited input manifest and filtered public questions. |
| `events.jsonl` | Incremental arrival/answer events, frame PTS, memory counts, trigger decisions, prompts, and response status. |
| `predictions.json` | Explicit and proactive answer records, including used memory IDs, token counts, timing, and eviction state. |
| `status.json` | Current phase or final counts, decode statistics, memory statistics, and GPU usage. |
| `memory.pt` | Final CPU memory `state_dict()` plus provenance, written after the session completes. |

The runner exits with code 2 when answer records are incomplete and reports execution exceptions separately. `memory_truncated` in a generated answer reports declared truncation or prior long-bank eviction; token selection/merging alone is not reported as an eviction. Inspect both status and predictions to assess a run. No pilot output or quality result is asserted by this README.

The Python API supports `bank.state_dict()` and `PersistentMemory.from_state_dict(state, device=...)`, preserving queues, positions, counters, the pending-cascade phase, and trigger state. The CLI deliberately has no implicit resume. A caller implementing continuation must also restore its stream cursor, pair/time numbering, remaining questions, and active proactive task; a bank checkpoint alone does not restore that scheduler state or prevent duplicate answers. Only load trusted local checkpoints.

Run the streaming CPU checks with the local sources selected explicitly:

```bash
cd <original-workspace-path>
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH="$PWD/models/qwen2-5-vl/src:$PWD/models/qwen-vl-utils/src:$PWD" \
  .venv/bin/python -m pytest tests/test_streaming_*.py -q
```

Tests use controlled token fixtures and tiny random official models to check causality, snapshot isolation, bounded queues, trigger behavior, repeated-query reuse, and sparse-position decoding. They do not establish pretrained-model answer quality. Real VideoMem benchmarking remains blocked by the absence of reviewed `query_time` observation cutoffs in the original QA and by provisional mappings for concatenated Day clips. Answer targets or evidence timestamps must not be substituted for those cutoffs. Audio, benchmark scoring, and paper-metric reproduction are outside this implementation.
