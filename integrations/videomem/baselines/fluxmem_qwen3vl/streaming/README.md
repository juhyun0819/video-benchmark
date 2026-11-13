> Export note: commands below document the original integration. Prepared pilot inputs, environments, raw results, and checkpoints are excluded. Supply your own inputs and paths. See the variant README for export setup and limitations.

**Persistent FluxMem streaming inference with Qwen3-VL-8B-Instruct**

This local variant replaces the paper/reference Qwen2.5-VL backbone with pretrained **Qwen3-VL-8B-Instruct**. It performs inference only: no training, fine-tuning, optimizer updates, or newly trained trigger. The [official FluxMem repository](https://github.com/ShareLab-SII/FluxMem) supplies the unchanged TAS/SDC kernels; the persistent queues, scheduler, finite long-memory option, and Qwen3 integration are local extensions. This variant does not establish reproduction of the paper's results.

Work in `<original-workspace-path>`, branch `videomem-qwen3vl8b-streaming`. The earlier `<original-workspace-path>` and `<original-workspace-path>` repositories are preserved separately. The entrypoint is `python -m streaming.run`, using the native Transformers Qwen3 model and processor.

The `.venv-qwen3` symlink points to the existing `<original-workspace-path>` environment, with Transformers **4.57.3** and Torch **2.9**. Treat that shared environment as read-only; installing or upgrading packages through the symlink would change EgoRAG's environment too. The Qwen3 entrypoint does not need the old Qwen2.5 model package on `PYTHONPATH`: [upstream_kernel.py](upstream_kernel.py) loads only the unchanged Torch kernel and utility files under an isolated module namespace.

Run the prepared pilot with a new output directory:

```bash
cd <original-workspace-path>
.venv-qwen3/bin/python -m streaming.run \
  --config configs/qwen3vl8b_streaming_pilot.json \
  --output local_runs/NEW
```

`local_runs/NEW` must not already exist. Choose a different output path for each execution. The [pilot configuration](../configs/qwen3vl8b_streaming_pilot.json) specifies:

| Setting | Pilot value |
| --- | --- |
| Backbone | `<local-model-or-verification-path>` |
| Weight verification | `<local-model-or-verification-path>` |
| Physical GPU | 4, exposed alone to Torch |
| Sampling | 1 fps; longest sampled image side at most 512 pixels |
| Processor budget | 262,144 pixels per frame |
| Short / medium / long capacities | 8 / 64 / 8 sampled raw frames |
| Trigger threshold | `gamma = 0.5` |
| TAS policy | `upstream` |
| Generation | Greedy, up to 192 new tokens; proactive responses enabled |

The runner verifies local weights, loads bfloat16 weights with SDPA, and performs no model downloads. Explicit CLI options override config defaults; `--gpu 5`, `--proactive off`, or `--tas-policy paper` describe different experiments and should be recorded as such. A trigger threshold must be supplied through the config or CLI. Gamma 0.5 and the small finite long capacity are declared pilot settings, not paper benchmark defaults. The 512-pixel Qwen3 pilot is not a controlled speed comparison with the previous 448-pixel Qwen2.5 pilot.

The prepared inputs are `local_runs/qwen3_pilot_inputs_v1/manifest.json`, `questions.json`, and the predeclared `plan.json`. They use one real 90-second clip and synthetic questions at 10, 30, 30, and 90 seconds. The repeated 30-second questions exercise reuse of the same encoded memory. These are execution-pilot inputs, not benchmark accuracy inputs. This README makes no claim about completed pilot results or answer quality.

| Module | Responsibility |
| --- | --- |
| [video.py](video.py) | Sequential CPU decoding, exact PTS checks, source-local two-frame sampling, and image lifetime management. |
| [memory.py](memory.py) | Base persistent short/medium/long queues, TAS, SDC, backward trigger, finite eviction, and checkpoints. |
| [qwen3_memory.py](qwen3_memory.py) | `Qwen3PersistentMemory`: all three DeepStack levels aligned with the main projected tokens. |
| [qwen3_bridge.py](qwen3_bridge.py) | Native Qwen3 visual encoding, timestamp-separated prompts, sparse native RoPE, and decoding from stored main/DeepStack embeddings. |
| [upstream_kernel.py](upstream_kernel.py) | Model-free import of the unchanged released FluxMem kernels. |
| [session.py](session.py) | Observation cutoffs, response-before-cascade ordering, public prompts, and proactive task lifecycle. |
| [run.py](run.py), [checkpoint.py](checkpoint.py) | Config/CLI entrypoint, weight verification, GPU isolation, provenance, logs, and final memory checkpoint. |
| [../local_videomem/data.py](../local_videomem/data.py) | Manifest validation, chronological prefix selection, and public-question filtering. |

Every question requires a reviewed observation cutoff, for example:

```json
{
  "question_id": "example",
  "question": "Describe the visible setting and activities.",
  "choices": [],
  "query_time": {"day_id": "SmokeClip", "day_order": 1, "seconds": 30}
}
```

Only `question_id`, `question`, `choices`, and `query_time` survive public-question filtering. The model prompt receives the question and choices; filenames, source descriptions, ground-truth answers, reasons, target times, and evidence intervals remain outside it. The pilot private questions contain forbidden-field sentinels to exercise that separation. Empty choices request a short free-text answer; nonempty choices request an option letter.

At 1 fps, two sampled frames form one Qwen3 temporal grid. The official Qwen3 video processor and visual encoder run once per available pair. The encoder returns a final projected tensor and all three DeepStack tensors, each aligned row-for-row at the merged spatial resolution. The adapter passes the stored DeepStack levels to their native language-layer injections when answering. Aspect ratio is preserved during sampling, and the native processor applies its patch-compatible sizing within the configured pixel budget; 512 is a maximum sampled side, not a promise that every input is square.

Observation intervals are half-open: each included frame must precede the query cutoff. A pair whose second frame equals the cutoff is excluded. An observable first member without its second member contributes no tokens to that answer. Unpaired frames at source boundaries or the final cutoff are discarded, and pairs never span source files. CPU lookahead may decode the next pair, but the scheduler answers earlier questions before encoding or inserting it. The run stops at the last explicit question's horizon. The iterator retains at most two sampled RGB PIL images apart from codec/preprocessing buffers, and closes them when advanced or closed.

The arrival protocol is `ingest() → snapshot()/answer if needed → advance()`. Ingestion appends the complete main and DeepStack grid and computes backward novelty. A response on that arrival sees chronological long, medium, and short memory before overflow compression or eviction. A query between arrivals sees the previously completed bank. `advance()` applies TAS to short overflow, SDC to medium overflow, then evicts the oldest long records when a finite capacity is configured. Snapshots own detached copies; later arrivals and answers do not change them.

All capacity arguments count **sampled raw frames**, converted into temporal-grid slots by integer division by two. Thus 8/64/8 means 4/32/4 grid slots; odd values round down. Short memory requires at least two sampled frames, while medium and long may be zero. A fully pruned grid still occupies a queue slot. A response before `advance()` may include one additional newly arrived grid.

**A finite `long_frames` value is necessary for bounded retained visual memory.** The pilot uses 8. Omitting this setting outside the pilot config leaves the CLI/base-memory default `None`, preserving the released algorithm's unlimited long-memory behavior. With finite capacities and fixed spatial dimensions, retained main tokens and all three auxiliary levels have a fixed upper bound. FIFO eviction removes an entire old record and every associated tensor. Reported retained tensor bytes include DeepStack tensors and the base memory's positions/caches. Model weights, decoding buffers, per-answer KV caches, accumulated answers, and audit output are additional process memory; finite eviction also loses older observations.

TAS policy is explicit:

| Policy | Selection behavior |
| --- | --- |
| `upstream` | Released inclusive `distance >= Otsu threshold` selection, with its additional drop when similarity to the next grid exceeds 0.8. |
| `paper` | Local strict `(backward > backward_threshold) OR (forward > forward_threshold)` selection, using separate thresholds without the extra 0.8 removal. |

Both policies use the released 128-bin Otsu implementation over `[0, 2]`, its first maximizing bin center, and its median fallback for a degenerate distribution. The first grid has no backward comparison. TAS decisions use **only the final main projection**; precisely the same selected rows are retained from every DeepStack level.

SDC also derives its spatial graph, connected components, means, and representative anchors from the main features. The released kernel remains authoritative for main output values and anchor coordinates. Each auxiliary level is averaged over the **identical main-derived components**, then stored at those official anchors. Auxiliary features do not change the graph or anchor selection. Degenerate Otsu cases remain unmerged, matching the released kernel.

The backward trigger uses the Boolean fraction `r = mean(backward_distance > Otsu(backward_distance))`. It fires only if `r > gamma` and a previous successful response exists; equality does not trigger. Trigger distances at or below `1e-6` are zeroed to suppress normalization roundoff, without changing stored TAS distances. Constant distances yield no strictly above-threshold tokens, and the first grid's score is zero. DeepStack features do not participate in this decision.

The last successfully answered explicit question remains the active proactive task. This task lifecycle is a local scheduler policy. Explicit queries take priority over a proactive trigger on the same arrival. Success requires EOS termination and nonempty decoded text. Empty memory produces `insufficient_observation`; an empty or length-limited answer produces `incomplete_generation` and does not arm or replace the active task.

Answering reconstructs native Qwen3 timestamp-separated video groups and uses the model's native `get_rope_index()` before selecting sparse retained coordinates. The stored time ID identifies a memory group; it is not a Qwen2.5 temporal RoPE coordinate. At 1 fps, pair `p` receives timestamp text `(2*p + 0.5)` seconds, so groups begin at 0.5, 2.5, 4.5 seconds. Each group uses Qwen3's local temporal coordinate and original spatial positions. Original Day/file PTS remains separate audit metadata; source gaps and Day transitions are compacted in the model timeline.

Generation substitutes the retained main embeddings and injects all three retained DeepStack levels, with no visual re-encoding or second memory compression. A fresh explicit greedy KV cache is created for every answer. Causal sequence positions count actual retained prompt tokens separately from native rotary positions. This path supports argmax decoding and EOS stopping; it does not implement every Hugging Face sampling/logits option or automatically truncate context. A bank requires fixed merged height, width, hidden dimension, dtype, and device, and rejects spatial-shape changes explicitly.

| Output | Meaning |
| --- | --- |
| `provenance.json` | Repository commit/branch, local/native model source hashes, verified model revision, runtime settings, GPU identity, and local policies. |
| `manifest.json`, `public_questions.json` | Audited manifest and filtered public questions. |
| `events.jsonl` | Arrival/answer events, frame PTS, memory counts, trigger decisions, prompts, and response status. |
| `predictions.json` | Explicit/proactive answer records, used memory IDs, token counts, timing, and eviction state. |
| `status.json` | Current phase or final counts, decode/memory statistics, and GPU usage. |
| `memory.pt` | Final CPU `Qwen3PersistentMemory.state_dict()` with provenance, including all three DeepStack levels. |

The runner exits with code 2 when answer records are incomplete; execution exceptions are reported separately in `status.json`. A generated answer's `memory_truncated` flag reports declared truncation or prior long-bank eviction. Inspect status and predictions together when assessing an execution.

The Python API supports `bank.state_dict()` and `Qwen3PersistentMemory.from_state_dict(state, device=...)`, preserving all four visual streams, queues, coordinates, counters, trigger state, and the pending-cascade phase. The CLI has no implicit resume. Resuming a session also requires its stream cursor, pair numbering, remaining questions, and active proactive task; a bank checkpoint alone does not restore these. Only load trusted local checkpoints.

Run the relevant CPU checks in the Qwen3 environment:

```bash
cd <original-workspace-path>
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv-qwen3/bin/python -m pytest -q \
  tests/test_local_videomem_data.py \
  tests/test_streaming_memory.py \
  tests/test_streaming_video.py \
  tests/test_streaming_session.py \
  tests/test_qwen3_memory.py \
  tests/test_qwen3_bridge.py
```

These tests use controlled tensors and tiny random Qwen3 models to check causality, exact released main-memory parity, DeepStack alignment and component means, FIFO bounds, serialization, native coordinates, and stored-embedding decoding. They do not establish pretrained-model answer quality. Historical Qwen2.5 bridge/runtime tests and [model_bridge.py](model_bridge.py) remain in this clone for reference; they are excluded from the Qwen3 runtime test command because they target the older model implementation and environment.

Original VideoMem benchmarking remains blocked by missing reviewed `query_time` observation cutoffs in the original QA and provisional mappings for concatenated Day clips. Answer targets and evidence timestamps must not be substituted for observation cutoffs. Audio, benchmark scoring, and paper-metric reproduction are outside this implementation.
