# Native visual-memory adapters

`baselines.streambench.adapters.native.create_method(config)` wraps existing memory implementations without modifying their source. The encoder and reader remain method-specific. Observation is question-free; answers use isolated native snapshots and fresh decoder KV caches and never call `mark_response()`.

| Method names | Encoder / representation | Native budget control | Continuation state |
|---|---|---|---|
| `fifo`, `fifo_qwen3vl` | Qwen3-VL-8B main projection, all three DeepStack levels, integer positions | Whole-grid oldest-first eviction | Grid queue, shape, counters, phase, timestamps |
| `reservoir`, `reservoir_qwen3vl` | Same four aligned streams | Fixed-size Algorithm R reservoir | Unsorted slots, all counters including rejected arrivals, exact local RNG |
| `streamforest` | Released Qwen2/SigLIP, raw 729×1152 features, native event memory | Native short windows/token counts and long-token quota; PEMF unchanged | Every manager buffer, last raw frame, event times, merge counts, similarity caches |
| `fluxmem_streaming`, `fluxmem` | Released Qwen2.5-VL projected visual grids | Native TAS and SDC; unbounded long memory by default | All three tiers, normalization/backward-distance caches, positions, trigger/phase/counters |
| `fluxmem_qwen3vl` | Qwen3-VL main projection plus three aligned DeepStack streams | Same main-feature TAS/SDC decisions and auxiliary component means | Base FluxMem state plus every DeepStack tensor |

`fluxmem_reference` is explicitly unsupported as a continual adapter: its released prefill path reconstructs memory from a full prefix for each question. Use `fluxmem_streaming` for the repository's existing incremental implementation. The Qwen3 backbone change and persistent FluxMem scheduler were already local extensions, not claimed official reproductions.

## Byte budgets

Set `memory_budget_bytes` in bytes, independently of token counts. Accounting uses live tensor storage, deduplicates aliases, and includes positions, metadata, counters, local RNG, and an unmatched frame that survives a chunk boundary. Frozen model parameters and temporary generation KV are excluded. Python allocator overhead is measured separately by the runner.

FIFO and Reservoir derive a fixed grid capacity after the first actual projection. The calculation includes the measured native record and non-record state, a 256-byte growth allowance per record, and 16 KiB plus the observed raw-frame size for pending/protocol growth. `pending_frame_reserve_bytes` can increase the frame reserve. Reservoir never changes this capacity after sampling starts. An explicit `capacity_grids`, `memory_frames`, or `memory_tokens` selects the existing native capacity instead. Subsequent actual accounting is authoritative: a larger later frame or metadata record can make the configuration unsupported; the adapter does not silently change reservoir sampling.

StreamForest uses the published windows/token sizes and merge weights by default and derives only `long_memory_tokens_quota` from the requested bytes. Its conservative bound includes the last uncompressed frame, all recent/event staging tokens, the long quota, and a 16 KiB metadata/protocol allowance. A quota must fit at least one maximum-sized event. `memory_config` can supply an exact native configuration instead. Hidden width, encoder size, and pretrained layers are never reduced to meet B.

FluxMem defaults to `long_frames=null`, retaining native unbounded long memory and reporting `unsupported_under_budget` when it exceeds B. An explicit finite `long_frames` enables the repository's existing complete-oldest-long-grid eviction extension. This must be reported as an extension, not as native unbounded FluxMem. `tas_policy="upstream"` is the default; `"paper"` is the pre-existing alternate selection rule.

All methods measure actual persistent bytes after arrivals and chunk finalization. A failed/overbudget `observe()` rolls back the complete pre-call bank, counters, and pending frame. Temporary rollback copies consume additional runtime RAM but cannot be retrieved by future questions. Native context-length restrictions still apply at QA: increasing B does not increase the pretrained language model's maximum context, and this adapter does not silently truncate memory.

## Continuous temporal grouping and checkpointing

Two-frame encoders receive adjacent frames from the common continuous stream. An unmatched frame is copied into an owned CPU tensor and preserved across chunks, source files, and day checkpoints; it is never padded or silently discarded. Consequently a pair may straddle Day1/Day2 or two files. This intentionally replaces the legacy iterator's source-boundary orphan discard. The first day's checkpoint cannot answer from the unmatched frame until its partner has arrived; it does contain everything necessary to continue its processing.

Group timestamps are the mean of the two original global timestamps. Qwen3 stores that time as native timestamp text and uses ordinal group identities. Qwen2.5 uses `int(mean_global_timestamp * temporal_tokens_per_second)` for native temporal coordinates, defaulting to two units/second. Regular sampling therefore requires `FPS <= 2 * temporal_tokens_per_second` (default maximum **4 FPS**). Each observation preflights the actual pending/incoming pair coordinates before encoding, rejecting collisions with the configured scale in its error message; irregular timestamps are checked too. StreamForest uses global frame IDs/timestamps while retaining the original manager's ordinal event bookkeeping.

Checkpoints capture all native bank attributes and grids, pending pixels and provenance, stream/group counters, resolved native capacities, compression/eviction counts, and Reservoir RNG. Tensor objects are serialized as one graph so shared storage survives. This specifically preserves StreamForest's alias between `last_frame_flat` and an uncompressed newest frame. Restoring to a different device copies each storage once and reconstructs its views. Resolved StreamForest quotas are installed into the native manager before restoring its buffers.

`compression_count` counts arrivals that removed visual rows through compression, excluding whole-record rejection/eviction. `eviction_count` reports the native number of replaced or evicted grids. Per-query metrics report payload bytes handed to the native reader and visual-token/item counts. No observation or QA operation stores question text in the memory.

## Local model loading

All model reads use `local_files_only=True`. The repository's existing native bridges are imported lazily. Bank/kernel imports use isolated module names, so CPU tests can exercise all methods together without mixing the incompatible `streaming` packages.

Example FIFO configuration:

```json
{
  "method": "fifo",
  "memory_budget_bytes": 67108864,
  "model_path": "/absolute/local/Qwen3-VL-8B-Instruct",
  "device": "cuda:0",
  "dtype": "bfloat16",
  "max_pixels": 262144,
  "max_new_tokens": 192
}
```

Use `reservoir` plus an explicit `seed` for Algorithm R. StreamForest additionally requires `verification_file`, checked by its existing pinned native checkpoint verifier; its local released model is `MCG-NJU/StreamForest-Qwen2-7B` at revision `6840f91e17346b4c406ba03b57d1a63582765c91`. Qwen2.5 uses the vendored released architecture and `Qwen/Qwen2.5-VL-7B-Instruct`. Qwen3 methods use the native installed Transformers architecture and `Qwen/Qwen3-VL-8B-Instruct`; use each baseline's documented compatible environment. Include an immutable model revision in experiment configuration. All native loaders also hash local configuration, tokenizer, and weight artifacts into a checkpoint fingerprint, detecting replaced weights behind an unchanged model path. This streaming read adds model-load I/O but does not materialize another model.

```bash
python -m baselines.streambench observe --help
python -m baselines.streambench evaluate --help
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest -q -p no:cacheprovider baselines/streambench/tests/test_streambench_native.py
```

The focused suite uses the actual FIFO, Reservoir, FluxMem, and StreamForest memory kernels with explicitly injected synthetic features. It covers chronological retention, immutable answers, bounded native capacities, overflow rollback, pending RGB ownership, exact aliases, native quota restoration, and fresh-process continuation. The isolated test environment also supports Qwen2/Qwen3 model imports and legacy native suites. These checks do not establish pretrained answer quality, GPU throughput, or successful full-size checkpoint inference; no substitute model is used.
