# Native offline QA parallel supervisor

`python -m baselines.ours_offline.parallel` launches long-lived native runner
processes, one explicitly allocated physical GPU per QA shard. Each process loads
its answer model once. This is independent QA data parallelism; it does not split
an online stateful stream. The current use is EgoLife's fixed 500-QA validation
manifest. OURS remains QA-pending and can use the same supervisor after its
reviewed manifest arrives.

The supervisor only uses existing local models and prepared input manifests.
It does not download weights, import torch, stop foreign jobs, restart workers,
reuse previous predictions, or change allocation while a run is active. Every
launch requires a fresh output directory. Smoke results are separate from final
results. Freeze the checkout before launching: worker provenance hashes every
Python file in this package, and mismatched code is rejected during merging.

## Configuration

Paths are absolute or relative to the configuration file. Substitute the actual
verified local model paths, 40-character revisions and ledgers:

```json
{
  "root": "/path/egolife",
  "canonical": "/path/egolife-500/questions.json",
  "sources": "/path/egolife-500/sources.json",
  "expected_questions": 500,
  "max_frames": 32,
  "cpu_threads": 4,
  "decode_workers": 4,
  "poll_seconds": 5,
  "models": [
    {
      "name": "qwen3_vl_8b",
      "backbone": "qwen3_vl",
      "model": "/path/Qwen3-VL-8B-Instruct",
      "model_revision": "REPLACE_WITH_IMMUTABLE_40_CHARACTER_REVISION",
      "ledger": "/path/qwen.verified.json",
      "gpus": [2, 3, 4],
      "max_new_tokens": 256,
      "attention": "sdpa",
      "dtype": "bfloat16",
      "transformers_version": "4.57.6"
    },
    {
      "name": "llava_onevision_7b",
      "backbone": "llava_onevision",
      "model": "/path/llava-onevision-qwen2-7b-ov-hf",
      "model_revision": "REPLACE_WITH_IMMUTABLE_40_CHARACTER_REVISION",
      "ledger": "/path/llava.verified.json",
      "gpus": [5, 6, 7],
      "max_new_tokens": 256,
      "attention": "sdpa",
      "dtype": "bfloat16",
      "transformers_version": "4.57.6"
    }
  ]
}
```

Each model may override `decode_workers`; otherwise it inherits the global
default of four. Frame decoding preserves native frame identities, ordering and
causal checks. Optional per-model `max_pixels` is forwarded explicitly as the
native Qwen whole-video pixel budget; `channels_last_3d` defaults to false. The
native worker currently fixes PyTorch CPU threads at four and interop threads at
one, so the supervisor requires `cpu_threads=4` and sets OMP/MKL/OpenBLAS/NumExpr
thread environments to four. Budget host CPU capacity across all workers.

Allocations must be disjoint; no GPU is selected automatically. Before starting
any worker the supervisor checks all requested GPUs via `nvidia-smi` and rejects
memory usage >=1024 MiB or nonzero utilization. Advisory per-user GPU locks also
exclude another instance of this supervisor. Each native worker rechecks GPU
memory occupancy after verifying its model ledger. Occupancy is inherently a
point-in-time check; an unrelated process can still start later.

## Plan, launch and inspect

Use the interpreter/environment intended for the actual models. Child processes
use the same `sys.executable`, full argv, repository cwd and isolated stdout log.
Planning validates sources/QA/local paths and allocation, but does not load
weights or check GPU occupancy:

```bash
python -m baselines.ours_offline.parallel plan \
  --config /path/parallel.json --output /path/new-plan

nohup python -u -m baselines.ours_offline.parallel run \
  --config /path/parallel.json --output /path/new-final-run \
  > /path/supervisor.log 2>&1 &

python -m baselines.ours_offline.parallel status --output /path/new-final-run
```

`plan.json` freezes input/scorer/code hashes, model ledger identities, expected QA
IDs, GPU allocations and exact worker argv. Atomic, fsynced `status.json` stores
supervisor/worker PIDs, return codes, elapsed time, live worker progress and
polled GPU memory/utilization. `peak_observed_memory_used_mib` is the maximum
**sampled nvidia-smi memory.used**, can miss peaks and includes all processes on
the GPU; it is not `torch.cuda.max_memory_allocated`. Progress throughput includes
ledger verification and model setup. A worker's `verifying_model` phase rehashes
local model files and can spend substantial time reading NAS storage. The first
QA can remain in `loading` status until decode/inference finishes because the
native runner writes progress after each QA.

Per-model `shard-NNN.log` and `shard-NNN/` contain native outputs. On termination
of all workers the supervisor writes:

- `MODEL/predictions.merged.jsonl`, in canonical QA order.
- `MODEL/summary.json`, with exact coverage audit, fixed denominator accuracy,
  actual accepted rows, synthesized missing rows and setup-inclusive throughput.

Merge checks worker input/scorer/code/model/ledger/decoder-concurrency identity,
expected shard membership, unique QA IDs, query cutoff, GT, frame chronology and
causality, then rescoring uses the shared scorer. Duplicate, unknown, malformed or
provenance-mismatched rows fail the audit. Missing/rejected expected rows become
explicit `status=error`, `supervisor_generated=true` records with empty output;
`synthesized_missing_or_rejected` counts them separately from actual worker rows.
These records never count as completed model answers. Errors and incomplete
generation remain incorrect in the full expected-QA denominator. A complete run
requires exact coverage, successful rows and zero exit codes from every worker.

A failed supervisor leaves recorded workers running; it never kills or restarts
them. Inspect their PIDs/logs before taking action. `merge --output RUN` can rerun
the audit after a terminal supervisor state; unknown/nonzero worker exit codes
prevent a complete status. Re-merging does not re-run inference or rewrite shard
outputs. It refuses an actively running supervisor.

```bash
python -m baselines.ours_offline.parallel merge --output /path/new-final-run
python -m unittest baselines.ours_offline.tests.test_parallel -v
```

Supervisor tests use controlled fake worker files/process handles and fake GPU
telemetry. They exercise allocation, launch arguments/environment, occupancy
failure before spawning, provenance/causal/duplicate audits, exact coverage and
fixed-denominator failure handling. They perform no GPU or model inference.

For the published Qwen frame-budget comparison, set top-level `max_frames: 768`
and `sampling_fps: 0.5`. With `sampling_fps` omitted/null, the original all-CFR
uniform sampler remains in effect. Both settings are recorded and verified during
merge. This aligns the published rate/cap while preserving this benchmark's causal
cutoff and native checkpoint spatial resize; it is not an exact-score reproduction
claim. The default GPU preflight still requires all configured GPUs to be free.

Merged summaries include `performance` with all questions, `no_audio_required`,
`audio_required`, and unknown flags. The new worker statuses have the same groups.
For live or historical runs, `python -m baselines.ours_offline.report --run DIR`
rescores completed rows against the canonical labels. Interim denominators are
explicit; a stopped 32-frame run must not be merged with a new 768-frame run.
