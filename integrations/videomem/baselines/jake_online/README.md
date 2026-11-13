# JAKE online evaluation

Evaluate EgoLife A1_JAKE with a shared causal QA manifest and
`baselines.streambench.scoring.score_answer`. Run modules from the repository root.
Videos, checkpoints, full predictions, environments, and credentials stay outside Git.

## Implementations

- **SimpleStream-4B / 8B:** `Qwen/Qwen3-VL-{4B,8B}-Instruct`, recent four frames
  sampled on each source clip's 1 Hz grid, greedy generation, at most 256 tokens.
  This is a **native raw multi-image adaptation**, not a claim of equivalence to
  the upstream cached-embedding builder. Audio is not supplied; audio/name-required
  questions remain in the denominator.
- **StreamTTT-4B:** official `zeyun-zhong/StreamTTT-4B`, revision
  `58f05be5c71549d6c4826325e6d454d2b0b59228`, with upstream source revision
  `96aed21cc4d78dc7e1787a53a0a95691f3cfd07a`. Uses native raw video inputs at
  2 fps, a 262144-pixel budget, recent KV length 4096 and official MC option logits.
  No additional training or visual-feature cache. All recurrent fields, KV, pending
  temporal pairs and position counters are checkpointed. Each QA forks/restores state.

The **16 GiB budget is for observation state**, not total GPU memory. The current
run uses 32-frame forward windows: 192 frames exceeded a 3090's memory, and 96
frames failed later at QA22. Resume from QA20 preserved all state tensors while
changing only the forward window size. Consequently the run has mixed window
sizes; it is not an untouched official benchmark configuration.

`placement.py` is a **custom, optional** placement of decoder layers 18–35 on a
second GPU. The upstream source remains unchanged. A short checkpoint-based test
found bitwise-identical continuation state for 1GPU and 2GPU at the same batch size;
QA fork/restore also passed. This is not a general numerical-equivalence guarantee.
The production run switched to this placement at QA30. Time is never sharded into
independent zero-state streams.

`prefetch.py` overlaps one raw-frame decode window with GPU work. Lookahead is
bounded by the current QA cutoff. It caches no visual embeddings. The runner checks
actual PTS and conservatively bounds raw buffer bytes against the observation budget.
Decode and update times overlap: **do not add them to estimate wall time**.

## Data and canonical time

Rack5 data root: `/convei_nas2/zzuhyun/hyunjun/datasets/egolife`.
All 6266 JAKE clips have size/SHA256 receipts. Day1 initially used `canonical/`;
all-days runs use `canonical_v2/` with `jake_qa_v1.json` (the JSON schema is v2),
`all_sources_v2.json` and the common scorer hash.

`prepare.py` makes the original Day1 index. Deployment-specific scripts under
`rack5/` create and audit the all-days manifest. Their absolute paths are intentional
Rack5 deployment paths; edit them for a different checkout. They require existing
raw QA, inventory, download receipts and the Day1 canonical index. They do not
silently download missing data. `prepare_all_jake.py` requires an absent output
`canonical_v2/` directory and the official EgoLife checkout.

The all-days conversion follows `time_to_frame_idx` in official EgoLife revision
`7a97157908757cc898c26835b718653055ecc5f5`,
`EgoRAG/egorag/utils/util.py`: `(HH*3600+MM*60+SS)*20+FF`.
For 26 QA timecodes with FF>=20, this carries the frame offset into later seconds;
raw annotation values and each conversion are retained. This applies official code
arithmetic rather than interpreting FF as centiseconds. Day1 cutoffs are unchanged.
Clip tails overlapping the next source are trimmed from StreamTTT's sample plan.

## Environment

StreamTTT: Python 3.12, torch 2.8.0+cu126, torchvision 0.23.0+cu126,
transformers 4.57.1, flash-attn 2.8.3 and causal-conv1d 1.5.4.
See `rack5/streamttt-inference-requirements.txt` for remaining inference pins;
install the matching official CUDA extension wheels separately. Training-only
DeepSpeed/PEFT are not required by this inference path.

SimpleStream native adapter: Python 3.11, torch 2.9.1+cu128,
transformers 4.57.6, flash-attn 2.8.3.post1. Both paths retain native Conv3D and use
channels-last storage to avoid a slow RTX3090 BF16 convolution path.

## Example commands

Use an appropriate environment's Python, free GPUs, existing verified model ledgers,
and a **new output directory** for each invocation.

```bash
DATA=/convei_nas2/zzuhyun/hyunjun/datasets/egolife
MODELS=/convei_nas2/zzuhyun/hyunjun/models
SETUP=/home/zzuhyun/hyunjun/jake_setup

python -m baselines.jake_online.run_simple \
  --root "$DATA" --canonical "$DATA/canonical_v2" \
  --source-index all_sources_v2.json --days 1 2 3 4 5 6 7 \
  --model "$MODELS/Qwen3-VL-8B-Instruct" \
  --ledger "$SETUP/Qwen3-VL-8B-Instruct.verification.json" \
  --output /path/to/new-simple-run --gpu 4 --shard 0 --shards 4 --budget-gib 16

# Resume the verified two-GPU deployment from the QA30 checkpoint.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m baselines.jake_online.run_ttt \
  --root "$DATA" --canonical "$DATA/canonical_v2" \
  --source-index all_sources_v2.json --days 1 2 3 4 5 6 7 \
  --model "$MODELS/StreamTTT-4B" \
  --ledger "$SETUP/StreamTTT-4B.verification.json" \
  --upstream /home/zzuhyun/hyunjun/upstream/StreamTTT \
  --output /path/to/new-streamttt-run --gpu 0 --secondary-gpu 1 \
  --window-frames 32 --prefetch-raw --audit-state \
  --resume /convei_nas2/zzuhyun/hyunjun/jake_runs/streamttt_all_days_b16_v2/checkpoint_qa030
```

A SimpleStream shard must have a distinct output directory and GPU. Merge by unique
QA ID and verify exact coverage of all 500 questions. Use the 4B model/ledger for
SimpleStream-4B. StreamTTT is **one** continuous stream, not one independent run per day.
For an explicitly audited change of forward size, `--resume-window-frames-from`
checks that all other model identities match and continuation tensor values survive.

Per-run `provenance.json`, `status.json`, `predictions.jsonl`, update records and
resume audits provide settings, inputs, errors, scores and timing. Checkpoint files
contain trusted local Torch serialization and integrity hashes. Never load untrusted
checkpoint files.

## Results and validation

Run-specific reports, accuracy snapshots, and live job state are not included in
this source handoff. Produce them in an external output directory for the exact
dataset and model configuration being evaluated. A partial trial must not be
compared directly with a completed full-set score.

```bash
python -m unittest baselines.jake_online.test_inputs -q
```

Runtime checks additionally cover native video token reconstruction, causal decoded
PTS, source receipts, checkpoint integrity, state restoration and QA-state preservation.
Full-data audits require the separately supplied source videos and canonical QA.
