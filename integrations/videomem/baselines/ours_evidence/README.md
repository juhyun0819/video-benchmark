# GT Evidence Clip reference for ours

This reference uses privileged evidence annotations to select original video
frames, then answers with Qwen3-VL-8B or LLaVA-OneVision-7B through the shared
[native video backend](../ours_offline/README.md). Only frames, actual timestamps,
question and choices reach the model. Answer labels and evidence metadata stay
in the evaluator. All methods use the same canonical QA file and scorer.

The target is **ours**. QA and evidence annotations are pending. Prepare the video
source index with `ours_offline prepare` now; run this reference after real QA and
evidence arrive. Missing evidence is a per-question error and remains in the
denominator. There is no substitute QA or guessed neighboring evidence.

## Evidence input

Each canonical question may include exactly one of these `evidence` forms:

```json
{"evidence": {"points": [120.5, 310.25]}}
```

```json
{"evidence": {"clips": ["clips/event-a.mp4", "clips/event-b.mp4"]}}
```

```json
{"evidence": {"intervals": [{"start_time": 120.5, "end_time": 135.0}]}}
```

Times are global seconds on the source index's declared timeline. There is no
EgoLife date or FPS assumption. `points` select the containing original source
clips; `clips` select verified root-relative source paths. Both sample the full
selected clip up to `query_time`. For day-long source files this can mean an
entire day: use explicit `intervals` for bounded event clips. Intervals are
half-open `[start_time, end_time)` and must be covered by recorded sources and
end no later than the query. They are never expanded into full-day evidence.

Alternatively, supply `--evidence /path/evidence.json` with an object containing
`qa_manifest_sha256` and `entries`, each entry having `qa_id` and `evidence`.
The hash must match the exact canonical QA file. Duplicate or unknown IDs fail
validation. Inline evidence is used when no sidecar is supplied.

Uniform sampling distributes the explicit frame budget across selected clips,
requesting at least two frames per clip. Original frame rates and actual decoded
timestamps are preserved. Overlapping tails belong to the later source. Future
evidence, source gaps, missing evidence and insufficient frame budgets produce
explicit errors. Labels never select evidence implicitly.

## Plan and run after QA arrives

```bash
DATA=/convei_nas2/zzuhyun/hyunjun/datasets/videomem
INPUT=/home/zzuhyun/hyunjun/ours_setup/input_v1

python -m baselines.ours_evidence plan \
  --root "$DATA" --manifest /path/to/reviewed-ours-qa.json \
  --sources "$INPUT/sources.json" --max-frames 64 \
  --output /path/to/new-evidence-plan

python -m baselines.ours_evidence run \
  --root "$DATA" --manifest /path/to/reviewed-ours-qa.json \
  --sources "$INPUT/sources.json" --max-frames 64 \
  --output /path/to/new-evidence-run \
  --backbone qwen3_vl --model "$MODEL_PATH" \
  --model-revision "$MODEL_REVISION" --ledger "$MODEL_LEDGER" --gpu 2
```

The model environment, immutable revision and trusted verification ledger use
the shared offline backend contract. No model downloads or training are started.
`--days`, `--shard` and `--shards` select independent QA subsets while retaining
earlier source history. Output directories must be new.

Plan/prediction/provenance/status artifacts record evidence policy, selected
sources, actual timestamps, prediction, raw output, GT, correctness, latency and
errors. Failed or incomplete outputs count as incorrect. No pretrained accuracy
or performance result is claimed by the implementation.

## EgoLife validation only

The EgoLife timestamp conversion helper lives under `tests/`; the production
entry point consumes only the generic schema above. Tests cover source gaps,
overlaps, multiple evidence points, half-open intervals, manifest binding and
the model input boundary.

```bash
python -m unittest discover -s baselines/ours_evidence/tests -v
```
