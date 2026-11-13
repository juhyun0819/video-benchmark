# Continuous memory benchmark

Observe Day1–Day7 as one chronological experience, save cumulative checkpoints,
and answer questions independently from any checkpoint. Adapters keep each
method's native representation and account for all retrievable persistent bytes.
Methods without a native bounded retention policy report unsupported budgets.

Run commands from the **repository root** using the selected method's environment.
The directory contains `adapters/`, `configs/`, `tests/`, `docs/` and attributed
upstream code in `vendor/`. Frozen model weights and local run artifacts are not
included. Full pretrained inference dependencies vary by method.

```bash
python -m baselines.streambench list-methods
python -m baselines.streambench prepare \
  --sources baselines/streambench/configs/sources.local.json \
  --output /path/input/frames.jsonl --fps 1
python -m baselines.streambench observe \
  --config baselines/streambench/configs/streamforest.local.json \
  --manifest /path/input/frames.jsonl --output /path/runs/sf-pilot \
  --budget-mib 256 --chunk-size 8 --stop-after-day 2
python -m baselines.streambench evaluate \
  --config baselines/streambench/configs/streamforest.local.json \
  --checkpoint /path/runs/sf-pilot/checkpoint_D1_D2 \
  --questions /path/reviewed/questions.json --output /path/results/answers.jsonl
```

Create the local configurations from the supplied [templates](configs), replacing
all paths with verified local inputs. The evaluation budget must match the
checkpoint. To continue observation, use `observe --resume /path/to/checkpoint`
with the same manifest/config/budget and a fresh output directory. Questions and
labels never enter observation. No seven-day run starts automatically.

Read [implementation and accounting notes](../../IMPLEMENTATION_NOTES.md),
[executed validation and limitations](docs/streambench_validation.md), and the
method notes: [native visual memories](docs/streambench_native.md),
[EgoRAG/VideoTree](docs/streambench_text_tree.md),
[StreamTTT/Cambrian-S](docs/streambench_recurrent.md),
[ReKV](docs/streambench_rekv.md), [AKS](docs/streambench_aks.md).
The [SimpleStream Qwen2.5-VL-7B adaptation](../simplestream/README.md) is available
as `method: "simplestream"`, with an explicit recent-frame window and native
position handling. Its standalone CLI can also read the existing verified
Qwen visual archive; the common `observe` command continues to use PIL frames.

Structural tests, using an isolated compatible environment:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q baselines/streambench/tests
# Include the optional pinned official StreamTTT source tests:
STREAMBENCH_STREAMTTT_SOURCE=/path/to/StreamTTT \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q baselines/streambench/tests
```

These tests use actual native memory kernels and explicit lightweight fixtures.
They do not establish pretrained accuracy or seven-day throughput.
