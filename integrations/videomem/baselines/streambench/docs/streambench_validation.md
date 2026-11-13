# Validation record — 2026-11-10

Structural tests and pretrained execution provide different evidence. No full
seven-day run, full budget sweep, or benchmark accuracy result is claimed.

## Structural validation

After relocation, the complete benchmark suite passed **109 tests in 52.07
seconds, with zero skips**, including optional tests against the pinned official
StreamTTT source at `/tmp/videomem-audit-streamttt`. The exact command from the
repository root was:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 \
  STREAMBENCH_STREAMTTT_SOURCE=/tmp/videomem-audit-streamttt \
  ../streambench-test-env/bin/python -m pytest -q -p no:cacheprovider \
  baselines/streambench/tests \
  --junitxml=../streambench-validation/streambench-relocated-tests.xml
```

The pre-relocation suite also passed 109 tests. The tests exercise
physical-storage accounting, native compression/eviction,
byte limits, chronological common input, cumulative checkpoint/resume, cursor
and RNG preservation, question isolation, exception restoration, model identity
checks, no-evidence abstentions, and explicit unsupported budgets. Native visual
memory, EgoRAG, VideoTree, ReKV and AKS tests include fresh-process continuation.
Cambrian's Recall Memory also has a fresh-process state test. The optional tiny
native StreamTTT multimodal test restores cache and all 12 TTT fields in the
**same process**; a full native StreamTTT fresh-process continuation test remains
unverified. The reduced dimensions belong only to labeled structural tests,
never to the production adapter's pretrained architecture.

Separate focused validation earlier in implementation included **164 passing
tests** across the existing EgoRAG suite and the then-current text/tree tests.
After the production-role, artifact-identity and abstention fixes, the text/tree
and AKS suites passed **38 tests in 15.00 seconds**. Those historical counts are
subsets or earlier suite revisions, not additional independent accuracy runs.

## Actual pretrained StreamForest smoke

An existing local pretrained StreamForest model completed a CPU float32 smoke
using **2 synthetic color frames across 2 logical days and 2 QA calls**. The
report labels this `test_only: true` and `pretrained_weights_loaded: true`.
Reported wall time was
**104.70 seconds** and maximum persistent memory was **7,322,524 bytes** under a
64 MiB budget. The two QA outputs matched. This verifies
that narrow end-to-end path; it does not establish meaningful QA accuracy,
long-horizon retention, GPU performance, or full-day capacity.

This was a **pre-relocation historical run** from the repository root:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=8 \
  ../streambench-test-env/bin/python ../streambench-validation/smoke_streamforest_pretrained.py
```

The local artifacts were inspected at
`/home/zzuhyun/현준/streambench-validation/streamforest-pretrained-smoke-result.json`
and `streamforest-pretrained-smoke.log` in that same directory. The external
script retains the old import path and is not shipped as a portable command.
No model weights, videos, smoke outputs or environment files are published.

## Remaining execution limits

Pretrained inference for the other adapters remains unverified in this work:
FIFO/Reservoir, FluxMem variants, EgoRAG, VideoTree, StreamTTT, Cambrian-S, ReKV
and AKS. Their native source/API integration and structural tests are implemented;
missing compatible dependencies or checkpoints raise errors, with no substitute
model. Official method-specific environments must be validated separately.

The existing QA package also needs reviewed public alias/time mapping before
scientific checkpoint scoring. Larger model smoke tests, native StreamTTT
fresh-process inference, complete budget pilots, and seven-day quality/latency
experiments remain future validation work. See the [implementation notes](../../../IMPLEMENTATION_NOTES.md)
and linked method notes for deviations and unsupported settings.

## Source publication check scope

The existing export checker does **not** pass over the entire tracked repository:
it reports 31 pre-existing entries in `data/daywise_annotations_qa_v2_20261108`
for JSONL/non-source or oversized-file rules. That historical package is unchanged.
Publication checking for this change uses the same unmodified checker against a
temporary Git index containing only newly added or modified staged files. All
66 changed source files passed, with no failures. This is not a whole-repository
pass. First-party changes also pass `git diff --cached --check`; vendored sources
retain upstream whitespace to preserve their verified source hashes.
