# EgoRAG and VideoTree continuous-memory adapters

These adapters retain their text/tree representations. Neither inspected
upstream implementation defines an online, byte-bounded eviction policy. Both
therefore retain the complete required archive and return
`unsupported_under_budget` when it exceeds B. Changing VideoTree's query breadth
does **not** reduce its persistent feature archive. Neither adapter retrains a
model or claims that structural tests establish model accuracy.

| Method | Encoder | Retained representation | Native budget policy | Minimum memory | QA reads | Training |
|---|---|---|---|---|---|---|
| EgoRAG wrapper | Explicit configured VLM; normalized OpenCLIP text embeddings | Caption sentences, 10-minute/hour/day summaries, SQLite embeddings/index/call cache, metadata, retained sampled JPEGs | Retain all; reject overflow | Measured empty SQLite pages plus configuration; grows with observations | Native temporal selection and sentence retrieval, then retained JPEG evidence | None |
| VideoTree wrapper | Official local EVA-CLIP-8B `encode_image` | Every observed feature row, caption, timestamp | Retain all; reject overflow | Measured metadata; grows with observations | Native cosine k-means, relevance-guided Ward hierarchy, representative captions | None |

## Sources and exact scope

EgoRAG reuses the existing causal `baselines/egorag/videomem_adapter/Pipeline`
and its unmodified upstream retrieval function imports. The bundled EgoLife
revision is
[`7a97157908757cc898c26835b718653055ecc5f5`](https://github.com/EvolvingLMMs-Lab/EgoLife/tree/7a97157908757cc898c26835b718653055ecc5f5).
See the existing [upstream audit](../../egorag/videomem_docs/upstream_audit.md)
for paper/code differences. The new wrapper does not replace its model backends.

VideoTree source is
[`Ziyang412/VideoTree@605ec8e917f9a0f95d37a73a0b50bf0e8cfe076e`](https://github.com/Ziyang412/VideoTree/tree/605ec8e917f9a0f95d37a73a0b50bf0e8cfe076e).
Eight verbatim function definitions from `kmeans_pytorch/__init__.py`,
`adaptive_breath_expansion.py`, and `depth_expansion.py` are included under
`baselines/streambench/vendor/videotree`. The MIT license, original file SHA256s,
individual function SHA256s, original line numbers and vendored-file SHA256 are
recorded alongside them. Import wiring alone is adapted; SoftDTW is not part of
the supported cosine path. The original CLI is not imported: it has hard-coded
dataset paths, CUDA placement, mismatched GPT constructor arguments and script
argument errors.

The official extraction script loads `BAAI/EVA-CLIP-8B` and the
`openai/clip-vit-large-patch14` image processor, calls `encode_image`, and stores
one projected row per sampled frame. The inspected EVA model repository head is
[`0e4dca944e8ece27eb9dfe4a488c0ed0c4644fc9`](https://huggingface.co/BAAI/EVA-CLIP-8B/tree/0e4dca944e8ece27eb9dfe4a488c0ed0c4644fc9).
Its [configuration](https://huggingface.co/BAAI/EVA-CLIP-8B/blob/0e4dca944e8ece27eb9dfe4a488c0ed0c4644fc9/config.json)
specifies 224-pixel input and a 1280-dimensional projection. The wrapper checks
the actual returned shape against the loaded configuration, preserves the
actual dtype, and accounts actual tensor storage. It never assumes that a
paper's token count is its byte count.

The [paper configuration](https://arxiv.org/html/2405.19209v3#S4) uses GPT-4
version 1106, EVA-CLIP-8B, LaViLa for EgoSchema and 1 FPS. Its EgoSchema breadth
starts at 8 and doubles up to 32 until at least four representatives have
relevance 3; depth is up to three with branch width four. The released code
consumes precomputed LaViLa captions and does not contain LaViLa inference or a
trained VideoTree checkpoint. The framework itself is training-free.

## Necessary adaptations and limitations

VideoTree is originally an offline, **question-adaptive** algorithm. There is no
official online insertion, inter-day merge, or storage eviction algorithm.
`observe` now appends query-independent features and captions; `answer` builds a
temporary tree over the retained archive. This permits streaming ingestion
without question leakage, but query computation still scans the full feature
archive and Ward clustering can be expensive on long sequences. Do not describe
this as a new native online tree-memory algorithm.

The production factory provides actual local EVA inference and explicit
VLM/LLM backend calls. Its configured per-sample captioner replaces upstream
precomputed LaViLa captions. The backend may use the repository's local Qwen3-VL
implementation or an explicitly configured compatible API. This substitution
and the JSON relevance/answer prompts are deviations, recorded in checkpoint
identity and answer metadata. The factory does not silently supply a captioner,
weights, credentials, test double, or fallback model. Exact LaViLa captioner
reproduction is **not** implemented by the production factory; a separately
verified caller can provide that callable through the Python adapter interface.

VideoTree keeps native cosine k-means, its 15-iteration limit, Euclidean distance
for first-level representative choice, Ward linkage for depth expansion, and
cosine-to-mean depth representatives. The wrapper makes these explicit repairs:

- Use zero-based frame/caption IDs. Upstream's prompt formatter subtracts one
  and drops frame zero despite clustering returning zero-based indices.
- Map relevance scores from temporally sorted representative order back to
  cluster IDs. Upstream indexes this sorted list directly by cluster label.
- Bound k by the available prefix length. A one-cluster prefix uses the exact
  mean solution because upstream `squeeze()` loses the distance matrix axis.
- Preserve singleton high-relevance children that upstream skips, and remove
  duplicate selected frame IDs. These are query-time correctness repairs.
- Use CPU clustering, fixed recorded seeds and scoped RNG state for deterministic
  structural replay; hardware differences may affect floating-point ties.

EgoRAG captions each common input chunk, splitting only if a chunk crosses a day
ID. Use a 30-frame chunk at 1 FPS to approximate the existing adapter's 30-second
caption unit. Short terminal chunks remain observations. The wrapper uses the
common sampled images, without opening source videos again. JPEG compression
at quality 95 and first-sample-relative day origins are declared preprocessing
adaptations. Set `frame_period_seconds` to the common sampling interval. The
native HHMMSScc schema requires each day interval to remain below 24 hours.
Summary creation is performed at chunk finalization, independent of questions.

The existing wrapper's SQLite exact-cosine path is supported; persistent Chroma
ANN snapshots are explicitly rejected rather than partially serialized. The
last-time path retains the existing SQLite squared-L2 substitute for native
Chroma ranking. This is an existing adapter deviation, not a new claim of
paper-identical ANN retrieval. The existing visual-answer variant remains:
sampled JPEG bytes are kept in memory, included in B, and checkpointed.

## Complete state and accounting

EgoRAG schema 1 contains the in-memory SQLite image (all captions, sentence rows,
embedding JSON, index markers, summaries and observation-call cache), native
provenance and namespaces, manifest, per-day origins, last frame, current
cutoff, configuration, sampling interval and every retained JPEG plus metadata.
The fixed index namespace persists while the observed manifest grows. There is
no day-boundary reindex or previous-video replay. SQL rows are identical across
continuous/resumed execution; SQLite file-change header counters can differ
without changing any record or retrieval result.

SQLite's complete allocated page image is charged to `text_memory_bytes`,
including its text-serialized embeddings and index overhead. It is the sole
live store, with no duplicate disk database. JPEG payloads and their frame
metadata are in `visual_memory_bytes`; configuration, manifest, origins and
provenance are in `metadata_bytes`. This categorization deliberately includes
all SQLite pages rather than undercounting only caption characters. SQLite
internal allocator overhead is runtime RAM. Audit traces and provenance files
are never read back as model memory and are excluded.

VideoTree schema 1 contains the actual CPU feature tensor, captions, complete
common timestamps, algorithm parameters/seed, upstream revision, backend/model
identity and test-only flag. It has no partial frame buffer: each observe call
encodes and captions every supplied sample before returning. Frozen encoder and
language-model weights are excluded; the entire feature archive is counted.
Temporary query trees, relevance scores and decoded tensors are discarded.

The production VideoTree identity fingerprints all local loader inputs: encoder
weights/shards, EVA Python code/configuration, processor configuration, and local
language-model weights/tokenizers. Each file records its byte count and SHA-256;
declared immutable revisions are recorded separately from those measured hashes.
The native language backend's `cache_identity` is included. Its verified manifest
is reused when present; otherwise the wrapper hashes the local language directory.
Reuse checks reject changed file inventories or filesystem stat anchors, and a
new process rehashes the files before comparing checkpoint identity. The guard
does not detect an adversarial rewrite preserving every recorded filesystem stat.
API backends record configured model/revision and a hash of the resolved endpoint,
without credentials; server-side alias changes still require operator versioning.

EgoRAG QA uses native `query` and `answer`, tracks unique persistent cache/index
records returned plus retrieved JPEG bytes, and restores the observation SQLite
image in `finally`, even if inference fails. This is logical payload read volume,
not filesystem page I/O or repeated cache-hit traffic. VideoTree read volume
includes the full feature tensor used in clustering, unique captions read during
breadth/final reasoning, and selected timestamp records. Both return
`query_read_bytes`, retrieved-item counts and retrieved identifiers. The common
runner additionally restores the complete checkpoint and RNG per question.
An empty archive or native EgoRAG retrieval miss returns an explicit abstention:
`answer: ""`, `status: "no_observed_evidence"`. It remains a scored miss when a
label exists, without terminating evaluation of later questions.

## Configuration and execution

For EgoRAG, make a JSON config with `method: "egorag"`,
`memory_budget_bytes: 268435456`, `frame_period_seconds: 1.0`, and
`pipeline_config` pointing to an existing fully specified EgoRAG JSON config.
That native config must explicitly select `index_engine: "sqlite"` and actual
model/embedding backends. For local Qwen3-VL the existing backend's GPU UUID
guard still applies; resolve the allowed physical GPUs 4 and 5 to UUIDs before
launching, since the common accounting module imports torch early.

For VideoTree the factory config is:

```json
{
  "method": "videotree",
  "memory_budget_bytes": 268435456,
  "encoder": {
    "checkpoint": "/absolute/local/EVA-CLIP-8B",
    "processor_checkpoint": "/absolute/local/clip-vit-large-patch14",
    "revision": "0e4dca944e8ece27eb9dfe4a488c0ed0c4644fc9",
    "processor_revision": "REPLACE_WITH_VERIFIED_40_CHARACTER_REVISION",
    "device": "cuda:0",
    "dtype": "float16"
  },
  "language_config": {
    "models": {
      "captioner": {"backend": "openai_compatible", "model": "CONFIGURED_VISION_MODEL", "revision": "CONFIGURED_REVISION", "temperature": 0, "max_tokens": 256},
      "relevance": {"backend": "openai_compatible", "model": "CONFIGURED_TEXT_MODEL", "revision": "CONFIGURED_REVISION", "temperature": 0, "max_tokens": 256},
      "answerer": {"backend": "openai_compatible", "model": "CONFIGURED_TEXT_MODEL", "revision": "CONFIGURED_REVISION", "temperature": 0, "max_tokens": 256}
    }
  }
}
```

Supply verified local checkpoints and explicit endpoint environment variables
accepted by `videomem_adapter.backends.Models`. Local loading uses
`local_files_only=True`; the wrapper does not download models. EVA's reviewed
checkpoint implementation requires `trust_remote_code=True`. The placeholder
config deliberately fails until real model identities are supplied.
The public `relevance` role maps to the native backend's supported `evidence`
dispatch role; its relevance prompt remains the VideoTree-specific JSON prompt.

```bash
python -m baselines.streambench observe --config egorag-stream.json --manifest frames.jsonl --output runs/egorag --chunk-size 30 --stop-after-day 1
python -m baselines.streambench observe --config videotree-stream.json --manifest frames.jsonl --output runs/videotree --chunk-size 30 --stop-after-day 1
python -m baselines.streambench evaluate --config videotree-stream.json --checkpoint runs/videotree/checkpoint_D1 --questions questions_day1.json --output answers.jsonl
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q baselines/streambench/tests/test_streambench_text_tree.py
```

The focused tests use explicitly labeled pixel/model doubles with real SQLite
retrieval and official clustering kernels. They cover both adapters' fresh
process checkpoint continuation, day persistence, accounting, rejection rather
than fabricated eviction, QA isolation including failures, chronological input,
and tiny-prefix tree repairs. No real-model accuracy, latency or minimum GPU
memory has been measured in this implementation session.
