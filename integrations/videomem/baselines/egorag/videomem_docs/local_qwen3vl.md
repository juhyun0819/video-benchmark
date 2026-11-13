# Local Qwen3-VL backend

`hf_local` runs Qwen3-VL directly through Transformers without an API server,
endpoint, or API credentials. The ignored configuration
`configs/videomem_qwen3vl.local.json` can be created from the supplied
`configs/videomem_qwen3vl.example.json`. Replace all placeholder paths with your
own model files and verification manifest. Its pinned checkpoint identifiers are:

| Component | Checkpoint | Pinned revision |
| --- | --- | --- |
| Caption, evidence, summary, retrieval and answer models | `/absolute/path/to/models/Qwen3-VL-8B-Instruct` | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
| OpenCLIP text embeddings | `/absolute/path/to/models/CLIP-ViT-B-32-laion2B-s34B-b79K/open_clip_model.safetensors` | `1a25a446712ba5ee05982a381eed697ef9b435cf` |

The configured OpenCLIP architecture is `ViT-B-32`, with the official
`laion2b_s34b_b79k` weights. Its SHA256 is
`ac4f8c4b88af6d963118cbf40ad93176d092abbedfcb752601ae1866352656e6`.
Embeddings run on CPU, and the production index remains Chroma.

## Runtime and devices

Use the isolated `baselines/egorag/.venv-qwen3vl` environment. Its intended
runtime is PyTorch 2.9.0 with CUDA 12.8, Transformers 4.57.3, and Accelerate 1.12.0;
no installed runtime or model files are bundled with this export. Importing the
backend or validating ordinary settings does not import Torch or allocate GPUs.
Checkpoint integrity verification does read the local files on CPU.

Only physical NVIDIA GPUs **4 and 5** are permitted. For a fresh Python process:

```bash
cd baselines/egorag
CUDA_VISIBLE_DEVICES=4,5 .venv-qwen3vl/bin/python -m videomem_adapter.cli create \
  --config configs/videomem_qwen3vl.local.json \
  --manifest videomem_runs/manifest.json --work videomem_runs/qwen3vl --limit 2
```

Before importing Torch, the backend resolves physical indices 4 and 5 through
`nvidia-smi` and replaces that numeric mask with their UUIDs. It rejects other
devices, reversed UUID order, or an unverifiable identity. If Torch has already
been imported, launch with the exact GPU 4/5 UUID mask instead; a late change from
numeric IDs is rejected because CUDA may already have read that mask. The
backend also checks the CUDA runtime's actual device UUIDs before loading weights.
It never uses `device_map="auto"` or a CPU/offload fallback.

Captioner and evidence share a lazy model replica on logical `cuda:0` (physical
GPU 4). The remaining roles, including the visual final answerer, share a second
replica on logical `cuda:1` (physical GPU 5). Each replica loads only when one of
its roles is first called. The configuration uses bfloat16 and SDPA attention.
Processes have independent caches; run the bounded integration tool below to
reuse both replicas through all stages in one process.

Frames are decoded by the existing bounded FFmpeg path on CPU, then opened as
RGB PIL images on CPU. The processor receives timestamp/Day/clip labels followed
by their corresponding images, with image processing explicitly on CPU. Only
the processed tensors move to the selected GPU. The configuration keeps at most
30 frames per 30-second clip, resized to at most 512 pixels on the longer side;
processor pixel bounds are 65,536–262,144 per image. A smoke configuration may
reduce the frame count; its saved config identifies that difference.

## Checkpoint verification and cache identity

Every local model role in the supplied config references
`/absolute/path/to/models/.downloads/qwen3_vl_8b_instruct/verification.json`. Supply this file using the schema validated by `videomem_adapter/local_integrity.py` and
exercised by `tests/test_videomem_local_integrity.py`; private download reports are not bundled. Pipeline construction verifies its revision and streams
the local files again, checking each source SHA256 or Git blob hash and computing
an actual SHA256 identity. The weight index must reference verified shards.
Tokenizer, processor, and other loadable model files must also be covered.

Verification is shared across roles and replicas within a `Models` instance.
The pinned local OpenCLIP file is also checked against its configured SHA256.
Actual checkpoint identities participate in pipeline provenance and cache keys;
a declared revision string alone is not treated as proof of file contents.
File-state checks before inference and cache reuse reject changes to the device,
inode, size, modification time, or change time of verified files. These checks
depend on filesystem timestamp resolution; an in-place edit within one timestamp
tick that preserves every checked value is not detected without another full
hash pass. Keep checkpoints immutable during a process, and start a new process
to verify a changed checkpoint and create its cache namespace.

The `verified_manifest` option is optional for custom `hf_local` configurations.
Omitting it records `verification: not_configured`; it does **not** prove that the
local files match the declared model revision. The supplied production config
requires the verification manifest and fails while the download is incomplete.

## Bounded integration check

`videomem_tools/local_qwen_smoke.py` exercises captioning, summaries, retrieval,
video evidence, a visual answer, and caption/embedding resume using a short real
MP4. It requires actual local models, official OpenCLIP embeddings, and Chroma.
It rejects fixture model or embedding settings. It asks an explicitly labeled
synthetic smoke question and does not select or score benchmark QAs.

```bash
CUDA_VISIBLE_DEVICES=4,5 .venv-qwen3vl/bin/python videomem_tools/local_qwen_smoke.py \
  --source /absolute/path/to/a/short-real-clip.mp4 \
  --config configs/videomem_qwen3vl.local.json \
  --work videomem_runs/qwen3vl_smoke --cutoff 60
```

The tool requires at most four 30-second virtual clips and an explicit cutoff
between 30 and 90 seconds. Download completion and runtime preparation are
prerequisites. Inspect `result.json`, the saved config, and `run/model_calls.jsonl`
for the outcome. Preparing this backend and passing mocked tests do not establish
that real inference or the integration check has passed.

Local traces include the role, declared revision, actual checkpoint digest,
physical GPU identity, package versions, CPU image handling, generation settings,
token counts, and duration. They retain text prompts and frame hashes/timestamps,
while excluding image bytes. `max_tokens` maps to `max_new_tokens`; reaching the
limit without an EOS token raises an execution failure and never returns a
partial answer as successful. Empty outputs and model errors also fail without
changing models or choosing a fallback answer.

## Validation limits

The CPU contract tests mock Torch and Transformers and do not load weights:

```bash
python -m pytest -q tests/test_videomem_*.py
```

Real inference results, when available, belong in the smoke artifacts. Neither
these tests nor a bounded smoke establishes multi-day model quality, benchmark
accuracy, full-run cost, final Day-render alignment, audio understanding, or
numerical reproduction of the EgoLife paper. The existing causal query-time and
label-isolation requirements still apply.

Implementation follows the pinned checkpoint's local `README.md` example:
`Qwen3VLForConditionalGeneration`, `AutoProcessor.apply_chat_template` with
tokenization and generation prompting, followed by generated-prefix removal.
The original API-compatible and test-fixture backends remain separately selected
by their existing `backend` settings.
