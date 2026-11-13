# EgoRAG and benchmark source export

`baselines/egorag` contains the causal VideoMem adapter, its local-HF and API
backends, the bounded integration tool, synthetic tests, configuration examples,
and the seven upstream files required by its AST loader and provenance function.
The adapter's inference code and all test source files are byte-for-byte copies.
The original EgoLife revision is `7a97157908757cc898c26835b718653055ecc5f5` from
<https://github.com/EvolvingLMMs-Lab/EgoLife>. Its S-Lab License 1.0 is retained
beside the source. This export does not relicense that code.

`benchmark` contains the local Streamlit dataset-construction prototype,
synthetic demonstration, public TeleEgo download/assembly utility, dependencies,
and tests. `config/teleego_p4.json` is a pinned public-source download inventory;
it contains paths and content identifiers, not videos or narration text. The
utility performs downloads only when explicitly invoked by its operator.
Its main entrypoint is `streamlit run app.py` from the `benchmark` directory.
The `demo.py` script generates explicitly synthetic test videos and QA.

All of `lifelong_mem` is excluded. Neither export includes local media, real QA
or labels, annotations, source-video manifests, model files, environments,
caches, generated outputs, evaluation reports, or raw model logs. The model
example derived from the local Qwen settings replaces filesystem paths with
`/absolute/path/to/models`; its content identifiers and GPU policy are retained.
Users must supply their own inputs, review question cutoffs independently of
labels, and configure models before inference.

The original local-Qwen backend permits physical GPUs 4 and 5 only. The export
preserves that behavior and makes no claim of arbitrary-host GPU portability.
All software validation performed during export uses offline CPU fixtures.

[`egorag_export_manifest.json`](egorag_export_manifest.json) records original
and exported SHA256 values for every copied source, test, configuration, and
adapted document. Changed documents remove workstation paths and local dataset
status; no inference implementation was rewritten. The new overview READMEs
are export documentation and have no source-file hash.

## Validation

Run each suite from its own project directory to preserve the existing import
layout. Use `PYTHONDONTWRITEBYTECODE=1` and `pytest -p no:cacheprovider` when a
clean source tree is required. Both suites use generated synthetic media.
Local model tests mock model loading; they do not establish real-inference
quality, GPU correctness on another host, or benchmark accuracy.

Export validation on 2026-11-09 passed from the copied directories:

- EgoRAG: **146 passed** in 14.72 seconds.
- Benchmark prototype: **7 passed** in 17.23 seconds.
- All 55 copied-file hashes matched the export inventory; no copied file
  contained the original workstation root, and neither project contained
  generated media, databases, model weights, bytecode, or model-call logs.

The suites used the existing project environments with automatic pytest plugin
loading disabled, `CUDA_VISIBLE_DEVICES` empty, and Hugging Face/Transformers
offline modes enabled. No GPU model, external inference service, dataset download,
or private dataset was used. Temporary test artifacts stayed outside the export.
