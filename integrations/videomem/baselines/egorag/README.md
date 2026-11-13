# EgoRAG adapter for causal video memory

The supported entrypoint is `python -m videomem_adapter.cli`. It provides video
preparation, caption memory, summaries, retrieval, visual answering, and evaluation
with an explicit observation cutoff. Read [the adapter guide](videomem_docs/README.md)
for the protocol, configuration, and commands.

From the repository root, a CPU-only contract check is:

```bash
cd baselines/egorag
python -m venv .venv
.venv/bin/python -m pip install 'numpy>=1.26,<2' 'Pillow>=10,<12' 'pytest>=8,<9'
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m videomem_adapter.cli smoke --work /tmp/videomem-egorag-smoke
```

FFmpeg and FFprobe must be installed. Tests create synthetic media in temporary
directories. The fixture model and embedding backend do not download weights,
call an API, or use a GPU. The smoke is a software contract check; it does not
measure model quality or benchmark accuracy.

`requirements-videomem.in` supplies the additional production index and OpenCLIP
dependencies. API-backed runs require user-selected model IDs, revisions, and
environment variables. The optional [local Qwen backend](videomem_docs/local_qwen3vl.md)
requires additional model/runtime setup and preserves the original restriction to
physical NVIDIA GPUs 4 and 5. All paths in its example configuration are placeholders.

The `main.py`, `create_database.py`, and `egorag/` files are unchanged source
references used by the adapter's AST loader and provenance hashes. This is a
selected export, so use the adapter CLI rather than those upstream scripts.
The original project is [EgoLife](https://github.com/EvolvingLMMs-Lab/EgoLife),
revision `7a97157908757cc898c26835b718653055ecc5f5`. Retained upstream code is
subject to the included [S-Lab License 1.0](LICENSE), which permits noncommercial
use under its stated conditions. The upstream package metadata is not included.

No videos, real benchmark questions, private labels, model weights, or evaluation
results are supplied. Export provenance is recorded in
[`docs/egorag_export_manifest.json`](../../docs/egorag_export_manifest.json).
