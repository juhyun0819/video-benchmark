# Source handoff validation — 2026-11-09

This repository is an isolated source export. Original experiment checkouts,
running evaluation processes, datasets, caches, and model files were not changed.

## Checks run on exported copies

| Component | Offline CPU tests passed |
| --- | ---: |
| EgoRAG adapter | 146 |
| Generic benchmark prototype | 7 |
| FluxMem reference bridge | 21 |
| FluxMem streaming extension | 92 |
| FluxMem Qwen3-VL adaptation | 141 |
| Day1 orchestration/evaluation subset | 54 |

These are per-directory suites; shared tests occur in multiple variants. The
counts are not independent semantic-quality samples or new benchmark results.
Tests used synthetic fixtures, mocked backends, controlled tensors, and tiny
random models where applicable. No pretrained inference, GPU inference,
finetuning, or external inference API was run for this export.

- 155 Python source files passed syntax parsing.
- 400 source/export hash comparisons passed against the recorded inventories.
- Staged-source checks found no matching common credential patterns, excluded
  runtime directories, model/media files, binary blobs, or oversized files.
- Each vendored component retains its upstream license and pinned revision.
- Exported FluxMem provenance records the upstream revision separately from the
  new repository revision; an archive without Git metadata records null for
  the latter. Inference algorithms are unchanged by this metadata adjustment.

## Limits

External media, real QA/labels, model weights, checkpoint-verification records,
and frozen pilot outputs are intentionally absent. Historical tests that need
an installed pretrained processor or private assets were not included in the
self-contained validation subset. Configure local paths and GPU policies before
running on another machine. This source export is not a ready-made dataset or
an independently reproduced accuracy result.
