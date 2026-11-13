# Adapted sparse Day1 comparison

The paired condition uses released Qwen2.5-VL-7B FluxMem enabled versus disabled on identical 1,024 midpoint samples at 224 square pixels, identical questions and greedy BF16/SDPA generation. The observation cutoff is the end of the logically concatenated Day1. These are sampled frames, not full per-second observation. All supplied Day1 QA concern the final cooking clips; old long-term recall is untested. Some questions contain no sample within their public input-alias interval.

`run_v2.py` uses video-before-question layout. Both arms apply the same runtime `config.sliding_window=None` correction to honor the checkpoint's full-attention setting; original upstream source and weights remain untouched. Both use official `video_chunk_t=1` and original `generate`, without substituting cached projected embeddings.

`prepare.py` freezes question selection, frame sampling, public inputs and separate evaluation labels. `evaluate.py` alone opens gold labels. `check_pair.py` audits paired conditions. `prepare_full.py`, `run_multi_gpu.py`, `launch_full.py`, `progress_full.py`, and `aggregate_full.py` support frozen multi-GPU continuation and integrity-checked aggregation. The original v1 output-contract failure and later v2 runs are separate conditions. No historical outcomes or answer pools are bundled.

Preparation needs user-owned mapped QA and media. See `prepare.py` for the timeline/QA schema and CLI flags; local defaults must be overridden. Full continuation expects preserved pilot artifacts and cannot run from this source export alone. The original full continuation was active during export; this package claims no completed full-run result.

This is an adapted task: end-of-day access supplies later information to anticipation questions, labels were not revalidated against sparse samples, and logical concatenation is not guaranteed equivalent to a separate rendered day. It does not exercise persistent streaming, proactive triggers, finite eviction, or training. See [setup and dependencies](README.md).
