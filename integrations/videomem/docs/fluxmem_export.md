# FluxMem source export

Four independent import roots preserve the original integrations: `fluxmem_reference` (released Qwen2.5 implementation plus local adapter), `fluxmem_streaming` (custom persistent extension), `fluxmem_qwen3vl` (custom DeepStack-aware backbone adaptation), and `fluxmem_day1` (sparse paired offline condition). Each variant README documents dependencies, entrypoints, and limitations.

Upstream source is pinned to `6fdbf65b6f2af1a94ca41d542be5be86a565d338` from ShareLab-SII/FluxMem under Apache-2.0. Required model source and the small original QA wrapper are copied unchanged. Local source includes previously untracked modules and tests. Runners separate pinned upstream provenance from this repository's commit. Export documentation and the Qwen3 example paths were rewritten; inference algorithms are unchanged.

`fluxmem_export_manifest.json` records original paths, original/export SHA256, file classification, and whether export bytes changed. The original working trees and active Day1 jobs were not modified. No media, real questions, answer labels, raw results, logs, weights, checkpoints, Git history, environments, or upstream training/evaluation tree is included.

Hard-coded original paths and GPU allowlists remain in source and require deliberate porting or explicit supported overrides. The two model families require separate pinned Transformers environments. Historical tests requiring external processor/media files are distinguished from self-contained CPU checks; see validation in the manifest. No claim is made that all paper methods or published benchmark results are reproduced.

The export Git commit/branch metadata is optional: it is null for archives or unborn checkouts. This changes provenance handling only. Self-contained CPU suites passed for reference (21), streaming (92), Qwen3 (141), and Day1 (54): 308 checks total. All 112 Python files passed compilation without writing bytecode.
