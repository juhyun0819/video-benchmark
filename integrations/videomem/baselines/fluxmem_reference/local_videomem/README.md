# Local VideoMem adapter

Inference-only adapter for the unchanged released Qwen2.5-VL FluxMem model. `data.py` validates video manifests, chronological observation prefixes, and public questions. `run.py` runs one prefix per question; it does not retain memory between questions.

Run from this variant directory. Install the inference runtime described in `README.md`, then use `python local_videomem/run.py --help`. Supply your own `--manifest`, `--questions`, `--model`, `--verification`, and a new `--output` directory. Source defaults retain the original workstation paths for auditability; override them explicitly. The GPU guard currently permits physical indices 4 and 5 and rejects occupied devices. Port that host policy deliberately before using other devices.

Questions contain only `question_id`, `question`, a string-list `choices`, and `query_time` (`day_id`, `day_order`, `seconds`). Observation intervals exclude the cutoff. Ground-truth answers and evidence metadata do not enter prompts. `prepare_manifest()` accepts user-owned media paths and records fingerprints; no original media or QA is included here.

`launch_when_ready.py` and `watch.py` are historical workstation helpers with local download-marker assumptions. Prefer the direct runner when porting. `check_smoke.py` validates a supplied execution directory. There are no bundled smoke inputs or results.
