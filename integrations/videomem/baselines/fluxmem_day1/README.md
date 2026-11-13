# fluxmem_day1

Adapted offline end-of-Day1 paired experiment using released Qwen2.5-VL FluxMem ON/OFF. It samples 1,024 temporal midpoints across a 39,016.8-second day. Sparse coverage and end-of-day access change the source QA task; this is not the original HD-EPIC protocol.

This source-only export includes local modules/tests and the required unchanged upstream source packages. It excludes datasets, actual QA, weights, output records, environments, and checkpoints. Upstream: [ShareLab-SII/FluxMem](https://github.com/ShareLab-SII/FluxMem), commit `6fdbf65b6f2af1a94ca41d542be5be86a565d338`, Apache-2.0; retain `LICENSE` and source notices. `evaluation/lmms-eval/lmms_eval/models/qwen2_5_vl_fluxmem.py` is included as a prompt-layout reference, not a complete installed evaluation suite.

## Setup

Work from this directory and create a separate Python 3.11 environment. Install `requirements.inference.txt`; select suitable Torch/CUDA wheels for your machine. FFmpeg and FFprobe are required for media decoding tests and manifest preparation. The recorded Transformers version is 4.51.3; Qwen2.5 and Qwen3 variants should use separate environments.

```bash
python -m pip install -r requirements.inference.txt
python -m pip install --no-deps -e models/qwen2-5-vl -e models/qwen-vl-utils
```

`--no-deps` avoids the upstream package metadata replacing the recorded inference runtime with older Torch and training packages.

Supply local `Qwen/Qwen2.5-VL-7B-Instruct` files at revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`. The runners never download models. Supply a verification JSON containing `repo_id`, `revision`, and `verified_files` entries with `path`, `bytes`, and `hash` (SHA256 or Git-blob SHA1); each loaded file must match the recorded digest. Qwen3 additionally checks coverage of required config/tokenizer/weight files. No verification records or model files are bundled.

## Running and portability

Use `python -m day1_compare.prepare --help` and `python -m day1_compare.run_v2 --help`. Preparation requires your own mapped Day1 QA/timeline and media in the documented source schema. Full continuation helpers require frozen pilot artifacts; none are included. [Day1 details](README_DAY1.md) explain the experiment contract.

Source defaults retain original host paths and GPU policy for auditability. Explicit CLI/config paths override model and input defaults. The normal GPU guard permits physical indices 4/5; Day1 continuation supports 1–5. Port these policies deliberately. Do not invoke historical launch helpers unchanged on a new host. Runners now record the pinned original `upstream_commit` separately from `repository_commit`, which refers to this export repository.

Tests include controlled tensors, synthetic media, and tiny random models; two Day1 processor tests and the historical runtime-validation helper require local external assets. Run self-contained CPU subsets with `CUDA_VISIBLE_DEVICES=`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `PYTHONDONTWRITEBYTECODE=1`; pytest `-p no:cacheprovider` avoids repository caches. These checks do not establish pretrained answer quality.
