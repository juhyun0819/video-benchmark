# fluxmem_qwen3vl

Custom training-free Qwen3-VL-8B backbone adaptation. It preserves all three native DeepStack feature levels while sharing main-feature TAS/SDC decisions. This changes the paper backbone and is not a reproduction of published results.

This source-only export includes local modules/tests and the required unchanged upstream source packages. It excludes datasets, actual QA, weights, output records, environments, and checkpoints. Upstream: [ShareLab-SII/FluxMem](https://github.com/ShareLab-SII/FluxMem), commit `6fdbf65b6f2af1a94ca41d542be5be86a565d338`, Apache-2.0; retain `LICENSE` and source notices. `evaluation/lmms-eval/lmms_eval/models/qwen2_5_vl_fluxmem.py` is included as a prompt-layout reference, not a complete installed evaluation suite.

## Setup

Work from this directory and create a separate Python 3.11 environment. Install `requirements.inference.txt`; select suitable Torch/CUDA wheels for your machine. FFmpeg and FFprobe are required for media decoding tests and manifest preparation. The recorded Transformers version is 4.57.3; Qwen2.5 and Qwen3 variants should use separate environments.

```bash
python -m pip install -r requirements.inference.txt
```

The Qwen3 adapter loads only the upstream Torch compression kernels through an isolated namespace; it uses native Transformers Qwen3 model/processor code.

Supply local `Qwen/Qwen3-VL-8B-Instruct` files at revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`. The runners never download models. Supply a verification JSON containing `repo_id`, `revision`, and `verified_files` entries with `path`, `bytes`, and `hash` (SHA256 or Git-blob SHA1); each loaded file must match the recorded digest. Qwen3 additionally checks coverage of required config/tokenizer/weight files. No verification records or model files are bundled.

## Running and portability

Use `python -m streaming.run --help` and [streaming/README.md](streaming/README.md). Supply manifests, reviewed query cutoffs, explicit trigger threshold, local model verification, and a fresh output directory. `configs/qwen3vl8b_streaming_pilot.json` is an example with placeholder paths, not a ready-to-run dataset. A finite long-memory capacity is required for bounded retained visual memory.

Source defaults retain original host paths and GPU policy for auditability. Explicit CLI/config paths override model and input defaults. The normal GPU guard permits physical indices 4/5; Day1 continuation supports 1–5. Port these policies deliberately. Do not invoke historical launch helpers unchanged on a new host. Runners now record the pinned original `upstream_commit` separately from `repository_commit`, which refers to this export repository.

Tests include controlled tensors, synthetic media, and tiny random models; two Day1 processor tests and the historical runtime-validation helper require local external assets. Run self-contained CPU subsets with `CUDA_VISIBLE_DEVICES=`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `PYTHONDONTWRITEBYTECODE=1`; pytest `-p no:cacheprovider` avoids repository caches. These checks do not establish pretrained answer quality.
