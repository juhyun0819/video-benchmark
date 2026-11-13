"""The unchanged v2 inference protocol with explicitly authorized GPU placement.

The pilot entrypoint still allows only GPUs 4 and 5. This full-run entrypoint
allows physical GPUs 1 through 5 following the user's expanded authorization.
Every question's ON/OFF pair stays on the same GPU in launch_full. No inference
setting, model weight, prompt, frame, FluxMem operation, or decoding is changed.
"""
from __future__ import annotations

import os
import subprocess

from day1_compare import run, run_v2

ALLOWED_GPUS = (1, 2, 3, 4, 5)


def bind_gpu(index):
    if index not in ALLOWED_GPUS:
        raise ValueError('Only physical GPUs 1 through 5 are authorized')
    raw = subprocess.check_output([
        'nvidia-smi', f'--id={index}',
        '--query-gpu=index,uuid,name,memory.used',
        '--format=csv,noheader,nounits',
    ], text=True)
    parts = [part.strip() for part in raw.strip().split(',')]
    if len(parts) != 4 or int(parts[0]) != index or not parts[1].startswith('GPU-'):
        raise RuntimeError('Cannot verify physical GPU identity')
    if int(parts[3]) > 1024:
        raise RuntimeError('Selected GPU is occupied; no existing process will be stopped')
    os.environ['CUDA_VISIBLE_DEVICES'] = parts[1]
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    return {'physical_index': index, 'uuid': parts[1], 'name': parts[2]}


def main():
    run.assemble_inputs = run_v2.assemble_inputs
    run.bind_gpu = bind_gpu
    run.main(allowed_gpus=ALLOWED_GPUS)


if __name__ == '__main__':
    main()
