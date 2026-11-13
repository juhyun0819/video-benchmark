"""Check exactly the staged source snapshot; never print secret matches."""
from __future__ import annotations

import argparse
import json
from pathlib import PurePosixPath
import re
import subprocess

BLOCKED_DIRS = {
    '.git', '.venv', '.pytest_cache', '__pycache__', 'local_runs', 'videomem_runs',
    'outputs', 'artifacts', 'datasets', 'checkpoints', 'ckpt', 'logs', 'frames',
}
BLOCKED_EXTENSIONS = {
    '.mp4', '.mov', '.mkv', '.avi', '.webm', '.wav', '.mp3', '.png', '.jpg',
    '.jpeg', '.webp', '.gif', '.safetensors', '.pt', '.pth', '.bin', '.gguf',
    '.onnx', '.npy', '.npz', '.zip', '.tar', '.gz', '.parquet', '.db', '.jsonl',
}
PATTERNS = {
    'openai_style_key': rb'\bsk-[A-Za-z0-9_-]{24,}',
    'huggingface_token': rb'\bhf_[A-Za-z0-9]{24,}',
    'github_token': rb'\bgh[pousr]_[A-Za-z0-9]{30,}',
    'github_fine_grained_token': rb'\bgithub_pat_[A-Za-z0-9_]{30,}',
    'private_key': rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'aws_access_id': rb'\bAKIA[0-9A-Z]{16}\b',
    'url_password': rb'https?://[^\s/:]+:[^\s/@]+@',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staged', action='store_true', required=True)
    parser.parse_args()
    entries = subprocess.check_output(['git', 'ls-files', '-s', '-z']).split(b'\0')
    failures, count, total = [], 0, 0
    for entry in entries:
        if not entry:
            continue
        meta, name = entry.split(b'\t', 1)
        mode, object_id, stage = meta.decode().split()
        path = name.decode()
        part = PurePosixPath(path)
        reasons = []
        if stage != '0' or mode not in ('100644', '100755'):
            reasons.append('non_regular_or_unmerged_entry')
        raw = subprocess.check_output(['git', 'cat-file', 'blob', object_id]) if mode != '160000' else b''
        count += 1
        total += len(raw)
        if any(p in BLOCKED_DIRS or p.startswith('.venv') for p in part.parts):
            reasons.append('runtime_directory')
        if part.suffix.lower() in BLOCKED_EXTENSIONS:
            reasons.append('non_source_file')
        if part.name != '.env.example' and (part.name == '.env' or part.name.startswith('.env.') or part.suffix in ('.env', '.pem', '.key')):
            reasons.append('secret_file')
        if len(raw) > 2 * 1024 * 1024:
            reasons.append('oversized_source_blob')
        if b'\0' in raw:
            reasons.append('binary_blob')
        for rule, pattern in PATTERNS.items():
            if re.search(pattern, raw):
                reasons.append(rule)
        if reasons:
            failures.append({'path': path, 'rules': reasons})
    print(json.dumps({'files': count, 'bytes': total, 'passed': not failures, 'failures': failures}, indent=2))
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
