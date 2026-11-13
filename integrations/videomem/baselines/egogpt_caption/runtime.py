"""Freeze the verified model assets, upstream source and installed runtime."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess


def file_state(path):
    stat = Path(path).stat()
    return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
            'ctime_ns': stat.st_ctime_ns, 'inode': stat.st_ino, 'device': stat.st_dev}


def memory_runtime_identity(runtime):
    return {key:runtime.get(key) for key in ('model_revision','vision_revision','upstream_revision',
                                           'planned_versions','versions')}


def memory_snapshot(runtime):
    policy=runtime.get('memory_policy','native_gpu')
    if policy=='native_gpu':return {'policy':policy}
    if policy!='offload_idle_encoders':raise ValueError('Unknown caption memory policy')
    evidence=runtime.get('memory_validation',{})
    try:
        data=Path(evidence['path']).read_bytes()
        report=json.loads(data)
    except (KeyError,OSError,ValueError) as exc:
        raise ValueError('Missing or invalid memory validation') from exc
    if hashlib.sha256(data).hexdigest()!=evidence.get('sha256'):
        raise ValueError('Changed memory validation receipt')
    if (report.get('status')!='passed' or report.get('fidelity_passed') is not True
            or report.get('memory_fit') is not True):
        raise ValueError('Unsuccessful memory validation')
    if report.get('runtime_identity')!=memory_runtime_identity(runtime):
        raise ValueError('Runtime differs from memory validation')
    directory=Path(__file__).parent
    for filename,key in [('memory.py','helper_sha256'),('native.py','native_sha256')]:
        if hashlib.sha256((directory/filename).read_bytes()).hexdigest()!=report.get(key):
            raise ValueError('Caption helper differs from memory validation')
    try:
        observations=Path(report['observations_path']).read_bytes()
    except (KeyError,OSError) as exc:
        raise ValueError('Memory validation observations missing') from exc
    if hashlib.sha256(observations).hexdigest()!=report.get('observations_sha256'):
        raise ValueError('Memory validation observations changed')
    return {'policy':policy,'validation':report}


def snapshot(runtime):
    memory=memory_snapshot(runtime)
    versions = {name: importlib.metadata.version(name) for name in runtime['versions']}
    if versions != runtime['versions']:
        raise ValueError('Installed package versions differ from verified runtime')
    if platform.python_version() != runtime['planned_versions']['python']:
        raise ValueError('Python version differs from verified runtime')
    assets = {}
    for prefix, directory in [('model', 'model_path'), ('vision', 'vision_path')]:
        receipt_path = Path(runtime[prefix + '_verification'])
        receipt = json.loads(receipt_path.read_text())
        if (receipt['revision'] != runtime[prefix + '_revision']
                or receipt['directory'] != runtime[directory]):
            raise ValueError('Model verification receipt identity mismatch')
        for entry in receipt['verified_files']:
            path = Path(receipt['directory']) / entry['path']
            if file_state(path) != entry['state']:
                raise ValueError(f'Verified asset changed: {path}')
        assets[prefix] = receipt
    upstream = Path(runtime['upstream_path'])
    revision = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != runtime['upstream_revision']:
        raise ValueError('Upstream revision differs from verified runtime')
    if subprocess.run(['git', '-C', str(upstream), 'diff', '--quiet', 'HEAD', '--', '.']).returncode:
        raise ValueError('Upstream EgoGPT tracked files are modified')
    source_hashes = {str(path.relative_to(upstream)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(upstream.rglob('*.py'))}
    return {'assets': assets, 'upstream_sources': source_hashes, 'versions': versions,'memory':memory,
            'python': platform.python_version(),
            'ffmpeg_version': subprocess.check_output(['ffmpeg', '-version'], text=True),
            'ffprobe_version': subprocess.check_output(['ffprobe', '-version'], text=True)}
