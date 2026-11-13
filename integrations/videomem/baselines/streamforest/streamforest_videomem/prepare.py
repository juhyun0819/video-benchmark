"""Build the shared causal manifest from explicit local video source mappings."""
import argparse
import json
from pathlib import Path

from .shared import load_shared


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', required=True, help='JSON containing a videos list')
    parser.add_argument('--output', required=True, help='New manifest JSON path')
    parser.add_argument('--fingerprint-cache-dir')
    args = parser.parse_args(argv)
    output = Path(args.output).expanduser()
    if output.exists():
        parser.error('output already exists; choose a fresh manifest path')
    load_shared()
    from local_videomem.data import prepare_manifest
    manifest = prepare_manifest(Path(args.sources).expanduser(),
                                fingerprint_cache_dir=args.fingerprint_cache_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        handle.write(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(str(output))


if __name__ == '__main__':
    main()
