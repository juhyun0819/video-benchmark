#!/usr/bin/env python3
"""Syncs generated data with a private Hugging Face dataset repo, so another server can continue the work.

Usage: python3 scripts/sync_data.py <push|pull> [--root <repo root>] [--overwrite]

Only the folders in FOLDERS are synced, to data/<folder> in the repo. Source media and raw annotations
stay out of it: fetch those from their own datasets. Push refuses unless the repo is private.
Auth: HF_TOKEN or the token saved by `huggingface-cli login`.
"""
import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import HfHubHTTPError

REPO = "zzu819/videomem"
FOLDERS = ("catalog", "compositions", "exports", "renders")
BLOCKED_FOR_ANONYMOUS = (401, 403, 404)


class SyncError(Exception):
    pass


def assert_private(authed, anonymous, repo):
    """Raises unless the repo is marked private and an anonymous client really cannot read it."""
    if not authed.dataset_info(repo).private:
        raise SyncError(f"{repo} is not private: refusing to upload licensed data")
    try:
        anonymous.dataset_info(repo)
    except HfHubHTTPError as error:
        if error.response.status_code in BLOCKED_FOR_ANONYMOUS:
            return
        raise
    raise SyncError(f"{repo} is readable by anonymous clients: refusing to upload licensed data")


def folders_to_push(data_dir):
    """Returns the allowed folders that hold at least one file."""
    return [
        name
        for name in FOLDERS
        if (data_dir / name).is_dir() and any(path.is_file() for path in (data_dir / name).rglob("*"))
    ]


def assert_pull_safe(data_dir, overwrite):
    """Compositions are edited by hand, so a pull must not silently replace them."""
    compositions = data_dir / "compositions"
    if not overwrite and compositions.is_dir() and any(compositions.iterdir()):
        raise SyncError(f"{compositions} already has files: pass --overwrite to replace them")


def push(root):
    api = HfApi()
    assert_private(api, HfApi(token=False), REPO)
    data_dir = root / "data"
    names = folders_to_push(data_dir)
    if not names:
        raise SyncError(f"nothing to push in {data_dir}")
    for name in names:
        api.upload_folder(
            repo_id=REPO,
            repo_type="dataset",
            folder_path=data_dir / name,
            path_in_repo=f"data/{name}",
            commit_message=f"Sync data/{name}",
        )
        print(f"pushed data/{name}")


def pull(root, overwrite):
    assert_pull_safe(root / "data", overwrite)
    snapshot_download(
        REPO,
        repo_type="dataset",
        allow_patterns=[f"data/{name}/*" for name in FOLDERS],
        local_dir=root,
    )
    print(f"pulled into {root / 'data'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["push", "pull"])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--overwrite", action="store_true", help="let pull replace local compositions")
    args = parser.parse_args()
    try:
        if args.command == "push":
            push(args.root)
        else:
            pull(args.root, args.overwrite)
    except SyncError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
