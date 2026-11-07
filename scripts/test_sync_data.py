"""Tests for sync_data.py. Run with: python3 -m pytest scripts/test_sync_data.py"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
from huggingface_hub.utils import HfHubHTTPError

sys.path.insert(0, str(Path(__file__).parent))
from sync_data import SyncError, assert_private, assert_pull_safe, folders_to_push


def http_error(status):
    response = requests.Response()
    response.status_code = status
    return HfHubHTTPError(f"status {status}", response=response)


class FakeApi:
    def __init__(self, private=True, error=None):
        self.private = private
        self.error = error

    def dataset_info(self, repo):
        if self.error:
            raise self.error
        return SimpleNamespace(private=self.private)


def test_private_repo_blocked_for_anonymous_passes():
    assert_private(FakeApi(private=True), FakeApi(error=http_error(401)), "a/b")


def test_public_repo_is_refused():
    with pytest.raises(SyncError, match="not private"):
        assert_private(FakeApi(private=False), FakeApi(error=http_error(401)), "a/b")


def test_repo_visible_to_anonymous_is_refused():
    with pytest.raises(SyncError, match="anonymous"):
        assert_private(FakeApi(private=True), FakeApi(private=True), "a/b")


def test_unknown_anonymous_failure_is_refused():
    with pytest.raises(HfHubHTTPError):
        assert_private(FakeApi(private=True), FakeApi(error=http_error(500)), "a/b")


def test_push_takes_only_allowed_folders_that_exist(tmp_path):
    for name in ["catalog", "exports", "media", "raw", "samples"]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "x.json").write_text("{}")
    (tmp_path / "compositions").mkdir()

    assert folders_to_push(tmp_path) == ["catalog", "exports"]


def test_pull_refuses_to_replace_local_compositions(tmp_path):
    (tmp_path / "compositions").mkdir()
    (tmp_path / "compositions" / "a.json").write_text("{}")

    with pytest.raises(SyncError, match="--overwrite"):
        assert_pull_safe(tmp_path, overwrite=False)


def test_pull_allows_overwrite_when_asked(tmp_path):
    (tmp_path / "compositions").mkdir()
    (tmp_path / "compositions" / "a.json").write_text("{}")

    assert_pull_safe(tmp_path, overwrite=True)


def test_pull_into_empty_data_folder_is_safe(tmp_path):
    assert_pull_safe(tmp_path, overwrite=False)
