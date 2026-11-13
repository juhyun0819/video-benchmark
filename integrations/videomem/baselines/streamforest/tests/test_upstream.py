"""Pinned source integrity and namespace isolation for native inference."""
import json
import shutil
import sys
from types import ModuleType

import pytest

from streamforest_videomem import upstream


def test_vendored_source_is_complete_and_unchanged():
    record = upstream.verify_source()
    assert record['commit'] == upstream.UPSTREAM_COMMIT
    assert record['modifications'] == []
    paths = {row['path'] for row in record['files']}
    assert len(paths) == 38
    assert {'LICENSE', 'llava/model/multimodal_projector/memory_manager.py',
            'llava/model/language_model/llava_qwen.py'} <= paths


def test_changed_upstream_source_is_rejected(tmp_path, monkeypatch):
    root = tmp_path / 'vendor'
    shutil.copytree(upstream.VENDOR_ROOT, root)
    monkeypatch.setattr(upstream, 'VENDOR_ROOT', root)
    source = root / 'llava/model/multimodal_projector/memory_manager.py'
    source.write_text(source.read_text() + '\n# local mutation\n')
    with pytest.raises(RuntimeError, match='Vendored source changed'):
        upstream.verify_source()


def test_source_manifest_cannot_escape_vendor(tmp_path, monkeypatch):
    root = tmp_path / 'vendor'
    root.mkdir()
    record = upstream.verify_source()
    record['files'][0]['path'] = '../external.py'
    (root / 'UPSTREAM.json').write_text(json.dumps(record))
    monkeypatch.setattr(upstream, 'VENDOR_ROOT', root)
    with pytest.raises(ValueError, match='Invalid upstream source path'):
        upstream.verify_source()


def test_foreign_llava_namespace_is_rejected(monkeypatch):
    foreign = ModuleType('llava')
    foreign.__file__ = '/some/other/model/llava/__init__.py'
    monkeypatch.setitem(sys.modules, 'llava', foreign)
    with pytest.raises(RuntimeError, match='another project'):
        upstream.load_native()
