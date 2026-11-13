import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from baselines.egogpt_caption import runtime


@pytest.fixture
def configuration(tmp_path, monkeypatch):
    config = {'versions': {'example': '1.0'}, 'planned_versions': {'python': '3.10.20'},
              'upstream_path': str(tmp_path / 'upstream'), 'upstream_revision': 'abc'}
    Path(config['upstream_path']).mkdir()
    (Path(config['upstream_path']) / 'model.py').write_text('original = True\n')
    for name in ['model', 'vision']:
        directory = tmp_path / name
        directory.mkdir()
        asset = directory / 'weights.bin'
        asset.write_bytes(b'verified checkpoint')
        receipt = tmp_path / (name + '.json')
        receipt.write_text(json.dumps({'revision': name + '-revision', 'directory': str(directory),
                                      'verified_files': [{'path': asset.name, 'state': runtime.file_state(asset)}]}))
        config.update({name + '_path': str(directory), name + '_revision': name + '-revision',
                       name + '_verification': str(receipt)})
    monkeypatch.setattr(runtime.importlib.metadata, 'version', lambda name: '1.0')
    monkeypatch.setattr(runtime.platform, 'python_version', lambda: '3.10.20')
    monkeypatch.setattr(runtime.subprocess, 'check_output', lambda args, **kw: 'abc\n' if args[0]=='git' else 'ffmpeg')
    monkeypatch.setattr(runtime.subprocess, 'run', lambda args: SimpleNamespace(returncode=0))
    return config


def test_verified_asset_modification_is_rejected(configuration):
    runtime.snapshot(configuration)
    asset = Path(configuration['model_path']) / 'weights.bin'
    asset.write_bytes(b'different checkpoint')
    with pytest.raises(ValueError, match='asset changed'):
        runtime.snapshot(configuration)


def test_changed_dependency_version_is_rejected(configuration, monkeypatch):
    monkeypatch.setattr(runtime.importlib.metadata, 'version', lambda name: '2.0')
    with pytest.raises(ValueError, match='package versions'):
        runtime.snapshot(configuration)


def test_dirty_upstream_rejected_despite_same_commit(configuration, monkeypatch):
    monkeypatch.setattr(runtime.subprocess, 'run', lambda args: SimpleNamespace(returncode=1))
    with pytest.raises(ValueError, match='tracked files are modified'):
        runtime.snapshot(configuration)


def test_new_untracked_python_source_changes_fingerprint(configuration):
    before = runtime.snapshot(configuration)
    (Path(configuration['upstream_path']) / 'shadow.py').write_text('changed = True\n')
    assert runtime.snapshot(configuration) != before


def memory_validation(configuration, tmp_path):
    from baselines.egogpt_caption.archive import atomic_json, object_hash
    import hashlib
    directory=Path(runtime.__file__).parent
    raw=tmp_path/'observations.json'
    raw.write_text('{"observations":"verified independently"}\n')
    report={'status':'passed','fidelity_passed':True,'memory_fit':True,
            'runtime_identity':runtime.memory_runtime_identity(configuration),
            'helper_sha256':hashlib.sha256((directory/'memory.py').read_bytes()).hexdigest(),
            'native_sha256':hashlib.sha256((directory/'native.py').read_bytes()).hexdigest(),
            'observations_path':str(raw),'observations_sha256':hashlib.sha256(raw.read_bytes()).hexdigest()}
    path=tmp_path/'memory-validation.json';atomic_json(path,report)
    configuration.update(memory_policy='offload_idle_encoders',
                         memory_validation={'path':str(path),'sha256':object_hash(report)})
    return report,path


def test_offload_requires_verified_fidelity_and_memory_report(configuration):
    configuration['memory_policy']='offload_idle_encoders'
    with pytest.raises(ValueError,match='memory validation'):
        runtime.snapshot(configuration)


def test_offload_report_is_frozen_in_runtime(configuration,tmp_path):
    report,path=memory_validation(configuration,tmp_path)
    result=runtime.snapshot(configuration)
    assert result['memory']['policy']=='offload_idle_encoders'
    assert result['memory']['validation']==report
    path.write_text('{}')
    with pytest.raises(ValueError,match='memory validation'):
        runtime.snapshot(configuration)


def test_offload_report_rejects_unvalidated_helper(configuration,tmp_path):
    from baselines.egogpt_caption.archive import atomic_json, object_hash
    report,path=memory_validation(configuration,tmp_path)
    report['helper_sha256']='unvalidated'
    atomic_json(path,report)
    configuration['memory_validation']['sha256']=object_hash(report)
    with pytest.raises(ValueError,match='helper differs'):
        runtime.snapshot(configuration)


def test_offload_report_rejects_changed_observations(configuration,tmp_path):
    report,_=memory_validation(configuration,tmp_path)
    Path(report['observations_path']).write_text('{}')
    with pytest.raises(ValueError,match='observations changed'):
        runtime.snapshot(configuration)


def test_offload_report_rejects_different_runtime(configuration,tmp_path):
    memory_validation(configuration,tmp_path)
    configuration['model_revision']='different-model'
    with pytest.raises(ValueError,match='Runtime differs from memory validation'):
        runtime.snapshot(configuration)
