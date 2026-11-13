import os
import sys

import pytest

from baselines.egogpt_caption.run import reexec_clean_libraries


def test_gpu_worker_restarts_before_imports_with_bundled_library_paths(monkeypatch):
    monkeypatch.setenv('LD_LIBRARY_PATH','/usr/local/cuda-11.0/lib64:')
    monkeypatch.setenv('LD_PRELOAD','/unrelated/libcudnn.so')
    monkeypatch.setenv('OMP_NUM_THREADS','4')
    calls=[]
    def replace(executable, argv, env):
        calls.append((executable,argv,env))
        raise RuntimeError('process replaced')
    monkeypatch.setattr(os,'execve',replace)
    argv=['worker','--output','/archive','--index','0','--gpu','7']
    with pytest.raises(RuntimeError,match='process replaced'):
        reexec_clean_libraries(argv)
    executable,args,env=calls[0]
    assert executable==sys.executable
    assert args==[sys.executable,'-m','baselines.egogpt_caption',*argv]
    assert 'LD_LIBRARY_PATH' not in env and 'LD_PRELOAD' not in env
    assert env['OMP_NUM_THREADS']=='4'


def test_already_clean_worker_does_not_reexec(monkeypatch):
    monkeypatch.delenv('LD_LIBRARY_PATH',raising=False)
    monkeypatch.delenv('LD_PRELOAD',raising=False)
    def unexpected(*args):raise AssertionError('unnecessary restart')
    monkeypatch.setattr(os,'execve',unexpected)
    assert reexec_clean_libraries(['worker']) is False
