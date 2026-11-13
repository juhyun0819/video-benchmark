import os
from unittest.mock import patch

import pytest

from day1_compare import run_multi_gpu


@pytest.mark.parametrize('index', [0, 6, -1])
def test_unapproved_gpu_never_queried(index):
    with patch.object(run_multi_gpu.subprocess, 'check_output') as query:
        with pytest.raises(ValueError):
            run_multi_gpu.bind_gpu(index)
        query.assert_not_called()


def test_occupied_gpu_is_not_reclaimed():
    with patch.object(run_multi_gpu.subprocess, 'check_output', return_value='2, GPU-example, A100, 2048\n'):
        with pytest.raises(RuntimeError, match='occupied'):
            run_multi_gpu.bind_gpu(2)


def test_free_gpu_is_bound_by_uuid():
    with patch.dict(os.environ, {}, clear=False):
        with patch.object(run_multi_gpu.subprocess, 'check_output', return_value='1, GPU-example, A100, 4\n'):
            gpu = run_multi_gpu.bind_gpu(1)
        assert gpu == {'physical_index': 1, 'uuid': 'GPU-example', 'name': 'A100'}
        assert os.environ['CUDA_VISIBLE_DEVICES'] == 'GPU-example'
        assert os.environ['HF_HUB_OFFLINE'] == '1'


def test_entrypoint_preserves_v2_assembly_and_passes_only_gpu_allowlist():
    with patch.object(run_multi_gpu.run, 'main') as main:
        with patch.object(run_multi_gpu.run, 'bind_gpu'), patch.object(run_multi_gpu.run, 'assemble_inputs'):
            run_multi_gpu.main()
            assert run_multi_gpu.run.assemble_inputs is run_multi_gpu.run_v2.assemble_inputs
            assert run_multi_gpu.run.bind_gpu is run_multi_gpu.bind_gpu
            main.assert_called_once_with(allowed_gpus=(1, 2, 3, 4, 5))
