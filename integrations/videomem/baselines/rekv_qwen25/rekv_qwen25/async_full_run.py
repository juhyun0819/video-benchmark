"""Production supervision using the separately validated CPU writer process.

The existing supervisor retains its coverage, resume, failure and final-audit
behavior. This entry point selects the asynchronous worker only after the
current configuration has a completed real asynchronous scaling/fidelity report.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path

from . import full_run
from .coordinator import sha256_file


def validate_pipeline_report(config, validation):
    if config.get('writer_mode') != 'cpu_process_overlap_v1' or config.get('max_pending_blocks') != 1:
        raise ValueError('Asynchronous production requires the validated one-pending-block CPU process writer')
    root = Path(validation['config_path']).parent
    path = root/'benchmarks/async_scaling_report.json'
    report = json.loads(path.read_text())
    if (report.get('schema'), report.get('status'), report.get('writer_mode'), report.get('max_pending_blocks')) != (
            'rekv_async_scaling_v1', 'complete', 'cpu_process_overlap_v1', 1):
        raise ValueError('Asynchronous scaling has not completed successfully')
    proof = report.get('validation', {})
    if any(proof.get(key) != validation[key] for key in ('config_id', 'config_path', 'config_file_sha256')):
        raise ValueError('Asynchronous scaling used a different frozen configuration')
    stages = report.get('stages', [])
    if [stage.get('gpu_count') for stage in stages] != [2, 4] or any(
            stage.get('status') != 'complete' or stage.get('config_id') != validation['config_id'] for stage in stages):
        raise ValueError('Required asynchronous GPU stages are incomplete')
    for stage in stages:
        workers = stage.get('workers', [])
        if len(workers) != stage['gpu_count'] or any(
                worker.get('writer_drained') is not True or worker.get('writer_exitcode') != 0
                or worker.get('writer_cuda_initialized') is not False
                or worker.get('compute_write_overlap_seconds', 0) <= 0
                or worker.get('overlapped_observation_calls', 0) <= 0
                for worker in workers):
            raise ValueError('The real benchmark did not establish drained CPU writers and GPU/write overlap')
    for field, expected_count, metadata_flag in [
            ('cross_pipeline_fidelity', 12, 'metadata_equivalent_except_config'),
            ('cross_gpu_fidelity', 4, 'metadata_exact')]:
        checks = report.get(field, {})
        rows = checks.get('comparisons', [])
        if checks.get('status') != 'passed' or len(rows) != expected_count or any(
                row.get('status') != 'passed' or row.get('all_tensor_sha256_exact') is not True
                or row.get('all_tensor_geometry_exact') is not True or row.get(metadata_flag) is not True
                for row in rows):
            raise ValueError(f'Asynchronous output fidelity is incomplete: {field}')
    return dict(report_path=str(path), report_sha256=sha256_file(path))


@contextmanager
def asynchronous_supervision():
    """Scope worker selection and validation to this supervisor process."""
    original_command = full_run.worker_command
    original_validation = full_run.validate_config

    def command(*args, **kwargs):
        result = original_command(*args, **kwargs)
        module = result.index('-m') + 1
        if result[module] != 'rekv_qwen25.runner':
            raise ValueError('Unexpected base worker module')
        result[module] = 'rekv_qwen25.async_runner'
        return result

    def validate(*args, **kwargs):
        config, validation = original_validation(*args, **kwargs)
        validation = dict(validation)
        validation['asynchronous_writer_validation'] = validate_pipeline_report(config, validation)
        return config, validation

    full_run.worker_command, full_run.validate_config = command, validate
    try:
        yield
    finally:
        full_run.worker_command, full_run.validate_config = original_command, original_validation


def main():
    with asynchronous_supervision():
        full_run.main()


if __name__ == '__main__':
    main()
