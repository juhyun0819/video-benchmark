"""Real spawn/IPC with tiny CPU encoders: ordering, bounds and failure cleanup."""
import json
import time

from PIL import Image
import pytest
import torch

from baselines.feature_memory.parallel import ParallelEncodingError, _device_spec, _memory_metrics, iter_encoded


class FakeEncoder:
    def __init__(self, model_path, device='cpu', dtype='bfloat16'):
        self.model_path = str(model_path)
        self.dtype = getattr(torch, dtype)
        self.metadata = {'fixture': True, 'device': device}

    def encode(self, image):
        number = image.getpixel((0, 0))[0]
        if self.model_path == 'delay':
            time.sleep(0.5 if number == 0 else 0.005)
        if self.model_path == 'pause' and number > 0:
            time.sleep(0.8)
        if self.model_path == 'fail' and number == 3:
            raise ValueError('intentional encoder failure')
        features = torch.full((2, 3), number, dtype=self.dtype)
        return features, {'encode_seconds': 0.0, 'grid_thw': [1, 2, 4], 'fixture_number': number}


def fake_size(image):
    return 2 * 3 * 2


def frame_stream(count, lifecycle):
    lifecycle['closed'] = False
    lifecycle['decoded'] = 0
    lifecycle['images'] = []
    try:
        for number in range(count):
            image = Image.new('RGB', (8, 8), (number, 20, 30))
            lifecycle['images'].append(image)
            lifecycle['decoded'] += 1
            try:
                yield {'frame': image, 'frame_id': number, 'timestamp': float(number),
                       'decode_seconds': 0.01, 'day': 'Day1', 'source_timestamp': float(number)}
            finally:
                image.close()
    finally:
        lifecycle['closed'] = True


def assert_images_closed(lifecycle):
    assert lifecycle['closed']
    for image in lifecycle['images']:
        with pytest.raises(ValueError, match='closed'):
            image.getpixel((0, 0))


def test_parallel_out_of_order_exactly_once_matches_reference():
    reference_lifecycle, parallel_lifecycle, stats = {}, {}, {}
    reference = list(iter_encoded(frame_stream(9, reference_lifecycle), 'plain', mode='sequential',
                                  devices=['cpu'], encoder_factory=FakeEncoder))
    result = list(iter_encoded(frame_stream(9, parallel_lifecycle), 'delay', mode='parallel',
                              devices=['cpu:0', 'cpu:1'], chunk_frames=2, max_inflight_chunks=3,
                              max_queue_bytes=100000, stats=stats, encoder_factory=FakeEncoder,
                              feature_size_estimator=fake_size, worker_timeout_seconds=60))
    assert [row['frame_id'] for row in result] == list(range(9))
    assert [row['timestamp'] for row in result] == list(map(float, range(9)))
    assert all('frame' not in row for row in result)
    assert all(row['encoder_worker_id'] in (0, 1) and row['encoder_device'] == 'cpu' for row in result)
    assert all(row['encoder_worker_id'] == 0 for row in reference)
    for actual, expected in zip(result, reference):
        torch.testing.assert_close(actual['features'], expected['features'], rtol=0, atol=0)
        assert actual['encoder'] == expected['encoder']
    assert stats['num_encoder_calls'] == stats['num_frames_encoded'] == stats['frames_yielded'] == 9
    assert stats['frames_decoded'] == parallel_lifecycle['decoded'] == 9
    assert stats['out_of_order_chunks'] > 0
    assert stats['peak_inflight_chunks'] <= 3
    assert stats['peak_queue_reserved_bytes'] <= 100000
    assert stats['queue_reserved_bytes'] == stats['ready_feature_bytes'] == 0
    assert stats['current_inflight_chunks'] == 0
    assert stats['worker_processes_alive_after_close'] == 0
    assert sum(w['encoder_calls'] for w in stats['workers'].values()) == 9
    assert sum(w.get('completed_chunks', 0) for w in stats['workers'].values()) == stats['chunks_completed']
    for worker in stats['workers'].values():
        if worker.get('completed_chunks', 0):
            assert worker['min_after_chunk_current_cuda_allocated_bytes'] == 0
            assert worker['max_after_chunk_current_cuda_allocated_bytes'] == 0
            assert worker['min_after_chunk_current_cpu_rss_bytes'] > 0
            assert worker['max_after_chunk_current_cpu_rss_bytes'] >= worker['min_after_chunk_current_cpu_rss_bytes']
            assert worker['last_timestamp'] == float(worker['last_frame_id'])
    assert stats['feature_files_written'] == 0
    json.dumps(stats)
    assert_images_closed(reference_lifecycle)
    assert_images_closed(parallel_lifecycle)


def test_byte_cap_splits_chunks_and_backpressures():
    lifecycle, stats = {}, {}
    result = list(iter_encoded(frame_stream(5, lifecycle), 'plain', mode='parallel',
                              devices=['cpu:0', 'cpu:1'], chunk_frames=5, max_inflight_chunks=3,
                              max_queue_bytes=4000, stats=stats, encoder_factory=FakeEncoder,
                              feature_size_estimator=fake_size, worker_timeout_seconds=60))
    assert [r['frame_id'] for r in result] == list(range(5))
    assert stats['byte_backpressure_events'] > 0
    assert stats['chunks_dispatched'] >= 3
    assert stats['peak_queue_reserved_bytes'] <= 4000
    assert stats['worker_processes_alive_after_close'] == 0
    assert_images_closed(lifecycle)


def test_worker_failure_fails_without_retry_and_cleans_up():
    lifecycle, stats = {}, {}
    with pytest.raises(ParallelEncodingError, match='intentional encoder failure'):
        list(iter_encoded(frame_stream(12, lifecycle), 'fail', mode='parallel',
                          devices=['cpu:0', 'cpu:1'], chunk_frames=2, max_inflight_chunks=2,
                          stats=stats, encoder_factory=FakeEncoder,
                          feature_size_estimator=fake_size, worker_timeout_seconds=60))
    assert stats['state'] == 'failed'
    assert stats['worker_processes_alive_after_close'] == 0
    assert stats['chunks_dispatched'] <= 4  # No automatic retry/re-encoding.
    assert_images_closed(lifecycle)


def test_early_consumer_close_releases_chunks_and_workers():
    lifecycle, stats = {}, {}
    encoded = iter_encoded(frame_stream(20, lifecycle), 'plain', mode='parallel',
                           devices=['cpu:0', 'cpu:1'], chunk_frames=2, max_inflight_chunks=2,
                           stats=stats, encoder_factory=FakeEncoder,
                           feature_size_estimator=fake_size, worker_timeout_seconds=60)
    first = next(encoded)
    assert first['frame_id'] == 0
    assert 1 <= stats['current_inflight_chunks'] <= 2
    encoded.close()
    assert stats['state'] == 'closed_early'
    assert stats['worker_processes_alive_after_close'] == 0
    assert stats['queue_reserved_bytes'] == stats['ready_feature_bytes'] == 0
    assert stats['current_inflight_chunks'] == 0
    assert lifecycle['decoded'] <= 4
    assert_images_closed(lifecycle)


def test_reject_too_small_byte_cap_and_close_source():
    lifecycle, stats = {}, {}
    with pytest.raises(ValueError, match='One frame needs'):
        list(iter_encoded(frame_stream(2, lifecycle), 'plain', mode='parallel',
                          devices=['cpu'], max_queue_bytes=10, stats=stats,
                          encoder_factory=FakeEncoder, feature_size_estimator=fake_size,
                          worker_timeout_seconds=60))
    assert stats['worker_processes_alive_after_close'] == 0
    assert_images_closed(lifecycle)


def test_device_indices_respect_existing_visibility(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3,7')
    assert _device_spec('0') == ('cuda:0', '3')
    assert _device_spec('cuda:1') == ('cuda:0', '7')
    with pytest.raises(ValueError, match='outside'):
        _device_spec('2')
    assert _device_spec('cpu:4') == ('cpu', None)


def test_gpu_metrics_use_selected_logical_device(monkeypatch):
    seen = []
    def metric(device):
        seen.append(device)
        return 123
    for name in ('max_memory_allocated', 'memory_allocated', 'memory_reserved'):
        monkeypatch.setattr(torch.cuda, name, metric)
    assert _memory_metrics('cuda:3')['peak_cuda_allocated_bytes'] == 123
    assert seen == ['cuda:3', 'cuda:3', 'cuda:3']


def test_slow_consumer_snapshot_time_is_not_a_worker_timeout(monkeypatch):
    import baselines.feature_memory.parallel as parallel
    lifecycle, stats = {}, {}
    encoded = iter_encoded(frame_stream(2, lifecycle), 'pause', mode='parallel',
                           devices=['cpu'], chunk_frames=1, max_inflight_chunks=1,
                           stats=stats, encoder_factory=FakeEncoder,
                           feature_size_estimator=fake_size, worker_timeout_seconds=60)
    assert next(encoded)['frame_id'] == 0
    # Simulate a long snapshot while the consumer holds the yielded frame.
    # Spawned workers retain their real clock and take 0.8s for the next frame.
    original_clock = time.perf_counter
    monkeypatch.setattr(parallel.time, 'perf_counter', lambda: original_clock() + 1000)
    assert [row['frame_id'] for row in encoded] == [1]
    assert stats['state'] == 'completed'
    assert_images_closed(lifecycle)
