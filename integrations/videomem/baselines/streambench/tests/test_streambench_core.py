import copy
import json
import random
import tempfile
import unittest
from pathlib import Path

import torch

try:
    from baselines.streambench.accounting import measure_state, DiskState
    from baselines.streambench.method import StreamingMemoryMethod, UnsupportedBudget
except ImportError:
    measure_state = DiskState = StreamingMemoryMethod = UnsupportedBudget = None


class AccountingTests(unittest.TestCase):
    def test_tensor_alias_storage_counted_once(self):
        self.assertIsNotNone(measure_state, 'common byte accounting must be implemented')
        x = torch.zeros(32, dtype=torch.float32)
        result = measure_state({'visual_memory_bytes': [x[:2], x[8:]], 'kv_memory_bytes': x})
        self.assertEqual(result['persistent_total_bytes'], 128)
        self.assertEqual(result['visual_memory_bytes'], 128)
        self.assertEqual(result['kv_memory_bytes'], 0)

    def test_independent_copies_and_text_metadata_are_charged(self):
        self.assertIsNotNone(measure_state)
        x = torch.zeros(5, dtype=torch.float32)
        result = measure_state({'visual_memory_bytes': [x, x.clone()], 'text_memory_bytes': '한글',
                                'metadata_bytes': {'timestamp': 1.5}})
        self.assertEqual(result['visual_memory_bytes'], 40)
        self.assertEqual(result['text_memory_bytes'], 6)
        self.assertGreaterEqual(result['metadata_bytes'], 8)
        self.assertEqual(result['persistent_total_bytes'], sum(result[k] for k in (
            'visual_memory_bytes','kv_memory_bytes','text_memory_bytes','fast_weight_bytes','metadata_bytes')))

    def test_retrievable_disk_state_and_unknown_objects(self):
        self.assertIsNotNone(measure_state)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'index.bin'
            path.write_bytes(b'abc'*100)
            size = measure_state({'metadata_bytes': DiskState(path)})['persistent_total_bytes']
            self.assertGreaterEqual(size, 300)
        with self.assertRaises(TypeError):
            measure_state({'metadata_bytes': object()})

    def test_random_partial_storage_overlap_matches_byte_union_across_categories(self):
        # frombuffer exposes separate Torch storages over partially overlapping
        # byte ranges, unlike Tensor slicing which retains the entire storage.
        backing = bytearray(2048)
        rng = random.Random(731)
        for trial in range(16):
            spans = [(0, 32), (32, 64), (16, 48), (0, 2048), (100, 100)]
            spans += [(start, min(2048, start + rng.randrange(1, 160)))
                      for start in (rng.randrange(2048) for _ in range(90))]
            rng.shuffle(spans)
            tensors = [torch.frombuffer(backing, dtype=torch.uint8, offset=start, count=end-start)
                       if end > start else torch.empty(0, dtype=torch.uint8)
                       for start, end in spans]
            split = len(spans) // 2
            covered = set()
            expected = []
            for part in (spans[:split], spans[split:]):
                before = len(covered)
                for start, end in part:
                    covered.update(range(start, end))
                expected.append(len(covered) - before)
            result = measure_state({'visual_memory_bytes': tensors[:split],
                                    'kv_memory_bytes': tensors[split:]})
            self.assertEqual(result['visual_memory_bytes'], expected[0], trial)
            self.assertEqual(result['kv_memory_bytes'], expected[1], trial)
            self.assertEqual(result['persistent_total_bytes'], len(covered), trial)

    def test_many_disjoint_storage_intervals_order_and_late_bridge(self):
        # Independent small storages in one backing allocation model a large
        # native tier. A late overlapping bridge must charge only its gaps.
        count = 2048
        backing = bytearray(count * 4)
        starts = list(range(0, len(backing), 4))
        for order in (starts, list(reversed(starts)), random.Random(419).sample(starts, len(starts))):
            disjoint = [torch.frombuffer(backing, dtype=torch.uint8, offset=start, count=2)
                        for start in order]
            bridge = torch.frombuffer(backing, dtype=torch.uint8)
            result = measure_state({'visual_memory_bytes': disjoint, 'kv_memory_bytes': bridge,
                                    'fast_weight_bytes': disjoint})
            self.assertEqual(result['visual_memory_bytes'], count * 2)
            self.assertEqual(result['kv_memory_bytes'], count * 2)
            self.assertEqual(result['fast_weight_bytes'], 0)
            self.assertEqual(result['persistent_total_bytes'], len(backing))


if __name__ == '__main__':
    unittest.main()

class CheckpointTests(unittest.TestCase):
    def test_serialization_preserves_views_and_detects_corruption(self):
        from baselines.streambench.checkpoint import write_checkpoint,read_checkpoint
        x=torch.arange(32,dtype=torch.float32)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'checkpoint_D1'
            write_checkpoint(path,{'schema_version':1,'state':{'a':x,'b':x[8:]}})
            restored=read_checkpoint(path)['state']
            self.assertEqual(restored['a'].untyped_storage().data_ptr(), restored['b'].untyped_storage().data_ptr())
            self.assertEqual(measure_state({'visual_memory_bytes':restored})['persistent_total_bytes'],130)
            with (path/'state.pt').open('ab') as f:f.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'integrity'):
                read_checkpoint(path)

    def test_numpy_and_torch_aliases_are_not_double_counted(self):
        import numpy as np
        x=np.zeros(100,dtype=np.float32)
        tensor=torch.from_numpy(x)
        measured=measure_state({'visual_memory_bytes':[x[1:],tensor]})
        self.assertEqual(measured['persistent_total_bytes'],400)

class ReviewRegressionTests(unittest.TestCase):
    def test_numpy_views_charge_the_entire_external_backing_allocation(self):
        import numpy as np
        raw=bytearray(4096)
        view=np.frombuffer(raw,dtype=np.uint8,count=1)
        self.assertEqual(measure_state({'visual_memory_bytes':view})['persistent_total_bytes'],4096)
        tensor=torch.zeros(32,dtype=torch.float32)
        view=tensor[:1].numpy()
        self.assertEqual(measure_state({'visual_memory_bytes':view})['persistent_total_bytes'],128)

    def test_checkpoint_rejects_unowned_external_state_instead_of_losing_aliases(self):
        import numpy as np
        from streambench_fixture import CounterMemory
        method=CounterMemory(65536);method.initialize()
        original=np.arange(32,dtype=np.float32)
        method.values=[original,original[8:]]
        with self.assertRaisesRegex(TypeError,'NumPy'):
            method.snapshot()
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'external';p.write_bytes(b'retrievable')
            method.values=[DiskState(p)]
            with self.assertRaisesRegex(TypeError,'DiskState'):
                method.snapshot()
