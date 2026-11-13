"""Tiny lossless CPU media test; no model or pretrained inference."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from baselines.ours_offline.inputs import decode_samples, uniform_samples


@unittest.skipUnless(importlib.util.find_spec('av'), 'PyAV is an optional inference dependency')
class NativeDecoderTests(unittest.TestCase):
    def test_actual_pts_and_pixels_across_day_gap(self):
        import av
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            sources=[]
            for name,day,start,color in [('one.mkv',1,.4,20),('two.mkv',2,86400.1,100)]:
                with av.open(str(root/name),'w') as container:
                    stream=container.add_stream('ffv1',rate=4)
                    stream.width=stream.height=16;stream.pix_fmt='bgr0'
                    for i in range(4):
                        frame=av.VideoFrame.from_ndarray(np.full((16,16,3),color+i,dtype=np.uint8),format='rgb24')
                        for packet in stream.encode(frame):container.mux(packet)
                    for packet in stream.encode():container.mux(packet)
                sources.append({'path':name,'day':day,'start_time':start,'frames':4,'fps':4,'duration':1})
            selected=uniform_samples(sources,86400.6,10)
            frames,records=decode_samples(root,selected,86400.6)
            self.assertEqual(len(frames),7)
            self.assertEqual([int(np.asarray(f)[0,0,0]) for f in frames],[20,21,22,23,100,101,102])
            for actual,wanted in zip(records,selected):
                self.assertAlmostEqual(actual['actual_global_timestamp'],wanted['requested_global_timestamp'])
                self.assertLessEqual(actual['actual_global_timestamp'],86400.6+1e-8)
            parallel_frames,parallel_records=decode_samples(root,selected,86400.6,workers=4)
            self.assertEqual(parallel_records,records)
            for sequential,parallel in zip(frames,parallel_frames):
                np.testing.assert_array_equal(np.asarray(sequential),np.asarray(parallel))
