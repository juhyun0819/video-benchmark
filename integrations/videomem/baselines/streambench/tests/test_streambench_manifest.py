import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import av
try:
    from baselines.streambench.manifest import prepare_manifest, iter_chunks
except ImportError:
    prepare_manifest=iter_chunks=None


def make_video(path):
    with av.open(str(path),mode='w') as out:
        stream=out.add_stream('mpeg4',rate=4)
        stream.width=32
        stream.height=32
        stream.pix_fmt='yuv420p'
        for i in range(12):
            frame=av.VideoFrame.from_ndarray(np.full((32,32,3),i*15,dtype=np.uint8),format='rgb24')
            for packet in stream.encode(frame): out.mux(packet)
        for packet in stream.encode(): out.mux(packet)

class ManifestTests(unittest.TestCase):
    def test_identical_samples_and_resume_skips_old_video(self):
        self.assertIsNotNone(prepare_manifest)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for day in (1,2): make_video(root/f'day{day}.mp4')
            sources={'stream_id':'s','days':[{'day':d,'path':str(root/f'day{d}.mp4')} for d in (1,2)]}
            manifest=root/'frames.jsonl'
            metadata=prepare_manifest(sources,manifest,fps=1)
            self.assertEqual(metadata['frame_count'],6)
            ids=[]
            for frames, stamps in iter_chunks(manifest,chunk_size=2):
                self.assertLessEqual(len(frames),2)
                self.assertEqual(len({r['day'] for r in stamps}),1)
                ids.extend((r['global_frame_id'],r['global_timestamp']) for r in stamps)
            other=[]
            for _,stamps in iter_chunks(manifest,chunk_size=1):
                other.extend((r['global_frame_id'],r['global_timestamp']) for r in stamps)
            self.assertEqual(ids,other)
            (root/'day1.mp4').unlink()
            resumed=[r['global_frame_id'] for _,stamps in iter_chunks(manifest,chunk_size=2,start_after_frame_id=2) for r in stamps]
            self.assertEqual(resumed,[3,4,5])

    def test_source_change_detected_and_manifest_labels_rejected(self):
        self.assertIsNotNone(prepare_manifest)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); source=root/'day1.mp4';make_video(source)
            manifest=root/'frames.jsonl'
            prepare_manifest({'stream_id':'s','days':[{'day':1,'path':str(source)}]},manifest,fps=1)
            with source.open('ab') as f:f.write(b'x')
            with self.assertRaisesRegex(ValueError,'Source'):
                next(iter_chunks(manifest))

if __name__=='__main__': unittest.main()

class RepeatedSourceTests(unittest.TestCase):
    def test_same_file_can_be_observed_on_two_explicit_days(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'source.mp4';make_video(path)
            manifest=root/'frames.jsonl'
            prepare_manifest({'stream_id':'s','days':[{'day':1,'path':str(path)},{'day':2,'path':str(path)}]},manifest)
            rows=[r for _,stamp in iter_chunks(manifest) for r in stamp]
            self.assertEqual([r['day'] for r in rows],[1,1,1,2,2,2])
