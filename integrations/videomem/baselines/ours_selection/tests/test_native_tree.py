"""CPU integration with the actual vendored VideoTree kernels, no model weights."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from baselines.ours_selection.captions import CaptionManifest
from baselines.ours_selection.selection import evaluate_question


@unittest.skipUnless(importlib.util.find_spec('scipy') is not None,'Native VideoTree kernels require SciPy')
class NativeTreeTests(unittest.TestCase):
    def test_actual_ward_tree_causal_caption_filter_and_raw_vlm_bridge(self):
        class Archive:
            identity='native-tree-fixture'
            times=[float(i) for i in range(12)]
            features=np.array([[1+np.sin(i),1+np.cos(i),.1+i/12.,1.] for i in range(12)],dtype=np.float32)
            records=[{'global_frame_id':i,'actual_global_timestamp':float(i)} for i in range(12)]
            def eligible(self,cutoff):return [i for i,t in enumerate(self.times) if t<=cutoff]
            def verify_selected_sources(self,root,ids):pass
        class Relevance:
            seen=[]
            def score(self,question,choices,captions):
                assert choices==['first','second','third','fourth']
                self.seen.extend(int(c.split('-')[1]) for c in captions)
                return [3]*len(captions)
        class Backend:
            backbone='llava_onevision'
            calls=[]
            def answer(self,frames,stamps,question,choices):
                self.calls.append(([s['global_frame_id'] for s in stamps],list(choices)))
                return {'prediction':'B','raw_output':'B','complete':True}
        archive=Archive();relevance=Relevance();backend=Backend()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'captions.json'
            captions=[{'global_frame_id':i,'global_timestamp':float(i),'observed_until':float(i),
                'available_at':float(i),'text':f'caption-{i}'} for i in range(12)]
            captions[3].update(observed_until=10.,available_at=10.)
            path.write_text(json.dumps({'schema':'ours_videotree_captions_v1','archive_sha256':archive.identity,
                'producer':{'model':'fixture-captioner','question_conditioned':False},'captions':captions}))
            manifest=CaptionManifest(path,archive)
            q={'qa_id':'native-tree','query_time':7.,'question':'Which visible object?',
               'choices':['first','second','third','fourth'],'answer':'B','correct_idx':1,
               'reason':'SECRET','target_time':'SECRET'}
            result=evaluate_question('.',archive,q,method='videotree',scorer=relevance,
                parameters={'initial_clusters':2,'maximum_clusters':4,'high_relevance_threshold':4,
                            'branch_width':2,'seed':1},backend=backend,max_frames=8,captions=manifest,
                decoder=lambda root,record,cutoff:(object(),dict(record)))
        self.assertEqual(result['status'],'ok',result.get('error'))
        self.assertTrue(result['correct'])
        self.assertEqual(result['selection']['candidate_count'],7)
        self.assertEqual(result['selection']['cluster_count'],4)
        self.assertTrue(relevance.seen)
        self.assertTrue(set(relevance.seen)<={0,1,2,4,5,6,7})
        self.assertTrue(set(result['selected_frame_ids'])<={0,1,2,4,5,6,7})
        self.assertEqual(backend.calls[0][0],result['selected_frame_ids'])
        json.dumps(result)  # Real native hierarchy must remain JSON-serializable.


if __name__=='__main__':unittest.main()
