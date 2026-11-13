import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from baselines.ours_offline.prepare import write_prepared
from baselines.ours_offline.inputs import load_inputs
from baselines.ours_offline.run import main


def probe(fps, frames, duration_ts, time_base):
    return json.dumps({'streams':[{'avg_frame_rate':fps,'r_frame_rate':fps,
        'duration_ts':duration_ts,'time_base':time_base,'nb_frames':str(frames),'width':16,'height':16}]})


class DataOnlyPreparationTests(unittest.TestCase):
    def test_fractional_fps_concatenation_and_empty_pending_qa(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ['one.mp4','two.mp4']:(root/name).write_bytes(b'metadata-fixture')
            spec=root/'source-list.json';spec.write_text(json.dumps({'timeline':'concatenate','sources':[{'path':'one.mp4'},{'path':'two.mp4','day':12}]}))
            output=root/'prepared'
            with patch('baselines.ours_offline.prepare.subprocess.check_output',side_effect=[probe('30000/1001',300,300300,'1/30000'),probe('25',250,10,'1/1')]):
                result=write_prepared(root,spec,output)
            sources=json.loads((output/'sources.json').read_text())['sources']
            self.assertEqual(sources[0]['fps_fraction'],'30000/1001')
            self.assertEqual(sources[1]['start_time_fraction'],'1001/100')
            pending=json.loads((output/'questions.pending.json').read_text())
            self.assertEqual(pending['questions'],[]);self.assertEqual(result['qa_evaluated'],0)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['validate','--root',str(root),'--sources',str(output/'sources.json'),'--canonical',str(output/'questions.pending.json')]),0)
            with self.assertRaisesRegex(ValueError,'QA not supplied'):
                load_inputs(root,output/'questions.pending.json',output/'sources.json')
            pending['questions']=[{'qa_id':'ready','day':12,'query_time':15,'question':'Reviewed question','choices':['a','b'],'correct_idx':1}]
            qa=output/'reviewed.json';qa.write_text(json.dumps(pending))
            rows=load_inputs(root,qa,output/'sources.json')[2]
            self.assertEqual(rows[0]['answer'],'B')

    def test_explicit_timeline_does_not_compress_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'a.mp4').write_bytes(b'fixture')
            spec=root/'spec.json';spec.write_text(json.dumps({'timeline':'explicit','sources':[{'path':'a.mp4','start_time':1234.5}]}))
            with patch('baselines.ours_offline.prepare.subprocess.check_output',return_value=probe('25',100,4,'1/1')):
                write_prepared(root,spec,root/'out')
            source=json.loads((root/'out/sources.json').read_text())['sources'][0]
            self.assertEqual(source['start_time'],1234.5)
