import json

import pytest

from baselines.egogpt_caption import launch


def test_smoke_failure_prevents_full_worker_launch(tmp_path,monkeypatch):
    runtime = tmp_path/'runtime.json'
    runtime.write_text(json.dumps({'status':'ready','python':'test-python'}))
    output = tmp_path/'output'
    commands = []
    class Process:
        pid=123
        def __init__(self,args,**kwargs):
            commands.append(args)
            if 'prepare' in args:
                target=tmp_path/'output_smoke';target.mkdir(exist_ok=True)
                (target/'manifest.json').write_text(json.dumps({'clips':[
                    {'day_id':'Day1','clip_id':'day1_000000'},
                    {'day_id':'Day1','clip_id':'day1_000001'}]}))
            self.returncode=2 if 'baselines.egogpt_caption.audit' in args else 0
        def poll(self):return self.returncode
    monkeypatch.setattr(launch.subprocess,'Popen',Process)
    monkeypatch.setattr(launch.time,'sleep',lambda seconds:None)
    monkeypatch.setattr(launch.signal,'signal',lambda *args:None)
    with pytest.raises(RuntimeError,match='smoke-audit failed'):
        launch.launch(tmp_path/'inventory.json',runtime,output,tmp_path/'audio',[0,1])
    workers=[c for c in commands if 'worker' in c]
    assert len(workers)==1 and '--clip-ids' in workers[0]
    status=json.loads((output/'pipeline.json').read_text())
    assert status['phase']=='error' and status['failed_phase']=='verifying_smoke'


def test_repeated_gpu_indices_rejected_before_preparation(tmp_path):
    with pytest.raises(ValueError,match='distinct'):
        launch.launch('inventory','runtime',tmp_path/'output','audio',[0,0])
    assert not (tmp_path/'output').exists()
