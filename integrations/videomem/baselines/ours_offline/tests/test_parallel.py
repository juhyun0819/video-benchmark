"""CPU-only supervisor tests; worker files are controlled fixtures, no GPUs/models."""
import json
import os
from pathlib import Path
import tempfile
import unittest

from baselines.ours_offline.inputs import sha256
from baselines.ours_offline.parallel import (atomic_json,build_plan,expected_model,merge_all,
    normalize_config,read_json,supervise)


class ParallelTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        video=self.root/'clip.mp4';video.write_bytes(b'no decode in supervisor tests')
        stat=video.stat()
        source={'path':'clip.mp4','start_time':0.,'duration':10.,'fps':1.,'frames':10,
                'file_state':[stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns]}
        self.sources=self.root/'sources.json';atomic_json(self.sources,{'sources':[source]})
        scorer=Path(__file__).resolve().parents[2]/'streambench/scoring.py'
        self.questions=[{'qa_id':f'Q{i}','query_time':float(i+2),'question':'Which?',
            'choices':['alpha','beta'],'answer':'B','correct_idx':1} for i in range(5)]
        canonical=self.root/'qa.json'
        atomic_json(canonical,{'questions':self.questions,'scorer_sha256':sha256(scorer),'sources_sha256':sha256(self.sources)})
        models=[]
        for name,backbone,gpus in [('qwen','qwen3_vl',[9000,9001]),('llava','llava_onevision',[9002])]:
            folder=self.root/name;folder.mkdir();ledger=self.root/(name+'.ledger.json');atomic_json(ledger,{'fixture':True})
            models.append({'name':name,'backbone':backbone,'model':str(folder),'ledger':str(ledger),
                'model_revision':'a'*40,'gpus':gpus})
        self.config={'root':str(self.root),'canonical':str(canonical),'sources':str(self.sources),
            'max_frames':32,'expected_questions':5,'models':models,'poll_seconds':.1}
        self.path=self.root/'parallel.json';atomic_json(self.path,self.config)

    def plan(self,name='run'):
        return build_plan(normalize_config(self.path),self.root/name)[0]

    def write_worker(self,plan,worker,*,status='ok',omit_last=False):
        model=next(m for m in plan['config']['models'] if m['name']==worker['model'])
        folder=Path(worker['output']);folder.mkdir(parents=True)
        provenance={key:plan[key] for key in ('canonical_sha256','sources_sha256','scorer_sha256','code_sha256')}
        provenance.update({key:worker[key] for key in ('qa_ids','shard','shards')})
        provenance.update(max_frames=32,decode_workers=model['decode_workers'],model=expected_model(model),
                          verification={'ledger_sha256':model['ledger_sha256']})
        atomic_json(folder/'provenance.json',provenance)
        selected=[q for q in self.questions if q['qa_id'] in worker['qa_ids']]
        if omit_last:selected=selected[:-1]
        with (folder/'predictions.jsonl').open('w') as stream:
            for q in selected:
                row={'method':'Offline-'+model['backbone'],'qa_id':q['qa_id'],'query_time':q['query_time'],
                    'GT':'B','prediction':'B','raw_output':'B','correct':True,'status':status,'error':None,
                    'actual_frame_timestamps':[{'actual_global_timestamp':0.},{'actual_global_timestamp':1.}],
                    'answer_latency':.01,'stream_update_time':0.}
                stream.write(json.dumps(row)+'\n')
        atomic_json(folder/'status.json',{'status':'complete','completed_qa':len(selected)})

    @staticmethod
    def states(plan):
        return {f"{w['model']}:{w['shard']}":{'returncode':0,'started_at':1.,'finished_at':3.} for w in plan['workers']}

    def test_supervisor_launches_disjoint_shards_with_controlled_environment(self):
        plan=self.plan();calls=[]
        class Process:
            def __init__(self,pid):self.pid=pid;self.polls=0
            def poll(self):
                self.polls+=1
                return None if self.polls==1 else 0
        def popen(argv,**kwargs):
            worker=next(w for w in plan['workers'] if w['argv']==argv)
            self.write_worker(plan,worker);calls.append((argv,kwargs))
            return Process(7000+len(calls))
        samples=lambda gpus:{str(g):{'memory_used_mib':1234,'utilization_percent':75,'sampled_at':1.} for g in gpus}
        code=supervise(plan,popen=popen,occupancy=lambda g:{'gpu':g,'memory_mib':0},metrics=samples,sleeper=lambda _:None)
        self.assertEqual(code,0);self.assertEqual(len(calls),3)
        for argv,kw in calls:
            self.assertEqual(argv[0],plan['interpreter']);self.assertIn('--shard',argv)
            self.assertEqual(argv[argv.index('--decode-workers')+1],'4')
            self.assertTrue(kw['start_new_session']);self.assertNotIn('CUDA_VISIBLE_DEVICES',kw['env'])
            self.assertEqual(kw['env']['OMP_NUM_THREADS'],'4')
            self.assertEqual(kw['env']['CUDA_DEVICE_ORDER'],'PCI_BUS_ID')
        state=read_json(Path(plan['output'])/'status.json')
        self.assertEqual(state['status'],'complete')
        self.assertEqual(state['gpu_metrics']['9000']['peak_observed_memory_used_mib'],1234)
        for summary in state['summaries']:
            self.assertTrue(summary['exact_unique_coverage']);self.assertEqual(summary['correct'],5)
            self.assertEqual(summary['synthesized_missing_or_rejected'],0)
            rows=[json.loads(line) for line in (Path(plan['output'])/summary['model']/'predictions.merged.jsonl').read_text().splitlines()]
            self.assertEqual([r['qa_id'] for r in rows],plan['qa_ids'])

    def test_all_requested_gpus_are_checked_before_any_spawn(self):
        plan=self.plan();calls=[]
        def occupied(gpu):
            if gpu==9001:raise RuntimeError('occupied')
            return {'gpu':gpu}
        with self.assertRaisesRegex(RuntimeError,'occupied'):
            supervise(plan,popen=lambda *a,**kw:calls.append(a),occupancy=occupied)
        self.assertEqual(calls,[])
        self.assertEqual(read_json(Path(plan['output'])/'status.json')['status'],'supervisor_failed')

    def test_missing_and_incomplete_keep_fixed_denominator_and_explicit_origin(self):
        plan=self.plan()
        for worker in plan['workers']:
            self.write_worker(plan,worker,omit_last=worker['model']=='qwen' and worker['shard']==0,
                              status='incomplete' if worker['model']=='llava' else 'ok')
        summaries=merge_all(plan,self.states(plan));qwen,llava=summaries
        self.assertEqual(qwen['expected'],5);self.assertEqual(qwen['correct'],4)
        self.assertEqual(qwen['accuracy_over_expected'],.8)
        self.assertFalse(qwen['exact_unique_coverage']);self.assertEqual(qwen['synthesized_missing_or_rejected'],1)
        rows=[json.loads(line) for line in (Path(plan['output'])/'qwen/predictions.merged.jsonl').read_text().splitlines()]
        missing=[r for r in rows if r.get('supervisor_generated')]
        self.assertEqual(len(missing),1);self.assertEqual(missing[0]['status'],'error')
        self.assertEqual(llava['correct'],0);self.assertTrue(llava['exact_unique_coverage'])
        self.assertEqual(llava['errors_or_incomplete'],5)

    def test_duplicate_unknown_future_and_provenance_mismatch_rejected(self):
        plan=self.plan()
        for worker in plan['workers']:self.write_worker(plan,worker)
        first=Path(plan['workers'][0]['output'])/'predictions.jsonl'
        rows=[json.loads(line) for line in first.read_text().splitlines()]
        rows.append(dict(rows[0]));rows.append(dict(rows[0],qa_id='NOT_IN_MANIFEST'))
        rows[1]['actual_frame_timestamps'][1]['actual_global_timestamp']=1000.
        first.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        llava=Path(plan['workers'][-1]['output'])/'provenance.json'
        provenance=read_json(llava);provenance['model']['model_revision']='b'*40;atomic_json(llava,provenance)
        summaries=merge_all(plan,self.states(plan));qwen,llava=summaries
        self.assertEqual(qwen['duplicate_ids'],['Q0']);self.assertFalse(qwen['exact_unique_coverage'])
        self.assertGreaterEqual(len(qwen['audit_errors']),3)
        self.assertEqual(llava['accepted_worker_rows'],0);self.assertEqual(llava['synthesized_missing_or_rejected'],5)
        self.assertEqual(llava['accuracy_over_expected'],0.)

    def test_bad_allocations_and_wrong_manifest_count_are_prelaunch_errors(self):
        self.config['models'][1]['gpus']=[9000];atomic_json(self.path,self.config)
        with self.assertRaisesRegex(ValueError,'disjoint'):normalize_config(self.path)
        self.config['models'][1]['gpus']=[9002];self.config['expected_questions']=500;atomic_json(self.path,self.config)
        with self.assertRaisesRegex(ValueError,'expected_questions'):self.plan()
        self.config['cpu_threads']=2;atomic_json(self.path,self.config)
        with self.assertRaisesRegex(ValueError,'four PyTorch'):normalize_config(self.path)


if __name__=='__main__':unittest.main()
