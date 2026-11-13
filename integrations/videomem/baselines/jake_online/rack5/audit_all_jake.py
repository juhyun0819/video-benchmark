from concurrent.futures import ThreadPoolExecutor
import json,sys
from pathlib import Path
sys.path.insert(0,'/home/zzuhyun/hyunjun/videomem-baselines')
from baselines.jake_online.inputs import recent_samples,decode_sample,sha256
from baselines.jake_online.stream_frames import sample_plan
root=Path('/convei_nas2/zzuhyun/hyunjun/datasets/egolife');canonical=root/'canonical_v2'
manifest=json.loads((canonical/'jake_qa_v1.json').read_text());sources=json.loads((canonical/'all_sources_v2.json').read_text())['sources']
plan=sample_plan(sources,2)
assert all(a['requested_global_timestamp']<b['requested_global_timestamp'] for a,b in zip(plan,plan[1:]))
def audit(q):
 samples=recent_samples(sources,q['query_time'],4)
 actual=[decode_sample(root,s,q['query_time'])[1] for s in samples]
 assert len(actual)==4 and all(s['actual_global_timestamp']<=q['query_time']+1e-8 for s in actual)
 return {'qa_id':q['qa_id'],'query_time':q['query_time'],'frames':actual}
with ThreadPoolExecutor(max_workers=8) as pool:rows=list(pool.map(audit,manifest['questions']))
report={'status':'passed','qa_count':len(rows),'frame_count':sum(len(r['frames']) for r in rows),'future_frames':0,'stream_frames':len(plan),'stream_days':sorted(set(s['day'] for s in plan)),'qa_sha256':sha256(canonical/'jake_qa_v1.json'),'rows':rows}
Path('/home/zzuhyun/hyunjun/jake_setup/all-days-causal-audit.json').write_text(json.dumps(report,indent=2))
print({k:v for k,v in report.items() if k!='rows'},flush=True)
