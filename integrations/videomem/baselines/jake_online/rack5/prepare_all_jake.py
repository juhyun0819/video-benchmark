import ast
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,'/home/zzuhyun/hyunjun/videomem-baselines')
from baselines.jake_online.inputs import sha256,global_time
root=Path('/convei_nas2/zzuhyun/hyunjun/datasets/egolife')
out=root/'canonical_v2'
out.mkdir(exist_ok=False)
old=root/'canonical'
manifest=json.loads((old/'jake_qa_v1.json').read_text())
upstream=Path('/home/zzuhyun/hyunjun/upstream/EgoLife')
util=upstream/'EgoRAG/egorag/utils/util.py'
# Execute only the pure official timestamp function, without API dependencies.
tree=ast.parse(util.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='time_to_frame_idx')
namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(util),'exec'),namespace)
convert=namespace['time_to_frame_idx']
changes=[]
for q in manifest['questions']:
 value=(q['day']-1)*86400+convert(q['original_query_time']['time'],20)/20
 if q['query_time'] is not None:assert abs(q['query_time']-value)<1e-8
 else:
  changes.append({'qa_id':q['qa_id'],'original':q['original_query_time'],'global_seconds':value})
  q.update(query_time=value,time_status='official_frame_offset_overflow_normalized')
manifest['questions'].sort(key=lambda q:(q['query_time'],q['qa_id']))
manifest.update(schema='jake_canonical_qa_v2',time_review_qa_ids=[],timecode='Official time_to_frame_idx(HHMMSSFF,20)/20; FF carried as frame offset, including FF>=20',
 time_conversion={'upstream_revision':subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip(),'file':'EgoRAG/egorag/utils/util.py','function':'time_to_frame_idx','source_sha256':sha256(util),'normalizations':changes,'annotation_intent':'Apply official conversion arithmetic; do not infer centiseconds or edit raw QA.'},parent_manifest_sha256=sha256(old/'jake_qa_v1.json'))
(out/'jake_qa_v1.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
(out/'scoring_v1.py').write_bytes((old/'scoring_v1.py').read_bytes())
existing={s['path']:s for s in json.loads((old/'day1_sources_v1.json').read_text())['sources']}
inv=json.loads((root/'_download/tools/remote_inventory.json').read_text())
rows=[r for r in inv['files'] if r['participant']=='A1_JAKE'];assert len(rows)==6266

def probe(r):
 path=root/r['path'];s=path.stat()
 receipt=json.loads((root/'_download/receipts'/(r['path'].replace('/','__')+'.ready.json')).read_text())
 state=[s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
 assert s.st_size==r['size'] and receipt['sha256']==r['lfs_sha256'] and receipt['file_state']==state
 if r['path'] in existing:
  assert existing[r['path']]['file_state']==state;return existing[r['path']]
 d=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=avg_frame_rate,duration,nb_frames,width,height','-of','json',str(path)]))['streams'][0]
 assert Fraction(d['avg_frame_rate'])==20
 day=int(r['day'][3:])
 return {'path':r['path'],'day':day,'start_time':global_time(r['day'],path.stem.split('_')[-1]),'duration':float(d['duration']),'frames':int(d['nb_frames']),'width':d['width'],'height':d['height'],'fps':20,'bytes':r['size'],'sha256':r['lfs_sha256'],'file_state':state}
with ThreadPoolExecutor(max_workers=12) as pool:sources=sorted(pool.map(probe,rows),key=lambda s:s['start_time'])
(out/'all_sources_v2.json').write_text(json.dumps({'sources':sources},indent=2)+'\n')
(out/'day1_sources_v1.json').write_bytes((old/'day1_sources_v1.json').read_bytes())
report={'status':'complete','qa_count':500,'verified_sources':len(sources),'normalized_timecodes':changes,'manifest_sha256':sha256(out/'jake_qa_v1.json'),'sources_sha256':sha256(out/'all_sources_v2.json')}
(out/'preparation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
