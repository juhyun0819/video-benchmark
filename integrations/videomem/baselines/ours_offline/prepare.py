"""Prepare source metadata before QA exists; no frames or model are evaluated."""
from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import subprocess

from .inputs import finite_time, sha256, validate_sources


def prepare_sources(root, source_list):
    root = Path(root).resolve()
    spec = json.loads(Path(source_list).read_text())
    timeline = spec.get('timeline')
    rows = spec.get('sources')
    if timeline not in ('concatenate','explicit') or not isinstance(rows,list) or not rows:
        raise ValueError('source-list requires timeline concatenate|explicit and nonempty sources')
    sources, cursor = [], Fraction(0)
    for item in rows:
        path = Path(item.get('path',''))
        absolute=(root/path).resolve()
        if path.is_absolute() or '..' in path.parts or not absolute.is_relative_to(root) or not absolute.is_file():
            raise ValueError('Each source path must identify an existing file inside root')
        if timeline=='concatenate' and 'start_time' in item:
            raise ValueError('Do not mix explicit start_time with concatenate timeline')
        result=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=avg_frame_rate,r_frame_rate,duration,duration_ts,time_base,nb_frames,width,height',
            '-of','json',str(absolute)],text=True))
        if len(result.get('streams',[]))!=1:
            raise ValueError(f'Expected one selected video stream: {path}')
        stream=result['streams'][0]
        fps=Fraction(stream['avg_frame_rate'])
        if fps<=0 or fps!=Fraction(stream['r_frame_rate']):
            raise ValueError(f'CFR frame-rate metadata required: {path}')
        try:frames=int(stream['nb_frames'])
        except (KeyError,ValueError) as exc:
            raise ValueError(f'Exact frame count metadata required; no duration-based frame guessing: {path}') from exc
        duration=(Fraction(stream['duration_ts'])*Fraction(stream['time_base'])
                  if stream.get('duration_ts') is not None else Fraction(stream['duration']))
        if duration<=0 or frames<1 or abs(duration-Fraction(frames)/fps)>1/fps:
            raise ValueError(f'Inconsistent CFR stream duration/frame count: {path}')
        start=cursor if timeline=='concatenate' else finite_time(item.get('start_time'),'source start_time')
        state=absolute.stat()
        record={'path':str(path),'start_time':float(start),'start_time_fraction':str(start),
            'duration':float(duration),'duration_fraction':str(duration),'fps':float(fps),'fps_fraction':str(fps),
            'frames':frames,'width':int(stream['width']),'height':int(stream['height']),
            'bytes':state.st_size,'file_state':[state.st_ino,state.st_size,state.st_mtime_ns,state.st_ctime_ns]}
        if 'day' in item:record['day']=item['day']
        sources.append(record);cursor=start+duration
    sources=validate_sources(root,sources)
    return {'schema':'ours_sources_v1','timeline':timeline,'source_list_sha256':sha256(source_list),
        'verification':'ffprobe video stream metadata plus local file stat; no full-file SHA256 receipt claimed',
        'sources':sources}


def write_prepared(root, source_list, output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    inventory=prepare_sources(root,source_list)
    output.mkdir(parents=True,exist_ok=False)
    source_path=output/'sources.json';source_path.write_text(json.dumps(inventory,indent=2)+'\n')
    scorer=Path(__file__).parents[1]/'streambench/scoring.py'
    pending={'schema':'ours_canonical_qa_v1','status':'awaiting_QA',
             'scorer_sha256':sha256(scorer),'sources_sha256':sha256(source_path),'questions':[]}
    (output/'questions.pending.json').write_text(json.dumps(pending,indent=2)+'\n')
    status={'status':'sources_ready_awaiting_QA','sources':len(inventory['sources']),
            'frames':sum(s['frames'] for s in inventory['sources']),
            'duration_seconds':sum(s['duration'] for s in inventory['sources']),
            'inference_performed':False,'qa_evaluated':0}
    status['days'] = {str(day): {'sources':sum(s.get('day')==day for s in inventory['sources']),
        'frames':sum(s['frames'] for s in inventory['sources'] if s.get('day')==day)}
        for day in sorted({s['day'] for s in inventory['sources'] if 'day' in s})}
    (output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    return status
