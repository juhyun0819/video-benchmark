"""Pinned public-source downloader and evidence-preserving Day 1 assembler."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import quote
from src.common import load, save, probe

ROOT = Path(__file__).resolve().parent

def parse_srt(text):
    def seconds(value):
        h,m,s = value.replace(',', '.').split(':')
        return int(h)*3600+int(m)*60+float(s)
    rows, issues = [], []
    for block in re.split(r'\n\s*\n', text.replace('\r\n','\n').strip()):
        match = re.search(r'(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})', block)
        if not match:
            if block.strip(): issues.append({'type':'malformed_block','text':block})
            continue
        start,end = map(seconds,match.groups())
        body = block[match.end():].strip()
        row={'annotation_index':len(rows)+1,'source_start':start,'source_end':end,'text':body}
        if end <= start or not body:
            issues.append({'type':'invalid_interval_or_empty_text',**row})
            continue
        rows.append(row)
    if any(b['source_start']<a['source_start'] for a,b in zip(rows,rows[1:])):
        issues.append({'type':'non_chronological_srt_order','action':'sorted by original start timestamp; timestamps unchanged'})
    return sorted(rows,key=lambda r:(r['source_start'],r['source_end'])),issues

def download(config, videos=False):
    base=ROOT/'datasets/TeleEgo-Source'
    def fetch(f):
        target=base/f['path']
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.stat().st_size==f['size']:
            print('Already present:', f['path'],flush=True)
            return
        partial=target.with_suffix(target.suffix+'.partial')
        url=f"https://huggingface.co/datasets/{config['repo_id']}/resolve/{config['revision']}/{quote(f['path'])}?download=true"
        print('Downloading:',f['path'],f['size'],flush=True)
        subprocess.run(['curl','--fail','--location','--retry','4','--connect-timeout','30','--speed-limit','1024','--speed-time','120','--continue-at','-','--output',str(partial),url],check=True)
        if partial.stat().st_size!=f['size']: raise ValueError('Downloaded size mismatch: '+f['path'])
        expected=f.get('lfs',{}).get('oid')
        if expected:
            digest=hashlib.sha256()
            with partial.open('rb') as stream:
                for chunk in iter(lambda:stream.read(8*1024*1024),b''): digest.update(chunk)
            if digest.hexdigest()!=expected.removeprefix('sha256:'): raise ValueError('SHA256 mismatch: '+f['path'])
        partial.replace(target)
    selected = [f for f in config['files'] if videos or not f['path'].endswith('.mp4')]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, selected))

def assemble(config, render=False):
    base=ROOT/'datasets/TeleEgo-Source'
    output=ROOT/'outputs/teleego_p4_day1'
    output.mkdir(parents=True,exist_ok=True)
    segments, annotations, issues = [],[],[]
    cursor=0.0
    missing=[]
    for name in config['order']:
        relative=f'teleego_srts/P4/{name}/{name}_narration.srt'
        srt=base/relative
        if not srt.exists(): raise ValueError('Download annotations first: '+relative)
        rows,problems=parse_srt(srt.read_text(encoding='utf-8-sig'))
        issues.extend({'source':name,**problem} for problem in problems)
        path=base/f'source_videos/P4/{name}_video.mp4'
        info=probe(path) if path.exists() else None
        if info is None: missing.append(str(path))
        observed=max((r['source_end'] for r in rows),default=0)
        duration=info['duration'] if info else observed
        segment={'clip_id':name,'source_dataset':'TeleEgo','source_participant':'P4',
                 'source_day':name.split('-')[0],'source_recorded_at':None,'source_video':str(path),
                 'synthetic_day':1,'source_start':0,'source_end':duration,
                 'stream_start':cursor,'stream_end':cursor+duration,
                 'annotation_observation_end':observed,'media_verified':info is not None,
                 'timing_basis':'ffprobe' if info else 'SRT estimate only','media':info,
                 'annotation_path':relative,'annotation_sha256':hashlib.sha256(srt.read_bytes()).hexdigest()}
        segments.append(segment)
        for row in rows:
            valid=row['source_end']<=duration+0.05
            if not valid: issues.append({'type':'annotation_exceeds_video','source':name,**row})
            annotations.append({**row,'clip_id':name,'source_day':segment['source_day'],
                                'synthetic_day':1,'stream_start':cursor+row['source_start'],
                                'stream_end':cursor+row['source_end'],'within_media_bounds':valid,
                                'provenance':relative,'annotation_type':'official_narration',
                                'verification':'source_annotation; not independently visually reviewed'})
        cursor+=duration
    ready=not missing
    data={'schema_version':'teleego-1.0','persona_id':'teleego_p4','synthetic_persona':True,
          'source_participant':'P4','synthetic_day':1,'synthetic_start_time':'08:00:00',
          'source_revision':config['revision'],'repo_id':config['repo_id'],
          'description':'Multiple original days reordered; not a continuous real-world day',
          'media_verified':ready,'duration':cursor,'at_least_12_hours':cursor>=43200,
          'duration_status':'measured' if ready else 'provisional SRT estimate',
          'segments':segments,'missing_videos':missing,'rendered':False}
    save(output/'source_manifest.json',data)
    save(output/'annotations.json',annotations)
    save(output/'validation.json',{'issues':issues,'media_verified':ready,
          'manual_review_required':['Visual continuity','Source-date clues in speech','Narration accuracy','Unobserved gaps between annotations are not labeled activities']})
    # Extractive source-annotation QA: no broad activity label inferred from filenames.
    qa=[]
    if ready:
        for i,row in enumerate(annotations):
            if row['within_media_bounds']:
                qa.append({'qa_id':f'teleego_{i:05d}','question_type':'annotation_recall',
                           'question':f"What does the source narration report in segment {row['clip_id']} at source seconds {row['source_start']}–{row['source_end']}?",
                           'answer':row['text'],'answer_language':'zh','provenance':row['provenance'],
                           'evidence':{'clip_id':row['clip_id'],'stream_start':row['stream_start'],'stream_end':row['stream_end']},
                           'review_status':'annotation-backed; visual review pending'})
    save(output/'annotation_qa.json',qa)
    if render:
        if not ready: raise ValueError('Full source videos must be downloaded before rendering')
        if not data['at_least_12_hours']: raise ValueError('Measured stream is shorter than 12 hours')
        render_day(data,output)
    print(json.dumps({'duration_hours':cursor/3600,'annotations':len(annotations),'missing_videos':len(missing),'output':str(output)},indent=2),flush=True)
    return data

def render_day(data,output):
    from src.stream_generator import concat
    parts=output/'normalized'
    parts.mkdir(exist_ok=True)
    paths=[]
    for i,s in enumerate(data['segments']):
        target=parts/f'{i:02d}.mp4'
        # Completed parts may be resumed; all frames are decoded by FFmpeg during normalization.
        if not target.exists():
            tmp=target.with_name(target.stem+'.partial.mp4')
            subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-i',s['source_video'],
                '-map','0:v:0','-vf','scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,fps=24,setsar=1',
                '-an','-c:v','libx264','-preset','veryfast','-crf','25','-threads','4',str(tmp)],check=True)
            tmp.replace(target)
        duration=probe(target)['duration']
        if abs(duration-(s['source_end']-s['source_start']))>0.15: raise ValueError('Normalized duration mismatch')
        paths.append(target)
        print('Normalized:',target,flush=True)
    final=output/'day_01.mp4'
    concat(paths,final)
    actual=probe(final)['duration']
    if abs(actual-data['duration'])>0.5: raise ValueError('Final duration mismatch')
    data.update(rendered=True,actual_duration=actual,render_settings={'resolution':'640x360','fps':24,'audio':False})
    save(output/'source_manifest.json',data)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--download',choices=['annotations','all'])
    parser.add_argument('--render',action='store_true')
    args=parser.parse_args()
    config=load(ROOT/'config/teleego_p4.json')
    if args.download: download(config,args.download=='all')
    assemble(config,args.render)
