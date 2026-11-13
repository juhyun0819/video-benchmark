"""Resolve generic annotated evidence to causal original video frames."""
from fractions import Fraction
import bisect
import json
import math
from pathlib import Path
from baselines.ours_offline.inputs import sha256


class EvidenceError(ValueError):
    """An annotation cannot identify a complete causal evidence input."""


def trimmed_sources(sources):
    ordered=sorted(sources,key=lambda s:s['start_time'])
    result=[]
    for i,source in enumerate(ordered):
        row=dict(source)
        start=Fraction(row.get('start_time_fraction',str(row['start_time'])))
        duration=Fraction(row.get('duration_fraction',str(row['duration'])))
        end=start+duration
        if i+1<len(ordered):
            following=ordered[i+1]
            end=min(end,Fraction(following.get('start_time_fraction',str(following['start_time']))))
        if end<=start:raise EvidenceError('Sources must have distinct positive temporal spans')
        row.update(duration=float(end-start),duration_fraction=str(end-start))
        result.append(row)
    return result


def evidence_clips(sources, points, cutoff):
    if not points or not math.isfinite(cutoff):
        raise EvidenceError('Evidence needs finite points and a query cutoff')
    ordered = trimmed_sources(sources)
    starts = [source['start_time'] for source in ordered]
    if any(a >= b for a, b in zip(starts, starts[1:])):
        raise EvidenceError('Source start times must be distinct')
    selected = {}
    for point in points:
        if isinstance(point,bool) or not isinstance(point,(int,float)) or not math.isfinite(point) or point<0 or point > cutoff + 1e-8:
            raise EvidenceError(f'Future evidence point {point} exceeds query cutoff {cutoff}')
        index = bisect.bisect_right(starts, point)-1
        if index < 0:
            raise EvidenceError(f'No source clip contains evidence point {point}')
        source = dict(ordered[index])
        end = Fraction(source.get('start_time_fraction',str(source['start_time'])))+Fraction(source['duration_fraction'])
        if point >= end:
            raise EvidenceError(f'No source clip contains evidence point {point}; annotation lies in a gap')
        selected[source['path']] = source
    return sorted(selected.values(), key=lambda source: source['start_time'])


def evidence_samples(sources, points, cutoff, max_frames):
    from baselines.ours_offline.inputs import uniform_samples
    clips = evidence_clips(sources, points, cutoff)
    if type(max_frames) is not int or max_frames < 2*len(clips):
        raise EvidenceError(f'Frame budget {max_frames} cannot allocate two frames to each of {len(clips)} evidence clips')
    base, extra = divmod(max_frames, len(clips))
    samples = []
    for i, source in enumerate(clips):
        samples.extend(uniform_samples([source], cutoff, base+int(i < extra)))
    samples.sort(key=lambda item: item['requested_global_timestamp'])
    if {sample['source'] for sample in samples} != {clip['path'] for clip in clips}:
        raise EvidenceError('Sampling failed to cover every evidence clip')
    if any(sample['requested_global_timestamp'] > cutoff+1e-8 for sample in samples):
        raise EvidenceError('Future frame selected')
    return clips, samples


def load_evidence(path, manifest, manifest_path):
    """Accept inline QA evidence, or a sidecar bound to the exact QA manifest."""
    ids = {q['qa_id'] for q in manifest['questions']}
    if path is None:
        return {q['qa_id']:q.get('evidence') for q in manifest['questions']}
    document = json.loads(Path(path).read_text())
    if not isinstance(document, dict) or document.get('qa_manifest_sha256') != sha256(manifest_path):
        raise EvidenceError('Evidence sidecar belongs to a different QA manifest')
    entries = document.get('entries')
    if not isinstance(entries, list):raise EvidenceError('Evidence entries must be a list')
    result = {}
    for row in entries:
        if not isinstance(row, dict):raise EvidenceError('Evidence entries must be objects')
        key = row.get('qa_id')
        if key not in ids or key in result:raise EvidenceError('Unknown or duplicate evidence QA ID')
        result[key] = row.get('evidence')
    return result


def reference_samples(sources, evidence, cutoff, max_frames):
    """Resolve generic points, source clips, or half-open evidence intervals."""
    from baselines.ours_offline.inputs import uniform_samples
    if not isinstance(evidence, dict):raise EvidenceError('QA evidence is not available yet')
    kinds = [key for key in ('points','clips','intervals') if key in evidence]
    if len(kinds) != 1:raise EvidenceError('Use exactly one evidence form: points, clips or intervals')
    kind = kinds[0]
    values = evidence[kind]
    if not isinstance(values, list) or not values:raise EvidenceError('Evidence must be a nonempty list')
    if kind == 'points':return evidence_samples(sources, values, cutoff, max_frames)
    trimmed = trimmed_sources(sources)
    intervals = None
    if kind == 'clips':
        by_path={s['path']:s for s in trimmed}
        if any(not isinstance(path,str) or path not in by_path for path in values):
            raise EvidenceError('Evidence clip is not in the verified source manifest')
        clips=sorted((by_path[path] for path in set(values)),key=lambda s:s['start_time'])
        if any(s['start_time']>cutoff for s in clips):raise EvidenceError('Future evidence clip')
    else:
        intervals=[];selected={}
        for span in values:
            if not isinstance(span, dict):raise EvidenceError('Evidence intervals must be objects')
            a,b=span.get('start_time'),span.get('end_time')
            if any(isinstance(x,bool) or not isinstance(x,(float,int)) or not math.isfinite(x) or x<0 for x in (a,b)) or a>=b:
                raise EvidenceError('Evidence intervals need finite 0<=start_time<end_time')
            if b>cutoff+1e-8:raise EvidenceError('Future evidence interval')
            covered=a
            for s in trimmed:
                end=s['start_time']+s['duration']
                if end<=a or s['start_time']>=b:continue
                if s['start_time']>covered+1e-8:raise EvidenceError('Evidence interval contains an unrecorded source gap')
                covered=max(covered,min(end,b));selected[s['path']]=s
            if covered<b-1e-8:raise EvidenceError('Evidence interval not covered by sources')
            intervals.append((a,math.nextafter(float(b),-math.inf)))
        clips=sorted(selected.values(),key=lambda s:s['start_time'])
    if not clips or type(max_frames) is not int or max_frames<2*len(clips):
        raise EvidenceError('Need at least two requested frames per evidence clip')
    base,extra=divmod(max_frames,len(clips));samples=[]
    for i,s in enumerate(clips):
        samples.extend(uniform_samples([s],cutoff,base+int(i<extra),intervals=intervals))
    samples.sort(key=lambda s:s['requested_global_timestamp'])
    if {s['source'] for s in samples}!={s['path'] for s in clips}:raise EvidenceError('Incomplete evidence coverage')
    return clips,samples
