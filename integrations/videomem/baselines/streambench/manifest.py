"""One immutable sampled-PTS manifest; bounded decoding for every adapter."""
from fractions import Fraction
import json
import math
from pathlib import Path

from .checkpoint import file_digest, write_json


def _state(path):
    st=Path(path).stat()
    return {'size':st.st_size,'mtime_ns':st.st_mtime_ns}


def prepare_manifest(sources, output, *, fps=1):
    """Scan one source per day, recording actual frame PTS without storing pixels.

    Day timestamps concatenate actual source durations; they are experience time,
    not invented 24-hour wall-clock gaps. No questions/annotations are accepted.
    """
    import av
    if not isinstance(sources,dict) or set(sources)-{'stream_id','days'}:
        raise ValueError('Sources accepts only stream_id and days, never QA/labels')
    stream_id=sources.get('stream_id')
    days=sources.get('days')
    if not isinstance(stream_id,str) or not stream_id or not isinstance(days,list) or not days:
        raise ValueError('Sources needs stream_id and a nonempty days list')
    if isinstance(fps,bool) or not math.isfinite(fps) or fps<=0:
        raise ValueError('fps must be positive and finite')
    rate=Fraction(str(fps))
    output=Path(output)
    if output.exists() or output.with_suffix(output.suffix+'.meta.json').exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name(output.name+'.partial')
    if temporary.exists():
        raise FileExistsError(temporary)
    count=0
    global_offset=Fraction(0)
    source_records=[]
    try:
        with temporary.open('w') as target:
            for expected_day,source in enumerate(days,1):
                if not isinstance(source,dict) or set(source)-{'day','path','source_id'}:
                    raise ValueError('Each source accepts only day, path, source_id')
                if type(source.get('day')) is not int or source['day']!=expected_day:
                    raise ValueError('Days must be exactly 1,2,... in chronological order')
                path=Path(source['path']).expanduser().resolve()
                before=_state(path)
                digest=file_digest(path)
                source_id=source.get('source_id',f'day{expected_day}')
                if not isinstance(source_id,str) or not source_id or source_id in {s['source_id'] for s in source_records}:
                    raise ValueError('source_id must be distinct nonempty text')
                next_sample=Fraction(0)
                previous=None
                duration=Fraction(0)
                sampled=0
                with av.open(str(path)) as container:
                    if not container.streams.video:
                        raise ValueError(f'No video stream: {path}')
                    stream=container.streams.video[0]
                    stream.codec_context.thread_count=1
                    origin=Fraction(stream.start_time or 0)*Fraction(stream.time_base)
                    nominal=Fraction(1,1)/Fraction(stream.average_rate or 30)
                    for index,frame in enumerate(container.decode(stream)):
                        if frame.pts is None or frame.time_base is None:
                            raise ValueError('Every frame needs an exact PTS')
                        actual=Fraction(frame.pts)*Fraction(frame.time_base)-origin
                        if actual<0 or (previous is not None and actual<previous):
                            raise ValueError('Source presentation timestamps are not chronological')
                        previous=actual
                        frame_ticks=getattr(frame,'duration',None)
                        frame_duration=Fraction(frame_ticks)*Fraction(frame.time_base) if frame_ticks else nominal
                        duration=max(duration,actual+frame_duration)
                        if actual<next_sample:
                            continue
                        row={'stream_id':stream_id,'day':expected_day,'global_frame_id':count,
                             'source_id':source_id,'source_path':str(path),'source_frame_index':index,
                             'source_timestamp':float(actual),'global_timestamp':float(global_offset+actual),
                             'source_timestamp_fraction':str(actual),'global_timestamp_fraction':str(global_offset+actual),
                             'pts':frame.pts,'time_base':str(frame.time_base),'source_state':before,'source_sha256':digest}
                        target.write(json.dumps(row,ensure_ascii=False)+'\n')
                        count+=1;sampled+=1
                        next_sample=Fraction(math.floor(actual*rate)+1,1)/rate
                    if stream.duration is not None:
                        duration=max(duration,Fraction(stream.duration)*Fraction(stream.time_base))
                if not sampled or _state(path)!=before:
                    raise ValueError(f'Source empty or changed while preparing: {path}')
                source_records.append({'day':expected_day,'path':str(path),'source_id':source_id,
                                       'source_sha256':digest,'source_state':before,'sampled_frames':sampled,
                                       'duration_seconds':float(duration)})
                global_offset+=duration
        temporary.rename(output)
        metadata={'schema_version':1,'stream_id':stream_id,'fps':float(rate),'frame_count':count,
                  'sources':source_records,'manifest_sha256':file_digest(output),
                  'sampling':'first_actual_pts_at_or_after_each_source_fps_grid_target',
                  'global_time':'concatenated_source_duration','pixels_preloaded':False}
        write_json(output.with_suffix(output.suffix+'.meta.json'),metadata)
        return metadata
    except BaseException:
        if temporary.exists(): temporary.unlink()
        raise


def manifest_info(path):
    path=Path(path)
    metadata=json.loads(path.with_suffix(path.suffix+'.meta.json').read_text())
    if metadata.get('schema_version')!=1 or file_digest(path)!=metadata.get('manifest_sha256'):
        raise ValueError('Frame manifest integrity/schema mismatch')
    return metadata


def iter_chunks(manifest, *, chunk_size=8, start_after_frame_id=-1, stop_after_day=None):
    """Yield owned PIL frames until next iteration/close, no prior-day decode.

    Original pixels are shared as input; each adapter applies its own processor.
    Resuming scans small manifest metadata, not previous videos/features.
    """
    import av
    if type(chunk_size) is not int or chunk_size<=0:
        raise ValueError('chunk_size must be a positive integer')
    metadata=manifest_info(manifest)
    expected=0
    previous_time=None
    container=decoder=None
    source_key=None
    decoded_index=-1
    current_frame=None
    images,records=[],[]
    allowed={'stream_id','day','global_frame_id','source_id','source_path','source_frame_index',
             'source_timestamp','global_timestamp','source_timestamp_fraction','global_timestamp_fraction',
             'pts','time_base','source_state','source_sha256'}
    try:
        with Path(manifest).open() as stream:
            for line in stream:
                row=json.loads(line)
                if set(row)-allowed or row.get('stream_id')!=metadata['stream_id']:
                    raise ValueError('Frame manifest has unknown/private fields or a mismatched stream')
                if row['global_frame_id']!=expected:
                    raise ValueError('Manifest frame IDs must be contiguous')
                expected+=1
                stamp=Fraction(row['global_timestamp_fraction'])
                if previous_time is not None and stamp<=previous_time:
                    raise ValueError('Manifest timestamps must increase')
                previous_time=stamp
                if row['global_frame_id']<=start_after_frame_id:
                    continue
                if stop_after_day is not None and row['day']>stop_after_day:
                    break
                path=Path(row['source_path'])
                if images and (len(images)>=chunk_size or row['day']!=records[-1]['day']):
                    try: yield images,records
                    finally:
                        for image in images:image.close()
                        images,records=[],[]
                key=(row['day'],row['source_id'],path)
                if key!=source_key:
                    if container:container.close()
                    if _state(path)!=row['source_state'] or file_digest(path)!=row['source_sha256']:
                        raise ValueError(f'Source changed since manifest creation: {path}')
                    container=av.open(str(path))
                    video=container.streams.video[0]
                    video.codec_context.thread_count=1
                    decoder=iter(container.decode(video))
                    decoded_index=-1
                    source_key=key
                target=row['source_frame_index']
                if type(target) is not int or target<=decoded_index:
                    raise ValueError('Source sampled frame indices must increase')
                while decoded_index<target:
                    try:current_frame=next(decoder)
                    except StopIteration as exc:raise ValueError('Source ended before a manifest frame') from exc
                    decoded_index+=1
                if current_frame.pts!=row['pts'] or str(current_frame.time_base)!=row['time_base']:
                    raise ValueError('Decoded frame PTS differs from common manifest')
                if _state(path)!=row['source_state']:
                    raise ValueError('Source changed during decoding')
                images.append(current_frame.to_image().convert('RGB'))
                # Pass only temporal/public metadata, never a retrievable source
                # video path: adapters must retain/account evidence themselves.
                records.append({k:row[k] for k in ('stream_id','day','global_frame_id','source_id',
                                                    'source_frame_index','source_timestamp','global_timestamp')})
            if images:
                try:yield images,records
                finally:
                    for image in images:image.close()
                    images,records=[],[]
    finally:
        for image in images:image.close()
        if container:container.close()
