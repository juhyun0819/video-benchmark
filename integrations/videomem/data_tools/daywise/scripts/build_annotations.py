"""Preserve original annotations and map them to the current uncut Day clips."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
import copy
import csv
import hashlib
import json
import os
import shutil
import subprocess

OUT = Path(os.environ.get('VIDEOMEM_PACKAGE_OUTPUT', 'work/daywise_package')).resolve()
ROOT = Path(os.environ.get('VIDEOMEM_WORK_ROOT', 'work/daywise_inputs')).resolve()
MEDIA = Path(os.environ.get('VIDEOMEM_MEDIA_ROOT', 'work/media')).resolve()
CONVERSION = Path(os.environ.get('HD_EPIC_CONVERSION_ROOT', 'work/hd_epic/conversion')).resolve()
MANIFEST = MEDIA / 'manifest.json'
ANNOTATIONS = ROOT / 'annotations_adapted_v1/all_clips_with_provisional_offsets.jsonl'
BASIS = 'current_clip_order_at_1x_without_cuts_gaps_or_overlaps'


def dec(value):
    return Decimal(str(value))


def stamp(seconds):
    ms = round(dec(seconds) * 1000)
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    seconds, ms = divmod(ms, 1000)
    return f'{hours:02}:{minutes:02}:{seconds:02}.{ms:03}'


def timecode(seconds):
    frames = dec(seconds) * 25
    assert frames == frames.to_integral_value(), seconds
    frames = int(frames)
    return f'{frames//90000:02}:{frames//1500%60:02}:{frames//25%60:02}:{frames%25:02}'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def probe(row):
    path = Path(row['output'])
    assert path.is_file() and path.stat().st_size == row['bytes'], path
    data = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_entries',
        'stream=codec_type,width,height,r_frame_rate,avg_frame_rate,duration,nb_frames:format=duration,size',
        '-of', 'json', str(path)]))
    video = next(s for s in data['streams'] if s['codec_type'] == 'video')
    assert video['width'] == video['height'] == 720
    assert video['r_frame_rate'] == video['avg_frame_rate'] == '25/1'
    assert dec(video['duration']) == dec(row['output_duration_seconds'])
    assert dec(video['duration']) * 25 == dec(video['nb_frames'])
    return path.name, data


def main():
    manifest = json.loads(MANIFEST.read_text())
    old_timeline = json.loads((ROOT / 'videomem_timetable.json').read_text())
    original_rows = {Path(r['output']).name: r for r in manifest['rows']}
    assert len(original_rows) == len(manifest['rows']) == 70
    with ThreadPoolExecutor(max_workers=4) as pool:
        probes = dict(pool.map(probe, manifest['rows']))
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / 'provenance/current_clip_probes.json', probes)
    positions = defaultdict(Decimal)
    segments = []
    for old in sorted(old_timeline['segments'], key=lambda r: (r['day'], r['sequence'])):
        row = original_rows[Path(old['prepared_clip']).name]
        assert row['day'] == old['day'] and row['source_id'] == old['source_id']
        assert dec(row['source_start_seconds']) == dec(old['source_interval_start_seconds'])
        assert dec(row['source_end_seconds']) == dec(old['source_interval_end_seconds'])
        probe_data = probes[Path(row['output']).name]
        video = next(s for s in probe_data['streams'] if s['codec_type'] == 'video')
        start = positions[row['day']]
        end = start + dec(video['duration'])
        s = copy.deepcopy(old)
        s.update(
            clip_key=f'day{old["day"]}_clip{old["sequence"]:02}',
            previous_prepared_clip=old['prepared_clip'],
            prepared_clip=str(Path(row['output']).relative_to(MEDIA)),
            clip_filename=Path(row['output']).name,
            bytes=row['bytes'], video_duration_seconds=float(dec(video['duration'])),
            container_duration_seconds=float(dec(probe_data['format']['duration'])),
            concat_start_seconds=float(start), concat_end_seconds=float(end),
            concat_start=stamp(start), concat_end=stamp(end),
            concat_start_timecode_25fps=timecode(start), concat_end_timecode_25fps=timecode(end),
            duration_timecode_25fps=timecode(end-start),
            provisional_concat_start_seconds=float(start), provisional_concat_end_seconds=float(end),
            provisional_concat_start=stamp(start), provisional_concat_end=stamp(end),
            boundary_status='computed_from_current_clip_durations_render_not_verified',
            mapping_basis=BASIS, final_video_start_seconds=None, final_video_end_seconds=None,
            actual_render_alignment_verified=False, user_reported_timing_aligned=True,
            media_processing=copy.deepcopy(row.get('media_processing', {})),
        )
        positions[row['day']] = end
        segments.append(s)
    timeline = {
        'schema_version': 'daywise_current_clips_timeline_v2',
        'basis': BASIS, 'time_origin': 'each_Day_playback_starts_at_00:00:00',
        'interval_convention': '[start,end); end is the boundary after the last frame',
        'fps': 25, 'user_reported_timing_aligned': True,
        'actual_render_alignment_verified': False,
        'previous_uploaded_day_video_timings_used': False,
        'segments': segments,
        'days': [{'day': d, 'duration_seconds': float(positions[d]),
                  'duration': stamp(positions[d]), 'duration_timecode_25fps': timecode(positions[d]),
                  'clip_count': sum(s['day'] == d for s in segments)} for d in range(1, 8)],
    }
    write_json(OUT / 'timeline.json', timeline)
    fields = ['day','sequence','clip_key','clip_filename','activity','concat_start','concat_end',
              'concat_start_timecode_25fps','concat_end_timecode_25fps','video_duration_seconds',
              'source_id','source_interval_start_seconds','source_interval_end_seconds','prepared_clip']
    write_csv(OUT / 'timeline.csv', segments, fields)
    by_key = {s['clip_key']: s for s in segments}
    mapped, raw_ids = [], set()
    for line in ANNOTATIONS.open(encoding='utf-8'):
        old = json.loads(line)
        assert old['id'] not in raw_ids
        raw_ids.add(old['id'])
        s = by_key[old['clip_key']]
        assert old['source_id'] == s['source_id'] and old['day'] == s['day']
        lo, hi = dec(old['clip_start_seconds']), dec(old['clip_end_seconds'])
        assert 0 <= lo <= hi <= dec(s['video_duration_seconds'])
        start, end = dec(s['concat_start_seconds']) + lo, dec(s['concat_start_seconds']) + hi
        row = copy.deepcopy(old)
        row.update(schema_version='daywise_preserved_annotation_v2',
                   previous_prepared_clip=old['prepared_clip'], prepared_clip=s['prepared_clip'],
                   previous_alignment_status=old['alignment_status'],
                   day_start_seconds=float(start), day_end_seconds=float(end),
                   day_start=stamp(start), day_end=stamp(end),
                   provisional_day_start_seconds=float(start), provisional_day_end_seconds=float(end),
                   alignment_status='mapped_to_current_clip_concatenation', mapping_basis=BASIS,
                   final_boundaries_verified=False, actual_render_alignment_verified=False,
                   annotation_content_reverified_on_1hz_video=False,
                   original_annotation_sha256=hashlib.sha256(json.dumps(old['original_annotation'],ensure_ascii=False,sort_keys=True).encode()).hexdigest())
        assert row['text'] == old['text'] and row['original_annotation'] == old['original_annotation']
        mapped.append(row)
    assert len(mapped) == 47372
    mapped.sort(key=lambda r: (r['day'], r['day_start_seconds'], r['day_end_seconds'], r['layer'], r['id']))
    write_jsonl(OUT / 'annotations_all.jsonl', mapped)
    counts = {}
    for day in range(1, 8):
        folder = OUT / f'Day{day}'
        folder.mkdir(parents=True, exist_ok=True)
        rows = [r for r in mapped if r['day'] == day]
        day_clips = [s for s in segments if s['day'] == day]
        counts[str(day)] = {'annotations': len(rows), 'layers': dict(Counter(r['layer'] for r in rows)),
                            'clips': len(day_clips), 'duration': stamp(positions[day]),
                            'duration_seconds': float(positions[day])}
        write_jsonl(folder / 'annotations.jsonl', rows)
        write_csv(folder / 'annotations.csv', rows,
                  ['id','day_start','day_end','day_start_seconds','day_end_seconds','layer','text',
                   'clip_key','prepared_clip','source_id','annotation_id','clip_start_seconds',
                   'clip_end_seconds','clipped_at_source_clip_boundary','is_point_annotation','alignment_status'])
        write_csv(folder / 'timeline.csv', day_clips, fields)
        write_json(folder / 'timeline.json', {'day': day, 'duration': stamp(positions[day]),
                                             'basis': BASIS, 'segments': day_clips})
        lines = [f'Day{day} 주석 원문 / 재생시간 기준',
                 '현재 파일을 순서대로 1배속·무공백·무중첩 연결한 시간. 최종 편집 영상과의 대조는 미실시.',
                 '00:00:00은 Day 영상 시작이며 원본 촬영시각/실제 자정이 아님.',
                 '주석 text와 원본 정보는 수정하지 않음. 층별 주석은 서로 겹칠 수 있음.', '']
        for r in rows:
            lines += [f'[{r["day_start"]}–{r["day_end"]}] [{r["layer"]}] {r["text"]}',
                      f'  {r["clip_key"]} | {r["source_id"]} | {r["annotation_id"]}']
        (folder / 'annotations.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        srt_dir = folder / 'subtitles'
        srt_dir.mkdir(exist_ok=True)
        for layer in sorted({r['layer'] for r in rows}):
            lines = []
            selected = [r for r in rows if r['layer'] == layer and not r['is_point_annotation'] and r['day_end_seconds'] > r['day_start_seconds']]
            for i, r in enumerate(selected, 1):
                lines += [str(i), f'{r["day_start"].replace(".",",")} --> {r["day_end"].replace(".",",")}',
                          r['text'], '']
            (srt_dir / f'{layer}.srt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write_json(OUT / 'annotation_summary.json', {'instances':len(mapped),'source_text_changed':0,
                                               'source_original_annotation_changed':0,'by_day':counts})
    write_json(OUT / 'annotation_validation.json', {
        'result':'pass','current_media_files_probed':70,'all_video_tracks_720_square_25fps':True,
        'annotation_instances_checked':len(mapped),'unique_instance_ids':len(raw_ids),
        'all_clip_relative_ranges_valid':True,'original_text_and_records_preserved':True,
        'all_day_offsets_computed_using_video_stream_duration':True,
        'actual_render_alignment_verified':False,'semantic_revalidation_performed':False})
    provenance = OUT / 'provenance'
    sources = [MANIFEST, ANNOTATIONS, ROOT / 'videomem_timetable.json',
               ROOT / 'annotations_adapted_v1/summary.json',
               ROOT / 'annotations_adapted_v1/mapping_errors.json',
               CONVERSION / 'conversion_manifest.json',
               CONVERSION / 'summary.json']
    write_json(provenance / 'annotation_source_hashes.json', [{'path':str(p),'sha256':digest(p),'bytes':p.stat().st_size} for p in sources])
    shutil.copy2(ROOT / 'annotations_adapted_v1/summary.json', provenance / 'previous_annotation_summary.json')
    shutil.copy2(ROOT / 'annotations_adapted_v1/README.txt', provenance / 'previous_annotation_readme.txt')
    previous_provenance = ROOT / 'annotations_adapted_v1/provenance'
    if previous_provenance.exists():
        shutil.copytree(previous_provenance, provenance / 'original_annotation_provenance', dirs_exist_ok=True)
    print(json.dumps({'annotations':len(mapped),'clips':len(segments),'days':timeline['days']},ensure_ascii=False))


if __name__ == '__main__':
    main()
