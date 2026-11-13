"""Package unchanged official HD-EPIC Q&A for the current seven-day clip timeline.

No network access, model calls, new questions, or new answers. TIME-tag shifts
are a separate view; original records remain byte-equivalent after canonical
JSON serialization. The final edited video has not been frame-verified.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
import copy
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess

OUT = Path(os.environ.get('VIDEOMEM_PACKAGE_OUTPUT', 'work/daywise_package')).resolve()
ROOT = Path(os.environ.get('VIDEOMEM_WORK_ROOT', 'work/daywise_inputs')).resolve()
RAW = ROOT / 'qa_work/raw_hd_epic'
REVISION = '9a47a428a90d83a2fd5d87356afe97bef5649989'
TIME_TAG = re.compile(r'<TIME\s+(\d+:\d+:\d+(?:\.\d+)?)\s+(video\s+\d+)>')
SOURCE_ROOT = Path(os.environ.get('HD_EPIC_ORIGINAL_ROOT', 'work/hd_epic/original_mp4')).resolve()


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as file:
        for row in rows:
            file.write(dumps(row) + '\n')


def seconds(value):
    parts = str(value).split(':')
    if len(parts) != 3:
        raise ValueError('Expected HH:MM:SS time')
    h, m, s = map(Decimal, parts)
    if not all(x.is_finite() for x in (h, m, s)) or h < 0 or not 0 <= m < 60 or not 0 <= s < 60:
        raise ValueError('Invalid time')
    return h * 3600 + m * 60 + s


def stamp(value):
    milliseconds = round(Decimal(str(value)) * 1000)
    h, remainder = divmod(milliseconds, 3600000)
    m, remainder = divmod(remainder, 60000)
    s, ms = divmod(remainder, 1000)
    return f'{h:02d}:{m:02d}:{s:02d}.{ms:03d}'


def walk_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from walk_strings(item)


def transform_strings(value, function):
    if isinstance(value, str):
        return function(value)
    if isinstance(value, list):
        return [transform_strings(item, function) for item in value]
    raise ValueError('Unsupported question or answer type')


def probe_original(source_id):
    path = SOURCE_ROOT / f'{source_id}.mp4'
    data = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=duration,width,height,avg_frame_rate,nb_frames', '-of', 'json', str(path)
    ], text=True))
    return source_id, {'source_path': str(path), 'probe': data['streams'][0]}


def write_csv(path, rows):
    fields = ['qa_id', 'days', 'source_file', 'question_original', 'choices_original',
              'correct_idx', 'correct_answer_original', 'question_day_time',
              'choices_day_time', 'correct_answer_day_time', 'mapped_inputs', 'status']
    with path.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                'qa_id': row['qa_id'], 'days': dumps(row['days']), 'source_file': row['source_file'],
                'question_original': row['question'], 'choices_original': dumps(row['choices']),
                'correct_idx': row['correct_idx'], 'correct_answer_original': dumps(row['correct_answer']),
                'question_day_time': row['day_time_view']['question'],
                'choices_day_time': dumps(row['day_time_view']['choices']),
                'correct_answer_day_time': dumps(row['day_time_view']['correct_answer']),
                'mapped_inputs': dumps(row['mapped_inputs']), 'status': row['status'],
            })


def main():
    timeline_path = OUT / 'timeline.json'
    timeline_bytes = timeline_path.read_bytes()
    timetable = json.loads(timeline_bytes)
    segments = {}
    for segment in timetable['segments']:
        if segment['source_id'].startswith('HD_EPIC/'):
            source_id = segment['source_id'].split('/', 1)[1]
            assert source_id not in segments, 'Source used more than once; ambiguous mapping'
            segments[source_id] = segment
    assert len(segments) == 9
    assert set(s['day'] for s in segments.values()) == set(range(1, 8))
    with ThreadPoolExecutor(max_workers=4) as executor:
        original_video_metadata = dict(executor.map(probe_original, sorted(segments)))
    tree_path = ROOT / 'qa_work/hd_epic_full_tree.json'
    tree = json.loads(tree_path.read_bytes())
    files = [r for r in tree['tree'] if r['path'].startswith('vqa-benchmark/') and r['path'].endswith('.json')]
    assert len(files) == 30
    accepted, review, excluded = [], [], []
    total = unrelated = 0
    source_files = []
    unique_ids = set()
    originals = {}
    for entry in sorted(files, key=lambda row: row['path']):
        path = RAW / Path(entry['path']).name
        data = path.read_bytes()
        assert len(data) == entry['size']
        assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == entry['sha']
        source_files.append({'file': entry['path'], 'git_blob_sha1': entry['sha'],
                             'sha256': sha(data), 'bytes': len(data)})
        questions = json.loads(data)
        for qa_id, original in questions.items():
            assert qa_id not in unique_ids
            unique_ids.add(qa_id)
            total += 1
            inputs = original['inputs']
            ids = {item['id'] for item in inputs.values()}
            if not ids.intersection(segments):
                unrelated += 1
                continue
            originals[qa_id] = original
            row = {
                'qa_id': qa_id, 'dataset': 'HD-EPIC', 'source_revision': REVISION,
                'source_file': entry['path'],
                'source_url': f'https://github.com/hd-epic/hd-epic-annotations/blob/{REVISION}/{entry["path"]}',
                'question': copy.deepcopy(original['question']), 'choices': copy.deepcopy(original['choices']),
                'correct_idx': original['correct_idx'],
                'correct_answer': copy.deepcopy(original['choices'][original['correct_idx']]),
                'original_qa': copy.deepcopy(original), 'original_qa_sha256': sha(dumps(original).encode()),
                'days': sorted({segments[source]['day'] for source in ids.intersection(segments)}),
                'timeline_basis': 'current_prepared_clips_name_order_1x_no_gaps',
                'edit_alignment_status': 'user_reported_reassembled',
                'actual_render_alignment_verified': False,
                'answer_reverified_on_rendered_video': False,
                'one_hz_visual_evidence_reverified': False,
            }
            missing = sorted(ids - segments.keys())
            if missing:
                row.update(status='excluded_missing_original_inputs', missing_source_video_ids=missing)
                excluded.append(row)
                continue
            issues, mapped_inputs, tags = [], {}, []
            assert isinstance(original['question'], str)
            assert len(original['choices']) == 5 and type(original['correct_idx']) is int and 0 <= original['correct_idx'] < 5
            for alias, item in inputs.items():
                segment = segments[item['id']]
                source_start = Decimal(str(segment['source_interval_start_seconds']))
                source_end = Decimal(str(segment['source_interval_end_seconds']))
                day_offset = Decimal(str(segment['concat_start_seconds'])) - source_start
                image = alias.startswith('image ')
                if image and set(item) == {'id', 'time'}:
                    start = end = seconds(item['time'])
                    full = False
                elif alias.startswith('video ') and set(item) == {'id'}:
                    start, end, full = source_start, source_end, True
                elif alias.startswith('video ') and set(item) == {'id', 'start_time', 'end_time'}:
                    start, end, full = seconds(item['start_time']), seconds(item['end_time']), False
                else:
                    issues.append({'code': 'UNSUPPORTED_INPUT_SCHEMA', 'alias': alias, 'fields': sorted(item)})
                    continue
                if end < start or start < source_start or end > source_end or (image and start >= source_end):
                    issues.append({'code': 'INPUT_OUTSIDE_SELECTED_SOURCE_INTERVAL', 'alias': alias,
                                   'source_start_seconds': float(start), 'source_end_seconds': float(end),
                                   'selected_source_start_seconds': float(source_start),
                                   'selected_source_end_seconds': float(source_end)})
                probe = original_video_metadata[item['id']]['probe']
                original_duration = Decimal(probe['duration'])
                mapped_inputs[alias] = {
                    'original_input': copy.deepcopy(item), 'original_source_video_id': item['id'],
                    'input_type': 'image_frame' if image else 'video_clip',
                    'whole_original_video_requested': full,
                    'source_start_seconds': float(start), 'source_end_seconds': float(end),
                    'source_video_zero_in_day_seconds': float(day_offset),
                    'day': segment['day'], 'day_video_logical_name': f'Day{segment["day"]}',
                    'day_start_seconds': float(day_offset + start), 'day_end_seconds': float(day_offset + end),
                    'day_start': stamp(day_offset + start), 'day_end': stamp(day_offset + end),
                    'prepared_clip': segment['prepared_clip'],
                    'original_video_duration_seconds': float(original_duration),
                    'prepared_minus_original_duration_ms': float((source_end - source_start - original_duration) * 1000),
                    'visual_evidence_warning': 'Current cooking clips contain one sampled image per second held at 25fps. Original 30fps frame equivalence and answer validity have not been reverified.',
                    'time_alignment_status': 'user_reported_reassembled_current_concat_unverified_render',
                }
                if full:
                    mapped_inputs[alias]['whole_video_scope_note'] = (
                        'The original input requests the complete original video. The mapped span covers the prepared selected clip; '
                        'its slight frame-grid duration difference and 1Hz sampling are preserved as caveats, not judged answer-equivalent.'
                    )
            def replace_tags(text):
                matches = list(TIME_TAG.finditer(text))
                if text.count('<TIME') != len(matches):
                    issues.append({'code': 'UNPARSED_TIME_TAG', 'text': text})
                def replace(match):
                    value, alias = match.groups()
                    if alias not in mapped_inputs:
                        issues.append({'code': 'UNKNOWN_TIME_TAG_INPUT', 'alias': alias})
                        return match.group(0)
                    mapped = mapped_inputs[alias]
                    segment = segments[mapped['original_source_video_id']]
                    time = seconds(value)
                    lo = Decimal(str(segment['source_interval_start_seconds']))
                    hi = Decimal(str(segment['source_interval_end_seconds']))
                    if not lo <= time <= hi:
                        issues.append({'code': 'TIME_TAG_OUTSIDE_SELECTED_SOURCE_INTERVAL', 'alias': alias,
                                       'source_seconds': float(time), 'selected_source_end_seconds': float(hi)})
                    shifted = Decimal(str(mapped['source_video_zero_in_day_seconds'])) + time
                    tags.append({'input_alias': alias, 'source_tag': match.group(0),
                                 'source_seconds': float(time), 'day': mapped['day'],
                                 'day_seconds': float(shifted), 'day_time': stamp(shifted)})
                    return f'<TIME {stamp(shifted)} {alias}>'
                return TIME_TAG.sub(replace, text)
            question_day = transform_strings(original['question'], replace_tags)
            choices_day = transform_strings(original['choices'], replace_tags)
            if any('<BBOX' in text for text in walk_strings([original['question'], original['choices']])):
                issues.append({'code': 'BBOX_COORDINATE_MAPPING_UNVERIFIED',
                               'reason': 'Official pixel coordinates preserved; resizing/crop/layout and 1Hz frame evidence not reverified.'})
            row.update(mapped_inputs=mapped_inputs, mapped_time_references=tags,
                       day_time_view={'question': question_day, 'choices': choices_day,
                                      'correct_idx': original['correct_idx'],
                                      'correct_answer': choices_day[original['correct_idx']]},
                       issues=issues,
                       input_scope_policy='Use only each mapped input alias clip/frame; showing the entire Day changes the original task.')
            if issues:
                row['status'] = 'needs_input_or_coordinate_review'
                review.append(row)
            else:
                row['status'] = 'mapped_original_qa_unverified_visual_answers'
                accepted.append(row)

    write_jsonl(OUT / 'qa_all.jsonl', accepted)
    write_jsonl(OUT / 'qa_review_needed_all.jsonl', review)
    write_jsonl(OUT / 'qa_excluded_missing_sources_all.jsonl', excluded)
    day_counts = {}
    for day in range(1, 8):
        selected = [row for row in accepted if day in row['days']]
        selected_review = [row for row in review if day in row['days']]
        selected_excluded = [row for row in excluded if day in row['days']]
        day_counts[str(day)] = {'mapped': len(selected), 'review_needed': len(selected_review),
                                'excluded_missing_inputs': len(selected_excluded)}
        folder = OUT / f'Day{day}'
        write_json(folder / 'qa.json', selected)
        write_csv(folder / 'qa.csv', selected)
        write_jsonl(folder / 'qa_review_needed.jsonl', selected_review)
        write_jsonl(folder / 'qa_excluded_missing_sources.jsonl', selected_excluded)
        lines = [f'Day{day} 원문 HD-EPIC Q&A — 현재 클립 순서의 Day 시각',
                 '새 문항·새 정답 생성 없이 공식 question/choices/correct_idx 원문 보존. 아래 TIME 태그만 Day 시각으로 표시.',
                 '현재 파일명 순서·1배속·무공백 연결 기준. 사용자 재편집 완료 보고 기준이며 최종 렌더 프레임 대조는 미실시.',
                 '요리 영상은 1초당 한 이미지로 변환되어 빠른 동작·정확한 프레임 증거·정답 타당성을 다시 검증하지 않음.',
                 'video 1/video 2는 지정 구간만 의미하며 Day 전체를 입력으로 주면 원문 과업이 달라짐.',
                 '여러 Day 입력을 요구하는 문항은 각 관련 Day 파일에 함께 보존. 전체 집계는 qa_id 기준.',
                 'NextMe 업무·출퇴근·헬스 구간의 원문 Q&A는 확인한 원본 릴리스에 없어 새로 만들지 않음.', '']
        for row in selected:
            lines.append(row['qa_id'])
            for alias, inp in row['mapped_inputs'].items():
                lines.append(f'{alias}: Day{inp["day"]} {inp["day_start"]}–{inp["day_end"]} ({inp["input_type"]})')
            lines.append('Q: ' + row['day_time_view']['question'])
            for index, choice in enumerate(row['day_time_view']['choices']):
                lines.append(f'{index}: ' + (choice if isinstance(choice, str) else dumps(choice)))
            answer = row['day_time_view']['correct_answer']
            lines.extend([f'정답 인덱스(0부터): {row["correct_idx"]}',
                          'A: ' + (answer if isinstance(answer, str) else dumps(answer)), ''])
        (folder / 'qa.txt').write_text('\n'.join(lines) + '\n')

    all_rows = accepted + review + excluded
    original_preserved = all(row['original_qa'] == originals[row['qa_id']]
        and row['question'] == originals[row['qa_id']]['question']
        and row['choices'] == originals[row['qa_id']]['choices']
        and row['correct_idx'] == originals[row['qa_id']]['correct_idx']
        and row['original_qa_sha256'] == sha(dumps(originals[row['qa_id']]).encode()) for row in all_rows)
    assert original_preserved
    assert total == unrelated + len(all_rows)
    assert len({row['qa_id'] for row in all_rows}) == len(all_rows)
    for row in accepted + review:
        scrub = lambda text: TIME_TAG.sub(lambda match: '<TIME ' + match.group(2) + '>', text)
        assert transform_strings(row['question'], scrub) == transform_strings(row['day_time_view']['question'], scrub)
        assert transform_strings(row['choices'], scrub) == transform_strings(row['day_time_view']['choices'], scrub)
        assert row['day_time_view']['correct_idx'] == row['correct_idx']
        assert row['day_time_view']['correct_answer'] == row['day_time_view']['choices'][row['correct_idx']]
    mappings_valid = True
    for row in accepted:
        for inp in row['mapped_inputs'].values():
            segment = segments[inp['original_source_video_id']]
            assert inp['day_start_seconds'] >= segment['concat_start_seconds'] - 1e-8
            assert inp['day_end_seconds'] <= segment['concat_end_seconds'] + 1e-8
        for ref in row['mapped_time_references']:
            inp = row['mapped_inputs'][ref['input_alias']]
            assert abs(ref['day_seconds'] - (inp['source_video_zero_in_day_seconds'] + ref['source_seconds'])) < 1e-7
    validation = {
        'schema_version': 'daywise_original_qa_validation_v2',
        'official_source_git_blob_hashes_checked': len(source_files),
        'original_records_and_answers_preserved': original_preserved,
        'unique_qa_ids_across_status_partitions': True,
        'scan_partition_counts_reconcile': True,
        'accepted_input_ranges_inside_current_clips': mappings_valid,
        'time_tag_offset_arithmetic_checked': True,
        'day_time_view_changes_only_time_tags': True,
        'day_time_view_choice_order_and_answer_index_preserved': True,
        'new_questions_generated': 0, 'new_answers_generated': 0,
        'actual_render_alignment_verified': False,
        'visual_answer_correctness_revalidated': False,
        'one_hz_visual_evidence_reverified': False,
        'validation_scope': 'Structural provenance, lossless original QA preservation, and timestamp arithmetic only.',
    }
    summary = {
        'schema_version': 'official_hd_epic_qa_day1_7_current_concat_v2',
        'new_questions_generated': 0, 'source_revision': REVISION,
        'timeline_sha256': sha(timeline_bytes),
        'timeline_basis': 'current_prepared_clips_name_order_1x_no_gaps',
        'edit_alignment_status': 'user_reported_reassembled',
        'source_video_count': len(segments), 'official_source_files': len(source_files),
        'official_questions_scanned': total,
        'unrelated_to_selected_day1_7_sources': unrelated,
        'questions_intersecting_selected_sources': len(all_rows),
        'all_original_input_sources_available': len(accepted) + len(review),
        'mapped_questions': len(accepted), 'needs_review': len(review),
        'excluded_missing_source_questions': len(excluded),
        'review_issue_counts': dict(Counter(issue['code'] for row in review for issue in row['issues'])),
        'questions_by_day': day_counts,
        'questions_spanning_multiple_days': sum(len(row['days']) > 1 for row in accepted),
        'sum_day_mapped_occurrences': sum(c['mapped'] for c in day_counts.values()),
        'categories': dict(Counter(Path(row['source_file']).stem for row in accepted)),
        'nextme_status': 'No prewritten original NextMe QA in the inspected pinned release; NextAct prediction records are not relabeled as QA.',
        'original_answers_preserved': True,
        'actual_render_alignment_verified': False,
        'visual_answer_correctness_revalidated': False,
        'one_hz_visual_evidence_reverified': False,
    }
    write_json(OUT / 'qa_summary.json', summary)
    write_json(OUT / 'qa_validation.json', validation)
    write_json(OUT / 'provenance/qa_sources.json', {'revision': REVISION, 'files': source_files,
        'source_tree_sha256': sha(tree_path.read_bytes()), 'timeline_sha256': sha(timeline_bytes),
        'build_script_sha256': sha(Path(__file__).read_bytes())})
    write_json(OUT / 'provenance/qa_original_video_probes.json', original_video_metadata)
    nextme_source = ROOT / 'qa_work/nextme_official_metadata/nextact_inventory.json'
    if nextme_source.exists():
        shutil.copyfile(nextme_source, OUT / 'provenance/qa_nextme_nextact_not_qa_inventory.json')
    print(dumps(summary))


if __name__ == '__main__':
    main()
