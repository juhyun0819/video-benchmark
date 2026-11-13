"""Create readable indexes and portable archives after annotation/QA validation."""
from pathlib import Path
import csv
import hashlib
import json
import os
import zipfile

ROOT = Path(os.environ.get('VIDEOMEM_PACKAGE_OUTPUT', 'work/daywise_package')).resolve()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def jsonl(path):
    return [json.loads(line) for line in path.open(encoding='utf-8') if line.strip()]


def compact(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def diagnostic_text(path, title):
    rows = jsonl(path)
    lines = [title, '원문 질문·정답을 보존한 검토/제외 목록이며 시간 매핑 완료 목록과 구분함.', '']
    for r in rows:
        lines += [r['qa_id'], '상태: ' + r['status'], 'Q: ' + compact(r['question'])]
        for i, choice in enumerate(r['choices']):
            lines.append(f'{i}: {compact(choice)}')
        lines += [f'원문 정답 인덱스(0부터): {r["correct_idx"]}', 'A: ' + compact(r['correct_answer']),
                  '확인 사항: ' + compact(r.get('issues', r.get('missing_source_video_ids', []))), '']
    path.with_suffix('.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def archive(path, files, base):
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for file in sorted(files):
            z.write(file, arcname=str(file.relative_to(base)))
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None, path
    return {'file': path.name, 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    a = json.loads((ROOT / 'annotation_summary.json').read_text())
    q = json.loads((ROOT / 'qa_summary.json').read_text())
    t = json.loads((ROOT / 'timeline.json').read_text())
    rows = []
    for d in range(1, 8):
        ann, qa = a['by_day'][str(d)], q['questions_by_day'][str(d)]
        rows.append({'day': f'Day{d}', 'duration': ann['duration'], 'clips': ann['clips'],
                     'annotations': ann['annotations'], 'original_qa_mapped': qa['mapped'],
                     'qa_review_needed': qa['review_needed'],
                     'qa_missing_inputs_occurrences': qa['excluded_missing_inputs']})
        folder = ROOT / f'Day{d}'
        diagnostic_text(folder / 'qa_review_needed.jsonl', f'Day{d}: 추가 확인이 필요한 원문 Q&A')
        diagnostic_text(folder / 'qa_excluded_missing_sources.jsonl', f'Day{d}: 필요한 원본 입력이 없어 제외한 Q&A')
        lines = [
            f'Day{d} 주석 + 원문 Q&A',
            f'총 재생 시간: {ann["duration"]} / 영상 {ann["clips"]}개',
            f'주석 {ann["annotations"]:,}개 / 시간 매핑 Q&A {qa["mapped"]:,}문항',
            f'추가 검토 Q&A {qa["review_needed"]}문항 / 입력 누락 Q&A {qa["excluded_missing_inputs"]}문항(여러 Day에 중복 참조 가능)',
            '',
            '바로 읽기: annotations.txt(주석), qa.txt(질문·선택지·정답·시간)',
            '엑셀: annotations.csv, qa.csv, timeline.csv (UTF-8 BOM)',
            '프로그램: annotations.jsonl(원문 포함), qa.json(원문 및 Day 시간 변환본)',
            '자막: subtitles/*.srt (L1~L5, HD_action/activity/recipe_step/recipe_prep 각각 분리)',
            '검토/제외: qa_review_needed.jsonl/.txt, qa_excluded_missing_sources.jsonl/.txt',
            '',
            '시간 기준',
            '- 현재 Day 폴더 파일을 timeline.csv 순서대로 1배속·무공백·무중첩 연결.',
            '- 파일명에 반올림된 시간이 아닌 실제 25fps 영상 트랙 길이로 누적.',
            '- Day 00:00:00은 재생 시작. 원본 촬영 날짜/시각과 다름.',
            '- 사용자가 시간 정렬을 완료했다고 알렸지만 .kdenlive/최종 렌더와 직접 대조하지 못함.',
            '- 예전 업로드 Day 영상의 짧아진 길이나 Day6 오차를 비례 보정해 적용하지 않음.',
            '- 실제 편집에서 컷·배속·공백·전환 겹침이 있으면 timeline.csv와 대조 후 재매핑 필요.',
            '',
            '주석/정답 보존',
            '- 주석 text와 원문 original_annotation 유지. 일부 상위 주석은 클립 밖 맥락을 포함할 수 있음.',
            '- 주석 층끼리 시간 구간이 겹칠 수 있으며 건수는 서로 다른 행동의 수가 아님.',
            '- 길이 0인 시점 주석은 JSONL/CSV/TXT에 보존하고 SRT에서는 제외.',
            '- Q&A는 공식 HD-EPIC 요리 영상 원문만 사용. 새로운 질문/정답/번역을 만들지 않음.',
            '- NextMe 출퇴근·업무·헬스는 주석이 있으나 확인한 릴리스에 대응하는 기존 문장형 Q&A는 없음.',
            '- 원문 question/choices/correct_idx 그대로. day_time_view에서만 TIME 태그를 Day 시각으로 이동.',
            '- 정답 인덱스는 0부터 시작. 중첩 선택지 목록도 원문 구조 그대로 보존.',
            '- video 1/video 2는 mapped_inputs의 해당 구간만 의미하며 Day 전체로 대체하지 말 것.',
            '- 1초당 1이미지로 변환한 요리 영상에서 짧은 동작/시선/순간 프레임의 정답 유효성은 재검증하지 않음.',
            '- 시간 매핑/파일 구조 검사 완료와 시각적 정답 검증 완료는 다름.',
            '- 영상 파일 자체는 이 압축파일에 포함하지 않음.',
            '',
        ]
        (folder / 'README.txt').write_text('\n'.join(lines), encoding='utf-8')
    summary = {
        'schema_version': 'daywise_annotations_qa_bundle_v2',
        'days': rows, 'annotation_instances': a['instances'],
        'original_qa_mapped_unique': q['mapped_questions'],
        'original_qa_review_needed_unique': q['needs_review'],
        'original_qa_missing_inputs_unique': q['excluded_missing_source_questions'],
        'new_questions_generated': 0, 'original_text_and_answers_preserved': True,
        'source_media_changed': False, 'old_annotation_packages_overwritten': False,
        'timeline_basis': t['basis'], 'actual_render_alignment_verified': False,
        'answers_reverified_on_one_hz_video': False,
    }
    write_json(ROOT / 'summary.json', summary)
    with (ROOT / 'day_summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        'Day1~Day7 주석 및 공식 원문 Q&A 묶음',
        '',
        f'주석: {a["instances"]:,}개 / 시간 매핑 Q&A: {q["mapped_questions"]:,}문항',
        f'추가 확인: {q["needs_review"]}문항 / 필요한 원본 입력 누락: {q["excluded_missing_source_questions"]}개 고유 문항',
        '새 질문 생성 0개. 원문과 정답 유지.',
        '',
        '각 Day 폴더에서 annotations.txt와 qa.txt를 열면 바로 읽을 수 있음.',
        'CSV는 엑셀용, JSON/JSONL은 원문·출처를 포함한 프로그램용. subtitles/는 층별 SRT.',
        '',
        'Day | 총 길이 | 주석 | 시간 매핑 Q&A | 추가 확인 Q&A',
    ]
    for row in rows:
        lines.append(f'{row["day"]} | {row["duration"]} | {row["annotations"]:,} | {row["original_qa_mapped"]:,} | {row["qa_review_needed"]}')
    lines += [
        '',
        '이번 시간 기준은 현재 70개 파일의 영상 트랙 길이를 순서대로 합친 새 연결본이다.',
        '사용자의 시간 정렬 완료 보고를 기록하되, 최종 .kdenlive/렌더 영상과의 직접 대조는 미실시.',
        '원본 순서·1배속·공백/추가컷/겹침 없음이 전제이며, 예전 업로드 영상 길이에 맞춘 보정은 하지 않음.',
        'timeline.csv에서 클립별 시작/끝을 확인할 수 있음. HH:MM:SS.mmm는 밀리초, 별도 timecode 열 HH:MM:SS:FF는 25fps 프레임.',
        '',
        'Q&A 범위',
        '- 현재 요리 영상에 대응하는 공식 HD-EPIC 원문 질문/선택지/정답. 언어/의미/정답 인덱스 유지.',
        '- NextMe 출퇴근·업무·식사·헬스는 L1~L5 주석 제공. 확인한 공식 릴리스에 대응 문장형 Q&A가 없어 새로 생성하지 않음.',
        '- QA missing-input 목록은 같은 문항이 여러 Day에 참조되므로 Day별 수의 합과 고유 67문항은 다름.',
        '- 좌표 미확인 102문항과 시간태그 범위 문제 1문항은 qa_review_needed로 분리.',
        '- 1Hz 요리 영상에서 짧은 동작·시선·순간 이미지·좌표의 답을 다시 시각 판정하지 않았음.',
        '- 구조/원문 보존/시각 계산 검증은 완료하더라도 최종 영상용 검증된 정답셋이라는 뜻은 아님.',
        '',
        '전체 파일',
        '- annotations_all.jsonl: 전 Day 주석 원문과 매핑.',
        '- qa_all.jsonl: 시간 매핑된 고유 Q&A.',
        '- qa_review_needed_all.jsonl / qa_excluded_missing_sources_all.jsonl: 추가 확인/입력 누락.',
        '- day_summary.csv / summary.json: Day별 집계.',
        '- timeline.json/csv: 현재 클립 기준 연결 시간표.',
        '- annotation_validation.json / qa_validation.json / independent_review.json: 검사 결과 및 범위.',
        '- provenance/: 원문 파일 해시와 이전 자료의 이력. previous_* 파일은 이번 사용 지침이 아닌 보존 이력.',
        '- scripts/: 재현용 코드. 원본 서버 자료와 영상이 있는 환경에서 실행 가능.',
        '',
        '현재 영상과 기존 주석/Q&A 자료를 변경하지 않았음. 영상 파일은 압축파일에 포함하지 않음.',
        '원본 데이터 출처와 이용 조건은 그대로 적용됨.',
        '',
    ]
    (ROOT / 'README.txt').write_text('\n'.join(lines), encoding='utf-8')
    all_files = [p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name != 'checksums.sha256']
    checks = [f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT)}' for p in sorted(all_files)]
    (ROOT / 'checksums.sha256').write_text('\n'.join(checks)+'\n', encoding='utf-8')
    archives = []
    for d in range(1, 8):
        folder = ROOT / f'Day{d}'
        archives.append(archive(ROOT.parent / f'Day{d}_annotations_qa_20261108.zip',
                                [p for p in folder.rglob('*') if p.is_file()], ROOT))
    archives.append(archive(ROOT.parent / 'Day1-7_annotations_qa_20261108.zip',
                            [p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts], ROOT.parent))
    write_json(ROOT.parent / 'daywise_archives_20261108.json', archives)
    print(json.dumps({'summary':summary,'archives':archives},ensure_ascii=False))


if __name__ == '__main__':
    main()
