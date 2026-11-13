"""Create common QA and a verified Day1 video index without modifying raw data."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess

from .inputs import canonical_questions, global_time, sha256, recent_samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root, out = args.root, args.output
    out.mkdir(parents=True, exist_ok=True)
    raw = root / 'EgoLifeQA/EgoLifeQA_A1_JAKE.json'
    qa_receipt = json.loads((root / '_download/qa_manifest.json').read_text())
    assert sha256(raw) == qa_receipt['sha256']
    questions = canonical_questions(json.loads(raw.read_text()))
    assert len(questions) == 500
    scorer = Path(__file__).parents[1] / 'streambench/scoring.py'
    manifest = {'schema': 'jake_canonical_qa_v1', 'participant': 'A1_JAKE',
        'source_revision': qa_receipt['revision'], 'raw_qa_sha256': sha256(raw),
        'timecode': 'HHMMSSFF; FF/20 seconds; global=(day-1)*86400+clock',
        'cutoff': 'actual frame timestamp <= query_time',
        'scorer': 'baselines.streambench.scoring.score_answer',
        'scorer_sha256': sha256(scorer),
        'time_review_qa_ids': [q['qa_id'] for q in questions if q['query_time'] is None],
        'questions': questions}
    inventory = json.loads((root / '_download/tools/remote_inventory.json').read_text())
    rows = [r for r in inventory['files'] if r['participant'] == 'A1_JAKE' and r['day'] == 'DAY1']
    assert len(rows) == 828

    def probe(row):
        path = root / row['path']
        receipt = json.loads((root / '_download/receipts' /
            (row['path'].replace('/', '__') + '.ready.json')).read_text())
        s = path.stat()
        assert s.st_size == row['size'] and receipt['sha256'] == row['lfs_sha256']
        assert receipt['file_state'] == [s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]
        data = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=avg_frame_rate,duration,nb_frames,width,height', '-of', 'json', str(path)]))['streams'][0]
        assert Fraction(data['avg_frame_rate']) == 20
        return {'path': row['path'], 'day': 1,
            'start_time': global_time('DAY1', path.stem.split('_')[-1]),
            'duration': float(data['duration']), 'frames': int(data['nb_frames']),
            'width': data['width'], 'height': data['height'], 'fps': 20,
            'bytes': row['size'], 'sha256': row['lfs_sha256'], 'file_state': receipt['file_state']}

    with ThreadPoolExecutor(max_workers=8) as pool:
        sources = sorted(pool.map(probe, rows), key=lambda s: s['start_time'])
    # Fractional opening timestamp is corroborated by the actual media boundary.
    assert abs(sources[0]['start_time'] + sources[0]['duration'] - sources[1]['start_time']) < 1e-6
    day1 = [q for q in questions if q['day'] == 1]
    for q in day1:
        points = recent_samples(sources, q['query_time'])
        assert len(points) == 4 and all(p['requested_global_timestamp'] <= q['query_time'] for p in points)
    (out / 'jake_qa_v1.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    (out / 'day1_sources_v1.json').write_text(json.dumps({'sources': sources}, indent=2) + '\n')
    print(json.dumps({'qa_total': len(questions), 'day1_qa': len(day1), 'day1_videos': len(sources),
        'qa_sha256': sha256(out / 'jake_qa_v1.json'), 'day1_sources_sha256': sha256(out / 'day1_sources_v1.json'),
        'sampling': 'source-local 1Hz; most recent four; actual PTS checked at decode'}, indent=2))


if __name__ == '__main__':
    main()
