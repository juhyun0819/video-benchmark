"""Read-only download and real-video smoke status monitor (no GPU work)."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def show(work, download):
    d = read(download/'status.json')
    phase = d.get('phase', 'not_started')
    done, total = d.get('downloaded_bytes', 0), d.get('total_bytes', 0)
    if phase == 'complete': done = total
    percent = 100 * done / total if total else 0
    elapsed = d.get('elapsed_seconds', 0)
    eta = (total-done)/(done/elapsed) if done and elapsed and phase == 'downloading' else None
    print(datetime.now().astimezone().isoformat(timespec='seconds'))
    if phase in ('downloading', 'complete'):
        print(f"Qwen2.5-VL-7B: {phase} · {done/1e9:.2f}/{total/1e9:.2f} GB · {percent:.1f}%"
              + (f' · approx {eta/60:.1f} min remaining' if eta is not None else ''))
    else:
        print('Qwen2.5-VL-7B:', phase, d.get('current_file', ''))
    print('Launcher:', read(work/'launcher_status.json').get('phase', 'not_started'))
    for gpu, label in [(4, 'fluxmem'), (5, 'uncompressed_control')]:
        s = read(work/label/'status.json')
        print(f"GPU{gpu} {label}: {s.get('phase', 'waiting')} · {s.get('completed', 0)}/{s.get('total', 1)}")
        for row in read(work/label/'predictions.json') or []:
            print(' ', row['status'], (row.get('prediction') or row.get('error') or '')[:240])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=ROOT/'local_runs/smoke_v1')
    parser.add_argument('--download', type=Path, default=Path('/home/kai/models/.downloads/qwen25vl_7b_fluxmem'))
    parser.add_argument('--interval', type=float, default=5)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    try:
        while True:
            show(args.work, args.download)
            if args.once: break
            print(flush=True)
            time.sleep(max(1, args.interval))
    except KeyboardInterrupt:
        pass
