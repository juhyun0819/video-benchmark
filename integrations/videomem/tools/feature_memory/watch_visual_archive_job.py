#!/usr/bin/env python3
"""Observe an existing archive producer after stopping the memory consumer.

This watcher never starts, signals, or restarts any GPU/CPU worker. The original
producer continues with its existing PID and open log files.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import time


def read_json(path):
    return json.loads(Path(path).read_text())


def process_identity(pid):
    try:
        fields = Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()
        return {'state': fields[0], 'start_ticks': int(fields[19])}
    except FileNotFoundError:
        return None


def publish(path, value):
    temporary = path.with_name(path.name + '.watcher.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-status', type=Path, required=True)
    args = parser.parse_args()
    with args.job_status.with_suffix('.watcher.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        job = read_json(args.job_status)
        if job.get('memory_construction_state') != 'stopped_by_user':
            raise ValueError('Explicit memory stop receipt is required')
        archive = Path(job['archive'])
        producer_pid = int(job['producer_pid'])
        expected_start = int(job['producer_start_ticks'])
        while True:
            # Preserve independently added probe information on every refresh.
            job = read_json(args.job_status)
            job.update(launcher_pid=os.getpid(), phase='visual_archive_only',
                       updated_at_utc=datetime.now(timezone.utc).isoformat())
            status = read_json(archive / 'status.json')
            current = process_identity(producer_pid)
            alive = (current is not None and current['state'] != 'Z'
                     and current['start_ticks'] == expected_start)
            if (archive / 'complete.json').exists() and status.get('status') == 'completed':
                job.update(state='completed', phase='visual_archive_complete',
                           archive_completion_receipt=str(archive / 'complete.json'))
            elif status.get('status') == 'failed' or not alive:
                job.update(state='failed', phase='visual_archive_failed',
                           error='Producer failed or exited without its completion receipt')
            else:
                job['state'] = 'running'
            publish(args.job_status, job)
            if job['state'] != 'running':
                print(json.dumps(job), flush=True)
                return 0 if job['state'] == 'completed' else 1
            time.sleep(15)


if __name__ == '__main__':
    raise SystemExit(main())
