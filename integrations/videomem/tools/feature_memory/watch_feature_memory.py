#!/usr/bin/env python3
"""Read-only monitor for visual archives, memory construction and KV probes."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time


PROJECT = Path(__file__).resolve().parents[2].parent
DEFAULT_JOB = PROJECT / 'evaluation' / 'feature_memory_byte_job.json'
STAGED_JOB = DEFAULT_JOB.with_name('feature_memory_staged_job.json')
MAX_JSON_BYTES = 4 * 1024**2
MAX_TAIL_BYTES = 2 * 1024**2


def read_json(path):
    # Status/plan files have bounded schemas. Never load an accidentally selected
    # growing index or feature shard into the terminal process.
    with Path(path).open('rb') as handle:
        raw = handle.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError(f'JSON status exceeds {MAX_JSON_BYTES} bytes: {path}')
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError(f'Expected a JSON object: {path}')
    return result


def optional_json(path):
    try:
        return read_json(path)
    except FileNotFoundError:
        return {}


def recent_progress(path, limit=11):
    # A growing log is never reread in full. Ignore a partial first/last line.
    with path.open('rb') as handle:
        handle.seek(0, 2)
        size = handle.tell()
        offset = max(0, size - MAX_TAIL_BYTES)
        handle.seek(offset)
        raw = handle.read(MAX_TAIL_BYTES)
    lines = raw.splitlines()
    if offset:
        lines = lines[1:]
    rows = []
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError):
            continue
        if isinstance(row, dict) and 'frames_processed' in row and 'elapsed_seconds' in row:
            rows.append(row)
        if len(rows) == limit:
            break
    return list(reversed(rows))


def gpu_lines():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
        details = result.stdout.strip() or result.stderr.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        details = f'GPU status unavailable: {exc}'
    return ['', 'GPU index, utilization %, memory MiB', details]


def state_lines(rows):
    lines = ['Method / budget    Actual GB / cap GB    Effective ratio    Update seconds (cumulative)']
    for row in rows:
        lines.append(f"{row['method']:10s} {row['label']:5s}  {row['actual_memory_bytes']/1e9:9.3f} / {row['hard_budget_bytes']/1e9:7.2f}"
                     f"    {row.get('effective_retention_ratio', 0):8.2%}       {row.get('memory_update_time', 0):9.2f}")
    return lines


def rate_and_eta(frames, total, average, history=()):
    recent = average
    if len(history) > 1:
        first, last = history[0], history[-1]
        seconds = last['elapsed_seconds'] - first['elapsed_seconds']
        if seconds > 0:
            recent = (last['frames_processed'] - first['frames_processed']) / seconds
    eta = f'{max(0, total - frames) / recent / 3600:.2f} h' if recent > 0 else 'pending'
    if total and frames >= total:
        eta = '0.00 h (frames done; finalization may remain)'
    return recent, eta


def frame_line(frames, total):
    return f'{frames:,} / {total:,} ({100 * frames / total:.3f}%)' if total else f'{frames:,} / pending'


def kv_probe_lines(job):
    path = job.get('kv_probe_status_path')
    if not path:
        return []
    probe = optional_json(Path(path))
    lines = ['', f"KV probe: {probe.get('status', 'pending')}  phase: {probe.get('phase', 'pending')}"]
    if 'frames_processed' in probe:
        lines.append('Probe frames: ' + frame_line(probe['frames_processed'], probe.get('total_frames', 0)))
    for key in ('model', 'model_id', 'device', 'samples_completed', 'total_samples', 'elapsed_seconds', 'summary', 'output', 'error'):
        if key in probe:
            value = probe[key]
            rendered = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
            lines.append(f'  {key}: {rendered[:500]}')
    lines.append('KV status: ' + str(path))
    return lines


def staged_display(job, gpu=False):
    archive = Path(job['archive'])
    stopped = job.get('memory_construction_state') == 'stopped_by_user' or job.get('phase') == 'visual_archive_only'
    smoke = not stopped and job.get('phase') in ('cpu_smoke', 'validating_cpu_smoke') and bool(job.get('smoke_output'))
    run = Path(job['smoke_output'] if smoke else job['output'])
    consumer_label = 'Stage 2 smoke' if smoke else 'Stage 2 memory'
    producer = optional_json(archive / 'status.json')
    header = optional_json(archive / 'archive.json')
    # A cancelled stage is historical. Do not read its stale status/worker files
    # or imply that its old rates and memory rows are still live.
    consumer = {} if stopped else optional_json(run / 'status.json')
    plan = header.get('plan', {}) if stopped else optional_json(run / 'budget_plan.json') or header.get('plan', {})
    producer_plan = header.get('plan', plan)
    try:
        history = [] if stopped else recent_progress(run / 'progress.jsonl')
    except FileNotFoundError:
        history = []
    progress = history[-1] if history else consumer
    # status.json owns the current snapshot phase/frame count; the progress tail
    # supplies recent rates and the latest complete state rows.
    processed = consumer.get('frames_processed', progress.get('frames_processed', 0))
    total = consumer.get('total_frames', plan.get('planned_frames', 0))
    archived = producer.get('frames_processed', 0)
    published = producer.get('frames_published', archived)
    archive_total = producer.get('total_frames', producer_plan.get('planned_frames', 0))
    producer_rate = float(producer.get('total_throughput', 0) or 0)
    _, producer_eta = rate_and_eta(archived, archive_total, producer_rate)
    average = float(progress.get('total_throughput', 0) or 0)
    recent, consumer_eta = rate_and_eta(processed, total, average, history)
    lines = [datetime.now().astimezone().isoformat(timespec='seconds'),
             f"Job: {job.get('state', 'pending')}  phase: {job.get('phase', 'pending')}  Producer PID: {job.get('producer_pid', 'pending')}  Consumer PID: {'stopped by user' if stopped else job.get('consumer_pid', 'pending')}",
             '', f"Stage 1 archive: {producer.get('status', 'pending')}  phase: {producer.get('phase', 'pending')}",
             'Frames: ' + frame_line(archived, archive_total),
             f"Average: {producer_rate:.3f} fps   Elapsed: {producer.get('elapsed_seconds', 0) / 3600:.2f} h   Archive ETA: {producer_eta}",
             f"Payload: {producer.get('payload_bytes', 0) / 1e9:.3f} GB   Published: {published:,} frames / {producer.get('published_shards', 0)} shards   Reused frames: {producer.get('seed_frames_reused', 0):,}   New encoder calls: {producer.get('num_encoder_calls', 0):,}"]
    encoder = producer.get('encoder_stats', {})
    if encoder:
        lines.append(f"Encoder queue: {encoder.get('queue_depth', '?')} chunks / {float(encoder.get('queue_reserved_bytes') or 0) / 1e9:.3f} GB")
    if stopped:
        lines += ['', 'Stage 2 memory: stopped_by_user (bounded-memory construction stopped)',
                  *kv_probe_lines(job), '', 'Rates reflect published archive progress; monitor refresh does not imply a new sample.']
        if gpu:
            lines += gpu_lines()
        lines += ['', f'Archive: {archive}', f'Stopped memory run: {run}']
        return '\n'.join(lines)
    lines += ['', f"{consumer_label}: {consumer.get('status', 'pending')}  phase: {consumer.get('phase', 'pending')}",
              'Frames: ' + frame_line(processed, total),
              f"Average: {average:.3f} fps   Recent: {recent:.3f} fps   Elapsed: {progress.get('elapsed_seconds', 0) / 3600:.2f} h",
              f'Rate-only ETA: {consumer_eta} (future snapshot/compaction I/O excluded)',
              f'Published archive lead: {max(0, published - processed):,} frames',
              f"CPU family RSS: {progress.get('resources', {}).get('cpu_ram', 0) / 1e9:.2f} GB"]
    for key in ('snapshot_method', 'snapshot_label', 'snapshot_timestamp', 'snapshot_state', 'snapshot_progress'):
        if key in consumer:
            lines.append(f'{key}: {consumer[key]}')
    worker_names = ['lightweight'] + ['fluxmem_' + setting['label'] for setting in plan.get('settings', [])[:4]]
    worker_rows = []
    for name in worker_names:
        # Names come from the bounded four-setting plan; never enumerate or read
        # all state shards while rendering progress.
        if '/' in name or '\\' in name:
            continue
        worker = optional_json(run / 'worker_status' / (name + '.json'))
        if worker:
            details = ' '.join(str(worker[key]) for key in ('method', 'label', 'stage') if key in worker)
            worker_rows.append(f"  {name}: PID {worker.get('pid', '?')}  {worker.get('phase', '?')}  "
                               f"frames {worker.get('frames_seen', 0):,}  {details}".rstrip())
    if worker_rows:
        lines += ['CPU workers (live):', *worker_rows]
    elif consumer.get('worker_pids'):
        lines.append('CPU worker PIDs: ' + json.dumps(consumer['worker_pids'], sort_keys=True))
    workers = progress.get('worker_timings', consumer.get('worker_timings', {}))
    if workers:
        seconds = {name: round(value.get('seconds', 0), 3) if isinstance(value, dict) else value
                   for name, value in workers.items()}
        lines.append('Last batch worker seconds: ' + json.dumps(seconds, sort_keys=True))
    reference = plan.get('full_feature_bytes')
    if reference is not None:
        lines.append(f'Fixed budget reference: {reference / 1e9:.6f} GB (full video inventory)')
    lines += ['', *state_lines(progress.get('states', consumer.get('states', []))),
              '', 'Rates reflect published progress; monitor refresh does not imply a new sample.']
    lines += kv_probe_lines(job)
    if gpu:
        lines += gpu_lines()
    lines += ['', f'Archive: {archive}', f'Memory: {run}']
    if smoke:
        lines.append(f"Full run after smoke validation: {job['output']}")
    return '\n'.join(lines)


def display(job_path, gpu=False):
    job = read_json(job_path)
    if 'archive' in job:
        return staged_display(job, gpu)
    run = Path(job['output'])
    state = read_json(run / 'status.json')
    plan = read_json(run / 'budget_plan.json')
    history = recent_progress(run / 'progress.jsonl')
    progress = history[-1] if history else state
    frames = state.get('frames_processed', progress.get('frames_processed', 0))
    total = plan['planned_frames']
    elapsed = progress.get('elapsed_seconds', 0)
    average = progress.get('total_throughput', 0)
    recent, eta = rate_and_eta(frames, total, average, history)
    lines = [datetime.now().astimezone().isoformat(timespec='seconds'),
        f"State: {state.get('status', job.get('state'))}  phase: {state.get('phase', 'stream')}",
        'Frames: ' + frame_line(frames, total),
        f'Average: {average:.3f} fps   Recent: {recent:.3f} fps   Elapsed: {elapsed / 3600:.2f} h',
        f'Rate-only ETA: {eta} (future snapshot/compaction I/O excluded)',
        f"Decode compute: {progress.get('decode_fps', 0):.2f} fps   Encoder compute*: {progress.get('encoder_compute_fps', 0):.2f} fps",
        f"Queue: {progress.get('queue_depth', '?')} chunks / {float(progress.get('queue_reserved_bytes') or 0) / 1e9:.3f} GB",
        f"CPU family RSS: {progress.get('resources', {}).get('cpu_ram', 0) / 1e9:.2f} GB",
        '*Encoder compute uses SUM of worker compute times, not aggregate wall throughput.',
        'Rates reflect published progress; monitor refresh does not imply a new sample.',
        '', *state_lines(progress.get('states', []))]
    if gpu:
        lines += gpu_lines()
    lines += ['', str(run)]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-status', type=Path, default=STAGED_JOB if STAGED_JOB.exists() else DEFAULT_JOB)
    parser.add_argument('--interval', type=float, default=2)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be positive')
    try:
        while True:
            try:
                output = display(args.job_status, args.gpu)
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
                output = f'Waiting for readable progress: {exc}'
            if sys.stdout.isatty() and not args.once:
                print('\033[2J\033[H', end='')
            print(output, flush=True)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
