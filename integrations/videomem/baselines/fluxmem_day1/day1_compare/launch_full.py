"""Run five immutable Day1 shard pairs, one sequential pair per physical GPU.

Run this CPU-only supervisor in a durable user service. Each inference process
owns one GPU; a failed arm never cancels another arm and is never retried.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
GPUS = (1, 2, 3, 4, 5)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_plan(package: Path, new_root: Path) -> list[dict]:
    """Validate the frozen public partition before reserving any outputs."""
    package, new_root = Path(package).resolve(), Path(new_root).resolve()
    manifest = read_json(package / "run_manifest.json")
    remaining = manifest["question_ids"]["remaining"]
    reused = manifest["question_ids"]["reused"]
    all_ids = manifest["question_ids"]["all"]
    if len(all_ids) != len(set(all_ids)) or set(reused) & set(remaining):
        raise ValueError("Duplicate or overlapping frozen question IDs")
    if set(all_ids) != set(reused) | set(remaining):
        raise ValueError("Frozen population does not equal reused plus remaining")
    if len(remaining) != len(set(remaining)) or len(reused) != len(set(reused)):
        raise ValueError("Duplicate frozen continuation IDs")
    frozen_shards = {item["name"]: item for item in manifest["shards"]}
    if len(frozen_shards) != len(GPUS) or len(manifest["shards"]) != len(GPUS):
        raise ValueError("Expected exactly five frozen shard assignments")
    seen, plan = set(), []
    for gpu in GPUS:
        name = f"shard_{gpu:02d}"
        source = package / "shards" / name
        if not (source / "completed.json").is_file():
            raise ValueError(f"Incomplete prepared shard: {source}")
        frozen = frozen_shards[name]
        completion_bytes = (source / "completed.json").read_bytes()
        if hashlib.sha256(completion_bytes).hexdigest() != frozen["package"]["completed_sha256"]:
            raise ValueError(f"Frozen completion hash mismatch: {name}")
        completion = json.loads(completion_bytes)
        for filename, expected in completion["files"].items():
            if (Path(filename).name != filename or
                    hashlib.sha256((source / filename).read_bytes()).hexdigest() != expected or
                    frozen["package"]["files"].get(filename) != expected):
                raise ValueError(f"Frozen package file hash mismatch: {name}/{filename}")
        ids = [row["question_id"] for row in read_json(source / "public_questions.json")]
        if not ids or len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError(f"Empty or duplicate shard question IDs: {name}")
        seen.update(ids)
        order = ["on", "off"] if gpu % 2 else ["off", "on"]
        if (ids != frozen["question_ids"] or frozen["gpu"] != gpu or
                frozen["arm_order"] != order or
                ids != remaining[frozen["remaining_start"]:frozen["remaining_stop"]]):
            raise ValueError(f"Shard differs from the frozen assignment: {name}")
        plan.append({"shard": name, "gpu": gpu, "input": str(source),
                     "output": str(new_root / name), "planned": len(ids),
                     "arm_order": order})
    if seen != set(remaining):
        raise ValueError("Shards must exactly partition the frozen remaining questions")
    return plan


class RunState:
    """Serialize all thread updates, replacing the status file atomically."""
    def __init__(self, path: Path, data: dict):
        self.path, self.data, self.lock = path, data, threading.RLock()
        self.update()

    def update(self, worker=None, arm=None, **values):
        with self.lock:
            target = self.data
            if worker is not None:
                target = target["workers"][worker]
            if arm is not None:
                target = target["arms"][arm]
            target.update(values)
            self.data["updated_at"] = time.time()
            temp = self.path.with_suffix(".json.tmp")
            with temp.open("w", encoding="utf-8") as stream:
                json.dump(self.data, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            temp.replace(self.path)


def run_process(command, log_path: Path, on_start, *, popen=None) -> int:
    factory = subprocess.Popen if popen is None else popen
    with log_path.open("x", encoding="utf-8") as log:
        process = factory(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                          stdin=subprocess.DEVNULL, env=dict(os.environ, PYTHONUNBUFFERED="1"))
        on_start(process.pid)
        return process.wait()


def launch(package: Path, new_root: Path, pilot_on: Path, pilot_off: Path,
           combined: Path, *, python: Path = ROOT / ".venv/bin/python", popen=None) -> int:
    package, new_root, pilot_on, pilot_off, combined = (
        Path(path).absolute() for path in (package, new_root, pilot_on, pilot_off, combined))
    plan = make_plan(package, new_root)
    # Reserving the entire root also prevents accidentally rerunning partial
    # inference, logs, or an earlier supervisor after a crash.
    if new_root.exists() or new_root.is_symlink():
        raise FileExistsError(f"Use a fresh inference root; existing results are immutable: {new_root}")
    if combined.exists() or combined.is_symlink():
        raise FileExistsError(f"Combined output already exists: {combined}")
    if not Path(python).is_file():
        raise FileNotFoundError(f"Missing inference interpreter: {python}")
    new_root.mkdir(parents=True, exist_ok=False)
    logs = new_root / "logs"
    logs.mkdir()
    manifest = read_json(package / "run_manifest.json")
    state = RunState(new_root / "run_status.json", {
        "schema_version": "day1_full_v2_supervisor_v1", "phase": "running",
        "supervisor_pid": os.getpid(), "started_at": time.time(), "ended_at": None,
        "package": str(package), "new_root": str(new_root), "combined": str(combined),
        "pilot_on": str(pilot_on), "pilot_off": str(pilot_off),
        "counts": {"all": len(manifest["question_ids"]["all"]),
                   "reused": len(manifest["question_ids"]["reused"]),
                   "remaining": len(manifest["question_ids"]["remaining"])},
        "workers": {item["shard"]: {**item, "phase": "pending", "current_arm": None,
            "arms": {arm: {"phase": "pending", "pid": None, "started_at": None,
                            "ended_at": None, "exit_code": None,
                            "log": str(logs / f"{item['shard']}_{arm}.log")}
                     for arm in ("on", "off")}} for item in plan},
        "aggregate": {"phase": "pending", "pid": None, "exit_code": None},
        "policy": "One attempt per arm; preserve failures; no retries or cancellation of other jobs",
    })

    def worker(item):
        shard = item["shard"]
        failures = []
        for arm in item["arm_order"]:
            state.update(worker=shard, phase="running", current_arm=arm)
            command = [str(python), "-u", "-m", "day1_compare.run_multi_gpu",
                       "--input", item["input"], "--output", str(Path(item["output"]) / arm),
                       "--gpu", str(item["gpu"]), "--fluxmem", arm]
            state.update(worker=shard, arm=arm, phase="starting", command=command,
                         started_at=time.time())
            try:
                destination = Path(item["output"]) / arm
                if destination.exists() or destination.is_symlink():
                    raise FileExistsError(f"Refusing existing arm output: {destination}")
                code = run_process(command, logs / f"{shard}_{arm}.log",
                    lambda pid: state.update(worker=shard, arm=arm, phase="running", pid=pid),
                    popen=popen)
                state.update(worker=shard, arm=arm, phase="finished" if code == 0 else "failed",
                             exit_code=code, ended_at=time.time())
            except Exception as exc:
                code = -1
                state.update(worker=shard, arm=arm, phase="failed", exit_code=code,
                             error=f"{type(exc).__name__}: {exc}", ended_at=time.time())
            if code:
                failures.append(arm)
        state.update(worker=shard, phase="failed" if failures else "finished",
                     current_arm=None, ended_at=time.time())
        return bool(failures)

    failed = False
    with ThreadPoolExecutor(max_workers=len(GPUS), thread_name_prefix="day1_gpu") as pool:
        futures = {pool.submit(worker, item): item for item in plan}
        for future in as_completed(futures):
            try:
                failed = future.result() or failed
            except Exception as exc:
                failed = True
                state.update(worker=futures[future]["shard"], phase="failed",
                             error=f"{type(exc).__name__}: {exc}", ended_at=time.time())
    # Aggregate even if an arm failed: the offline auditor records the exact
    # failure or refuses an incomplete population without manufacturing rows.
    state.update(phase="aggregating")
    command = [str(python), "-u", "-m", "day1_compare.aggregate_full", "--package", str(package),
               "--pilot-on", str(pilot_on), "--pilot-off", str(pilot_off),
               "--new-root", str(new_root), "--output", str(combined)]
    aggregate = {"phase": "starting", "command": command, "started_at": time.time(),
                 "pid": None, "exit_code": None, "log": str(logs / "aggregate.log")}
    state.update(aggregate=aggregate)
    try:
        def aggregate_started(pid):
            aggregate.update(phase="running", pid=pid)
            state.update(aggregate=aggregate)
        code = run_process(command, logs / "aggregate.log", aggregate_started, popen=popen)
        aggregate.update(phase="finished" if code == 0 else "failed", exit_code=code,
                         ended_at=time.time())
    except Exception as exc:
        code = -1
        aggregate.update(phase="failed", exit_code=code, error=f"{type(exc).__name__}: {exc}",
                         ended_at=time.time())
    failed = failed or code != 0
    state.update(aggregate=aggregate, phase="failed" if failed else "finished",
                 exit_code=int(failed), ended_at=time.time())
    return int(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=ROOT / "artifacts/day1_full_v2")
    parser.add_argument("--new-root", type=Path, default=ROOT / "local_runs/day1_full_v2")
    parser.add_argument("--pilot-on", type=Path, default=ROOT / "local_runs/day1_v2_on")
    parser.add_argument("--pilot-off", type=Path, default=ROOT / "local_runs/day1_v2_off")
    parser.add_argument("--output", type=Path, default=ROOT / "local_runs/day1_full_v2_combined")
    parser.add_argument("--python", type=Path, default=ROOT / ".venv/bin/python")
    args = parser.parse_args()
    raise SystemExit(launch(args.package, args.new_root, args.pilot_on, args.pilot_off,
                            args.output, python=args.python))


if __name__ == "__main__":
    main()
