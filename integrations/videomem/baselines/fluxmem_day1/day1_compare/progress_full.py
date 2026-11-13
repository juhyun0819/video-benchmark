"""Read-only progress for the five-GPU Day1 run; no model or label imports."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def load(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def alive(pid):
    if not isinstance(pid, int):
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def snapshot(new_root: Path, *, now=None) -> dict:
    now = time.time() if now is None else now
    state = load(Path(new_root) / "run_status.json", None)
    if state is None:
        raise FileNotFoundError(f"Supervisor status not found: {new_root}/run_status.json")
    counts = state["counts"]
    totals = {arm: {"reused": counts["reused"], "new_complete": 0,
                    "new_attempted": 0, "total": counts["all"]} for arm in ("on", "off")}
    workers, errors, rates = [], [], {"on": [], "off": []}
    for name, worker in sorted(state["workers"].items()):
        completed_ids = {}
        item = {"shard": name, "gpu": worker["gpu"], "phase": worker["phase"],
                "current_arm": worker["current_arm"], "planned": worker["planned"], "arms": {}}
        if worker.get("error"):
            errors.append(f"{name}: {worker['error']}")
        for arm in ("on", "off"):
            run = worker["arms"][arm]
            output = Path(worker["output"]) / arm
            rows = load(output / "predictions.json", [])
            status = load(output / "status.json", {})
            ids = [row["question_id"] for row in rows]
            if len(ids) != len(set(ids)) or len(rows) > worker["planned"]:
                raise ValueError(f"Invalid prediction population in {name}/{arm}")
            complete = sum(row.get("status") == "complete" for row in rows)
            completed_ids[arm] = {row["question_id"] for row in rows if row.get("status") == "complete"}
            totals[arm]["new_complete"] += complete
            totals[arm]["new_attempted"] += len(rows)
            item["arms"][arm] = {"phase": status.get("phase", run["phase"]),
                "supervisor_phase": run["phase"], "pid": run.get("pid"),
                "alive": alive(run.get("pid")) if run["phase"] == "running" else False,
                "completed": complete, "attempted": len(rows), "exit_code": run.get("exit_code")}
            if run.get("error"):
                errors.append(f"{name}/{arm}: {run['error']}")
            elif run.get("exit_code") not in (None, 0):
                errors.append(f"{name}/{arm}: process exit {run['exit_code']}; log: {run['log']}")
            if run["phase"] == "running" and not item["arms"][arm]["alive"]:
                errors.append(f"{name}/{arm}: recorded worker PID is no longer alive")
            bad = sum(row.get("status") != "complete" for row in rows)
            if bad:
                errors.append(f"{name}/{arm}: {bad} truncated or execution-error result(s)")
            if len(rows) >= 2 and run.get("started_at") is not None:
                end = run.get("ended_at") or now
                rates[arm].append(max(0.0, end - run["started_at"]) / len(rows))
        item["paired_complete"] = len(completed_ids["on"] & completed_ids["off"])
        workers.append(item)
    aggregate = state.get("aggregate", {})
    if aggregate.get("phase") == "failed":
        errors.append(f"aggregate: {aggregate.get('error', 'exit ' + str(aggregate.get('exit_code')))}; log: {aggregate.get('log')}")
    if state["phase"] in ("running", "aggregating") and not alive(state.get("supervisor_pid")):
        errors.append("Supervisor PID is no longer alive; status may be stale")
    eta = None
    if not errors and all(rates.values()) and state["phase"] == "running":
        # Slowest observed end-to-end rate includes loading/preprocessing;
        # add 25% headroom and sum sequential arms on each GPU.
        eta = max(sum((item["planned"] - item["arms"][arm]["attempted"]) *
                      max(rates[arm]) * 1.25 for arm in ("on", "off")) for item in workers)
    return {"phase": state["phase"], "supervisor_pid": state.get("supervisor_pid"),
            "arms": totals, "workers": workers, "errors": errors,
            "estimated_seconds_remaining": eta,
            "eta_basis": "Slowest measured wall-time/question per arm plus 25%; excludes final aggregation; estimate, not a deadline",
            "aggregate_phase": aggregate.get("phase", "pending")}


def render(report):
    lines = [f"Day1 full v2: {report['phase']} (supervisor PID {report['supervisor_pid']})"]
    for worker in report["workers"]:
        arms = " | ".join(f"{arm.upper()} {worker['arms'][arm]['completed']}/{worker['planned']} "
                          f"({worker['arms'][arm]['phase']})" for arm in ("on", "off"))
        lines.append(f"GPU {worker['gpu']}: {worker['phase']} {worker['current_arm'] or '-'} | {arms} | "
                     f"paired {worker['paired_complete']}/{worker['planned']}")
    for arm, totals in report["arms"].items():
        lines.append(f"{arm.upper()} complete: reused {totals['reused']} + new {totals['new_complete']} = "
                     f"{totals['reused'] + totals['new_complete']}/{totals['total']} "
                     f"(new attempted {totals['new_attempted']})")
    eta = report["estimated_seconds_remaining"]
    if report["phase"] == "finished":
        lines.append("ETA: complete")
    else:
        lines.append("ETA: awaiting measurements or unavailable after errors" if eta is None else
                     f"ETA: approximately {eta / 60:.1f} minutes; {report['eta_basis']}")
    lines.append(f"Aggregation: {report['aggregate_phase']}")
    lines.extend(f"ERROR: {error}" for error in report["errors"])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", "--new-root", dest="new_root", type=Path,
                        default=ROOT / "local_runs/day1_full_v2")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--interval", type=float, default=0,
                        help="Refresh every N seconds until finished/failed; 0 prints once")
    args = parser.parse_args()
    if args.interval < 0:
        parser.error("--interval must be nonnegative")
    try:
        while True:
            report = snapshot(args.new_root)
            print(json.dumps(report, indent=2) if args.json else render(report), flush=True)
            if not args.interval or report["phase"] in ("finished", "failed"):
                break
            deadline = time.monotonic() + args.interval
            while (remaining := deadline - time.monotonic()) > 0:
                time.sleep(min(remaining, 60))
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
