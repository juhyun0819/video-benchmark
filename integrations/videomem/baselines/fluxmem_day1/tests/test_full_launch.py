"""Exercise scheduling, immutability and monitoring without CUDA or a model."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from day1_compare.launch_full import launch, make_plan
from day1_compare.progress_full import render, snapshot


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def package(tmp_path):
    package = tmp_path / "package"
    ids = [f"q{i}" for i in range(10)]
    manifest = {"question_ids": {"all": ["pilot"] + ids, "reused": ["pilot"], "remaining": ids},
                "shards": []}
    for gpu in range(1, 6):
        name = f"shard_{gpu:02d}"
        subset = ids[2 * (gpu - 1):2 * gpu]
        source = package / "shards" / name
        save(source / "public_questions.json", [{"question_id": q} for q in subset])
        digest = hashlib.sha256((source / "public_questions.json").read_bytes()).hexdigest()
        files = {"public_questions.json": digest}
        save(source / "completed.json", {"files": files})
        manifest["shards"].append({"name": name, "question_ids": subset, "gpu": gpu,
            "remaining_start": 2 * (gpu - 1), "remaining_stop": 2 * gpu,
            "arm_order": ["on", "off"] if gpu % 2 else ["off", "on"],
            "package": {"files": files, "completed_sha256": hashlib.sha256(
                (source / "completed.json").read_bytes()).hexdigest()}})
    save(package / "run_manifest.json", manifest)
    return package


class FakeFactory:
    def __init__(self, *, fail=None):
        self.fail, self.calls, self.finished = fail, [], []
        self.active = set()
        self.lock = threading.Lock()
        self.first_wave = threading.Barrier(5)

    def __call__(self, command, **kwargs):
        assert kwargs["stdin"] == subprocess.DEVNULL
        assert kwargs["stderr"] == subprocess.STDOUT
        if "day1_compare.aggregate_full" in command:
            assert len(self.finished) == 10
            factory = self
            class Aggregate:
                pid = os.getpid()
                def wait(self):
                    factory.calls.append(("aggregate",))
                    return 0
            return Aggregate()
        gpu = int(command[command.index("--gpu") + 1])
        arm = command[command.index("--fluxmem") + 1]
        source = Path(command[command.index("--input") + 1])
        output = Path(command[command.index("--output") + 1])
        with self.lock:
            assert gpu not in self.active, "Two concurrent arms must never share a GPU"
            assert (gpu, arm) not in self.calls, "No arm may be retried"
            first = not any(call[0] == gpu for call in self.calls)
            self.calls.append((gpu, arm))
            self.active.add(gpu)
        factory = self
        class Process:
            pid = os.getpid()
            def wait(self):
                if first:
                    factory.first_wave.wait(timeout=5)
                questions = json.loads((source / "public_questions.json").read_text())
                rows = [{"question_id": q["question_id"], "status": "complete",
                         "generation_seconds": 1.0} for q in questions]
                code = int((gpu, arm) == factory.fail)
                if code:
                    rows[-1]["status"] = "execution_error"
                save(output / "predictions.json", rows)
                save(output / "status.json", {"phase": "finished" if not code else "stopped_on_error"})
                with factory.lock:
                    factory.active.remove(gpu)
                    factory.finished.append((gpu, arm))
                return code
        return Process()


def run_fake(package, tmp_path, factory):
    root = tmp_path / "runs"
    code = launch(package, root, tmp_path / "pilot_on", tmp_path / "pilot_off",
                  tmp_path / "combined", python=Path(sys.executable), popen=factory)
    return code, root


def test_all_five_gpus_run_concurrently_and_each_pair_is_sequential(package, tmp_path):
    fake = FakeFactory()
    code, root = run_fake(package, tmp_path, fake)
    assert code == 0
    assert len(fake.calls) == 11
    for gpu in range(1, 6):
        assert [arm for device, arm in fake.calls[:-1] if device == gpu] == (
            ["on", "off"] if gpu % 2 else ["off", "on"])
    state = json.loads((root / "run_status.json").read_text())
    assert state["phase"] == "finished"
    assert state["aggregate"]["exit_code"] == 0
    assert all(run["pid"] and run["started_at"] <= run["ended_at"]
               for worker in state["workers"].values() for run in worker["arms"].values())
    report = snapshot(root)
    assert report["arms"]["on"]["new_complete"] == 10
    assert report["arms"]["off"]["new_complete"] == 10
    assert all(worker["paired_complete"] == 2 for worker in report["workers"])
    assert "reused 1 + new 10 = 11/11" in render(report)
    assert not report["errors"]


def test_failure_does_not_retry_or_stop_partner_or_other_gpus(package, tmp_path):
    fake = FakeFactory(fail=(1, "on"))
    code, root = run_fake(package, tmp_path, fake)
    assert code == 1
    assert len(fake.finished) == len(set(fake.finished)) == 10
    assert (1, "off") in fake.finished
    report = snapshot(root)
    assert report["phase"] == "failed"
    assert report["arms"]["on"]["new_complete"] == 9
    assert report["arms"]["off"]["new_complete"] == 10
    assert any("shard_01/on" in error for error in report["errors"])
    assert report["estimated_seconds_remaining"] is None


def test_refuse_existing_root_before_any_process(package, tmp_path):
    root = tmp_path / "runs"
    save(root / "shard_01/on/predictions.json", [{"sentinel": True}])
    fake = FakeFactory()
    with pytest.raises(FileExistsError, match="immutable"):
        run_fake(package, tmp_path, fake)
    assert not fake.calls
    assert json.loads((root / "shard_01/on/predictions.json").read_text()) == [{"sentinel": True}]


def test_refuse_duplicate_ids_and_tampered_package(package, tmp_path):
    path = package / "run_manifest.json"
    original = json.loads(path.read_text())
    modified = json.loads(path.read_text())
    modified["question_ids"]["remaining"].append("q0")
    save(path, modified)
    with pytest.raises(ValueError, match="Duplicate"):
        make_plan(package, tmp_path / "runs")
    save(path, original)
    save(package / "shards/shard_01/public_questions.json", [{"question_id": "q3"}])
    with pytest.raises(ValueError, match="hash mismatch"):
        make_plan(package, tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


def test_refuse_frozen_assignment_change(package, tmp_path):
    path = package / "run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["shards"][0]["gpu"] = 5
    save(path, manifest)
    with pytest.raises(ValueError, match="frozen assignment"):
        make_plan(package, tmp_path / "runs")


def test_monitor_cli_is_cpu_only(package, tmp_path):
    _, root = run_fake(package, tmp_path, FakeFactory())
    script = ("import runpy, sys; sys.argv = ['progress_full', '--root', sys.argv[1], '--json']; "
              "runpy.run_module('day1_compare.progress_full', run_name='__main__'); "
              "assert 'torch' not in sys.modules")
    result = subprocess.run([sys.executable, "-c", script, str(root)], check=True,
                            capture_output=True, text=True)
    assert json.loads(result.stdout)["phase"] == "finished"
