"""Bounded telemetry counters; no growing in-memory sample history."""
import os
import subprocess
import threading
import time

import psutil


class Telemetry:
    def __init__(self, interval=5):
        self.interval = interval
        self.stop_event = threading.Event()
        self.peak_cpu_ram = 0
        self.cpu_ram = 0
        self.gpus = {}
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()

    def _loop(self):
        while not self.stop_event.is_set():
            try:
                process = psutil.Process(os.getpid())
                ram = 0
                for child in [process, *process.children(recursive=True)]:
                    try:
                        ram += child.memory_info().rss
                    except psutil.Error:
                        pass
                with self.lock:
                    self.cpu_ram = ram
                    self.peak_cpu_ram = max(self.peak_cpu_ram, ram)
                result = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used',
                                         '--format=csv,noheader,nounits'], capture_output=True,
                                        text=True, timeout=3, check=False)
                if result.returncode == 0:
                    with self.lock:
                        for line in result.stdout.splitlines():
                            index, util, memory = [int(x.strip()) for x in line.split(',')]
                            record = self.gpus.setdefault(str(index), {'samples': 0, 'utilization_sum': 0,
                                                                     'max_utilization_percent': 0, 'peak_used_bytes': 0})
                            record['samples'] += 1
                            record['current_utilization_percent'] = util
                            record['current_used_bytes'] = memory * 2**20
                            record['utilization_sum'] += util
                            record['max_utilization_percent'] = max(record['max_utilization_percent'], util)
                            record['peak_used_bytes'] = max(record['peak_used_bytes'], memory * 2**20)
            except (OSError, ValueError, subprocess.SubprocessError, psutil.Error):
                pass
            self.stop_event.wait(self.interval)

    def report(self):
        with self.lock:
            return {'cpu_ram': self.cpu_ram, 'peak_cpu_ram': self.peak_cpu_ram,
                    'cpu_ram_definition': 'sampled sum of parent and child process RSS; shared pages may be counted per process',
                    'gpus': {k: {**v, 'mean_utilization_percent': v['utilization_sum'] / v['samples']}
                             for k, v in self.gpus.items()},
                    'gpu_metric_scope': 'device-wide nvidia-smi; worker-specific allocator peaks reported in encoder worker stats'}

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
