from __future__ import annotations

import csv
import json
import math
import os
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _proc_status(pid: int) -> tuple[int | None, int]:
    try:
        lines = (Path("/proc") / str(pid) / "status").read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    except OSError:
        return None, 0
    parent = None
    rss_kib = 0
    for line in lines:
        if line.startswith("PPid:"):
            parent = int(line.split()[1])
        elif line.startswith("VmRSS:"):
            rss_kib = int(line.split()[1])
    return parent, rss_kib


def process_tree(root_pid: int) -> set[int]:
    parents: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        parent, _ = _proc_status(pid)
        if parent is not None:
            parents[pid] = parent
    selected = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in selected and pid not in selected:
                selected.add(pid)
                changed = True
    return selected


def process_tree_rss_mb(root_pid: int) -> tuple[set[int], float]:
    pids = process_tree(root_pid)
    rss_kib = sum(_proc_status(pid)[1] for pid in pids)
    return pids, rss_kib / 1024.0


def gpu_memory_mb(pids: set[int]) -> float:
    executable = shutil.which("nvidia-smi")
    if executable is None or not pids:
        return 0.0
    try:
        result = subprocess.run(
            [
                executable,
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return 0.0
    total = 0.0
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) < 2:
            continue
        try:
            pid, memory = int(row[0].strip()), float(row[1].strip())
        except ValueError:
            continue
        if pid in pids:
            total += memory
    return total


def directory_size_mb(path: Path) -> float:
    total = 0
    if not path.exists():
        return 0.0
    for root, directories, files in os.walk(path, followlinks=False):
        directories[:] = [name for name in directories if not (Path(root) / name).is_symlink()]
        for name in files:
            file_path = Path(root) / name
            if file_path.is_symlink():
                continue
            try:
                total += file_path.stat().st_size
            except OSError:
                pass
    return total / (1024.0 * 1024.0)


def worker_event_timings(
    path: Path, *, command_started_monotonic: float | None = None
) -> dict[str, float | None]:
    result: dict[str, float | None] = {
        "model_ready_sec": None,
        "time_to_first_prediction_sec": None,
    }
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        elapsed = event.get("elapsed_sec")
        event_monotonic = event.get("monotonic_sec")
        if command_started_monotonic is not None and isinstance(
            event_monotonic, (int, float)
        ):
            elapsed = float(event_monotonic) - command_started_monotonic
        if not isinstance(elapsed, (int, float)) or not math.isfinite(float(elapsed)):
            continue
        elapsed = max(0.0, float(elapsed))
        if event.get("event") == "model_ready" and result["model_ready_sec"] is None:
            result["model_ready_sec"] = elapsed
        if (
            event.get("event") == "first_prediction"
            and result["time_to_first_prediction_sec"] is None
        ):
            result["time_to_first_prediction_sec"] = elapsed
    return result


class _Sampler:
    def __init__(self, pid: int, output_dir: Path, interval_sec: float) -> None:
        self.pid = pid
        self.output_dir = output_dir
        self.interval_sec = interval_sec
        self.stop_event = threading.Event()
        self.peak_cpu_ram_mb = 0.0
        self.peak_gpu_vram_mb = 0.0
        self.peak_temporary_disk_mb = 0.0
        self.sample_count = 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=max(5.0, self.interval_sec * 3.0))
        self._sample()

    def _sample(self) -> None:
        pids, rss_mb = process_tree_rss_mb(self.pid)
        self.peak_cpu_ram_mb = max(self.peak_cpu_ram_mb, rss_mb)
        self.peak_gpu_vram_mb = max(self.peak_gpu_vram_mb, gpu_memory_mb(pids))
        self.peak_temporary_disk_mb = max(
            self.peak_temporary_disk_mb, directory_size_mb(self.output_dir / "work")
        )
        self.sample_count += 1

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self._sample()
            self.stop_event.wait(self.interval_sec)


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    timed_out: bool
    started_at: str
    ended_at: str
    telemetry: dict[str, Any]


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def run_monitored_command(
    command: Sequence[str],
    *,
    cwd: str | Path,
    env: Mapping[str, str],
    stdout_path: str | Path,
    stderr_path: str | Path,
    output_dir: str | Path,
    timeout_sec: float,
    sample_interval_sec: float = 1.0,
) -> CommandResult:
    stdout_path = Path(stdout_path)
    stderr_path = Path(stderr_path)
    output_dir = Path(output_dir)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    started_monotonic = time.monotonic()
    timed_out = False
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            env=dict(env),
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )
        sampler = _Sampler(process.pid, output_dir, max(0.1, sample_interval_sec))
        sampler.start()
        try:
            process.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_group(process)
        finally:
            sampler.stop()
    wall_time = time.monotonic() - started_monotonic
    timings = worker_event_timings(
        output_dir / "worker_events.jsonl",
        command_started_monotonic=started_monotonic,
    )
    telemetry = {
        "wall_time_sec": wall_time,
        **timings,
        "peak_cpu_ram_mb": sampler.peak_cpu_ram_mb,
        "peak_gpu_vram_mb": sampler.peak_gpu_vram_mb,
        "peak_temporary_disk_mb": sampler.peak_temporary_disk_mb,
        "sample_count": sampler.sample_count,
    }
    return CommandResult(
        returncode=int(process.returncode),
        timed_out=timed_out,
        started_at=started_at,
        ended_at=utc_now(),
        telemetry=telemetry,
    )
