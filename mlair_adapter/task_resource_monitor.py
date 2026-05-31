"""PID-scoped CPU/RAM/GPU samples for MLAir Resource Usage Contract v1."""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from typing import Any

from mlair_adapter.usage_cost_math import cpu_delta_to_machine_percent

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]


def resource_monitor_enabled() -> bool:
    for key in ("ML_AIR_RESOURCE_MONITOR_ENABLED", "MLAIR_RESOURCE_MONITOR_ENABLED"):
        val = os.getenv(key, "").strip().lower()
        if val in ("0", "false", "no"):
            return False
    return True


def default_sample_interval_seconds() -> float:
    raw = (
        os.getenv("ML_AIR_RESOURCE_SAMPLE_INTERVAL")
        or os.getenv("MLAIR_RESOURCE_SAMPLE_INTERVAL_SEC")
        or "3"
    ).strip()
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 3.0


def cpu_percent_from_delta(delta_cpu_seconds: float, delta_wall_seconds: float) -> float:
    if delta_wall_seconds <= 0 or delta_cpu_seconds < 0:
        return 0.0
    return round((delta_cpu_seconds / delta_wall_seconds) * 100.0, 2)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


_nvml_lock = threading.Lock()
_nvml_initialized = False


def _nvml_init_once() -> bool:
    global _nvml_initialized
    if _nvml_initialized:
        return True
    with _nvml_lock:
        if _nvml_initialized:
            return True
        try:
            import pynvml  # type: ignore[import-untyped]

            pynvml.nvmlInit()
            _nvml_initialized = True
            return True
        except Exception:
            return False


def _gpu_stats_nvml(pids: set[int]) -> tuple[float | None, float | None]:
    if not pids or not _nvml_init_once():
        return None, None
    import pynvml  # type: ignore[import-untyped]

    util_peak: float | None = None
    mem_peak_mb: float | None = None
    try:
        for idx in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            device_util: float | None = None
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                device_util = float(util.gpu)
            except Exception:
                pass
            matched = False
            try:
                for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                    if int(proc.pid) not in pids:
                        continue
                    matched = True
                    used_mb = float(proc.usedGpuMemory) / (1024.0 * 1024.0)
                    if mem_peak_mb is None or used_mb > mem_peak_mb:
                        mem_peak_mb = used_mb
            except Exception:
                pass
            if matched and device_util is not None:
                util_peak = device_util if util_peak is None else max(util_peak, device_util)
            elif device_util is not None and util_peak is None and mem_peak_mb is not None:
                util_peak = device_util
    except Exception:
        return None, None
    return util_peak, mem_peak_mb


def _gpu_stats_torch() -> tuple[float | None, float | None]:
    try:
        import torch
    except ImportError:
        return None, None
    if not torch.cuda.is_available():
        return None, None
    try:
        mem_bytes = max(int(torch.cuda.memory_allocated()), int(torch.cuda.max_memory_allocated()))
        mem_mb = round(mem_bytes / (1024.0 * 1024.0), 2) if mem_bytes > 0 else None
        return None, mem_mb
    except Exception:
        return None, None


def _gpu_stats_for_pids(pids: set[int]) -> tuple[float | None, float | None]:
    util, mem = _gpu_stats_nvml(pids)
    if util is not None or mem is not None:
        return util, mem
    t_util, t_mem = _gpu_stats_torch()
    return t_util if util is None else util, t_mem if mem is None else mem


class TaskResourceMonitor:
    """Background sampler for a process tree (external worker)."""

    def __init__(
        self,
        *,
        task_id: str | None = None,
        interval_seconds: float | None = None,
        flush_interval_seconds: float | None = None,
    ) -> None:
        self.task_id = str(task_id or "").strip() or None
        self.interval_seconds = (
            interval_seconds if interval_seconds is not None else default_sample_interval_seconds()
        )
        self.flush_interval_seconds = flush_interval_seconds  # unused on external workers
        self._root_pid: int | None = None
        self._samples: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._started_at: float | None = None
        self._cpu_times0 = 0.0
        self._disk_io0: tuple[int, int] | None = None
        self._last_sample_wall: float | None = None
        self._last_sample_cpu: float | None = None
        self._gpu_seconds_acc = 0.0
        self._gpu_mem_mb_seconds_acc = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def root_pid(self) -> int | None:
        return self._root_pid

    def start(self, pid: int | None = None) -> None:
        if psutil is None or not resource_monitor_enabled():
            return
        if self._thread and self._thread.is_alive():
            self.attach_pid(int(pid if pid is not None else os.getpid()))
            return
        self.attach_pid(int(pid if pid is not None else os.getpid()))
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="task-resource-monitor",
            daemon=True,
        )
        self._thread.start()

    def attach_pid(self, pid: int) -> None:
        if psutil is None or not resource_monitor_enabled():
            return
        self._root_pid = int(pid)
        self._started_at = time.perf_counter()
        self._cpu_times0 = self._cpu_time_seconds_tree(self._root_pid)
        self._disk_io0 = self._disk_io_tree(self._root_pid)
        self._last_sample_wall = self._started_at
        self._last_sample_cpu = self._cpu_times0

    def stop(self) -> dict[str, Any]:
        return self.build_report()

    def sample_once(self) -> dict[str, Any] | None:
        if psutil is None or self._root_pid is None or not resource_monitor_enabled():
            return None
        procs = self._process_tree(self._root_pid)
        pids = {p.pid for p in procs}
        mem_bytes = self._memory_rss_bytes_tree(self._root_pid)
        gpu_util, gpu_mem_mb = _gpu_stats_for_pids(pids)
        with self._lock:
            cpu_pct = self._cpu_percent_since_last_tree_locked(self._root_pid)
            sample = {
                "sampled_at": _iso_now(),
                "cpu_percent": cpu_pct,
                "memory_mb": round(mem_bytes / (1024.0 * 1024.0), 2) if mem_bytes > 0 else 0.0,
                "gpu_util_percent": gpu_util,
                "gpu_memory_mb": gpu_mem_mb,
            }
            self._samples.append(sample)
            if gpu_util is not None and float(gpu_util) > 0:
                self._gpu_seconds_acc += self.interval_seconds
            if gpu_mem_mb is not None:
                self._gpu_mem_mb_seconds_acc += float(gpu_mem_mb) * self.interval_seconds
        return sample

    def build_report(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self.interval_seconds + 3.0)
        self._thread = None

        if not resource_monitor_enabled():
            return {}
        wall_seconds = 0.0
        if self._started_at is not None:
            wall_seconds = max(0.0, time.perf_counter() - self._started_at)

        with self._lock:
            samples = list(self._samples)

        cpu_seconds = 0.0
        disk_read = None
        disk_write = None
        if self._root_pid is not None and psutil is not None:
            cpu_seconds = max(0.0, self._cpu_time_seconds_tree(self._root_pid) - self._cpu_times0)
            disk_end = self._disk_io_tree(self._root_pid)
            if self._disk_io0 is not None and disk_end is not None:
                disk_read = max(0, disk_end[0] - self._disk_io0[0])
                disk_write = max(0, disk_end[1] - self._disk_io0[1])

        memory_rss_kb = None
        if samples:
            peak_mb = max(float(s["memory_mb"]) for s in samples if s.get("memory_mb") is not None)
            memory_rss_kb = int(peak_mb * 1024)
        elif self._root_pid is not None and psutil is not None:
            rss = self._memory_rss_bytes_tree(self._root_pid)
            if rss > 0:
                memory_rss_kb = int(rss / 1024)

        resource_usage: dict[str, Any] = {}
        if wall_seconds > 0:
            resource_usage["duration_ms"] = int(wall_seconds * 1000)
        if cpu_seconds > 0:
            resource_usage["cpu_time_seconds"] = round(cpu_seconds, 3)
        if memory_rss_kb is not None:
            resource_usage["memory_rss_kb"] = memory_rss_kb
        if disk_read is not None:
            resource_usage["disk_read_bytes"] = disk_read
        if disk_write is not None:
            resource_usage["disk_write_bytes"] = disk_write
        if self._gpu_seconds_acc > 0:
            resource_usage["gpu_seconds"] = round(self._gpu_seconds_acc, 3)
        if self._gpu_mem_mb_seconds_acc > 0:
            resource_usage["gpu_memory_mb_seconds"] = round(self._gpu_mem_mb_seconds_acc, 3)

        out: dict[str, Any] = {}
        if resource_usage:
            out["resource_usage"] = resource_usage
        if samples:
            out["usage_samples"] = samples
        return out

    def _sample_loop(self) -> None:
        if psutil is None or self._root_pid is None:
            return
        while not self._stop.is_set():
            self.sample_once()
            if self._stop.wait(self.interval_seconds):
                break

    def _cpu_percent_since_last_tree_locked(self, root_pid: int) -> float:
        now_wall = time.perf_counter()
        now_cpu = self._cpu_time_seconds_tree(root_pid)
        if self._last_sample_wall is None or self._last_sample_cpu is None:
            self._last_sample_wall = now_wall
            self._last_sample_cpu = now_cpu
            return 0.0
        delta_wall = now_wall - self._last_sample_wall
        delta_cpu = now_cpu - self._last_sample_cpu
        self._last_sample_wall = now_wall
        self._last_sample_cpu = now_cpu
        raw = cpu_percent_from_delta(delta_cpu, delta_wall)
        return cpu_delta_to_machine_percent(raw)

    @staticmethod
    def _process_tree(root_pid: int) -> list[Any]:
        if psutil is None:
            return []
        try:
            root = psutil.Process(root_pid)
        except psutil.NoSuchProcess:
            return []
        procs = [root]
        try:
            procs.extend(root.children(recursive=True))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        alive: list[Any] = []
        for proc in procs:
            try:
                if proc.is_running():
                    alive.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return alive

    def _cpu_time_seconds_tree(self, root_pid: int) -> float:
        total = 0.0
        for proc in self._process_tree(root_pid):
            try:
                ct = proc.cpu_times()
                total += float(ct.user + ct.system)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total

    def _memory_rss_bytes_tree(self, root_pid: int) -> int:
        total = 0
        for proc in self._process_tree(root_pid):
            try:
                total += int(proc.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total

    @staticmethod
    def _disk_io_tree(root_pid: int) -> tuple[int, int] | None:
        if psutil is None:
            return None
        read_total = 0
        write_total = 0
        seen = False
        try:
            root = psutil.Process(root_pid)
            procs = [root] + root.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None
        for proc in procs:
            try:
                io = proc.io_counters()
                read_total += int(io.read_bytes)
                write_total += int(io.write_bytes)
                seen = True
            except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                continue
        return (read_total, write_total) if seen else None
