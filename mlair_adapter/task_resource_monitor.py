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

# Min task VRAM delta (MB) before reporting GPU on Hub (skip idle CUDA context).
_GPU_REPORT_MIN_MB = float(os.getenv("MLAIR_GPU_REPORT_MIN_MB", "20"))
# Ignore per-task RSS delta spikes above this (bad baseline / absolute leak into heartbeats).
_MEM_DELTA_SANITY_MAX_MB = float(os.getenv("MLAIR_MEM_DELTA_SANITY_MAX_MB", "16384"))


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


def _nvml_device_used_mb(device_index: int = 0) -> float | None:
    if not _nvml_init_once():
        return None
    import pynvml  # type: ignore[import-untyped]

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return round(float(info.used) / (1024.0 * 1024.0), 2)
    except Exception:
        return None


def _nvml_device_util(device_index: int = 0) -> float | None:
    if not _nvml_init_once():
        return None
    import pynvml  # type: ignore[import-untyped]

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return float(util.gpu)
    except Exception:
        return None


def _host_pids_for_tree(pids: set[int]) -> set[int]:
    """Map container PIDs to host PIDs (NVML uses host namespace)."""
    host: set[int] = set()
    for pid in pids:
        try:
            with open(f"/proc/{pid}/status", encoding="utf-8") as fh:
                for line in fh:
                    if not line.startswith("NSpid:"):
                        continue
                    parts = line.split()
                    host.add(int(parts[-1]))
                    break
        except OSError:
            host.add(int(pid))
    return host or pids


def _nvml_compute_procs(device_index: int = 0) -> list[tuple[int, float]]:
    if not _nvml_init_once():
        return []
    import pynvml  # type: ignore[import-untyped]

    out: list[tuple[int, float]] = []
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            out.append((int(proc.pid), float(proc.usedGpuMemory) / (1024.0 * 1024.0)))
    except Exception:
        return []
    return out


def _nvml_total_compute_mem_mb(device_index: int = 0) -> float | None:
    procs = _nvml_compute_procs(device_index)
    if not procs:
        return None
    return round(sum(mb for _, mb in procs), 2)


def _nvml_pid_mem_mb(pids: set[int], *, device_index: int = 0) -> float | None:
    if not pids or not _nvml_init_once():
        return None
    host_pids = _host_pids_for_tree(pids)
    peak: float | None = None
    for proc_pid, used_mb in _nvml_compute_procs(device_index):
        if proc_pid not in host_pids:
            continue
        peak = used_mb if peak is None else max(peak, used_mb)
    return round(peak, 2) if peak is not None else None


def _gpu_util_nvidia_smi(device_index: int = 0) -> float | None:
    import shutil
    import subprocess

    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = subprocess.check_output(
            [
                smi,
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
                f"--id={device_index}",
            ],
            timeout=2,
            text=True,
        )
        line = out.strip().splitlines()[0].strip()
        return float(line) if line else None
    except Exception:
        return None


def _cuda_total_vram_mb() -> float | None:
    if not _nvml_init_once():
        return None
    try:
        import pynvml  # type: ignore[import-untyped]

        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return float(info.total) / (1024.0 * 1024.0)
    except Exception:
        return None


def _gpu_util_from_memory_mb(mem_mb: float | None) -> float | None:
    if mem_mb is None:
        return None
    try:
        mem = float(mem_mb)
    except (TypeError, ValueError):
        return None
    if mem != mem or mem <= 0:  # NaN or non-positive
        return None
    total_mb = _cuda_total_vram_mb()
    if total_mb is None or total_mb <= 0:
        return None
    return min(100.0, round((mem / total_mb) * 100.0, 1))


def _resolve_gpu_util(nvml_util: float | None, mem_mb: float | None) -> float | None:
    util = nvml_util
    if util is None or float(util) <= 0:
        util = _gpu_util_nvidia_smi()
    if util is None or float(util) <= 0:
        util = _gpu_util_from_memory_mb(mem_mb)
    if util is None:
        return None
    return round(float(util), 2)


def _gpu_stats_for_pids(
    pids: set[int],
    *,
    pid_mem_baseline_mb: float = 0.0,
    device_mem_baseline_mb: float = 0.0,
    compute_mem_baseline_mb: float = 0.0,
) -> tuple[float | None, float | None]:
    """NVML-only GPU stats (safe from monitor background thread)."""
    pid_mem = _nvml_pid_mem_mb(pids)
    device_used = _nvml_device_used_mb(0)
    compute_total = _nvml_total_compute_mem_mb(0)
    util = _nvml_device_util(0)

    pid_delta = max(0.0, float(pid_mem or 0.0) - float(pid_mem_baseline_mb))
    device_delta = max(0.0, float(device_used or 0.0) - float(device_mem_baseline_mb))
    compute_delta = max(0.0, float(compute_total or 0.0) - float(compute_mem_baseline_mb))
    mem = max(pid_delta, device_delta, compute_delta)

    if mem < _GPU_REPORT_MIN_MB:
        return None, None
    return _resolve_gpu_util(util, mem), round(mem, 2)


def _cuda_memory_allocated_mb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return round(float(torch.cuda.memory_allocated()) / (1024.0 * 1024.0), 2)
    except Exception:
        pass
    return None


def _cuda_max_memory_allocated_mb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return round(float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0), 2)
    except Exception:
        pass
    return None


def _sync_cuda_baseline() -> None:
    """Main thread: reset CUDA cache before task baseline (best-effort)."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


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
        self._mem_rss_baseline_bytes = 0
        self._mem_rss_peak_delta_bytes = 0
        self._gpu_pid_mem_baseline_mb = 0.0
        self._gpu_device_mem_baseline_mb = 0.0
        self._gpu_compute_mem_baseline_mb = 0.0
        self._cuda_alloc_baseline_mb = 0.0
        self._cuda_peak_seen_mb = 0.0
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
        pid = int(pid)
        _sync_cuda_baseline()
        pids = {p.pid for p in self._process_tree(pid)}
        self._root_pid = pid
        self._started_at = time.perf_counter()
        tree_rss = self._memory_rss_bytes_tree(pid)
        self._mem_rss_baseline_bytes = tree_rss
        self._mem_rss_peak_delta_bytes = 0
        self._gpu_pid_mem_baseline_mb = float(_nvml_pid_mem_mb(pids) or 0.0)
        self._gpu_device_mem_baseline_mb = float(_nvml_device_used_mb(0) or 0.0)
        self._gpu_compute_mem_baseline_mb = float(_nvml_total_compute_mem_mb(0) or 0.0)
        self._cuda_alloc_baseline_mb = float(_cuda_memory_allocated_mb() or 0.0)
        self._cpu_times0 = self._cpu_time_seconds_tree(pid)
        self._disk_io0 = self._disk_io_tree(pid)
        self._last_sample_wall = self._started_at
        self._last_sample_cpu = self._cpu_times0
        with self._lock:
            self._samples.clear()
            self._gpu_seconds_acc = 0.0
            self._gpu_mem_mb_seconds_acc = 0.0
            self._cuda_peak_seen_mb = 0.0

    def refresh_memory_baseline(self) -> None:
        """Re-baseline RSS after a phase change (e.g. GPU fail → CPU retry in same task)."""
        if psutil is None or self._root_pid is None:
            return
        self._mem_rss_baseline_bytes = self._memory_rss_bytes_tree(self._root_pid)
        self._mem_rss_peak_delta_bytes = 0

    def stop(self) -> dict[str, Any]:
        return self.build_report()

    def sample_once(self) -> dict[str, Any] | None:
        if psutil is None or self._root_pid is None or not resource_monitor_enabled():
            return None
        try:
            return self._sample_once_impl()
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning("resource sample_once failed: %s", exc)
            return None

    def _sample_once_impl(self) -> dict[str, Any] | None:
        procs = self._process_tree(self._root_pid)  # type: ignore[arg-type]
        pids = {p.pid for p in procs}
        mem_bytes = self._memory_rss_bytes_tree(self._root_pid)
        if self._mem_rss_baseline_bytes <= 0 and mem_bytes > 0:
            self._mem_rss_baseline_bytes = mem_bytes
        delta_bytes = max(0, mem_bytes - self._mem_rss_baseline_bytes)
        delta_mb = delta_bytes / (1024.0 * 1024.0)
        if delta_mb > _MEM_DELTA_SANITY_MAX_MB:
            delta_bytes = 0
            delta_mb = 0.0
        elif delta_bytes > self._mem_rss_peak_delta_bytes:
            self._mem_rss_peak_delta_bytes = delta_bytes
        gpu_util, nvml_mem = _gpu_stats_for_pids(
            pids,
            pid_mem_baseline_mb=self._gpu_pid_mem_baseline_mb,
            device_mem_baseline_mb=self._gpu_device_mem_baseline_mb,
            compute_mem_baseline_mb=self._gpu_compute_mem_baseline_mb,
        )
        cuda_mem = self._cuda_peak_delta_mb(apply_threshold=False)
        gpu_mem_mb: float | None = None
        gpu_util_out: float | None = gpu_util
        mem_candidates = [
            float(x)
            for x in (nvml_mem, cuda_mem)
            if x is not None and float(x) >= _GPU_REPORT_MIN_MB
        ]
        if mem_candidates:
            gpu_mem_mb = round(max(mem_candidates), 2)
            if gpu_util_out is None or float(gpu_util_out) <= 0:
                gpu_util_out = _resolve_gpu_util(None, gpu_mem_mb)
        else:
            gpu_util_out, gpu_mem_mb = None, None
        if gpu_mem_mb is not None and float(gpu_mem_mb) > self._cuda_peak_seen_mb:
            self._cuda_peak_seen_mb = float(gpu_mem_mb)
        elif cuda_mem is not None and float(cuda_mem) > self._cuda_peak_seen_mb:
            self._cuda_peak_seen_mb = float(cuda_mem)
        with self._lock:
            cpu_pct = self._cpu_percent_since_last_tree_locked(self._root_pid)
            sample = {
                "sampled_at": _iso_now(),
                "cpu_percent": cpu_pct,
                "memory_mb": round(delta_mb, 2),
                "gpu_util_percent": gpu_util_out,
                "gpu_memory_mb": gpu_mem_mb,
            }
            self._samples.append(sample)
            if gpu_util_out is not None and float(gpu_util_out) > 0:
                self._gpu_seconds_acc += self.interval_seconds
            elif gpu_mem_mb is not None and float(gpu_mem_mb) >= _GPU_REPORT_MIN_MB:
                self._gpu_seconds_acc += self.interval_seconds
            if gpu_mem_mb is not None:
                self._gpu_mem_mb_seconds_acc += float(gpu_mem_mb) * self.interval_seconds
        return sample

    def _cuda_peak_delta_mb(self, *, apply_threshold: bool = True) -> float | None:
        peak = _cuda_max_memory_allocated_mb()
        if peak is None:
            return None
        delta = max(0.0, float(peak) - float(self._cuda_alloc_baseline_mb))
        if apply_threshold and delta < _GPU_REPORT_MIN_MB:
            return None
        return round(delta, 2)

    def _merge_cuda_peak_sample(self) -> None:
        cuda_mb = self._cuda_peak_delta_mb(apply_threshold=False)
        if self._cuda_peak_seen_mb > 0:
            cuda_mb = max(float(cuda_mb or 0.0), float(self._cuda_peak_seen_mb))
        if cuda_mb is None or float(cuda_mb) < _GPU_REPORT_MIN_MB:
            return
        cuda_mb = round(float(cuda_mb), 2)
        util = _resolve_gpu_util(_nvml_device_util(0), cuda_mb)
        with self._lock:
            if self._samples and isinstance(self._samples[-1], dict):
                last = self._samples[-1]
                last["gpu_memory_mb"] = max(float(last.get("gpu_memory_mb") or 0), cuda_mb)
                if util is not None:
                    last["gpu_util_percent"] = max(
                        float(last.get("gpu_util_percent") or 0),
                        float(util),
                    )
            else:
                self._samples.append(
                    {
                        "sampled_at": _iso_now(),
                        "cpu_percent": 0.0,
                        "memory_mb": 0.0,
                        "gpu_util_percent": util,
                        "gpu_memory_mb": cuda_mb,
                    }
                )
            if util is not None and float(util) > 0:
                self._gpu_seconds_acc += self.interval_seconds
            self._gpu_mem_mb_seconds_acc += float(cuda_mb) * self.interval_seconds

    def build_report(self) -> dict[str, Any]:
        if not self._stop.is_set():
            self.sample_once()
        self._merge_cuda_peak_sample()
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
        peak_delta_bytes = self._mem_rss_peak_delta_bytes
        if samples:
            peak_mb = max(float(s["memory_mb"]) for s in samples if s.get("memory_mb") is not None)
            peak_delta_bytes = max(peak_delta_bytes, int(peak_mb * 1024))
        elif self._root_pid is not None and psutil is not None:
            rss = max(0, self._memory_rss_bytes_tree(self._root_pid) - self._mem_rss_baseline_bytes)
            peak_delta_bytes = max(peak_delta_bytes, rss)
        if peak_delta_bytes > 0:
            memory_rss_kb = int(peak_delta_bytes / 1024)

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
