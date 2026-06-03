import os
import unittest

from mlair_adapter.task_resource_monitor import cpu_percent_from_delta
from mlair_adapter.usage_cost_math import (
    cpu_delta_to_machine_percent,
    normalize_cpu_tree_percent,
)


class TaskResourceMonitorTest(unittest.TestCase):
    def test_cpu_percent_from_delta_one_core(self) -> None:
        self.assertAlmostEqual(cpu_percent_from_delta(1.0, 1.0), 100.0)

    def test_cpu_percent_from_delta_multi_core(self) -> None:
        self.assertAlmostEqual(cpu_percent_from_delta(3.5, 1.0), 350.0)

    def test_cpu_percent_from_delta_zero_wall(self) -> None:
        self.assertEqual(cpu_percent_from_delta(1.0, 0.0), 0.0)

    def test_cpu_delta_to_machine_percent_partial_core(self) -> None:
        # 44.6 psutil-units ≈ 0.446 cores on 8-core → ~5.6% machine (matches Task Manager)
        self.assertAlmostEqual(cpu_delta_to_machine_percent(44.6, logical_cpus=8), 5.58, places=1)

    def test_cpu_delta_to_machine_percent_full_machine(self) -> None:
        self.assertAlmostEqual(cpu_delta_to_machine_percent(800.0, logical_cpus=8), 100.0)

    def test_normalize_cpu_tree_percent_legacy_sum(self) -> None:
        self.assertAlmostEqual(normalize_cpu_tree_percent(792.0, logical_cpus=8), 99.0)

    def test_normalize_cpu_tree_percent_contract_v1(self) -> None:
        self.assertAlmostEqual(normalize_cpu_tree_percent(5.6, logical_cpus=8), 5.6)

    def test_gpu_stats_for_pids_uses_pid_memory_delta(self) -> None:
        from unittest import mock

        from mlair_adapter import task_resource_monitor as trm

        with mock.patch.object(trm, "_nvml_pid_mem_mb", return_value=1680.0):
            with mock.patch.object(trm, "_nvml_device_used_mb", return_value=1680.0):
                with mock.patch.object(trm, "_nvml_device_util", return_value=85.0):
                    util, mem = trm._gpu_stats_for_pids(
                        {123},
                        pid_mem_baseline_mb=300.0,
                        device_mem_baseline_mb=400.0,
                    )
        self.assertEqual(util, 85.0)
        self.assertEqual(mem, 1380.0)

    def test_gpu_util_from_memory_mb_none_is_safe(self) -> None:
        from mlair_adapter.task_resource_monitor import _gpu_util_from_memory_mb

        self.assertIsNone(_gpu_util_from_memory_mb(None))

    def test_build_report_merge_cuda_none_is_safe(self) -> None:
        from mlair_adapter import task_resource_monitor as trm
        from mlair_adapter.task_resource_monitor import TaskResourceMonitor

        mon = TaskResourceMonitor(interval_seconds=1.0)
        mon.start(os.getpid())
        mon.sample_once()
        report = mon.stop()
        self.assertIsInstance(report, dict)

    def test_gpu_stats_for_pids_zero_util_uses_memory_occupancy(self) -> None:
        from unittest import mock

        from mlair_adapter import task_resource_monitor as trm

        with mock.patch.object(trm, "_nvml_pid_mem_mb", return_value=1680.0):
            with mock.patch.object(trm, "_nvml_device_used_mb", return_value=1680.0):
                with mock.patch.object(trm, "_nvml_device_util", return_value=0.0):
                    with mock.patch.object(trm, "_cuda_total_vram_mb", return_value=3768.0):
                        with mock.patch.object(trm, "_gpu_util_nvidia_smi", return_value=0.0):
                            util, mem = trm._gpu_stats_for_pids(
                                {123},
                                pid_mem_baseline_mb=300.0,
                                device_mem_baseline_mb=300.0,
                            )
        self.assertEqual(mem, 1380.0)
        self.assertGreater(util or 0, 0.0)

    def test_gpu_stats_for_pids_below_threshold_returns_none(self) -> None:
        from unittest import mock

        from mlair_adapter import task_resource_monitor as trm

        with mock.patch.object(trm, "_nvml_pid_mem_mb", return_value=315.0):
            with mock.patch.object(trm, "_nvml_device_used_mb", return_value=315.0):
                with mock.patch.object(trm, "_nvml_device_util", return_value=85.0):
                    util, mem = trm._gpu_stats_for_pids(
                        {123},
                        pid_mem_baseline_mb=300.0,
                        device_mem_baseline_mb=300.0,
                    )
        self.assertIsNone(util)
        self.assertIsNone(mem)

    def test_sample_once_reports_task_scoped_rss_delta(self) -> None:
        from unittest import mock

        from mlair_adapter import task_resource_monitor as trm
        from mlair_adapter.task_resource_monitor import TaskResourceMonitor

        mon = TaskResourceMonitor(interval_seconds=1.0)
        mon._root_pid = os.getpid()
        baseline = 20 * 1024 * 1024 * 1024
        mon._mem_rss_baseline_bytes = baseline
        mon._mem_rss_peak_delta_bytes = 0
        with mock.patch.object(mon, "_memory_rss_bytes_tree", return_value=baseline + 3 * 1024**3):
            with mock.patch.object(mon, "_process_tree", return_value=[]):
                with mock.patch.object(mon, "_cpu_percent_since_last_tree_locked", return_value=5.0):
                    with mock.patch.object(trm, "_gpu_stats_for_pids", return_value=(None, None)):
                        with mock.patch.object(mon, "_cuda_peak_delta_mb", return_value=None):
                            sample = mon.sample_once()
        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertAlmostEqual(sample["memory_mb"], 3072.0, places=0)
        self.assertEqual(mon._mem_rss_peak_delta_bytes, 3 * 1024**3)

    def test_refresh_memory_baseline_resets_delta_peak(self) -> None:
        from unittest import mock

        from mlair_adapter.task_resource_monitor import TaskResourceMonitor

        mon = TaskResourceMonitor(interval_seconds=1.0)
        mon._root_pid = os.getpid()
        mon._mem_rss_baseline_bytes = 100
        mon._mem_rss_peak_delta_bytes = 5000
        with mock.patch.object(mon, "_memory_rss_bytes_tree", return_value=8 * 1024**3):
            mon.refresh_memory_baseline()
        self.assertEqual(mon._mem_rss_baseline_bytes, 8 * 1024**3)
        self.assertEqual(mon._mem_rss_peak_delta_bytes, 0)

    def test_gpu_stats_for_pids_cuda_context_noise_ignored(self) -> None:
        from unittest import mock

        from mlair_adapter import task_resource_monitor as trm

        with mock.patch.object(trm, "_nvml_pid_mem_mb", return_value=15.0):
            with mock.patch.object(trm, "_nvml_device_used_mb", return_value=15.0):
                with mock.patch.object(trm, "_nvml_device_util", return_value=5.0):
                    util, mem = trm._gpu_stats_for_pids(
                        {123},
                        pid_mem_baseline_mb=0.0,
                        device_mem_baseline_mb=0.0,
                    )
        self.assertIsNone(util)
        self.assertIsNone(mem)


if __name__ == "__main__":
    unittest.main()
