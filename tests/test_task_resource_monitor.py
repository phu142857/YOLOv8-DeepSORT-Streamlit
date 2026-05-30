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


if __name__ == "__main__":
    unittest.main()
