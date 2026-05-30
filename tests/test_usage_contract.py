import unittest

from mlair_adapter.usage_contract import contract_complete_resource_usage, contract_summary_from_report


class UsageContractTest(unittest.TestCase):
    def test_contract_summary_from_report(self) -> None:
        report = {
            "resource_usage": {
                "duration_ms": 60_000,
                "cpu_time_seconds": 120.0,
                "memory_rss_kb": 4 * 1024 * 1024,
            },
            "usage_samples": [
                {
                    "sampled_at": "2026-05-30T10:00:00+00:00",
                    "cpu_percent": 80.0,
                    "memory_mb": 4000.0,
                },
                {
                    "sampled_at": "2026-05-30T10:00:05+00:00",
                    "cpu_percent": 92.0,
                    "memory_mb": 4120.0,
                },
            ],
        }
        summary = contract_summary_from_report(report)
        self.assertAlmostEqual(summary["duration_seconds"], 60.0)
        self.assertAlmostEqual(summary["cpu_percent_peak"], 92.0)
        self.assertAlmostEqual(summary["memory_mb_peak"], 4120.0)

    def test_contract_complete_resource_usage_v1_fields(self) -> None:
        report = {
            "resource_usage": {"duration_ms": 1000, "cpu_time_seconds": 2.5},
            "usage_samples": [{"sampled_at": "2026-05-30T10:00:00+00:00", "cpu_percent": 50.0, "memory_mb": 100.0}],
        }
        ru = contract_complete_resource_usage(report)
        self.assertIn("cpu_percent_peak", ru)
        self.assertIn("duration_seconds", ru)
        self.assertIn("duration_ms", ru)


if __name__ == "__main__":
    unittest.main()
