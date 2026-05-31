import unittest
from unittest import mock

from mlair_adapter import train_device as td


class TrainDeviceTest(unittest.TestCase):
    def setUp(self) -> None:
        td.reset_gpu_train_failures()

    def test_auto_returns_cpu_or_cuda(self) -> None:
        dev = td.resolve_train_device()
        self.assertIn(dev, ("cpu", "mps", 0, "0"))

    def test_explicit_cpu(self) -> None:
        import os

        old = os.environ.get("CV_MLAIR_TRAIN_DEVICE")
        os.environ["CV_MLAIR_TRAIN_DEVICE"] = "cpu"
        try:
            self.assertEqual(td.resolve_train_device(), "cpu")
        finally:
            if old is None:
                os.environ.pop("CV_MLAIR_TRAIN_DEVICE", None)
            else:
                os.environ["CV_MLAIR_TRAIN_DEVICE"] = old

    def test_pick_train_device_uses_cpu_after_streak(self) -> None:
        with mock.patch.object(td, "gpu_fail_streak_threshold", return_value=3):
            for _ in range(3):
                td.record_gpu_train_failure()
            self.assertEqual(td.pick_lifecycle_train_device(), "cpu")

    def test_gpu_failure_increments_streak(self) -> None:
        td.record_gpu_train_failure()
        td.record_gpu_train_failure()
        self.assertEqual(td.consecutive_gpu_train_failures(), 2)

    def test_run_train_retries_cpu_after_threshold(self) -> None:
        calls: list[str | int] = []

        class FakeModel:
            def train(self, *, device, **kwargs):
                calls.append(device)
                if device != "cpu":
                    raise RuntimeError("CUDA OOM")
                return "ok"

        with mock.patch.object(td, "gpu_fail_streak_threshold", return_value=2):
            td.record_gpu_train_failure()
            with mock.patch.object(td, "pick_lifecycle_train_device", return_value=0):
                results, device = td.run_ultralytics_train_with_device_policy(
                    FakeModel(),
                    data="x.yaml",
                )
        self.assertEqual(results, "ok")
        self.assertEqual(device, "cpu")
        self.assertEqual(calls, [0, "cpu"])


if __name__ == "__main__":
    unittest.main()
