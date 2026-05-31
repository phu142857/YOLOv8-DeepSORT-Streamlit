import unittest

from mlair_adapter.train_device import resolve_train_device


class TrainDeviceTest(unittest.TestCase):
    def test_auto_returns_cpu_or_cuda(self) -> None:
        dev = resolve_train_device()
        self.assertIn(dev, ("cpu", "mps", 0, "0"))

    def test_explicit_cpu(self) -> None:
        import os

        old = os.environ.get("CV_MLAIR_TRAIN_DEVICE")
        os.environ["CV_MLAIR_TRAIN_DEVICE"] = "cpu"
        try:
            self.assertEqual(resolve_train_device(), "cpu")
        finally:
            if old is None:
                os.environ.pop("CV_MLAIR_TRAIN_DEVICE", None)
            else:
                os.environ["CV_MLAIR_TRAIN_DEVICE"] = old


if __name__ == "__main__":
    unittest.main()
