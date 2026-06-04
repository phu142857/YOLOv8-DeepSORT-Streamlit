"""Worker context uses train-ready version after detect publish."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mlair_adapter.run_workspace import save_state
from mlair_adapter.worker_context import plugin_context_from_lease_task
class WorkerContextDetectTest(unittest.TestCase):
    def test_train_version_overrides_lease_pin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            import mlair_adapter.run_workspace as rw

            old_root = rw.settings.artifact_root
            object.__setattr__(rw.settings, "artifact_root", root)

            save_state(
                "run-1",
                {
                    "detect_publish_ok": True,
                    "source_dataset_version_id": "input-v",
                    "train_dataset_version_id": "train-v",
                    "dataset_version_id": "train-v",
                },
            )
            task = {
                "task_id": "t1",
                "run_id": "run-1",
                "plugin": "cv_yolo_prepare",
                "payload": {
                    "context": {
                        "dataset_version_id": "input-v",
                        "model_id": "model-1",
                    }
                },
            }
            ctx = plugin_context_from_lease_task(task)
            self.assertEqual(ctx["dataset_version_id"], "train-v")
            self.assertEqual(ctx["source_dataset_version_id"], "input-v")
            object.__setattr__(rw.settings, "artifact_root", old_root)

    def test_merge_ok_overrides_for_prepare(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            import mlair_adapter.run_workspace as rw

            old_root = rw.settings.artifact_root
            object.__setattr__(rw.settings, "artifact_root", root)

            save_state(
                "run-2",
                {
                    "merge_ok": True,
                    "source_dataset_version_id": "input-v",
                    "train_dataset_version_id": "train-v",
                },
            )
            task = {
                "task_id": "t2",
                "run_id": "run-2",
                "plugin": "cv_yolo_prepare",
                "payload": {"context": {"dataset_version_id": "input-v", "model_id": "m"}},
            }
            ctx = plugin_context_from_lease_task(task)
            self.assertEqual(ctx["dataset_version_id"], "train-v")
            object.__setattr__(rw.settings, "artifact_root", old_root)


if __name__ == "__main__":
    unittest.main()
