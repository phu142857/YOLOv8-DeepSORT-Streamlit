"""Tests for detect split → Hub publish → train-ready merge helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from mlair_adapter.detect_dataset_publish import (
    build_detect_task_lineage,
    filter_rows_with_job_id,
    merge_train_manifest_rows,
    publish_detect_split_and_train_ready,
    write_manifest_csv,
)
from mlair_adapter.job_detections import write_frame_detection
from shared.artifacts import ArtifactStore


class DetectDatasetPublishTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = Path(tempfile.mkdtemp())

    def test_filter_rows_with_job_id(self) -> None:
        rows = [
            {"job_id": "a", "frame_index": "0"},
            {"job_id": "", "frame_index": "1"},
        ]
        out = filter_rows_with_job_id(rows)
        self.assertEqual(len(out), 1)

    def test_merge_train_manifest_rows(self) -> None:
        store = ArtifactStore(root=self._tmp / "artifacts")
        job_id = "job-merge"
        write_frame_detection(
            job_id,
            "000000",
            [{"class_name": "car", "confidence": 0.9, "xyxy": [1, 2, 3, 4]}],
            store=store,
        )
        write_frame_detection(
            job_id,
            "000001",
            [{"class_name": "bus", "confidence": 0.8, "xyxy": [5, 6, 7, 8]}],
            store=store,
        )
        labeled = [{"job_id": job_id, "frame_index": "000000", "image_uri": "http://x/0.jpg"}]
        missing = [{"job_id": job_id, "frame_index": "000001", "image_uri": "http://x/1.jpg"}]
        merged = merge_train_manifest_rows(labeled, missing, store=store)
        self.assertEqual(len(merged), 2)

    def test_write_manifest_csv_columns(self) -> None:
        path = self._tmp / "m.csv"
        n = write_manifest_csv(
            [{"job_id": "j", "frame_index": "1", "image_uri": "u", "source_file": "", "artifact_path": "1.jpg"}],
            path,
        )
        self.assertEqual(n, 1)
        text = path.read_text(encoding="utf-8")
        self.assertIn("image_uri", text)
        self.assertIn("job_id", text)

    def test_build_detect_task_lineage(self) -> None:
        lin = build_detect_task_lineage(
            source_version_id="src-v",
            model_id="model-1",
            run_id="run-1",
            detected_version_id="det-v",
            not_detected_version_id="nd-v",
            train_version_id="train-v",
        )
        self.assertEqual(len(lin["inputs"]), 2)
        self.assertGreaterEqual(len(lin["outputs"]), 3)
        names = {o["name"] for o in lin["outputs"]}
        self.assertIn("dataset_version_train_ready", names)

    def test_publish_detect_split_and_train_ready(self) -> None:
        client = MagicMock()
        client.upload_manifest_csv.side_effect = [
            {"dataset_id": "ds1", "dataset_version_id": "v-detected"},
            {"dataset_id": "ds2", "dataset_version_id": "v-not"},
            {"dataset_id": "ds3", "dataset_version_id": "v-train"},
        ]
        row = {
            "job_id": "j",
            "frame_index": "0",
            "image_uri": "http://x/0.jpg",
            "source_file": "",
            "artifact_path": "0.jpg",
        }
        out = publish_detect_split_and_train_ready(
            client,
            source_version_id="src",
            labeled_rows=[row],
            missing_rows=[],
            train_rows=[row],
            run_id="run-abc",
            model_id="m1",
        )
        self.assertEqual(out["train_dataset_version_id"], "v-train")
        self.assertEqual(client.upload_manifest_csv.call_count, 2)
        self.assertIsNotNone(out.get("lineage"))


if __name__ == "__main__":
    unittest.main()
