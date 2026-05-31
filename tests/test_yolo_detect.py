"""Unit tests for incremental lifecycle detect (skip + artifact merge)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mlair_adapter.job_detections import load_job_detections, write_frame_detection
from mlair_adapter.yolo_detect import scan_manifest_gaps
from shared.artifacts import ArtifactStore


class YoloDetectTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = Path(tempfile.mkdtemp())

    def test_scan_manifest_gaps_skips_labeled_frames(self) -> None:
        store = ArtifactStore(root=self._tmp / "artifacts")
        job_id = "job-a"
        layout = store.job_layout(job_id)
        store.write_json(
            layout["detections"] / "detections.json",
            {
                "frame": 0,
                "frame_index": "000000",
                "detections": [
                    {
                        "class_id": 2,
                        "class_name": "car",
                        "confidence": 0.9,
                        "xyxy": [10.0, 10.0, 50.0, 50.0],
                    }
                ],
            },
        )

        rows = [
            {"frame_index": "000000", "job_id": job_id, "image_uri": "http://x/0.jpg"},
            {"frame_index": "000001", "job_id": job_id, "image_uri": "http://x/1.jpg"},
        ]
        labeled, missing, _ = scan_manifest_gaps(rows, store=store)
        self.assertEqual(len(labeled), 1)
        self.assertEqual(labeled[0]["frame_index"], "000000")
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0]["frame_index"], "000001")

    def test_write_frame_detection_merges_jsonl(self) -> None:
        store = ArtifactStore(root=self._tmp / "artifacts2")
        job_id = "zip-import"
        write_frame_detection(
            job_id,
            "000000",
            [{"class_name": "car", "xyxy": [1, 2, 3, 4]}],
            store=store,
        )
        write_frame_detection(
            job_id,
            "000001",
            [{"class_name": "truck", "xyxy": [5, 6, 7, 8]}],
            store=store,
        )

        det_path = store.job_layout(job_id)["detections"] / "detections.jsonl"
        self.assertTrue(det_path.is_file())
        loaded = load_job_detections(job_id, store)
        self.assertIn("000000", loaded)
        self.assertIn("000001", loaded)
        self.assertEqual(loaded["000000"][0]["class_name"], "car")

    def test_write_frame_detection_updates_single_json(self) -> None:
        store = ArtifactStore(root=self._tmp / "artifacts3")
        job_id = "exec-job"
        layout = store.job_layout(job_id)
        store.write_json(
            layout["detections"] / "detections.json",
            {"frame": 0, "detections": []},
        )

        write_frame_detection(
            job_id,
            "000000",
            [{"class_name": "bus", "xyxy": [0, 0, 10, 10]}],
            store=store,
        )

        payload = json.loads((layout["detections"] / "detections.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["detections"][0]["class_name"], "bus")


if __name__ == "__main__":
    unittest.main()
