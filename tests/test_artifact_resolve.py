"""Tests for MLAir file:// artifact → local weights/detection fallback."""

from __future__ import annotations

import unittest
from pathlib import Path

from mlair_adapter.artifact_resolve import (
    parse_mlair_model_artifact_uri,
    resolve_local_detection_weights,
)


class TestArtifactResolve(unittest.TestCase):
    def test_parse_hub_default_uri(self) -> None:
        uri = "file:///mlair/artifacts/models/yolo/yolovn/yolov8n/v1"
        name, ver = parse_mlair_model_artifact_uri(uri)
        self.assertEqual(name, "yolov8n")
        self.assertEqual(ver, 1)

    def test_resolve_local_base_weights(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "yolov8n" / "base"
            base.mkdir(parents=True)
            pt = base / "weights.pt"
            pt.write_bytes(b"fake")
            found = resolve_local_detection_weights(model_name="yolov8n", detection_root=root)
            self.assertEqual(found, pt)


if __name__ == "__main__":
    unittest.main()
