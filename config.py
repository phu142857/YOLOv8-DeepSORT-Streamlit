#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     config.py
   @Author:        Luyao.zhang
   @Date:          2023/5/16
   @Description: configuration file
-------------------------------------------------
"""
from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())


# Source
SOURCES_LIST = ["Image", "Video", "Webcam"]


# DL model config — layout: weights/detection/{model}/{version}/weights.pt
DETECTION_MODEL_DIR = ROOT / "weights" / "detection"
DEFAULT_MODEL_SPEC = "yolov8n/base"

# Legacy flat paths (optional); prefer model/version folders.
YOLOv8n = DETECTION_MODEL_DIR / "yolov8n" / "base" / "weights.pt"
YOLOv8x = DETECTION_MODEL_DIR / "yolov8x" / "base" / "weights.pt"

# Fallback list when API discovery is unavailable (UI only).
DETECTION_MODEL_LIST = [
    "yolov8n/base",
    "yolov8s/base",
    "yolov8m/base",
    "yolov8l/base",
    "yolov8x/base",
]


OBJECT_COUNTER = None
OBJECT_COUNTER1 = None