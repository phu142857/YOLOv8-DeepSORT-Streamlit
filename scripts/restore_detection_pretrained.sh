#!/usr/bin/env bash
# Restore official Ultralytics COCO weights into weights/detection/{model}/pretrained/ (+ base/).
# Inference prefers pretrained/ so Hub sync to base/ does not break Vehicle Detection.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:-yolov8n}"
CONTAINER="${CV_API_CONTAINER:-cv-lifecycle-api}"

_restore_in_python() {
  local model="$1"
  local base_dir="$2"
  python3 - "$model" "$base_dir" <<'PY'
import sys
from pathlib import Path
from ultralytics import YOLO

model, base_dir = sys.argv[1], Path(sys.argv[2])
base = base_dir / model / "base" / "weights.pt"
pretrained = base_dir / model / "pretrained" / "weights.pt"
pretrained.parent.mkdir(parents=True, exist_ok=True)
base.parent.mkdir(parents=True, exist_ok=True)
m = YOLO(f"{model}.pt")
data = Path(m.ckpt_path).read_bytes()
pretrained.write_bytes(data)
base.write_bytes(data)
m2 = YOLO(str(pretrained))
print(f"Restored {model} COCO → {pretrained} (+ {base}), classes={len(m2.names)}")
PY
}

if python3 -c "import torch" 2>/dev/null; then
  _restore_in_python "$MODEL" "$ROOT/weights/detection"
elif docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER"; then
  echo "Using container $CONTAINER"
  docker exec -w /app "$CONTAINER" python3 -c "
from pathlib import Path
from ultralytics import YOLO
model = '${MODEL}'
pre = Path(f'/app/weights/detection/{model}/pretrained/weights.pt')
base = Path(f'/app/weights/detection/{model}/base/weights.pt')
pre.parent.mkdir(parents=True, exist_ok=True)
m = YOLO(f'{model}.pt')
data = Path(m.ckpt_path).read_bytes()
pre.write_bytes(data)
base.write_bytes(data)
print(f'Restored {model}, classes={len(YOLO(str(pre)).names)}')
"
else
  echo "ERROR: need torch locally or running $CONTAINER" >&2
  exit 1
fi

echo "Run a new Execution (select yolov8s/base or yolov8n after restore)."
