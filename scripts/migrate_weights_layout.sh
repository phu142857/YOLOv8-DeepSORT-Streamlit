#!/usr/bin/env bash
# Move flat weights/detection/*.pt → weights/detection/{model}/base/weights.pt
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DET="${ROOT}/weights/detection"
mkdir -p "$DET"

shopt -s nullglob
for f in "$DET"/*.pt; do
  base="$(basename "$f")"
  model="${base%.pt}"
  dest_dir="${DET}/${model}/base"
  mkdir -p "$dest_dir"
  dest="${dest_dir}/weights.pt"
  if [[ -f "$dest" ]]; then
    echo "skip (exists): $dest"
    continue
  fi
  mv "$f" "$dest"
  echo "migrated: $f -> $dest"
done

echo "Done. Models:"
find "$DET" -name '*.pt' | sort
