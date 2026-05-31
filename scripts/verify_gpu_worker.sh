#!/usr/bin/env bash
# Verify CV train worker sees CUDA (default stack via compose.yaml).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

COMPOSE=(./scripts/compose)

echo "=== Host GPU ==="
nvidia-smi -L 2>/dev/null || echo "WARN: nvidia-smi not on host"

echo ""
echo "=== Worker container state ==="
WORKER_PS=$("${COMPOSE[@]}" ps mlair-cv-train-worker 2>&1)
echo "$WORKER_PS" | tail -3

if echo "$WORKER_PS" | grep -qE 'cv-lifecycle-workload:latest[^-]|:latest[[:space:]]'; then
  echo ""
  echo "ERROR: worker is on CPU image (cv-lifecycle-workload:latest)."
  echo "  Expected cv-lifecycle-workload:gpu — rebuild stack:"
  echo "    docker compose build mlair-cv-train-worker"
  echo "    docker compose up -d --no-deps mlair-cv-train-worker"
  exit 1
fi

if ! echo "$WORKER_PS" | grep -qE 'Up|running'; then
  echo "ERROR: worker not running — check: ./scripts/compose logs mlair-cv-train-worker"
  exit 1
fi

echo ""
echo "=== CUDA check (in worker) ==="
"${COMPOSE[@]}" exec -T mlair-cv-train-worker python -c "
import torch
from mlair_adapter.train_device import resolve_train_device
print('torch_version', torch.__version__)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu_name', torch.cuda.get_device_name(0))
    x = torch.randn(2, 3, device='cuda')
    print('cuda_tensor_ok', str(x.device))
print('resolve_train_device', resolve_train_device())
if not torch.cuda.is_available():
    raise SystemExit(1)
"

echo ""
echo "PASS: cuda_available True"
