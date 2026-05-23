#!/usr/bin/env bash
# Build all MLAir :local images from ml-air, then CV API overlay (ml-air-api:cv-workload).
#
# Same tags as manual build in ml-air repo:
#   docker build -t ml-air-api:local -f api/Dockerfile .
#   docker build -t ml-air-scheduler:local -f scheduler/Dockerfile .
#   …
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MLAIR="${ML_AIR_SRC:-$ROOT/../ml-air}"

if [[ ! -f "$MLAIR/api/Dockerfile" ]]; then
  echo "ERROR: ml-air repo not found at: $MLAIR" >&2
  echo "Set ML_AIR_SRC to your ml-air checkout." >&2
  exit 1
fi

cd "$MLAIR"

echo "==> ml-air-api:local"
docker build -t ml-air-api:local -f api/Dockerfile .

echo "==> ml-air-scheduler:local"
docker build -t ml-air-scheduler:local -f scheduler/Dockerfile .

echo "==> ml-air-executor:local"
docker build -t ml-air-executor:local -f executor/Dockerfile .

echo "==> ml-air-frontend:local"
docker build -t ml-air-frontend:local -f frontend/Dockerfile .

echo "==> ml-air-realtime:local"
docker build -t ml-air-realtime:local -f realtime/Dockerfile .

export MLAIR_API_IMAGE=ml-air-api:local

cd "$ROOT"
echo "==> ml-air-api:cv-workload (CV plugins on ml-air-api:local)"
docker compose build api

echo "==> cv-lifecycle-workload (cv-api + train worker)"
docker compose build cv-api mlair-cv-train-worker

echo ""
echo "Done. Recreate MLAir + CV worker:"
echo "  docker compose up -d --force-recreate api scheduler executor realtime frontend mlair-cv-train-worker"
echo ""
echo "Verify tracking persist in running API:"
echo "  docker exec ml-air-api python -c \"from app.domains.orchestration import worker_task_service as w; print('tracking_ok', hasattr(w,'_persist_run_plugin_tracking'))\""
