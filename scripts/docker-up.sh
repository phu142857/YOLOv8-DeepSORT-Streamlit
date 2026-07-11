#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
  cp .env.docker.example .env
  echo "Created .env from .env.docker.example"
fi

echo "Pulling MLAir images (${MLAIR_REGISTRY:-ghcr.io/phu142857}/ml-air-*:${MLAIR_TAG:-latest})..."
./scripts/compose pull api scheduler executor frontend realtime postgres redis

echo "Building CV workload image (first run may take several minutes)..."
./scripts/compose build cv-api mlair-cv-train-worker

echo "Fixing MLAir artifact volume ownership (import .pt → model versions)..."
./scripts/compose rm -f mlair-artifact-init 2>/dev/null || true
./scripts/compose run --rm mlair-artifact-init 2>/dev/null || ./scripts/compose up -d mlair-artifact-init

echo "Starting stack..."
./scripts/compose up -d "$@"

if [[ "${CV_SKIP_POST_BOOTSTRAP:-0}" != "1" ]]; then
  echo ""
  echo "Post-start: MLAir permissions + model sync + pipeline..."
  ./scripts/post_stack_bootstrap.sh || {
    echo "Post bootstrap failed (stack is up). Retry: ./scripts/post_stack_bootstrap.sh" >&2
  }
fi

echo ""
echo "  CV UI        http://localhost:${CV_UI_PORT:-8501}"
echo "  CV API       http://localhost:${CV_API_PORT:-8000}/docs"
echo "  MLAir UI     http://localhost:${MLAIR_PORT:-8080}"
echo "  MLAir API    http://localhost:${ML_AIR_API_PORT:-8080}/health"
echo "  MLAir WS     ws://localhost:${MLAIR_PORT:-8080}/ws"
echo "  Grafana      http://localhost:${ML_AIR_GRAFANA_PORT:-33000}"
echo "  Prometheus   http://localhost:${ML_AIR_PROMETHEUS_PORT:-39090}"
