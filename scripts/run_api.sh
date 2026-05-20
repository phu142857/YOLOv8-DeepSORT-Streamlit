#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
exec uvicorn backend.main:app \
  --host "${CV_API_HOST:-0.0.0.0}" \
  --port "${CV_API_PORT:-8000}" \
  --reload
