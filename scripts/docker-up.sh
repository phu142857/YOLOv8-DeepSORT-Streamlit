#!/usr/bin/env bash
# Deprecated — use: docker compose build && docker compose up -d
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -f .env ]]; then
  cp .env.docker.example .env
  echo "Created .env from .env.docker.example"
fi
exec docker compose up -d --build "$@"
