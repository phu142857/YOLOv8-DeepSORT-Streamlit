#!/usr/bin/env bash
# Build CV image, start full stack, auto-fix MLAir perms + sync models + pipeline.
set -euo pipefail
cd "$(dirname "$0")/.."

ARGS=("$@")
if [[ ${#ARGS[@]} -eq 0 ]]; then
  ARGS=(-d)
fi

docker compose build cv-api cv-ui
./scripts/docker-up.sh "${ARGS[@]}"
./scripts/post_stack_bootstrap.sh
