#!/usr/bin/env bash
# After compose up / build: fix MLAir artifact perms, full model sync, pipeline bootstrap.
set -euo pipefail
cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
[[ -f .env ]] && set -a && source .env && set +a

CV_API_PORT="${CV_API_PORT:-8000}"
ML_AIR_API_PORT="${ML_AIR_API_PORT:-8080}"
CV_API_URL="${CV_API_URL:-http://127.0.0.1:${CV_API_PORT}}"
MLAIR_API_CONTAINER="${MLAIR_API_CONTAINER:-ml-air-api}"
MAX_WAIT="${BOOTSTRAP_MAX_WAIT_SEC:-180}"

wait_http() {
  local url="$1"
  local label="$2"
  local i=0
  while [[ "$i" -lt "$MAX_WAIT" ]]; do
    if curl -sf --max-time 3 "$url" >/dev/null 2>&1; then
      echo "  OK $label"
      return 0
    fi
    sleep 2
    i=$((i + 2))
  done
  echo "Timeout waiting for $label ($url)" >&2
  return 1
}

echo "==> Waiting for MLAir API..."
wait_http "http://127.0.0.1:${ML_AIR_API_PORT}/health" "ml-air-api"

echo "==> Fixing artifact volume permissions (model versions / .pt import)..."
MLAIR_API_CONTAINER="$MLAIR_API_CONTAINER" ./scripts/fix_mlair_model_artifacts_perm.sh

echo "==> Waiting for CV API..."
wait_http "${CV_API_URL}/health" "cv-api"

echo "==> MLAir registry core (models + pipeline)..."
SYNC_JSON=$(curl -sf -X POST "${CV_API_URL}/api/v1/registry/models/sync-full?force=1")
echo "$SYNC_JSON" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('  pushed:', d.get('pushed'), 'pulled:', d.get('pulled'), 'pruned:', d.get('state_pruned'))
fails = [r for r in (d.get('push_results') or []) if r.get('ok') is False]
if fails:
    print('  push errors:', fails[0].get('error', fails[0])[:200])
    sys.exit(1)
" || {
  echo "Model sync failed — check: docker logs ml-air-api | tail -30" >&2
  exit 1
}

if curl -sf "${CV_API_URL}/api/v1/runtime" | python3 -c "import json,sys; d=json.load(sys.stdin); exit(0 if d.get('mlair_configured') else 1)" 2>/dev/null; then
  echo "==> Pipeline mapping (Hub Train)..."
  curl -sf -X POST "${CV_API_URL}/api/v1/registry/pipeline/bootstrap" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('  pipeline:', d.get('pipeline_id'), 'skipped:', d.get('skipped', False))
" || echo "  (pipeline bootstrap skipped — non-fatal)"
fi

echo "==> Done. Hub: http://localhost:${ML_AIR_FRONTEND_PORT:-38080}"
