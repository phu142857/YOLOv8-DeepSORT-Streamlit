#!/usr/bin/env bash
# After compose up / build: fix MLAir artifact perms, full model sync, pipeline bootstrap.
set -euo pipefail
cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
[[ -f .env ]] && set -a && source .env && set +a

CV_API_PORT="${CV_API_PORT:-8000}"
ML_AIR_API_PORT="${ML_AIR_API_PORT:-8080}"
CV_API_URL="${CV_API_URL:-http://127.0.0.1:${CV_API_PORT}}"
MLAIR_API_CONTAINER="${MLAIR_API_CONTAINER:-mlair}"
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

if ! docker exec "$MLAIR_API_CONTAINER" python -c \
  "from app.domains.orchestration import worker_task_service as w; assert hasattr(w,'_persist_run_plugin_tracking')" \
  2>/dev/null; then
  echo "WARN: ml-air:latest image lacks tracking persist (Hub Metrics/Artifacts will be empty)." >&2
  echo "  Rebuild/replace the ml-air:latest image (owned by the ml-air project), then: docker compose up -d --force-recreate mlair" >&2
fi

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
fails = [r for r in (d.get('push_results') or []) if r.get('ok') is False and not r.get('skipped')]
skipped = sum(1 for r in (d.get('push_results') or []) if r.get('skipped'))
if skipped:
    print('  push skipped (already aligned):', skipped)
if fails:
    print('  push errors:', fails[0].get('error', fails[0])[:200])
    sys.exit(1)
if int(d.get('pushed') or 0) == 0 and int(d.get('pulled') or 0) == 0 and fails:
    sys.exit(1)
" || {
  echo "Model sync failed — check: docker logs ml-air-api | tail -30" >&2
  exit 1
}

if curl -sf "${CV_API_URL}/api/v1/runtime" | python3 -c "import json,sys; d=json.load(sys.stdin); exit(0 if d.get('mlair_configured') else 1)" 2>/dev/null; then
  echo "==> Pipeline + map all models (Hub Train with model)..."
  curl -sf -X POST "${CV_API_URL}/api/v1/registry/pipeline/bootstrap" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('  pipeline:', d.get('pipeline_id'), 'mode:', d.get('mode'), 'skipped:', d.get('skipped', False), 'republished:', d.get('republished', False))
" || echo "  (pipeline bootstrap skipped — non-fatal)"
  TOKEN="${CV_MLAIR_TOKEN:-${ML_AIR_TRACKING_TOKEN:-admin-token}}"
  curl -sf -X POST "http://127.0.0.1:${ML_AIR_API_PORT}/v1/tenants/${CV_MLAIR_TENANT:-default}/projects/${CV_MLAIR_PROJECT:-default_project}/plugins/reload" \
    -H "Authorization: Bearer ${TOKEN}" -H "Content-Type: application/json" -d '{}' >/dev/null 2>&1 \
    && echo "  plugins: reload ok" || echo "  plugins: reload skipped (build api with cv plugins)"
fi

echo "==> Done. Hub: http://localhost:${MLAIR_PORT:-8080}"
