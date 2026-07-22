#!/usr/bin/env bash
# Bootstrap models + pipeline on an EXTERNAL MLAir controller via cv-api on the worker VM.
set -euo pipefail
cd "$(dirname "$0")/.."

[[ -f .env ]] && set -a && source .env && set +a

CV_API_URL="${CV_API_URL:-http://127.0.0.1:${CV_API_PORT:-8000}}"
MLAIR_URL="${MLAIR_CONTROLLER_URL:-${CV_MLAIR_API_URL:-http://192.168.120.182:8080}}"
TENANT="${CV_MLAIR_TENANT:-yolo}"
PROJECT="${CV_MLAIR_PROJECT:-yoloVN}"
TOKEN="${CV_MLAIR_TOKEN:-}"

echo "==> Waiting for cv-api ${CV_API_URL}/health"
for i in $(seq 1 90); do
  curl -sf --max-time 3 "${CV_API_URL}/health" >/dev/null && break
  sleep 2
done
curl -sf "${CV_API_URL}/health" >/dev/null

echo "==> Sync models registry (cv-api → MLAir ${MLAIR_URL})"
curl -sf -X POST "${CV_API_URL}/api/v1/registry/models/sync-full?force=1" | python3 -m json.tool

if [[ -n "$TOKEN" ]]; then
  echo "==> Pipeline bootstrap (tenant=${TENANT} project=${PROJECT})"
  curl -sf -X POST "${CV_API_URL}/api/v1/registry/pipeline/bootstrap" | python3 -m json.tool

  echo "==> Reload plugins on controller"
  curl -sf -X POST "${MLAIR_URL}/v1/plugins/reload" \
    -H "Authorization: Bearer ${TOKEN}" \
    -H 'Content-Type: application/json' \
    -d '{}' && echo "plugins reload ok"
else
  echo "WARN: CV_MLAIR_TOKEN empty — skip pipeline bootstrap / plugin reload" >&2
  echo "      Run: MLAIR_URL=... ./scripts/provision-worker-sa.sh" >&2
fi

echo "==> Done. Hub scope: ${TENANT}/${PROJECT} @ ${MLAIR_URL}"
