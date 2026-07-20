#!/usr/bin/env bash
# Create a Hub PAT for the YOLO cv-api + worker (external controller).
# Run on a machine that can reach MLAir (laptop or controller VM).
set -euo pipefail

MLAIR_URL="${MLAIR_URL:-http://192.168.120.182:8080}"
ADMIN_USER="${MLAIR_ADMIN_USER:-admin}"
ADMIN_PASS="${MLAIR_ADMIN_PASSWORD:-admin-change-me}"
OUT_FILE="${1:-}"

echo "==> MLAir: ${MLAIR_URL}"
curl -sf --max-time 5 "${MLAIR_URL}/health" >/dev/null || {
  echo "MLAir not reachable at ${MLAIR_URL}/health" >&2
  exit 1
}

ACCESS=$(curl -sf -X POST "${MLAIR_URL}/v1/auth/login" \
  -H 'Content-Type: application/json' \
  -d "{\"username\":\"${ADMIN_USER}\",\"password\":\"${ADMIN_PASS}\"}" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')

PAT=$(curl -sf -X POST "${MLAIR_URL}/v1/auth/pats" \
  -H "Authorization: Bearer ${ACCESS}" \
  -H 'Content-Type: application/json' \
  -d '{"description":"yolo-cv-worker","expires_in_days":365}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["token"])')

if [[ -n "$OUT_FILE" ]]; then
  printf '%s' "$PAT" >"$OUT_FILE"
  chmod 0600 "$OUT_FILE"
  echo "Wrote token to ${OUT_FILE}"
else
  echo "CV_MLAIR_TOKEN=${PAT}"
fi
