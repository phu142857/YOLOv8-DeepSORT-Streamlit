#!/bin/sh
# One-shot: login to MLAir Identity and write a PAT for cv-api + worker.
set -eu

SECRETS_DIR="${SECRETS_DIR:-/run/stack-auth}"
TOKEN_FILE="${SECRETS_DIR}/CV_MLAIR_TOKEN"
MLAIR_URL="${MLAIR_URL:-http://mlair:8080}"
ADMIN_USER="${MLAIR_ADMIN_USER:-admin}"
ADMIN_PASS="${MLAIR_ADMIN_PASSWORD:-admin-change-me}"
MAX_WAIT="${STACK_AUTH_MAX_WAIT_SEC:-180}"

mkdir -p "$SECRETS_DIR"

if [ -s "$TOKEN_FILE" ]; then
  echo "[stack-auth] Reusing existing token at $TOKEN_FILE"
  exit 0
fi

apk add --no-cache curl python3 >/dev/null 2>&1

echo "[stack-auth] Waiting for MLAir at ${MLAIR_URL}/health ..."
i=0
while [ "$i" -lt "$MAX_WAIT" ]; do
  if curl -sf --max-time 3 "${MLAIR_URL}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 2
  i=$((i + 2))
done
if ! curl -sf --max-time 3 "${MLAIR_URL}/health" >/dev/null 2>&1; then
  echo "[stack-auth] MLAir API not ready" >&2
  exit 1
fi

echo "[stack-auth] Creating Hub PAT for CV stack..."
ACCESS=$(curl -sf -X POST "${MLAIR_URL}/v1/auth/login" \
  -H 'Content-Type: application/json' \
  -d "{\"username\":\"${ADMIN_USER}\",\"password\":\"${ADMIN_PASS}\"}" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')

PAT=$(curl -sf -X POST "${MLAIR_URL}/v1/auth/pats" \
  -H "Authorization: Bearer ${ACCESS}" \
  -H 'Content-Type: application/json' \
  -d '{"description":"yolo-cv-stack","expires_in_days":365}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["token"])')

printf '%s' "$PAT" >"$TOKEN_FILE"
chmod 0444 "$TOKEN_FILE"
echo "[stack-auth] Wrote ${TOKEN_FILE}"
