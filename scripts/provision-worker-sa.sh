#!/usr/bin/env bash
# Create a Service Account for external CV worker lease (correct tenant/project scope).
# Run on a machine that can reach MLAir Hub/API (controller or laptop).
set -euo pipefail

MLAIR_URL="${MLAIR_URL:-http://127.0.0.1:8080}"
ADMIN_USER="${MLAIR_ADMIN_USER:-admin}"
ADMIN_PASS="${MLAIR_ADMIN_PASSWORD:-admin-change-me}"
TENANT="${CV_MLAIR_TENANT:-yolo}"
PROJECT="${CV_MLAIR_PROJECT:-yoloVN}"
SA_NAME="${MLAIR_WORKER_SA_NAME:-cv-yolo-worker}"
OUT_FILE="${1:-}"

echo "==> MLAir: ${MLAIR_URL} scope=${TENANT}/${PROJECT}"

TOKEN=$(curl -sf -X POST "${MLAIR_URL}/v1/auth/login" \
  -H 'Content-Type: application/json' \
  -d "{\"username\":\"${ADMIN_USER}\",\"password\":\"${ADMIN_PASS}\"}" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')

SA_ID=$(curl -sf "${MLAIR_URL}/v1/service-accounts" \
  -H "Authorization: Bearer ${TOKEN}" \
  | python3 -c "
import json,sys,os
name=os.environ['SA_NAME']
for row in json.load(sys.stdin).get('items',[]):
    if row.get('name')==name:
        print(row['id']); break
" SA_NAME="${SA_NAME}" 2>/dev/null || true)

if [[ -z "${SA_ID:-}" ]]; then
  SA_ID=$(curl -sf -X POST "${MLAIR_URL}/v1/service-accounts" \
    -H "Authorization: Bearer ${TOKEN}" \
    -H 'Content-Type: application/json' \
    -d "{\"name\":\"${SA_NAME}\",\"description\":\"YOLO external worker\"}" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
  echo "Created SA ${SA_NAME} id=${SA_ID}"
else
  echo "Reusing SA ${SA_NAME} id=${SA_ID}"
fi

curl -sf -X PUT "${MLAIR_URL}/v1/service-accounts/${SA_ID}/permissions" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H 'Content-Type: application/json' \
  -d '{"permissions":["tasks:lease","tasks:heartbeat","tasks:complete","tasks:fail","logs:write","metrics:write","artifacts:write","usage:write"]}' \
  >/dev/null

# Idempotent: add scope if missing (list then POST only when needed)
HAS_SCOPE=$(curl -sf "${MLAIR_URL}/v1/service-accounts/${SA_ID}/scopes" \
  -H "Authorization: Bearer ${TOKEN}" \
  | python3 -c "
import json,sys,os
t,p=os.environ['TENANT'],os.environ['PROJECT']
for s in json.load(sys.stdin).get('items',[]):
    if s.get('tenant_id')!=t: continue
    if s.get('all_projects'): print('yes'); break
    if p in (s.get('project_ids') or []): print('yes'); break
" TENANT="${TENANT}" PROJECT="${PROJECT}" 2>/dev/null || true)

if [[ "${HAS_SCOPE:-}" != "yes" ]]; then
  curl -sf -X POST "${MLAIR_URL}/v1/service-accounts/${SA_ID}/scopes" \
    -H "Authorization: Bearer ${TOKEN}" \
    -H 'Content-Type: application/json' \
    -d "{\"tenant_id\":\"${TENANT}\",\"all_projects\":false,\"project_ids\":[\"${PROJECT}\"]}" \
    >/dev/null
  echo "Added scope ${TENANT}/${PROJECT}"
fi

SECRET=$(curl -sf -X POST "${MLAIR_URL}/v1/service-accounts/${SA_ID}/issue-secret" \
  -H "Authorization: Bearer ${TOKEN}" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["secret"])')

if [[ -n "${OUT_FILE}" ]]; then
  printf 'CV_MLAIR_TOKEN=%s\n' "${SECRET}" >"${OUT_FILE}"
  chmod 0600 "${OUT_FILE}"
  echo "Wrote ${OUT_FILE}"
else
  echo "CV_MLAIR_TOKEN=${SECRET}"
fi

echo "==> Set CV_MLAIR_TOKEN in worker .env and recreate mlair-cv-train-worker"
