#!/usr/bin/env bash
# End-to-end smoke: CV API upload → pipeline → lifecycle timeline → optional MLAir checks.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

API_URL="${CV_API_BASE_URL:-http://127.0.0.1:8000}"
MODEL="${CV_DEFAULT_MODEL:-yolov8n.pt}"

echo "== CV Lifecycle E2E demo =="
echo "API: $API_URL"

curl -sf "$API_URL/health" | python3 -m json.tool
echo ""

SAMPLE=""
for candidate in \
  "$ROOT/artifacts/uploads"/*/*.mp4 \
  "$ROOT/artifacts/uploads"/*/*.jpg \
  "$ROOT/demo/sample.mp4"
do
  if [[ -f "$candidate" ]]; then
    SAMPLE="$candidate"
    break
  fi
done

if [[ -z "$SAMPLE" ]]; then
  echo "No sample media under artifacts/uploads/ — upload a short mp4 first, then re-run."
  exit 1
fi

echo "Using sample: $SAMPLE"

UPLOAD_JSON=$(curl -sf -F "file=@${SAMPLE}" "$API_URL/api/v1/uploads")
UPLOAD_ID=$(echo "$UPLOAD_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['upload_id'])")
echo "Upload id: $UPLOAD_ID"

EXT="${SAMPLE##*.}"
SOURCE="video"
[[ "$EXT" =~ ^(jpg|jpeg|png)$ ]] && SOURCE="image"

JOB_JSON=$(curl -sf -X POST "$API_URL/api/v1/jobs" \
  -H "Content-Type: application/json" \
  -d "{\"source_type\":\"$SOURCE\",\"model_name\":\"$MODEL\",\"confidence\":0.5,\"upload_id\":\"$UPLOAD_ID\"}")
JOB_ID=$(echo "$JOB_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
echo "Job id: $JOB_ID"

curl -sf -X POST "$API_URL/api/v1/jobs/$JOB_ID/start" >/dev/null

echo "Polling job status…"
for _ in $(seq 1 600); do
  STATUS=$(curl -sf "$API_URL/api/v1/jobs/$JOB_ID" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
  echo "  status=$STATUS"
  case "$STATUS" in
    completed|failed|cancelled) break ;;
  esac
  sleep 2
done

echo ""
echo "== Job result =="
curl -sf "$API_URL/api/v1/jobs/$JOB_ID" | python3 -m json.tool

echo ""
echo "== Lifecycle timeline =="
curl -sf "$API_URL/api/v1/jobs/$JOB_ID/lifecycle" | python3 -m json.tool

if curl -sf "$API_URL/api/v1/mlair/status" | python3 -c "import sys,json; exit(0 if json.load(sys.stdin).get('configured') else 1)" 2>/dev/null; then
  echo ""
  echo "== MLAir status =="
  curl -sf "$API_URL/api/v1/mlair/status" | python3 -m json.tool
  DS=$(curl -sf "$API_URL/api/v1/mlair/datasets" | python3 -c "
import sys,json
items=json.load(sys.stdin).get('items',[])
print(items[0]['dataset_id'] if items else '')
" 2>/dev/null || true)
  if [[ -n "${DS:-}" ]]; then
    echo "Buffer for dataset $DS:"
    curl -sf "$API_URL/api/v1/mlair/datasets/$DS/buffer" | python3 -m json.tool || true
  fi
fi

echo ""
echo "Done. Job $JOB_ID"
