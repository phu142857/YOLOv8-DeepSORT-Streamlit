#!/usr/bin/env bash
# CV + MLAir AWS E2E workflow check — colored status table (Vet-AI style).
#
# Usage:
#   ./test_workflow.sh [dev]
#   ./test_workflow.sh dev --list-only   # URLs only, no HTTP checks
#   SKIP_ENV_URL_SYNC=1 ./test_workflow.sh dev
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV="dev"
LIST_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --list-only) LIST_ONLY=1; shift ;;
    -*) echo "Unknown option: $1" >&2; exit 1 ;;
    *) ENV="$1"; shift ;;
  esac
done

# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

DEPLOY_STATE="$(deploy_state_file "$ENV")"
if [[ -f "$DEPLOY_STATE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$DEPLOY_STATE"
  set +a
elif terraform -chdir="$(tf_dir)" output alb_dns_name >/dev/null 2>&1; then
  ALB_DNS_NAME="$(tf_output alb_dns_name "$ENV")"
  ALB_BASE_URL="http://${ALB_DNS_NAME}"
  APP_PRIVATE_IP="$(tf_output app_private_ip "$ENV")"
  APP_INSTANCE_ID="$(tf_output app_instance_id "$ENV")"
  AWS_REGION="${AWS_REGION:-ap-southeast-1}"
else
  echo "ERROR: no deploy state ($DEPLOY_STATE) and no terraform outputs."
  echo "Run: ./deploy.sh $ENV"
  exit 1
fi

ALB_DNS_NAME="${ALB_DNS_NAME:-$(tf_output alb_dns_name "$ENV")}"
ALB_BASE_URL="${ALB_BASE_URL:-http://${ALB_DNS_NAME}}"
BASE_URL="${ALB_BASE_URL%/}"

ENABLE_MLAIR="${ENABLE_MLAIR:-true}"
ENABLE_CV="${ENABLE_CV:-true}"

# Sync BASE_URL keys into .deploy/{env}.env (for scripts / CI)
sync_deploy_urls_to_env() {
  [[ -f "$DEPLOY_STATE" ]] || return 0
  [[ "${SKIP_ENV_URL_SYNC:-}" == "1" ]] && return 0
  local tmp
  tmp="$(mktemp)"
  grep -v -E '^(BASE_URL|ALB_BASE_URL|MLAIR_PUBLIC_BASE_URL|CV_PUBLIC_BASE_URL|CV_API_PUBLIC_BASE_URL)=' \
    "$DEPLOY_STATE" >"$tmp" 2>/dev/null || true
  {
    cat "$tmp"
    echo "BASE_URL=${BASE_URL}"
    echo "ALB_BASE_URL=${BASE_URL}"
    echo "MLAIR_PUBLIC_BASE_URL=${BASE_URL}"
    echo "CV_PUBLIC_BASE_URL=${BASE_URL}/cv"
    echo "CV_API_PUBLIC_BASE_URL=${BASE_URL}/cv-api"
  } >"$DEPLOY_STATE"
  rm -f "$tmp"
  chmod 600 "$DEPLOY_STATE"
}
sync_deploy_urls_to_env

# Colors (ANSI)
CLR_RESET="\033[0m"
CLR_DIM="\033[2m"
CLR_RED="\033[0;31m"
CLR_RED_BOLD="\033[1;31m"
CLR_GREEN="\033[0;32m"
CLR_YELLOW="\033[0;33m"

status_color() {
  local code="$1"
  case "$code" in
    2??) printf "%b" "$CLR_GREEN" ;;
    3??) printf "%b" "$CLR_YELLOW" ;;
    4??) printf "%b" "$CLR_RED" ;;
    5??) printf "%b" "$CLR_RED_BOLD" ;;
    000) printf "%b" "$CLR_DIM" ;;
    *)   printf "%b" "$CLR_DIM" ;;
  esac
}

http_get_code() {
  local url="$1"
  curl -sS -o /dev/null -m 15 -w "%{http_code}" "$url" 2>/dev/null || echo "000"
}

rows=()
add_row() { rows+=("$1|$2|$3|$4"); }

if [[ "$ENABLE_MLAIR" =~ ^(1|true|yes)$ ]]; then
  add_row "MLAir Hub UI" "GET" "${BASE_URL}/" "2..|3.."
  MLAIR_NEXT_ASSET="$(curl -sS -m 15 "${BASE_URL}/" 2>/dev/null | grep -oE '/_next/static/[^"?]+\.js' | head -1 || true)"
  if [[ -n "$MLAIR_NEXT_ASSET" ]]; then
    add_row "MLAir Hub _next static" "GET" "${BASE_URL}${MLAIR_NEXT_ASSET}" "200|2.."
  fi
  add_row "MLAir runtime config" "GET" "${BASE_URL}/mlair-runtime-config.js" "200"
  add_row "MLAir API health" "GET" "${BASE_URL}/health" "200"
  add_row "MLAir API OpenAPI docs" "GET" "${BASE_URL}/docs" "2.."
  add_row "MLAir API OpenAPI JSON" "GET" "${BASE_URL}/openapi.json" "200|2.."
  add_row "MLAir API tenants" "GET" "${BASE_URL}/v1/tenants" "401|403|2.."
  add_row "MLAir API pipelines" "GET" "${BASE_URL}/v1/tenants/default/projects/default_project/pipelines" "401|403|2..|404"
  add_row "MLAir API runs" "GET" "${BASE_URL}/v1/tenants/default/projects/default_project/runs" "401|403|2..|404"
  add_row "MLAir API datasets" "GET" "${BASE_URL}/v1/tenants/default/projects/default_project/datasets" "401|403|2..|404"
fi

if [[ "$ENABLE_CV" =~ ^(1|true|yes)$ ]]; then
  add_row "CV Streamlit UI" "GET" "${BASE_URL}/cv" "2..|3.."
  add_row "CV Streamlit health" "GET" "${BASE_URL}/cv/_stcore/health" "200|2.."
  # ALB forwards /cv-api/* without strip — backend /health is not at this path
  add_row "CV API via ALB prefix" "GET" "${BASE_URL}/cv-api/health" "404|502|503"
  add_row "CV API runtime (if mounted)" "GET" "${BASE_URL}/cv-api/api/v1/runtime" "200|404|502"
fi

if [[ "$LIST_ONLY" == "1" ]]; then
  echo "=== CV + MLAir deployed URLs (env=$ENV, list-only) ==="
  echo "ALB: ${ALB_DNS_NAME}"
  [[ -n "${APP_INSTANCE_ID:-}" ]] && echo "EC2: ${APP_INSTANCE_ID} (${APP_PRIVATE_IP:-private})"
  echo ""
  for row in "${rows[@]}"; do
    IFS='|' read -r name _ url _ <<<"$row"
    printf "  %-32s %s\n" "$name" "$url"
  done
  echo ""
  echo "BASE_URL=${BASE_URL}"
  echo "MLAIR Hub: ${BASE_URL}/"
  echo "CV UI:     ${BASE_URL}/cv"
  exit 0
fi

NAME_W=42
URL_W=88
printf "%-${NAME_W}s  %-${URL_W}s\n" "Name(status)" "URL"
printf "%-${NAME_W}s  %-${URL_W}s\n" "------------" "---"

total=0
passed=0
failed=0
failed_names=()

for row in "${rows[@]}"; do
  IFS='|' read -r name method url expected <<<"$row"
  code="$(http_get_code "$url")"

  if [[ "$code" =~ ^($expected)$ ]]; then
    ((passed += 1))
    mark="PASS"
  else
    ((failed += 1))
    failed_names+=("${name}:${code}(expected:${expected})")
    mark="FAIL"
  fi
  ((total += 1))

  c="$(status_color "$code")"
  left="${name}(${code},${mark})"
  printf "%b%-${NAME_W}s%b  %-${URL_W}s\n" "$c" "$left" "$CLR_RESET" "$url"
done

echo
echo "=== Deployed URLs (copy / ${DEPLOY_STATE}) ==="
echo "BASE_URL=${BASE_URL}"
echo "ALB_DNS_NAME=${ALB_DNS_NAME}"
if [[ "$ENABLE_MLAIR" =~ ^(1|true|yes)$ ]]; then
  echo "MLAIR_PUBLIC_BASE_URL=${BASE_URL}"
  echo "MLAir Hub UI: ${BASE_URL}/"
  echo "MLAir API:    ${BASE_URL}/health  ${BASE_URL}/v1/..."
fi
if [[ "$ENABLE_CV" =~ ^(1|true|yes)$ ]]; then
  echo "CV_PUBLIC_BASE_URL=${BASE_URL}/cv"
  echo "CV Streamlit: ${BASE_URL}/cv"
  echo "CV API ALB:   ${BASE_URL}/cv-api/  (prefix; health at EC2 :8000/health via SSM)"
fi
if [[ -n "${APP_INSTANCE_ID:-}" ]]; then
  echo "SSM: aws ssm start-session --target ${APP_INSTANCE_ID} --region ${AWS_REGION:-ap-southeast-1}"
fi
echo

if [[ "$failed" -eq 0 ]]; then
  printf "%bPASS%b %d/%d checks\n" "$CLR_GREEN" "$CLR_RESET" "$passed" "$total"
  exit 0
fi

printf "%bFAIL%b %d/%d checks passed\n" "$CLR_RED_BOLD" "$CLR_RESET" "$passed" "$total"
printf "Failed checks:\n"
for item in "${failed_names[@]}"; do
  printf "  - %s\n" "$item"
done
echo "Tip: 502/503 often means compose not up on EC2 — run: SKIP_BUILD=1 ./deploy.sh ${ENV}"
exit 1
