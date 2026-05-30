#!/usr/bin/env bash
# One-command deploy: Terraform → ECR → SSM compose (no manual export).
#
#   cd infrastructure && ./deploy.sh
#
# Optional: SKIP_BUILD=1 | SKIP_ANSIBLE=1 | TERRAFORM_ONLY=1 | ./deploy.sh staging
#
set -euo pipefail

ENV="${1:-dev}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_ANSIBLE="${SKIP_ANSIBLE:-0}"
TERRAFORM_ONLY="${TERRAFORM_ONLY:-0}"

INFRA="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib/common.sh
source "$INFRA/lib/common.sh"

echo "==> CV + MLAir AWS deploy (env=$ENV)"

require_commands terraform aws docker jq

ensure_tfvars "$ENV"
load_deploy_env "$ENV"
purge_scheduled_app_secret "$ENV"

echo "==> Terraform init"
"$INFRA/Scripts/terraform-init.sh" "$ENV"

echo "==> Terraform apply"
"$INFRA/Scripts/terraform-apply.sh" "$ENV" apply

STATE_FILE="$(write_deploy_state "$ENV")"
echo "==> Wrote state: $STATE_FILE"

export AWS_REGION
AWS_REGION="$(grep AWS_REGION "$STATE_FILE" | cut -d= -f2)"
export ECR_REGISTRY="$(grep ECR_REGISTRY "$STATE_FILE" | cut -d= -f2)"
export NAME_PREFIX="$(grep NAME_PREFIX "$STATE_FILE" | cut -d= -f2)"
export IMAGE_TAG="${IMAGE_TAG:-latest}"

if [[ "$TERRAFORM_ONLY" == "1" ]]; then
  echo "TERRAFORM_ONLY=1 — skipping build and ansible."
  echo "Next: IMAGE_TAG=$IMAGE_TAG ./deploy.sh $ENV  (or SKIP_BUILD=0)"
  exit 0
fi

if [[ "$SKIP_BUILD" != "1" ]]; then
  echo "==> Images: MLAir from GHCR (${MLAIR_REGISTRY:-ghcr.io/phu142857}), CV → ECR"
  export DEPLOY_ENV="$ENV"
  ALB_DNS_BUILD="$(tf_output alb_dns_name "$ENV")"
  if [[ -n "$ALB_DNS_BUILD" ]]; then
    export NEXT_PUBLIC_API_BASE_URL="http://${ALB_DNS_BUILD}"
    export NEXT_PUBLIC_MLAIR_REALTIME_WS="ws://${ALB_DNS_BUILD}/ws"
    echo "    NEXT_PUBLIC_API_BASE_URL=$NEXT_PUBLIC_API_BASE_URL (CV API overlay build only)"
  fi
  "$INFRA/Scripts/build-push-ecr.sh"
else
  echo "==> SKIP_BUILD=1"
fi

if [[ "$SKIP_ANSIBLE" != "1" ]]; then
  write_ansible_inventory "$ENV" >/dev/null || true
  APP_ID="$(terraform -chdir="$INFRA/Terraform" output -raw app_instance_id 2>/dev/null || true)"
  if [[ -n "$APP_ID" ]]; then
    wait_ssm_online "$APP_ID" "$AWS_REGION" 600 || {
      echo "SSM not ready — replace EC2 then re-run: terraform apply -replace=aws_instance.app" >&2
      exit 1
    }
  fi
  echo "==> App deploy on EC2 (SSM: compose + mlair-runtime-config.js)"
  "$INFRA/Scripts/ssm-deploy-stack.sh" "$ENV"
else
  echo "==> SKIP_ANSIBLE=1 (skipped EC2 compose deploy)"
fi

echo ""
if [[ "${SKIP_TEST_WORKFLOW:-0}" == "1" ]]; then
  echo "==> Deploy finished (SKIP_TEST_WORKFLOW=1 — tests run by caller)."
else
  echo "==> Deploy finished. Run: $INFRA/test_workflow.sh $ENV"
  "$INFRA/test_workflow.sh" "$ENV" || true
fi
