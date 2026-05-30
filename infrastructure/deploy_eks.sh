#!/usr/bin/env bash
set -euo pipefail

# One command — full hybrid AWS deploy (no extra steps):
#   Terraform (EKS + IRSA + EC2/RDS/Redis/EFS/ALB)
#   ECR build + EC2 compose (Hub/API on ALB)
#   EKS cv-train-worker + HPA
#   Smoke tests
#
# Usage (from anywhere):
#   /path/to/infrastructure/deploy_eks.sh
#   /path/to/infrastructure/deploy_eks.sh dev
#
# Optional: SKIP_BUILD=1 ./deploy_eks.sh dev
#
# Worker concurrency (EKS cv-train-worker):
#   CV_TRAIN_WORKER_MIN_REPLICAS=4   guaranteed execution slots (HPA floor)
#   CV_TRAIN_WORKER_MAX_REPLICAS=16  burst ceiling (CPU HPA until KEDA)
#   CV_TRAIN_WORKER_FIXED_POOL=N     disable HPA; exactly N replicas (legacy)

ENV="${1:-dev}"
INFRA="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib/common.sh
source "$INFRA/lib/common.sh"

require_commands terraform aws docker jq helm kubectl python3

export TF_VAR_enable_eks=true
export SKIP_ANSIBLE=0
export SKIP_BUILD="${SKIP_BUILD:-0}"
export SKIP_TEST_WORKFLOW=1
# Train workers run on EKS; do not start duplicate worker on EC2 compose.
export CV_USE_EKS_TRAIN_WORKERS=1

chmod +x "$INFRA/deploy.sh" "$INFRA/Scripts/"*.sh 2>/dev/null || true

echo "=========================================="
echo " MLAir hybrid deploy (env=$ENV)"
echo " EC2 (ALB) + EKS workers — single command"
echo "=========================================="

"$INFRA/deploy.sh" "$ENV"

echo ""
echo "==> EKS workers (Helm)"
"$INFRA/Scripts/eks-deploy-stack.sh" "$ENV"

echo ""
echo "==> Smoke tests"
"$INFRA/test_workflow.sh" "$ENV"

ALB="$(terraform -chdir="$INFRA/Terraform" output -raw alb_dns_name 2>/dev/null || true)"
NS="${NAME_PREFIX:-cv-mlair-${ENV}}"

echo ""
echo "=========================================="
echo " Done."
echo "   UI/API:  http://${ALB}/"
echo "   Workers: kubectl get hpa,pods -n ${NS}"
echo "=========================================="
