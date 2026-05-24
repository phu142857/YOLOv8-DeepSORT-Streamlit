#!/usr/bin/env bash
# Usage: ./terraform-apply.sh [dev] [plan|apply]
set -euo pipefail
ENV="${1:-dev}"
ACTION="${2:-apply}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=../lib/common.sh
source "$ROOT/lib/common.sh"
load_deploy_env "$ENV"
TF_DIR="$ROOT/Terraform"
VAR_FILE="$TF_DIR/environments/$ENV/terraform.tfvars"

if [[ ! -f "$VAR_FILE" ]]; then
  echo "Missing $VAR_FILE — copy from terraform.tfvars.example" >&2
  exit 1
fi

cd "$TF_DIR"
terraform "$ACTION" -var-file="$VAR_FILE" -var="environment=$ENV"

if [[ "$ACTION" == "apply" ]]; then
  echo ""
  terraform output
fi
