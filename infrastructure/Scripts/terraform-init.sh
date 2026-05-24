#!/usr/bin/env bash
# Usage: ./terraform-init.sh [dev]
set -euo pipefail
ENV="${1:-dev}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TF_DIR="$ROOT/Terraform"
BACKEND="$TF_DIR/environments/$ENV/backend.hcl"

cd "$TF_DIR"
if [[ -f "$BACKEND" ]]; then
  terraform init -backend-config="$BACKEND" -reconfigure
else
  echo "No backend.hcl — using local state (dev only)"
  terraform init
fi
