#!/usr/bin/env bash
# Deploy docker compose stack on EC2 via SSM (default app deploy path).
# Usage: ./ansible-deploy.sh [dev]
# Legacy name kept for compatibility — calls ssm-deploy-stack.sh.
set -euo pipefail
ENV="${1:-dev}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "$ROOT/Scripts/ssm-deploy-stack.sh" "$ENV"
