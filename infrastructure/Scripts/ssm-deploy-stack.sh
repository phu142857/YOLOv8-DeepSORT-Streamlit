#!/usr/bin/env bash
# Deploy docker compose stack to EC2 via SSM (no Ansible/boto3 on laptop).
# Usage: ./ssm-deploy-stack.sh [dev]
set -euo pipefail

ENV="${1:-dev}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=../lib/common.sh
source "$ROOT/lib/common.sh"
load_deploy_env "$ENV"
TF_DIR="$ROOT/Terraform"
COMPOSE_SRC="$ROOT/Docker/docker-compose.aws.yml"
RUNTIME_CFG_TPL="$ROOT/Docker/mlair-runtime-config.js.tpl"
BOOTSTRAP_SRC="$ROOT/../scripts/post_stack_bootstrap.sh"

INSTANCE_ID="$(terraform -chdir="$TF_DIR" output -raw app_instance_id)"
wait_ssm_online "$INSTANCE_ID" "$AWS_REGION" 600
ALB_DNS="$(terraform -chdir="$TF_DIR" output -raw alb_dns_name)"
RDS_EP="$(terraform -chdir="$TF_DIR" output -raw rds_endpoint)"
REDIS_EP="$(terraform -chdir="$TF_DIR" output -raw redis_endpoint)"
ECR_REG="$(terraform -chdir="$TF_DIR" output -raw ecr_registry_url)"
APP_ROOT="/opt/${NAME_PREFIX}"
SECRETS_ARN="$(terraform -chdir="$TF_DIR" output -raw secrets_manager_arn)"
MODELS_S3_BUCKET="$(terraform -chdir="$TF_DIR" output -raw models_s3_bucket 2>/dev/null || true)"

command -v aws >/dev/null || { echo "aws CLI required" >&2; exit 1; }
command -v jq >/dev/null || { echo "jq required" >&2; exit 1; }
[[ -f "$COMPOSE_SRC" ]] || { echo "missing $COMPOSE_SRC" >&2; exit 1; }

echo "==> SSM deploy to $INSTANCE_ID ($APP_ROOT)"

SECRETS_JSON="$(aws secretsmanager get-secret-value --secret-id "$SECRETS_ARN" --region "$AWS_REGION" --query SecretString --output text)"
DB_PASS="$(echo "$SECRETS_JSON" | jq -r .POSTGRES_PASSWORD)"
JWT="$(echo "$SECRETS_JSON" | jq -r .ML_AIR_JWT_HS256_SECRET)"
TRACK="$(echo "$SECRETS_JSON" | jq -r .ML_AIR_TRACKING_TOKEN)"
CV_TOKEN="$(echo "$SECRETS_JSON" | jq -r .CV_MLAIR_TOKEN)"

ENV_FILE="$(mktemp)"
: >"$ENV_FILE"
trap 'rm -f "$REMOTE_SCRIPT" "$ENV_FILE" "$RUNTIME_CFG_FILE"' EXIT
MLAIR_REGISTRY="${MLAIR_REGISTRY:-ghcr.io/phu142857}"
MLAIR_IMAGE_TAG="${MLAIR_IMAGE_TAG:-latest}"

# Quote values — RDS/JWT passwords often contain () and other shell metacharacters.
dotenv_set "$ENV_FILE" NAME_PREFIX "$NAME_PREFIX"
dotenv_set "$ENV_FILE" ECR_REGISTRY "$ECR_REG"
dotenv_set "$ENV_FILE" IMAGE_TAG "$IMAGE_TAG"
dotenv_set "$ENV_FILE" MLAIR_REGISTRY "$MLAIR_REGISTRY"
dotenv_set "$ENV_FILE" MLAIR_IMAGE_TAG "$MLAIR_IMAGE_TAG"
DB_URL="$(postgres_database_url mlair "$DB_PASS" "$RDS_EP" mlair)"
dotenv_set "$ENV_FILE" ML_AIR_DATABASE_URL "$DB_URL"
dotenv_set "$ENV_FILE" ML_AIR_REDIS_URL "redis://${REDIS_EP}:6379/0"
dotenv_set "$ENV_FILE" ML_AIR_JWT_HS256_SECRET "$JWT"
dotenv_set "$ENV_FILE" ML_AIR_TRACKING_TOKEN "$TRACK"
dotenv_set "$ENV_FILE" ML_AIR_MANIFEST_SIGNING_KEY "$JWT"
dotenv_set "$ENV_FILE" CV_MLAIR_TOKEN "$CV_TOKEN"
AUTH_TOKENS_JSON="$(mlair_auth_tokens_json "$CV_TOKEN")"
dotenv_set "$ENV_FILE" ML_AIR_AUTH_TOKENS_JSON "$AUTH_TOKENS_JSON"
dotenv_set "$ENV_FILE" CV_MLAIR_TRAIN_CALLBACK_TOKEN "$TRACK"
dotenv_set "$ENV_FILE" CV_MLAIR_PROMOTE_WEBHOOK_TOKEN "$TRACK"
dotenv_set "$ENV_FILE" NEXT_PUBLIC_API_BASE_URL "http://${ALB_DNS}"
dotenv_set "$ENV_FILE" NEXT_PUBLIC_MLAIR_REALTIME_WS "ws://${ALB_DNS}/ws"
dotenv_set "$ENV_FILE" ML_AIR_RUNTIME_API_BASE_URL "http://${ALB_DNS}"
dotenv_set "$ENV_FILE" ML_AIR_RUNTIME_REALTIME_BASE_URL "ws://${ALB_DNS}/ws"
dotenv_set "$ENV_FILE" CV_MLAIR_HUB_URL "http://${ALB_DNS}"
dotenv_set "$ENV_FILE" CV_API_BASE_URL "http://cv-api:8000"
dotenv_set "$ENV_FILE" ML_AIR_HTTP_TASK_ALLOWED_HOSTS "cv-api,api,localhost,127.0.0.1"
dotenv_set "$ENV_FILE" ML_AIR_SKIP_APPROVAL_FOR_PROMOTE "1"
dotenv_set "$ENV_FILE" ML_AIR_TASK_EXECUTION_MODE "external"
dotenv_set "$ENV_FILE" CV_MLAIR_SYNC_AFTER_TRAIN "0"
dotenv_set "$ENV_FILE" CV_MLAIR_AUTO_SYNC_MODELS "1"
dotenv_set "$ENV_FILE" CV_MLAIR_SYNC_ON_STARTUP "1"
dotenv_set "$ENV_FILE" CV_CLIENT_SAVE_TO_DATASET "1"
dotenv_set "$ENV_FILE" CV_MLAIR_PERSIST_INGEST_FRAMES "0"
if [[ -n "${MODELS_S3_BUCKET:-}" ]]; then
  dotenv_set "$ENV_FILE" CV_MODELS_S3_BUCKET "$MODELS_S3_BUCKET"
  dotenv_set "$ENV_FILE" CV_MODELS_S3_PREFIX "ml-models"
  dotenv_set "$ENV_FILE" CV_MODELS_S3_REGION "$AWS_REGION"
  dotenv_set "$ENV_FILE" CV_DETECTION_BOOTSTRAP_MODELS "yolov8n,yolov8s,yolov8m,yolov8l,yolov8x"
fi
if [[ -n "${GHCR_TOKEN:-}" ]]; then
  dotenv_set "$ENV_FILE" GHCR_TOKEN "$GHCR_TOKEN"
  dotenv_set "$ENV_FILE" GHCR_USER "${GHCR_USER:-phu142857}"
fi
ENV_B64="$(base64 -w0 "$ENV_FILE")"

RUNTIME_CFG_FILE="$(mktemp)"
sed "s|__ALB_DNS__|${ALB_DNS}|g" "$RUNTIME_CFG_TPL" >"$RUNTIME_CFG_FILE"
RUNTIME_CFG_B64="$(base64 -w0 "$RUNTIME_CFG_FILE")"

COMPOSE_B64="$(base64 -w0 "$COMPOSE_SRC")"
BOOTSTRAP_B64=""
if [[ -f "$BOOTSTRAP_SRC" ]]; then
  BOOTSTRAP_B64="$(base64 -w0 "$BOOTSTRAP_SRC")"
fi

REMOTE_SCRIPT="$(mktemp)"

cat >"$REMOTE_SCRIPT" <<REMOTE_EOF
#!/bin/bash
set -euxo pipefail
APP_ROOT="${APP_ROOT}"
COMPOSE_BIN="/usr/local/lib/docker/cli-plugins/docker-compose"
mkdir -p "\$APP_ROOT"
echo '${COMPOSE_B64}' | base64 -d > "\$APP_ROOT/docker-compose.aws.yml"
echo '${ENV_B64}' | base64 -d > "\$APP_ROOT/.env"
echo '${RUNTIME_CFG_B64}' | base64 -d > "\$APP_ROOT/mlair-runtime-config.js"
if [[ -n '${BOOTSTRAP_B64}' ]]; then
  echo '${BOOTSTRAP_B64}' | base64 -d > "\$APP_ROOT/post_stack_bootstrap.sh"
  chmod +x "\$APP_ROOT/post_stack_bootstrap.sh"
fi
chown -R ec2-user:docker "\$APP_ROOT"
chmod 600 "\$APP_ROOT/.env"
if [[ ! -x "\$COMPOSE_BIN" ]]; then
  mkdir -p /usr/local/lib/docker/cli-plugins
  curl -fsSL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64" \\
    -o "\$COMPOSE_BIN"
  chmod +x "\$COMPOSE_BIN"
fi
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REG}
mkdir -p /home/ec2-user/.docker
cp -f /root/.docker/config.json /home/ec2-user/.docker/config.json 2>/dev/null || true
chown -R ec2-user:ec2-user /home/ec2-user/.docker
cd "\$APP_ROOT"
# Optional GHCR login (public packages work without token). pipefail + grep miss = exit 1 without || true.
GHCR_TOKEN=""
GHCR_USER="phu142857"
if [[ -f .env ]]; then
  GHCR_TOKEN="\$(grep -m1 '^GHCR_TOKEN=' .env 2>/dev/null | cut -d= -f2- | sed 's/^"//;s/"$//' || true)"
  GHCR_USER="\$(grep -m1 '^GHCR_USER=' .env 2>/dev/null | cut -d= -f2- | sed 's/^"//;s/"$//' || true)"
  GHCR_USER="\${GHCR_USER:-phu142857}"
fi
if [[ -n "\$GHCR_TOKEN" ]]; then
  echo "\$GHCR_TOKEN" | docker login ghcr.io -u "\$GHCR_USER" --password-stdin
  cp -f /root/.docker/config.json /home/ec2-user/.docker/config.json 2>/dev/null || true
  chown -R ec2-user:ec2-user /home/ec2-user/.docker
fi
# Prefer standalone compose binary (ec2-user often lacks docker compose plugin)
COMPOSE=(sudo -u ec2-user "\$COMPOSE_BIN" -f docker-compose.aws.yml)
"\${COMPOSE[@]}" pull
"\${COMPOSE[@]}" up -d --remove-orphans
echo "Waiting for cv-api..."
for i in \$(seq 1 36); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null; then echo OK; break; fi
  sleep 10
done
if [[ -x "\$APP_ROOT/post_stack_bootstrap.sh" ]]; then
  sudo -u ec2-user env CV_API_URL=http://127.0.0.1:8000 ML_AIR_API_PORT=8080 "\$APP_ROOT/post_stack_bootstrap.sh" || true
fi
docker ps
REMOTE_EOF

SCRIPT_B64="$(base64 -w0 "$REMOTE_SCRIPT")"
CMD_ID="$(aws ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --document-name AWS-RunShellScript \
  --timeout-seconds 7200 \
  --parameters "commands=[\"echo ${SCRIPT_B64} | base64 -d | bash\"]" \
  --region "$AWS_REGION" \
  --query Command.CommandId --output text)"

echo "SSM command: $CMD_ID (waiting up to 60 min)..."
for _ in $(seq 1 120); do
  sleep 30
  STATUS="$(aws ssm get-command-invocation --command-id "$CMD_ID" --instance-id "$INSTANCE_ID" --region "$AWS_REGION" --query Status --output text)"
  echo "  status=$STATUS"
  [[ "$STATUS" == "Success" ]] && break
  [[ "$STATUS" == "Failed" || "$STATUS" == "Cancelled" || "$STATUS" == "TimedOut" ]] && {
    aws ssm get-command-invocation --command-id "$CMD_ID" --instance-id "$INSTANCE_ID" --region "$AWS_REGION" \
      --query '[StandardOutputContent,StandardErrorContent]' --output text
    exit 1
  }
done

aws ssm get-command-invocation --command-id "$CMD_ID" --instance-id "$INSTANCE_ID" --region "$AWS_REGION" \
  --query StandardOutputContent --output text | tail -40
echo "==> SSM deploy finished. Run: $ROOT/test_workflow.sh $ENV"
