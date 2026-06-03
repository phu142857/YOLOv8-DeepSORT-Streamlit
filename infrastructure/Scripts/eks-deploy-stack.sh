#!/usr/bin/env bash
set -euo pipefail

ENV="${1:-dev}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"

# shellcheck source=../lib/common.sh
source "$ROOT/lib/common.sh"
load_deploy_env "$ENV"

require_commands terraform aws helm kubectl jq python3

TF_DIR="$ROOT/Terraform"
CHART_DIR="$REPO_ROOT/charts/mlair-stack"

CLUSTER_NAME="$(terraform -chdir="$TF_DIR" output -raw eks_cluster_name 2>/dev/null || true)"
AWS_REGION="$(terraform -chdir="$TF_DIR" output -raw aws_region 2>/dev/null || echo "${AWS_REGION:-ap-southeast-1}")"

if [[ -z "$CLUSTER_NAME" ]]; then
  echo "ERROR: EKS cluster output empty after terraform apply." >&2
  echo "  Re-run: $ROOT/deploy_eks.sh $ENV" >&2
  exit 1
fi

echo "==> kubeconfig: $CLUSTER_NAME ($AWS_REGION)"
aws eks update-kubeconfig --name "$CLUSTER_NAME" --region "$AWS_REGION" >/dev/null

wait_eks_cluster_ready "$CLUSTER_NAME" "$AWS_REGION"
wait_eks_nodegroup_ready "$CLUSTER_NAME" "$AWS_REGION"
wait_eks_addon_active "$CLUSTER_NAME" "aws-efs-csi-driver" "$AWS_REGION"
reconcile_eks_efs_csi

ALB_DNS="$(terraform -chdir="$TF_DIR" output -raw alb_dns_name 2>/dev/null || true)"
APP_PRIVATE_IP="$(terraform -chdir="$TF_DIR" output -raw app_private_ip 2>/dev/null || true)"
RDS_EP="$(terraform -chdir="$TF_DIR" output -raw rds_endpoint 2>/dev/null || true)"
REDIS_EP="$(terraform -chdir="$TF_DIR" output -raw redis_endpoint 2>/dev/null || true)"
ECR_REG="$(terraform -chdir="$TF_DIR" output -raw ecr_registry_url 2>/dev/null || true)"
EFS_ID="$(terraform -chdir="$TF_DIR" output -raw efs_id 2>/dev/null || true)"
EFS_AP_MODELS="$(terraform -chdir="$TF_DIR" output -raw efs_access_point_models_id 2>/dev/null || true)"
EFS_AP_DATASETS="$(terraform -chdir="$TF_DIR" output -raw efs_access_point_datasets_id 2>/dev/null || true)"
EFS_AP_CV="$(terraform -chdir="$TF_DIR" output -raw efs_access_point_cv_artifacts_id 2>/dev/null || true)"
SECRETS_ARN="$(terraform -chdir="$TF_DIR" output -raw secrets_manager_arn 2>/dev/null || true)"

if [[ -z "$APP_PRIVATE_IP" ]]; then
  echo "ERROR: app_private_ip empty — EC2 app not provisioned." >&2
  exit 1
fi
if [[ -z "$SECRETS_ARN" || -z "$EFS_ID" ]]; then
  echo "ERROR: missing terraform outputs (secrets or EFS)." >&2
  exit 1
fi
# accessPointIds kept for docs; hybrid mount uses full filesystem + subPath (see pv-efs-shared.yaml).

SECRETS_JSON="$(aws secretsmanager get-secret-value --secret-id "$SECRETS_ARN" --region "$AWS_REGION" --query SecretString --output text)"
DB_PASS="$(echo "$SECRETS_JSON" | jq -r .POSTGRES_PASSWORD)"
JWT="$(echo "$SECRETS_JSON" | jq -r .ML_AIR_JWT_HS256_SECRET)"
TRACK="$(echo "$SECRETS_JSON" | jq -r .ML_AIR_TRACKING_TOKEN)"
CV_TOKEN="$(echo "$SECRETS_JSON" | jq -r .CV_MLAIR_TOKEN)"

DB_URL="$(postgres_database_url mlair "$DB_PASS" "$RDS_EP" mlair)"
AUTH_TOKENS_JSON="$(mlair_auth_tokens_json "$CV_TOKEN")"

NS="${K8S_NAMESPACE:-${NAME_PREFIX:-cv-mlair-${ENV}}}"
RELEASE="mlair"
IMAGE_PREFIX="${NAME_PREFIX:-cv-mlair-${ENV}}"
MLAIR_API_URL="${MLAIR_API_BASE_URL:-http://${APP_PRIVATE_IP}:8080}"

VALUES_FILE="$(mktemp)"
trap 'rm -f "$VALUES_FILE"' EXIT

export VALUES_FILE ALB_DNS ECR_REG EFS_ID EFS_AP_MODELS EFS_AP_DATASETS EFS_AP_CV IMAGE_TAG AUTH_TOKENS_JSON
export DB_URL REDIS_EP JWT TRACK CV_TOKEN IMAGE_PREFIX MLAIR_API_URL
export CV_TRAIN_WORKER_MIN_REPLICAS="${CV_TRAIN_WORKER_MIN_REPLICAS:-1}"
export CV_TRAIN_WORKER_MAX_REPLICAS="${CV_TRAIN_WORKER_MAX_REPLICAS:-16}"
export CV_TRAIN_WORKER_FIXED_POOL="${CV_TRAIN_WORKER_FIXED_POOL:-0}"
python3 <<'PY'
import json
import os

min_rep = int(os.environ.get("CV_TRAIN_WORKER_MIN_REPLICAS", "1"))
max_rep = int(os.environ.get("CV_TRAIN_WORKER_MAX_REPLICAS", "16"))
fixed = int(os.environ.get("CV_TRAIN_WORKER_FIXED_POOL", "0"))
if min_rep < 1:
    min_rep = 1
if max_rep < min_rep:
    max_rep = min_rep
if fixed < 0:
    fixed = 0
if fixed > 0:
    min_rep = fixed
    max_rep = fixed

out = {
    "global": {
        "publicBaseUrl": f"http://{os.environ['ALB_DNS']}",
        "realtimeWsUrl": f"ws://{os.environ['ALB_DNS']}/ws",
        "mlairApiBaseUrl": os.environ["MLAIR_API_URL"],
        "tenantId": "default",
        "projectId": "default_project",
        "databaseUrl": os.environ["DB_URL"],
        "redisUrl": f"redis://{os.environ['REDIS_EP']}:6379/0",
        "jwtSecret": os.environ["JWT"],
        "manifestSigningKey": os.environ["JWT"],
        "trackingToken": os.environ["TRACK"],
        "authTokensJson": os.environ["AUTH_TOKENS_JSON"],
        "cvMlairToken": os.environ["CV_TOKEN"],
        "ecrRegistry": os.environ["ECR_REG"],
        "namePrefix": os.environ["IMAGE_PREFIX"],
        "imageTag": os.environ.get("IMAGE_TAG", "latest"),
    },
    "storage": {
        "efs": {
            "createStorageClass": False,
            "useSharedAccessPoints": True,
            "fileSystemId": os.environ["EFS_ID"],
            "accessPointIds": {
                "models": os.environ["EFS_AP_MODELS"],
                "datasets": os.environ["EFS_AP_DATASETS"],
                "cvArtifacts": os.environ["EFS_AP_CV"],
            },
        }
    },
    "cvTrainWorker": {
        "replicas": min_rep,
        "resourceMonitorEnabled": True,
        "resourceSampleIntervalSec": 3,
        "trainBatch": int(os.environ.get("CV_TRAIN_WORKER_TRAIN_BATCH", "4")),
        "trainWorkers": int(os.environ.get("CV_TRAIN_WORKER_TRAIN_WORKERS", "2")),
        "trainMaxFrames": int(os.environ.get("CV_TRAIN_WORKER_TRAIN_MAX_FRAMES", "0")),
        "detectMaxFrames": int(os.environ.get("CV_TRAIN_WORKER_DETECT_MAX_FRAMES", "0")),
        "resources": {
            "requests": {"cpu": "500m", "memory": "2Gi"},
            "limits": {"cpu": "4", "memory": "16Gi"},
        },
    },
    "autoscaling": {
        "hpa": {
            "enabled": fixed <= 0,
            "minReplicas": min_rep,
            "maxReplicas": max_rep,
        }
    },
}

with open(os.environ["VALUES_FILE"], "w", encoding="utf-8") as f:
    json.dump(out, f)
PY

ensure_metrics_server

echo "==> namespace $NS"
kubectl create namespace "$NS" >/dev/null 2>&1 || true

delete_failed_efs_pvcs "$NS"
migrate_eks_efs_to_shared_access_points "$NS"

if helm status "$RELEASE" -n "$NS" >/dev/null 2>&1; then
  HELM_STATUS="$(helm status "$RELEASE" -n "$NS" -o json | jq -r '.info.status')"
  if [[ "$HELM_STATUS" == "pending-install" || "$HELM_STATUS" == "pending-upgrade" || "$HELM_STATUS" == "failed" ]]; then
    echo "==> Removing stuck Helm release (status=$HELM_STATUS)"
    helm uninstall "$RELEASE" -n "$NS" || true
    sleep 3
    delete_failed_efs_pvcs "$NS"
  fi
fi

echo "==> Helm (workers-only; shared EFS with EC2; API: $MLAIR_API_URL)"
helm upgrade --install "$RELEASE" "$CHART_DIR" \
  -n "$NS" \
  -f "$CHART_DIR/values-eks-workers-only.yaml" \
  -f "$VALUES_FILE" \
  --wait --timeout 25m

wait_mlair_pvcs_bound "$NS"

WORKER_DEPLOY="$(kubectl -n "$NS" get deploy -l app.kubernetes.io/component=cv-train-worker -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
if [[ -n "$WORKER_DEPLOY" ]]; then
  kubectl -n "$NS" rollout status "deploy/$WORKER_DEPLOY" --timeout=15m
  READY="$(kubectl -n "$NS" get deploy "$WORKER_DEPLOY" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo 0)"
  if [[ "${READY:-0}" -lt 1 ]]; then
    echo "ERROR: cv-train-worker not ready — logs:" >&2
    kubectl -n "$NS" logs -l app.kubernetes.io/component=cv-train-worker --tail=40 >&2 || true
    exit 1
  fi
fi

verify_eks_worker_model_mount "$NS"

kubectl -n "$NS" get deploy,hpa,pvc,pods
