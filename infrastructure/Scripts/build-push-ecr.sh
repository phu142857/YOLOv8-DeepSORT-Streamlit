#!/usr/bin/env bash
# Build/push images for AWS deploy.
#
# MLAir (default MLAIR_IMAGE_SOURCE=ghcr): pull from GHCR, no local ml-air build.
#   https://github.com/phu142857/ml-air → ghcr.io/phu142857/ml-air-{api,frontend,...}
#
# CV workload: always build from this repo and push to ECR.
#   ml-air-api-cv-workload = GHCR ml-air-api + integrations/mlair_cv_plugins
#
# Requires: ECR_REGISTRY, NAME_PREFIX, AWS_REGION
# Optional: MLAIR_REGISTRY, MLAIR_IMAGE_TAG, MLAIR_IMAGE_SOURCE=ghcr|local
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
INFRA="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=../lib/common.sh
source "$INFRA/lib/common.sh"

MLAIR_SRC="${MLAIR_SRC:-$(repo_root)/../ml-air}"
NAME_PREFIX="${NAME_PREFIX:-cv-mlair-dev}"
ECR_REGISTRY="${ECR_REGISTRY:?set ECR_REGISTRY from terraform output}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
AWS_REGION="${AWS_REGION:-ap-southeast-1}"
MLAIR_IMAGE_SOURCE="${MLAIR_IMAGE_SOURCE:-ghcr}"

load_deploy_env "${DEPLOY_ENV:-dev}" 2>/dev/null || true
MLAIR_REGISTRY="${MLAIR_REGISTRY:-ghcr.io/phu142857}"
MLAIR_IMAGE_TAG="${MLAIR_IMAGE_TAG:-latest}"

aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$ECR_REGISTRY"

tag_push() {
  local repo="$1" local_image="$2"
  local uri="${ECR_REGISTRY}/${NAME_PREFIX}/${repo}:${IMAGE_TAG}"
  docker tag "$local_image" "$uri"
  docker push "$uri"
  echo "Pushed $uri"
}

build_cv_overlay_and_push() {
  local api_base_image="$1"
  cd "$ROOT"
  export MLAIR_API_IMAGE="$api_base_image"
  docker compose build api
  tag_push ml-air-api-cv-workload ml-air-api:cv-workload
  # EKS train workers need cu124; same tag used by EC2 cv-api/cv-ui (CPU fallback OK).
  docker compose build mlair-cv-train-worker
  tag_push cv-lifecycle-workload cv-lifecycle-workload:gpu
}

if [[ "$MLAIR_IMAGE_SOURCE" == "local" ]]; then
  echo "==> MLAIR_IMAGE_SOURCE=local — build MLAir from $MLAIR_SRC"
  if [[ ! -f "$MLAIR_SRC/api/Dockerfile" ]]; then
    echo "ERROR: ml-air not found at $MLAIR_SRC" >&2
    exit 1
  fi
  cd "$MLAIR_SRC"
  docker build -t ml-air-api:latest -f api/Dockerfile .
  docker build -t ml-air-scheduler:latest -f scheduler/Dockerfile .
  docker build -t ml-air-executor:latest -f executor/Dockerfile .
  docker build -t ml-air-frontend:latest -f frontend/Dockerfile \
    --build-arg NEXT_PUBLIC_API_BASE_URL="${NEXT_PUBLIC_API_BASE_URL:-http://localhost:8080}" \
    --build-arg NEXT_PUBLIC_MLAIR_REALTIME_WS="${NEXT_PUBLIC_MLAIR_REALTIME_WS:-}" \
    .
  docker build -t ml-air-realtime:latest -f realtime/Dockerfile .

  tag_push ml-air-api ml-air-api:latest
  tag_push ml-air-scheduler ml-air-scheduler:latest
  tag_push ml-air-executor ml-air-executor:latest
  tag_push ml-air-frontend ml-air-frontend:latest
  tag_push ml-air-realtime ml-air-realtime:latest
  build_cv_overlay_and_push "ml-air-api:latest"
else
  echo "==> MLAIR_IMAGE_SOURCE=ghcr — pull MLAir from ${MLAIR_REGISTRY} (tag ${MLAIR_IMAGE_TAG})"
  for svc in api scheduler executor realtime frontend; do
    docker pull "$(mlair_ghcr_image "$svc")"
  done
  build_cv_overlay_and_push "$(mlair_ghcr_image api)"
fi

echo "Done. CV images on ECR; MLAir runtime images: ${MLAIR_REGISTRY}/ml-air-*:${MLAIR_IMAGE_TAG}"
