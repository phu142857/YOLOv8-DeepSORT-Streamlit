# `mlair-stack` Helm chart

Deploy MLAir + CV lifecycle components to Kubernetes (EKS-ready).

## Install (example)

```bash
kubectl create namespace mlair

helm upgrade --install mlair ./charts/mlair-stack \
  -n mlair \
  -f ./charts/mlair-stack/values-dev.yaml \
  --set global.publicBaseUrl="http://<alb-dns>" \
  --set global.realtimeWsUrl="ws://<alb-dns>/ws"
```

## Notes
- This chart assumes **managed RDS + ElastiCache** (provide `global.databaseUrl` / `global.redisUrl`).
- For EFS, use the **EFS CSI driver** with IRSA (`TF_VAR_enable_eks=true` terraform) and `storage.efs.createStorageClass=true`.
- Worker identity: `cvTrainWorker.workerIdFromPodName=true` ensures each pod has a unique `MLAIR_WORKER_ID`.

## Hybrid AWS deploy (recommended)
- **One command:** `infrastructure/deploy_eks.sh` (or `./deploy_eks.sh dev`)
- EC2: Hub, API, CV UI on ALB. EKS: `cv-train-worker` + HPA only (`values-eks-workers-only.yaml`).
- **EFS:** workers mount the same Terraform access points as EC2 (`/mnt/efs/mlair-models` → `/mlair/artifacts/models`). Do not use dynamic `efs-sc` PVCs for hybrid — train would not see Hub model artifacts.

### Worker concurrency vs autoscaling (hybrid EKS)

Three separate concerns — do not use CPU HPA as the only concurrency guarantee:

| Layer | Mechanism | Default (dev hybrid) |
|-------|-----------|----------------------|
| **Concurrency guarantee** | HPA `minReplicas` (= baseline pool) | **1** (use **4** for parallel-task QA) |
| **Autoscaling (burst)** | HPA `maxReplicas` + CPU | up to **16** |
| **Scheduling policy** | MLAir lease / queues | upstream (Phase 2–4) |

Env overrides (persist across `./deploy_eks.sh`):

```bash
CV_TRAIN_WORKER_MIN_REPLICAS=4   # parallel-task QA (default deploy uses 1)
CV_TRAIN_WORKER_MAX_REPLICAS=16  # burst ceiling
SKIP_BUILD=1 ./infrastructure/deploy_eks.sh dev
```

Legacy fixed pool (disables HPA): `CV_TRAIN_WORKER_FIXED_POOL=4` or overlay `values-eks-4-worker-pool.yaml`.

Phase 2 (not yet wired): KEDA on `pending_tasks` / Redis queue depth — see `templates/autoscaling/keda-cv-train-worker.yaml`.

