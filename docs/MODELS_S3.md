# Detection models — S3 source of truth + local cache

## Layout

**S3** (durable, survives `destroy` of EC2/EFS if bucket kept):

```text
s3://{bucket}/ml-models/detection/{model}/production.json
s3://{bucket}/ml-models/detection/{model}/v2/best.pt
```

`production.json`:

```json
{ "version": "v2", "sha256": "..." }
```

**Local** (inference cache on EFS / laptop):

```text
weights/detection/yolov8x/
  current/weights.pt    # optional mirror of S3 production
  previous/weights.pt   # rollback (old base before promote)
  base/weights.pt       # canonical — jobs use yolov8x/base
  production/weights.pt
  pretrained/weights.pt # COCO seed (inference prefers this for /base)
```

Inference **never** calls `YOLO("s3://...")` per request. Flow:

1. Startup / first job → download from S3 **once** (if manifest exists)
2. Else seed Ultralytics COCO into `pretrained/` + `base/`
3. Load YOLO singleton from local path

## Env

| Variable | Default | Meaning |
|----------|---------|---------|
| `CV_MODELS_S3_BUCKET` | (empty) | Enable S3 when set (Terraform output `models_s3_bucket`) |
| `CV_MODELS_S3_PREFIX` | `ml-models` | Key prefix |
| `CV_MODELS_S3_SYNC_ON_STARTUP` | `1` | Pull production manifest on API start |
| `CV_MODELS_S3_UPLOAD_ON_PROMOTE` | `1` | After Hub promote → upload `v{N}/best.pt` + manifest |
| `CV_DETECTION_BOOTSTRAP_MODELS` | `yolov8n,...,yolov8x` | Seed COCO if disk empty |

## Promote flow

```text
Train → MLAir promote → local base/ + production/
                    → S3 v{N}/best.pt + production.json
Next deploy (empty EFS) → startup reads manifest → download → base/
```

## API

```bash
curl -X POST http://localhost:8000/api/v1/registry/models/bootstrap
curl -X POST "http://localhost:8000/api/v1/registry/models/s3/sync?model=yolov8x"
```

## AWS

Terraform creates `{name_prefix}-models-{account_id}` and grants EC2 IAM read/write.
`ssm-deploy-stack.sh` writes `CV_MODELS_S3_BUCKET` into `.env` on the instance.

After destroy/recreate app stack, run `./deploy.sh` (terraform apply creates bucket) — first cv-api start seeds or pulls weights automatically.
