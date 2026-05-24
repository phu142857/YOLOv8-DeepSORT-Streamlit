# AWS Infrastructure — CV + MLAir Stack

Triển khai toàn bộ stack (`docker-compose.yml` local) lên AWS theo mô hình:

- **Terraform** — hạ tầng (VPC, RDS, ElastiCache, EFS, ALB, ECR, EC2, IAM, Secrets)
- **SSM deploy** — `.env`, `mlair-runtime-config.js`, `docker compose up` trên EC2 (qua `ssm-deploy-stack.sh`)
- **Docker** — `docker-compose.aws.yml` (RDS/Redis managed, ECR images, EFS volumes)
- **Scripts** — wrapper CLI (`init`, `apply`, `build-push`, `deploy`)

## Kiến trúc (tóm tắt)

```text
Internet → ALB (HTTPS)
              ├─ /hub/*     → ml-air-frontend :3000
              ├─ /api/mlair → ml-air-api :8080
              ├─ /cv-api/*  → cv-lifecycle-api :8000
              └─ /cv/*      → cv-lifecycle-ui :8501

EC2 (private) — Docker Compose
  api, scheduler, executor, realtime, frontend, cv-api, cv-ui, mlair-cv-train-worker

RDS PostgreSQL  ← thay container postgres
ElastiCache Redis ← thay container redis
EFS             ← artifacts, weights, model/dataset registry
GHCR            ← ml-air-{api,frontend,scheduler,executor,realtime} (phu142857/ml-air)
ECR             ← cv-lifecycle-workload, ml-air-api-cv-workload (CV plugins overlay)
```

## Thư mục

| Path | Mục đích |
|------|----------|
| [Terraform/](Terraform/README.md) | IaC AWS |
| [Ansible/](Ansible/README.md) | Provision app host |
| [Docker/](Docker/docker-compose.aws.yml) | Compose production |
| [Scripts/](Scripts/) | Automation |
| [docs/](docs/DEPLOYMENT.md) | Hướng dẫn từng bước |

## Quick start (một lệnh)

```bash
# Lần đầu: copy tfvars (deploy.sh cũng tự tạo nếu thiếu)
cp infrastructure/Terraform/environments/dev/terraform.tfvars.example \
   infrastructure/Terraform/environments/dev/terraform.tfvars

# Một lệnh duy nhất (từ repo root hoặc infrastructure/)
./deploy-aws.sh
# hoặc:
cd infrastructure && ./deploy.sh

./test_workflow.sh          # kiểm tra URL (mặc định env=dev)
./destroy.sh -y             # xóa hạ tầng
```

Instance types (`m7i-flex.large`, `db.t3.micro`) đã nằm trong `terraform.tfvars` và `deploy.sh` — **không cần `export`**.

### Tuỳ chọn deploy

| Biến / lệnh | Ý nghĩa |
|-------------|---------|
| `SKIP_BUILD=1 ./deploy.sh dev` | Chỉ hạ tầng + SSM app deploy (images đã có trên ECR/GHCR) |
| `MLAIR_IMAGE_TAG=v0.1.0 ./deploy.sh dev` | Pin tag GHCR khác `latest` (tùy chọn) |
| `MLAIR_IMAGE_SOURCE=local ./deploy.sh dev` | Build MLAir từ repo local `../ml-air` thay vì GHCR |
| `GHCR_TOKEN=ghp_... ./deploy.sh dev` | Login GHCR nếu package private |
| `SKIP_ANSIBLE=1 ./deploy.sh dev` | Chỉ Terraform + build/push (bỏ qua compose trên EC2) |
| `TERRAFORM_ONLY=1 ./deploy.sh dev` | Chỉ Terraform |
| `SKIP_ENV_URL_SYNC=1 ./test_workflow.sh dev` | Không ghi URL vào `.deploy/dev.env` |
| `./Scripts/ssm-deploy-stack.sh dev` | Chỉ bước app deploy (cùng logic như trong `deploy.sh`) |

Chi tiết từng bước: [Scripts/](Scripts/) và [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Yêu cầu

- AWS CLI v2, Terraform ≥ 1.5, Docker, `jq`
- Quyền tạo VPC, EC2, RDS, ElastiCache, EFS, ALB, ECR, IAM, Secrets Manager
- MLAir images: pull từ **GHCR** `ghcr.io/phu142857/ml-air-*` (không cần clone `ml-air` trừ khi `MLAIR_IMAGE_SOURCE=local`)
- EC2 app cần SSM agent online (deploy app qua Session Manager, không cần SSH)

## Lưu ý production

- Đổi toàn bộ secret trong Secrets Manager / `group_vars` (không dùng `admin-token`)
- Bật `enable_https` + ACM certificate trên ALB
- `docker compose down -v` **không** dùng trên AWS — dùng backup RDS/EFS
- GPU train: cân nhắc instance GPU riêng hoặc worker trên EC2 `g4dn` (chưa tự động trong module mặc định)
