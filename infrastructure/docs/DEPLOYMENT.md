# Hướng dẫn deploy AWS

## 0. Chuẩn bị

```bash
aws configure   # account + region
export AWS_REGION=ap-southeast-1
export PROJECT=cv-mlair
export ENV=dev
```

Chỉ cần repo này trên máy build (MLAir pull từ GHCR mặc định):

```text
ATE/WORK/YOLOv8-DeepSORT-Streamlit
```

Tùy chọn `MLAIR_IMAGE_SOURCE=local`: clone thêm `ATE/WORK/ml-air`.

## 1. Terraform

```bash
cd Infrastructure/Terraform/environments/dev
cp terraform.tfvars.example terraform.tfvars
# Chỉnh: instance_type, db_password (hoặc để Terraform random + Secrets Manager)
```

```bash
cd ../../Scripts
./terraform-init.sh dev
./terraform-apply.sh dev
```

Lưu outputs:

```bash
terraform -chdir=../Terraform output
```

## 2. Images (GHCR + ECR)

Mặc định `build-push-ecr.sh`:

1. **Pull** từ GHCR ([phu142857/ml-air](https://github.com/phu142857/ml-air)):  
   `ghcr.io/phu142857/ml-air-{api,frontend,scheduler,executor,realtime}:latest`
2. **Build & push ECR** (repo này):  
   `ml-air-api-cv-workload` (API + CV plugins), `cv-lifecycle-workload`

```bash
cd infrastructure/Scripts
export ECR_REGISTRY=$(terraform -chdir=../Terraform output -raw ecr_registry_url)
export NAME_PREFIX=cv-mlair-dev
# Mặc định MLAIR_IMAGE_TAG=latest trên GHCR
# Package private: export GHCR_TOKEN=ghp_...
./build-push-ecr.sh
```

EC2 `docker compose pull` lấy MLAir từ GHCR + CV từ ECR.

## 3. Ansible

```bash
# inventory từ Terraform
cd Infrastructure/Ansible
cp inventory/aws_ec2.yml.example inventory/aws_ec2.yml
# Điền ansible_host / instance_id

ansible-playbook -i inventory/aws_ec2.yml playbooks/site.yml \
  -e env=dev \
  -e ecr_registry=$ECR_REGISTRY
```

Playbook sẽ:

- Cài Docker + compose plugin
- Mount EFS
- Deploy `Infrastructure/Docker/docker-compose.aws.yml`
- Chạy `post_stack_bootstrap.sh` tương đương

## 4. Kiểm tra

| URL (qua ALB) | Dịch vụ |
|---------------|---------|
| `https://<alb>/` | MLAir Hub |
| `https://<alb>/cv/` | Streamlit CV |
| Health | `curl https://<alb>/cv-api/health` |

## 5. Cập nhật release

```bash
./build-push-ecr.sh
ansible-playbook ... playbooks/deploy-stack.yml -e image_tag=<git-sha>
```

## Rollback

- ECR giữ tag `previous` / `latest`
- Ansible: `image_tag=previous`
- RDS/EFS: snapshot trước upgrade (Terraform `backup_retention_period`)
