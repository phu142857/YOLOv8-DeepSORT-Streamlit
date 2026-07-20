# Terraform — CV + MLAir on AWS

## Layout

```text
Terraform/
├── main.tf              # root module wiring
├── variables.tf
├── outputs.tf
├── versions.tf
├── providers.tf
├── vpc.tf
├── security_groups.tf
├── rds.tf
├── elasticache.tf
├── efs.tf
├── ecr.tf
├── secrets.tf
├── iam.tf
├── ec2.tf
├── alb.tf
└── environments/
    └── dev/
        ├── terraform.tfvars.example
        └── backend.hcl.example
```

## Usage

```bash
export TF_VAR_environment=dev
terraform init -backend-config=environments/dev/backend.hcl
terraform plan -var-file=environments/dev/terraform.tfvars
terraform apply -var-file=environments/dev/terraform.tfvars
```

Hoặc dùng `../Scripts/terraform-apply.sh dev`.

## State backend (khuyến nghị)

Copy `environments/dev/backend.hcl.example` → `backend.hcl`, tạo S3 bucket + DynamoDB lock table trước khi `init`.
