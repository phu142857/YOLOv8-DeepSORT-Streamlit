# Kiến trúc AWS

## Thành phần

| Layer | Dịch vụ AWS | Thay thế local |
|-------|-------------|----------------|
| Compute | EC2 `t3.xlarge`+ (app) | Docker trên laptop |
| Database | RDS PostgreSQL 16 | `postgres` container |
| Cache / queue | ElastiCache Redis 7 | `redis` container |
| Artifacts | EFS (3 access points) | Docker volumes |
| Registry | ECR | Local `:local` images |
| Ingress | ALB + optional ACM | Published ports |
| Secrets | Secrets Manager | `.env` plaintext |
| Ops | SSM Session Manager | SSH local |

## Luồng dữ liệu

1. User → ALB → `cv-ui` / `frontend` / `cv-api` / `api`
2. `cv-api` ghi job artifacts → EFS `cv-artifacts`
3. Ingest → MLAir API → dataset frames trên EFS `mlair-datasets`
4. Train worker → đọc/ghi EFS + gọi MLAir API
5. MLAir API/scheduler/executor → RDS + Redis

## Mạng

- VPC `/16`, 2 AZ
- Public subnets: ALB
- Private subnets: EC2, RDS, ElastiCache, EFS mount targets
- NAT Gateway (outbound ECR pull, package updates)
- Security groups least-privilege giữa tiers

## HA (mức dev/staging)

Module mặc định **single-AZ** EC2 + RDS single instance để giảm chi phí. Production nên:

- RDS Multi-AZ
- ElastiCache replication group
- EC2 ASG min 2 + shared EFS (hoặc chuyển ECS Fargate — roadmap)
