# Ansible — deploy stack lên EC2

## Inventory

Sao chép `inventory/aws_ec2.yml.example` → `inventory/aws_ec2.yml` và điền IP/instance (từ `terraform output app_private_ip`).

Dùng SSM (khuyến nghị):

```yaml
ansible_connection: community.aws.aws_ssm
ansible_aws_ssm_instance_id: i-xxxxxxxx
```

Cài collection: `ansible-galaxy collection install community.aws`

## Playbooks

| Playbook | Mục đích |
|----------|----------|
| `site.yml` | Common + deploy full stack |
| `deploy-stack.yml` | Chỉ pull images + compose up |

## Biến

`group_vars/all.yml` — override `ecr_registry`, `name_prefix`, URLs ALB.

Secrets lấy từ AWS Secrets Manager trong role `cv-mlair-stack`.
