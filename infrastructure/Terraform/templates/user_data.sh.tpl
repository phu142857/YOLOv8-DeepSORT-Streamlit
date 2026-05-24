#!/bin/bash
set -euxo pipefail
dnf update -y
dnf install -y docker amazon-efs-utils jq amazon-ssm-agent
systemctl enable --now amazon-ssm-agent
systemctl enable --now docker
usermod -aG docker ec2-user

mkdir -p /mnt/efs/cv-artifacts /mnt/efs/mlair-models /mnt/efs/mlair-datasets
echo "${efs_id}:/cv-artifacts /mnt/efs/cv-artifacts efs _netdev,noresvport,tls,iam,accesspoint=${ap_cv} 0 0" >> /etc/fstab
echo "${efs_id}:/mlair-models /mnt/efs/mlair-models efs _netdev,noresvport,tls,iam,accesspoint=${ap_models} 0 0" >> /etc/fstab
echo "${efs_id}:/mlair-datasets /mnt/efs/mlair-datasets efs _netdev,noresvport,tls,iam,accesspoint=${ap_datasets} 0 0" >> /etc/fstab
mount -a

# Docker Compose v2 plugin
mkdir -p /usr/local/lib/docker/cli-plugins
curl -SL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64" \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

mkdir -p /opt/${name_prefix}
chown -R ec2-user:ec2-user /opt/${name_prefix} /mnt/efs

# ECR login helper (Ansible deploy completes stack)
aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${ecr_registry}
