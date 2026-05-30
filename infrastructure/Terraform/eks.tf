locals {
  eks_enabled = var.enable_eks
}

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  create = local.eks_enabled

  cluster_name    = "${local.name_prefix}-eks"
  cluster_version = var.eks_cluster_version

  vpc_id     = aws_vpc.main.id
  subnet_ids = aws_subnet.private[*].id

  cluster_endpoint_public_access = true

  enable_cluster_creator_admin_permissions = true
  enable_irsa                              = local.eks_enabled

  eks_managed_node_groups = {
    default = {
      instance_types = var.eks_node_instance_types
      min_size       = var.eks_node_min_size
      max_size       = var.eks_node_max_size
      desired_size   = var.eks_node_desired_size
    }
  }

  cluster_addons = local.eks_enabled ? {
    aws-efs-csi-driver = {
      most_recent              = true
      service_account_role_arn = module.efs_csi_irsa[0].iam_role_arn
    }
  } : {}
}

# IRSA: EFS CSI controller needs EFS permissions to provision dynamic volumes.
module "efs_csi_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  count = local.eks_enabled ? 1 : 0

  role_name             = "${local.name_prefix}-efs-csi"
  attach_efs_csi_policy = true

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:efs-csi-controller-sa"]
    }
  }
}

# EKS worker pods → shared RDS / Redis / EFS / EC2 MLAir API
resource "aws_security_group_rule" "rds_from_eks_nodes" {
  count = local.eks_enabled ? 1 : 0

  type                     = "ingress"
  from_port                = 5432
  to_port                  = 5432
  protocol                 = "tcp"
  security_group_id        = aws_security_group.rds.id
  source_security_group_id = module.eks.node_security_group_id
  description              = "PostgreSQL from EKS nodes"
}

resource "aws_security_group_rule" "redis_from_eks_nodes" {
  count = local.eks_enabled ? 1 : 0

  type                     = "ingress"
  from_port                = 6379
  to_port                  = 6379
  protocol                 = "tcp"
  security_group_id        = aws_security_group.redis.id
  source_security_group_id = module.eks.node_security_group_id
  description              = "Redis from EKS nodes"
}

resource "aws_security_group_rule" "efs_from_eks_nodes" {
  count = local.eks_enabled ? 1 : 0

  type                     = "ingress"
  from_port                = 2049
  to_port                  = 2049
  protocol                 = "tcp"
  security_group_id        = aws_security_group.efs.id
  source_security_group_id = module.eks.node_security_group_id
  description              = "NFS from EKS nodes"
}

resource "aws_security_group_rule" "app_api_from_eks_nodes" {
  count = local.eks_enabled ? 1 : 0

  type                     = "ingress"
  from_port                = 8080
  to_port                  = 8080
  protocol                 = "tcp"
  security_group_id        = aws_security_group.app.id
  source_security_group_id = module.eks.node_security_group_id
  description              = "MLAir API on EC2 from EKS train workers"
}
