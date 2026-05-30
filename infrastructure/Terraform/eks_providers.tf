provider "kubernetes" {
  host                   = try(module.eks.cluster_endpoint, null)
  cluster_ca_certificate = try(base64decode(module.eks.cluster_certificate_authority_data), null)
  token                  = try(data.aws_eks_cluster_auth.this[0].token, null)
}

provider "helm" {
  kubernetes {
    host                   = try(module.eks.cluster_endpoint, null)
    cluster_ca_certificate = try(base64decode(module.eks.cluster_certificate_authority_data), null)
    token                  = try(data.aws_eks_cluster_auth.this[0].token, null)
  }
}

data "aws_eks_cluster_auth" "this" {
  count = local.eks_enabled ? 1 : 0
  name  = module.eks.cluster_name
}

