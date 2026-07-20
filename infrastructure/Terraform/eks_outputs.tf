output "eks_cluster_name" {
  value       = try(module.eks.cluster_name, "")
  description = "EKS cluster name (when enable_eks=true)"
}

output "eks_cluster_endpoint" {
  value       = try(module.eks.cluster_endpoint, "")
  description = "EKS cluster API endpoint"
}

output "eks_cluster_oidc_issuer_url" {
  value       = try(module.eks.cluster_oidc_issuer_url, "")
  description = "OIDC issuer URL for IRSA"
}

