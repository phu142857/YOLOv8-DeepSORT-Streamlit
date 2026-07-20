output "vpc_id" {
  value = aws_vpc.main.id
}

output "alb_dns_name" {
  value = aws_lb.main.dns_name
}

output "app_instance_id" {
  value = aws_instance.app.id
}

output "app_private_ip" {
  value = aws_instance.app.private_ip
}

output "rds_endpoint" {
  value = aws_db_instance.main.address
}

output "redis_endpoint" {
  value = aws_elasticache_cluster.main.cache_nodes[0].address
}

output "efs_id" {
  value = aws_efs_file_system.main.id
}

output "efs_access_point_models_id" {
  value       = aws_efs_access_point.mlair_models.id
  description = "EFS access point for /mlair-models (shared with EC2)"
}

output "efs_access_point_datasets_id" {
  value       = aws_efs_access_point.mlair_datasets.id
  description = "EFS access point for /mlair-datasets (shared with EC2)"
}

output "efs_access_point_cv_artifacts_id" {
  value       = aws_efs_access_point.cv_artifacts.id
  description = "EFS access point for /cv-artifacts (shared with EC2)"
}

output "ecr_registry_url" {
  value = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com"
}

output "ecr_repository_urls" {
  value = { for k, r in aws_ecr_repository.repos : k => r.repository_url }
}

output "secrets_manager_arn" {
  value = aws_secretsmanager_secret.app.arn
}

output "database_url" {
  value     = "postgresql://${var.db_username}:${random_password.db.result}@${aws_db_instance.main.address}:5432/${var.db_name}"
  sensitive = true
}

output "redis_url" {
  value = "redis://${aws_elasticache_cluster.main.cache_nodes[0].address}:6379/0"
}

output "models_s3_bucket" {
  value = aws_s3_bucket.models.id
}
