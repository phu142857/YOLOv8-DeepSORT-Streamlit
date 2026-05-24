variable "aws_region" {
  type        = string
  description = "AWS region"
  default     = "ap-southeast-1"
}

variable "environment" {
  type        = string
  description = "Environment name (dev, staging, prod)"
}

variable "project_name" {
  type        = string
  description = "Resource name prefix"
  default     = "cv-mlair"
}

variable "vpc_cidr" {
  type    = string
  default = "10.20.0.0/16"
}

variable "node_instance_type" {
  type        = string
  description = "EC2 instance type (set via TF_VAR_node_instance_type)"
  default     = "m7i-flex.large"
}

variable "app_volume_size_gb" {
  type    = number
  default = 100
}

variable "key_name" {
  type        = string
  description = "EC2 SSH key pair name (optional if using SSM only)"
  default     = ""
}

variable "db_instance_class" {
  type        = string
  description = "RDS instance class (set via TF_VAR_db_instance_class)"
  default     = "db.t3.micro"
}

variable "db_allocated_storage_gb" {
  type    = number
  default = 20
}

variable "db_username" {
  type    = string
  default = "mlair"
}

variable "db_name" {
  type    = string
  default = "mlair"
}

variable "redis_node_type" {
  type    = string
  default = "cache.t3.micro"
}

variable "enable_https" {
  type    = bool
  default = false
}

variable "acm_certificate_arn" {
  type        = string
  description = "ACM cert ARN for ALB HTTPS (required when enable_https=true)"
  default     = ""
}

variable "allowed_cidr_blocks" {
  type        = list(string)
  description = "CIDR allowed to reach ALB"
  default     = ["0.0.0.0/0"]
}

variable "repository_names" {
  type = list(string)
  default = [
    "cv-lifecycle-workload",
    "ml-air-api-cv-workload",
    "ml-air-api",
    "ml-air-frontend",
    "ml-air-scheduler",
    "ml-air-executor",
    "ml-air-realtime",
  ]
}

variable "image_tag" {
  type    = string
  default = "latest"
}
