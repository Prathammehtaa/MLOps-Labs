variable "aws_region" {
  description = "AWS region where resources will be created"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Name of the project, used to name all resources"
  type        = string
  default     = "mlops-artifacts"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "artifact_retention_days" {
  description = "Number of days before old artifact versions are archived to Glacier"
  type        = number
  default     = 30
}