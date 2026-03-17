# Lab 4 - Terraform: MLOps Artifact Storage Infrastructure

## Overview
This lab uses Terraform to provision MLOps artifact storage infrastructure on AWS.
Instead of the reference lab's basic EC2 + VPC setup, this lab provisions an S3 bucket 
for storing ML model artifacts with versioning, lifecycle management, and IAM access control.

## Infrastructure Provisioned
- S3 bucket with versioning and lifecycle policy (archives to Glacier after 30 days)
- IAM role for SageMaker access
- IAM policy with least-privilege read/write access to the bucket
- IAM policy attachment wiring the role and policy together

## Prerequisites
- Terraform v1.14+ (`brew install hashicorp/tap/terraform`)
- AWS CLI v2 (`brew install awscli`)
- AWS credentials configured (`aws configure`)

## How to Run
```bash
# Initialize
terraform init

# Preview changes
terraform plan

# Apply
terraform apply

# Destroy when done
terraform destroy
```

## Verify in AWS Console
- **S3** → confirm `mlops-artifacts-dev-bucket` exists with versioning enabled and lifecycle rule configured
- **IAM → Roles** → confirm `mlops-artifacts-dev-role` exists with `mlops-artifacts-dev-policy` attached