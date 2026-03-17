output "bucket_name" {
  description = "Name of the ML artifacts S3 bucket"
  value       = aws_s3_bucket.ml_artifacts.id
}

output "bucket_arn" {
  description = "ARN of the ML artifacts S3 bucket"
  value       = aws_s3_bucket.ml_artifacts.arn
}

output "iam_role_arn" {
  description = "ARN of the ML role that can access the bucket"
  value       = aws_iam_role.ml_role.arn
}

output "iam_policy_arn" {
  description = "ARN of the ML artifacts access policy"
  value       = aws_iam_policy.ml_artifacts_policy.arn
}