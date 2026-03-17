provider "aws" {
  region = var.aws_region
}

resource "aws_s3_bucket" "ml_artifacts" {
  bucket = "${var.project_name}-${var.environment}-bucket"

  tags = {
    Name        = "${var.project_name}-${var.environment}-bucket"
    Environment = var.environment
    Project     = var.project_name
    ManagedBy   = "terraform"
  }
}

resource "aws_s3_bucket_versioning" "ml_artifacts_versioning" {
  bucket = aws_s3_bucket.ml_artifacts.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "ml_artifacts_lifecycle" {
  bucket = aws_s3_bucket.ml_artifacts.id

  depends_on = [aws_s3_bucket_versioning.ml_artifacts_versioning]

  rule {
    id     = "archive-old-artifacts"
    status = "Enabled"

    transition {
      days          = var.artifact_retention_days
      storage_class = "GLACIER"
    }

    noncurrent_version_transition {
      noncurrent_days = var.artifact_retention_days
      storage_class   = "GLACIER"
    }
  }
}

resource "aws_iam_role" "ml_role" {
  name = "${var.project_name}-${var.environment}-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = {
    Environment = var.environment
    Project     = var.project_name
    ManagedBy   = "terraform"
  }
}

resource "aws_iam_policy" "ml_artifacts_policy" {
  name        = "${var.project_name}-${var.environment}-policy"
  description = "Allows read and write access to the ML artifacts S3 bucket"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.ml_artifacts.arn,
          "${aws_s3_bucket.ml_artifacts.arn}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ml_role_policy_attachment" {
  role       = aws_iam_role.ml_role.name
  policy_arn = aws_iam_policy.ml_artifacts_policy.arn
}