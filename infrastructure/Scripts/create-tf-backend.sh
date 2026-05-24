#!/usr/bin/env bash
# One-time: create S3 bucket + DynamoDB table for Terraform state.
# Usage: ./create-tf-backend.sh cv-mlair-terraform-state-dev ap-southeast-1 cv-mlair-terraform-locks
set -euo pipefail
BUCKET="${1:?bucket name}"
REGION="${2:?region}"
TABLE="${3:?dynamodb lock table}"

aws s3api create-bucket \
  --bucket "$BUCKET" \
  --region "$REGION" \
  $( [[ "$REGION" != "us-east-1" ]] && echo --create-bucket-configuration LocationConstraint="$REGION" )

aws s3api put-bucket-versioning \
  --bucket "$BUCKET" \
  --versioning-configuration Status=Enabled

aws dynamodb create-table \
  --table-name "$TABLE" \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region "$REGION"

echo "Backend ready: bucket=$BUCKET table=$TABLE"
