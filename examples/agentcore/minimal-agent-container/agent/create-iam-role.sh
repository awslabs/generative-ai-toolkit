#!/bin/bash
set -e

ROLE_NAME="MinimalAgentCoreRuntimeRole"
POLICY_NAME="MinimalAgentCoreRuntimePolicy"

echo "Creating IAM role for minimal AgentCore Runtime agent..."

# Create the IAM role (or update if exists)
echo "Creating IAM role: $ROLE_NAME"
if ! aws iam get-role --role-name $ROLE_NAME >/dev/null 2>&1; then
  aws iam create-role \
    --role-name $ROLE_NAME \
    --assume-role-policy-document file://trust-policy.json \
    --description "IAM role for minimal AgentCore Runtime with CloudWatch logging"
  echo "Role $ROLE_NAME created"
else
  echo "Role $ROLE_NAME already exists, updating policies"
fi

# Attach managed policy for CloudWatch Application Signals
echo "Attaching managed policy: CloudWatchLambdaApplicationSignalsExecutionRolePolicy"
aws iam attach-role-policy \
  --role-name $ROLE_NAME \
  --policy-arn arn:aws:iam::aws:policy/CloudWatchLambdaApplicationSignalsExecutionRolePolicy 2>/dev/null || true

# Update inline policy
echo "Updating inline policy: $POLICY_NAME"
aws iam put-role-policy \
  --role-name $ROLE_NAME \
  --policy-name $POLICY_NAME \
  --policy-document file://inline-policy.json

# Get role ARN
ROLE_ARN=$(aws iam get-role --role-name $ROLE_NAME --query 'Role.Arn' --output text)

echo "IAM role ready!"
echo "Role Name: $ROLE_NAME"
echo "Role ARN: $ROLE_ARN"
echo "Use this Role ARN when creating the AgentCore Runtime"
