#!/bin/bash
set -e

ECR_REGISTRY="060337561279.dkr.ecr.eu-central-1.amazonaws.com"
ECR_REPO="cdk-hnb659fds-container-assets-060337561279-eu-central-1"
ROLE_ARN="arn:aws:iam::060337561279:role/MinimalAgentCoreRuntimeRole"
REGION="eu-central-1"
BASE_AGENT_NAME="minimal_agent"

# Create unique agent name with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
AGENT_NAME="${BASE_AGENT_NAME}_${TIMESTAMP}"

# Delete any existing agent runtimes starting with "minimal_agent"
echo "Deleting existing agent runtimes starting with '$BASE_AGENT_NAME'..."
EXISTING_RUNTIMES=$(aws bedrock-agentcore-control list-agent-runtimes \
  --region "$REGION" \
  --query "agentRuntimes[?starts_with(agentRuntimeName, '$BASE_AGENT_NAME')].agentRuntimeId" \
  --output text)

if [ -n "$EXISTING_RUNTIMES" ] && [ "$EXISTING_RUNTIMES" != "None" ]; then
  for RUNTIME_ID in $EXISTING_RUNTIMES; do
    if [ "$RUNTIME_ID" != "None" ]; then
      echo "Deleting endpoints for agent runtime: $RUNTIME_ID"
      # List and delete all non-DEFAULT endpoints for this runtime
      ENDPOINTS=$(aws bedrock-agentcore-control list-agent-runtime-endpoints \
        --agent-runtime-id "$RUNTIME_ID" \
        --region "$REGION" \
        --query "runtimeEndpoints[?name != 'DEFAULT'].name" \
        --output text)
      
      if [ -n "$ENDPOINTS" ] && [ "$ENDPOINTS" != "None" ]; then
        for ENDPOINT in $ENDPOINTS; do
          if [ "$ENDPOINT" != "None" ]; then
            echo "Deleting endpoint: $ENDPOINT"
            # Retry endpoint deletion up to 5 times with 10 second intervals
            for i in {1..5}; do
              if aws bedrock-agentcore-control delete-agent-runtime-endpoint \
                --agent-runtime-id "$RUNTIME_ID" \
                --endpoint-name "$ENDPOINT" \
                --region "$REGION" 2>/dev/null; then
                echo "Successfully deleted endpoint $ENDPOINT"
                break
              else
                echo "Attempt $i failed, waiting 10 seconds before retry..."
                sleep 10
              fi
            done
          fi
        done
        echo "Waiting for endpoint deletions to complete..."
        sleep 15
      fi
      
      echo "Deleting agent runtime: $RUNTIME_ID"
      aws bedrock-agentcore-control delete-agent-runtime \
        --agent-runtime-id "$RUNTIME_ID" \
        --region "$REGION" || echo "Failed to delete $RUNTIME_ID (may not exist)"
    fi
  done
  echo "Waiting for deletions to complete..."
  sleep 10
else
  echo "No existing runtimes found starting with '$BASE_AGENT_NAME'"
fi

# Create unique tag with timestamp
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
IMAGE_TAG="minimal-agent-${TIMESTAMP}"
VERSION=1

echo "Using image tag: $IMAGE_TAG"

echo "Building minimal AgentCore Runtime agent container..."
docker build -t minimal-agentcore-agent:latest --build-arg AGENT_VERSION=$VERSION .

echo "Pushing to ECR with unique tag..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_REGISTRY
docker tag minimal-agentcore-agent:latest $ECR_REGISTRY/$ECR_REPO:$IMAGE_TAG
docker push $ECR_REGISTRY/$ECR_REPO:$IMAGE_TAG

echo "Creating new AgentCore Runtime: $AGENT_NAME"
CONTAINER_URI="$ECR_REGISTRY/$ECR_REPO:$IMAGE_TAG"
echo "Container URI: $CONTAINER_URI"

RESPONSE=$(aws bedrock-agentcore-control create-agent-runtime \
  --agent-runtime-name "$AGENT_NAME" \
  --agent-runtime-artifact "containerConfiguration={containerUri=$CONTAINER_URI}" \
  --role-arn "$ROLE_ARN" \
  --network-configuration "networkMode=PUBLIC" \
  --region "$REGION" \
  --output json)

RUNTIME_ID=$(echo "$RESPONSE" | jq -r '.agentRuntimeId')
DEPLOYED_VERSION=$(echo "$RESPONSE" | jq -r '.agentRuntimeVersion')
echo "Successfully created new agent runtime: $RUNTIME_ID"
echo "Version: $DEPLOYED_VERSION"
echo "Container tag: $IMAGE_TAG"
