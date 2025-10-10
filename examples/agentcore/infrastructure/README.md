# AgentCore Integration Infrastructure

This CDK project defines the AWS infrastructure needed to deploy the AgentCore integration example to Amazon Bedrock AgentCore Runtime with **automated Docker container building**.

## Overview

The infrastructure includes:
- **Automated Docker image building** using CDK DockerImageAsset
- Container registry (ECR) for agent and MCP server images
- AgentCore Runtime configuration with built container images
- IAM roles and policies for Bedrock access
- CloudWatch logging and monitoring

## Prerequisites

- AWS CLI configured with appropriate permissions
- Node.js and npm installed
- AWS CDK CLI installed (`npm install -g aws-cdk`)
- **Docker installed and running** (for container builds)

## Quick Start

```bash
# Install dependencies
npm install

# Bootstrap CDK (first time only)
cdk bootstrap

# Deploy with automatic container building
cdk deploy
```

## Deployment Process

When you run `cdk deploy`, the following happens automatically:

1. **Container Build Phase**:
   - CDK builds Docker image from `../agent/Dockerfile`
   - CDK builds Docker image from `../mcp-server/Dockerfile`
   - Images are tagged with content-based hashes

2. **Container Push Phase**:
   - CDK creates ECR repositories (if needed)
   - CDK pushes built images to ECR

3. **Infrastructure Deployment**:
   - AgentCore runtimes created with built image URIs
   - IAM roles and policies deployed
   - Runtime endpoints created

4. **Output**:
   - Agent runtime endpoint ARN for invocation
   - Image URIs and repository information

## Useful Commands

### CDK Commands
- `npm run build`   - Compile TypeScript to JavaScript
- `npm run watch`   - Watch for changes and compile
- `cdk deploy`      - Deploy with automatic container building
- `cdk diff`        - Compare deployed stack with current state
- `cdk synth`       - Synthesize CloudFormation template
- `cdk destroy`     - Destroy the deployed stack

## Architecture

The infrastructure creates:

### Container Assets
- **Agent Image Asset**: Built from `../agent/` with weather agent code
- **MCP Server Image Asset**: Built from `../mcp-server/` with weather tools

### AgentCore Resources
- **Agent Runtime**: HTTP endpoint for agent invocation
- **MCP Server Runtime**: Internal MCP protocol endpoint
- **Runtime Endpoints**: Public endpoints for agent access

### Supporting Infrastructure
- **ECR Repositories**: Automatically created for container images
- **IAM Roles**: Least-privilege execution roles for runtimes
- **CloudWatch Logs**: Centralized logging for both runtimes

## Container Updates

To update containers after code changes:
```bash
# Simply redeploy - CDK detects changes and rebuilds
cdk deploy
```

CDK automatically:
- Detects changes in container source code
- Rebuilds only changed containers
- Updates AgentCore runtimes with new image versions
- Maintains zero-downtime deployments

## Testing the Deployment

After deployment, test your AgentCore integration:

```bash
# Get the agent endpoint from CDK outputs
AGENT_ENDPOINT=$(aws cloudformation describe-stacks \
  --stack-name AgentCoreIntegrationStack \
  --query 'Stacks[0].Outputs[?OutputKey==`AgentRuntimeEndpointArn`].OutputValue' \
  --output text)

# Run tests against the deployed endpoint
cd ../tests
python test_evaluation.py --type agentcore --agentcore-endpoint "$AGENT_ENDPOINT"
```