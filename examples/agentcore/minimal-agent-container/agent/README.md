# Minimal AgentCore Runtime Agent

A minimal containerized agent for Amazon Bedrock AgentCore Runtime that writes "Hello Message" to CloudWatch logs.

## Purpose

This minimal setup establishes the correct agent container configuration capable of writing to CloudWatch logs - a crucial step before further development. Without functional logging to CloudWatch, troubleshooting complex agents becomes impossible.

## Files

- `agent.py` - Minimal Python agent that logs to CloudWatch
- `Dockerfile` - Container definition for AgentCore Runtime
- `requirements.txt` - Python dependencies
- `create-iam-role.sh` - Script to create IAM role for AgentCore Runtime
- `build.sh` - Script to build the container
- `push.sh` - Script to push container to ECR

## Usage

### 1. Create IAM Role

```bash
./create-iam-role.sh
```

### 2. Build Container

```bash
./build.sh
```

### 3. Test Locally

```bash
docker run --rm minimal-agentcore-agent:latest
```

### 4. Push to ECR

```bash
./push.sh
```

## IAM Permissions

The IAM role includes essential permissions for:
- CloudWatch Logs (create log groups/streams, put log events)
- X-Ray tracing
- CloudWatch metrics
- AgentCore workload access tokens

## Container Features

- Based on Python 3.12-slim
- Non-root user (agentcore:1000)
- Health check endpoint
- CloudWatch logging configuration

## Next Steps

1. Deploy as AgentCore Runtime using the AWS console
2. Verify logs appear in CloudWatch
3. Add actual agent functionality
4. Implement proper error handling and monitoring
