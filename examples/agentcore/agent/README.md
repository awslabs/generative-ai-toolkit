# Weather Agent for AgentCore

Weather agent that integrates with Amazon Bedrock AgentCore Runtime and communicates with MCP servers in other AgentCore Runtimes.

## Architecture

```
User → Agent Runtime Endpoint → Agent Container → MCP Server Runtime Endpoint → MCP Server Container
```

Uses the AWS Bedrock AgentCore `InvokeAgentRuntime` API for runtime-to-runtime communication instead of direct HTTP connections.

## ⚠️ **Important**

**The `InvokeAgentRuntime` API only works with MCP servers deployed in AgentCore Runtimes, not regular containers.**

## Configuration

Required environment variable:
- `MCP_SERVER_ENDPOINT`: ARN of the MCP server runtime endpoint (set by CDK)

Optional:
- `BEDROCK_MODEL_ID`: Model to use (default: anthropic.claude-3-5-sonnet-20241022-v2:0)
- `AWS_REGION`: AWS region (default: us-east-1)

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
export MCP_SERVER_ENDPOINT=arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime-endpoint/mcp-server
export AWS_REGION=us-east-1

# Configure AWS credentials
aws configure

# Run agent
python app.py
```

## Container Build

```bash
docker build -t weather-agent .
```

## Key Components

- `weather_agent.py`: Main agent with BedrockConverseAgent + AgentCore MCP client
- `agentcore_mcp_client.py`: MCP client using AWS SDK for runtime communication
- `app.py`: Flask server with AgentCore endpoints (`/invocations`, `/ping`)
- `models.py`: Pydantic models for request/response handling

## AgentCore Endpoints

- `POST /invocations`: Main agent interaction (port 8080)
- `GET /ping`: Health check
- `GET /health`: Detailed status

The agent automatically discovers and registers tools from the MCP server at startup.