# AgentCore Integration Example

This example demonstrates how to build a weather agent with the Generative AI Toolkit and deploy it on Amazon Bedrock AgentCore Runtime. It showcases AI agent architecture with separate containerized components communicating via the Model Context Protocol (MCP).

## What is AgentCore?

Amazon Bedrock AgentCore Runtime provides a managed, serverless environment for deploying AI agents and their supporting services. Key advantages for this example:

- **Containerized Deployment**: Both agent and MCP server run in separate, scalable containers
- **Managed Infrastructure**: No server management, automatic scaling, and built-in monitoring
- **Enterprise Security**: IAM-based access control, Cognito authentication, and JWT token validation
- **Service Isolation**: Agent and tools run independently, enabling better reliability and scaling
- **Observability**: Built-in logging, tracing, and health monitoring

## Architecture

### Separated Components

**Agent Container** (`agent/`): 
- Weather assistant using `BedrockConverseAgent`
- Connects to MCP server for tool access
- Handles user interactions and conversation flow

**MCP Server Container** (`mcp-server/`):
- Provides weather tools via Model Context Protocol
- Modular tool architecture with separate modules per tool
- Independent scaling and deployment lifecycle

### Pydantic-Based Tools

This example demonstrates best practices for tool development:

- **Type Safety**: Pydantic models ensure robust input validation
- **Rich Documentation**: Model docstrings become tool descriptions for the LLM
- **JSON Schema Generation**: Automatic schema creation for tool parameters
- **Modular Design**: Each tool (`get_weather_tool.py`, `get_forecast_tool.py`) is self-contained
- **Maintainable Code**: Clear separation between models, business logic, and MCP registration

## Directory Structure

```
examples/agentcore/
├── agent/                    # Weather agent implementation
│   ├── agent.py             # Main agent with BedrockConverseAgent
│   ├── mcp_tool_manager.py  # MCP client integration
│   └── simple_mcp_client.py # AgentCore MCP client
├── mcp-server/              # Modular MCP tools server
│   ├── get_weather_tool.py  # Current weather tool + models
│   ├── get_forecast_tool.py # Forecast tool + models
│   └── mcp_server.py        # FastMCP server setup
├── infrastructure/          # CDK deployment stack
├── tests/                   # Comprehensive test suite
└── docker-compose.yml       # Local development environment
```

## Deployment Instructions

### Prerequisites

1. **AWS CLI configured** with appropriate permissions
2. **Node.js and npm** installed for CDK
3. **Docker** installed for container builds
4. **Python 3.13+** for local development

### Step-by-Step Deployment

#### 1. Install Dependencies

```bash
# Install CDK dependencies
cd infrastructure
npm install

# Install Python dependencies (optional, for local testing)
cd ../
pip install -r requirements.txt
```

#### 2. Configure Environment

```bash
# Set your AWS region
export AWS_REGION=us-east-1

# Choose a Bedrock model (ensure it's available in your region)
export BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
```

#### 3. Deploy Infrastructure

```bash
cd infrastructure

# Bootstrap CDK (first time only)
npx cdk bootstrap

# Optional: Set custom stack name (defaults to "{username}-agentcore-stack")
export CDK_STACK_NAME=MyWeatherAgent

# Deploy the stack
npx cdk deploy --all
```

This will create:
- Two AgentCore Runtime environments (agent and MCP server)
- ECR repositories for container images
- IAM roles and policies
- Cognito User Pool for authentication

#### 4. Build and Push Container Images

The CDK deployment automatically builds and pushes the Docker images to ECR. Monitor the deployment output for any build failures.

#### 5. Test the Deployment

```bash
# Run the test suite to verify deployment
cd ../tests

# Test agent deployment
python -m pytest agent/test_agent_deployment.py -v

# Test MCP server deployment
python -m pytest mcp_server/test_mcp_server_deployment.py -v
```

### Local Development

For local testing without AgentCore:

```bash
# Start MCP server
cd mcp-server
python mcp_server.py

# In another terminal, test the agent locally
cd ../tests
python -m pytest test_agent_local.py -v
```

## Getting Started

See `examples/agentcore/tests/` for comprehensive examples of:
- Local development and testing
- AgentCore deployment validation
- Tool schema verification
- End-to-end agent evaluation

The test suite demonstrates both local development workflows and deployment patterns, making it the best starting point for understanding how to use this example.