# AgentCore Integration Example

This example demonstrates how to integrate the Generative AI Toolkit with Amazon Bedrock AgentCore Runtime. It showcases a weather agent with Pydantic-based tools deployed via MCP server architecture, with both components running in separate AgentCore Runtimes.

## Overview

The example consists of four main components:

1. **Agent** (`agent/`): Weather agent using BedrockConverseAgent with MCP client integration
2. **MCP Server** (`mcp-server/`): Standalone server providing weather tools via MCP protocol
3. **Infrastructure** (`infrastructure/`): TypeScript CDK stack for deploying to AgentCore Runtime
4. **Tests** (`tests/`): Local and AgentCore testing capabilities

## Quick Start

### Local Development
1. Start MCP server: `python mcp-server/server.py`
2. Run agent locally: `python agent/app.py`
3. Test with toolkit evaluation

### AgentCore Deployment
1. Install CDK dependencies: `cd infrastructure && npm install`
2. Deploy infrastructure: `npm run deploy`
3. Build and push containers to ECR
4. Test via runtime endpoint

## Key Features

- Maintains toolkit tracing and evaluation features
- Pydantic-based tool definitions for best practices
- Minimal CDK configuration for AgentCore
- Local testing that matches AgentCore behavior
- Comprehensive evaluation using toolkit's Case and Expect classes

## Directory Structure

```
examples/agentcore/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── docker-compose.yml        # Local development
├── agent/                    # Weather agent implementation
├── mcp-server/              # MCP tools server
├── infrastructure/          # TypeScript CDK deployment
└── tests/                   # Testing suite
```

For detailed setup and deployment instructions, see the individual component directories.