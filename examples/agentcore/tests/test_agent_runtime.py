"""
Pytest tests for AgentCore runtime basic functionality.
Uses the InvokeAgentRuntime operation to test both agent and MCP server runtimes.
"""

import pytest


def test_agent_runtime_basic(bedrock_agentcore_client, agentcore_runtime_arns):
    """Test basic agent runtime functionality"""
    import json

    endpoint_arn = agentcore_runtime_arns["agent_runtime_arn"]

    if not endpoint_arn:
        pytest.fail("Agent runtime ARN not found in CDK stack outputs")

    # Prepare payload as JSON
    payload = json.dumps(
        {
            "prompt": "Hello! This is a test message to verify the agent runtime is working."
        }
    ).encode()

    import uuid

    response = bedrock_agentcore_client.invoke_agent_runtime(
        agentRuntimeArn=endpoint_arn,
        runtimeSessionId=str(uuid.uuid4()),
        payload=payload,
    )

    assert response is not None
    assert "ResponseMetadata" in response


def test_mcp_server_runtime_basic(bedrock_agentcore_client, agentcore_runtime_arns):
    """Test basic MCP server runtime functionality"""
    import json

    endpoint_arn = agentcore_runtime_arns["mcp_server_runtime_arn"]

    if not endpoint_arn:
        pytest.fail("MCP server runtime ARN not found in CDK stack outputs")

    # Prepare payload as JSON
    payload = json.dumps(
        {"prompt": "Hello MCP Server! Can you list your available tools?"}
    ).encode()

    response = bedrock_agentcore_client.invoke_agent_runtime(
        agentRuntimeArn=endpoint_arn,
        runtimeSessionId="test-mcp-server-basic-001",
        payload=payload,
    )

    assert response is not None
    assert "ResponseMetadata" in response


def test_agent_runtime_tool_query(bedrock_agentcore_client, agentcore_runtime_arns):
    """Test agent runtime with tool-related query"""
    import json

    endpoint_arn = agentcore_runtime_arns["agent_runtime_arn"]

    if not endpoint_arn:
        pytest.skip("Agent runtime ARN not found in CDK stack outputs")

    # Prepare payload as JSON
    payload = json.dumps(
        {"prompt": "What tools do you have available? Can you help me with a task?"}
    ).encode()

    response = bedrock_agentcore_client.invoke_agent_runtime(
        agentRuntimeArn=endpoint_arn,
        runtimeSessionId="test-agent-tools-001",
        payload=payload,
    )

    assert response is not None
    assert "ResponseMetadata" in response
