#!/usr/bin/env python3
"""
Pytest tests for AgentCore runtimes.
Gets runtime endpoints from CDK stack outputs via conftest.py fixtures.
"""

import pytest


def test_agent_runtime(bedrock_agentcore_client, agentcore_runtime_arns):
    """Test the agent runtime endpoint"""
    endpoint_arn = agentcore_runtime_arns["agent_runtime_endpoint_arn"]

    if not endpoint_arn:
        pytest.fail("Agent runtime endpoint ARN not found in CDK stack outputs")

    response = bedrock_agentcore_client.invoke_agent_runtime(
        runtimeEndpointArn=endpoint_arn,
        inputText="Hello! What tools do you have available?",
        sessionId="test-agent-001",
    )

    assert response is not None
    assert "ResponseMetadata" in response


def test_mcp_server_runtime(bedrock_agentcore_client, agentcore_runtime_arns):
    """Test the MCP server runtime endpoint"""
    endpoint_arn = agentcore_runtime_arns["mcp_server_runtime_endpoint_arn"]

    if not endpoint_arn:
        pytest.fail("MCP server runtime endpoint ARN not found in CDK stack outputs")

    response = bedrock_agentcore_client.invoke_agent_runtime(
        runtimeEndpointArn=endpoint_arn,
        inputText="Can you list your available MCP tools?",
        sessionId="test-mcp-server-001",
    )

    assert response is not None
    assert "ResponseMetadata" in response


def test_agent_runtime_with_streaming(bedrock_agentcore_client, agentcore_runtime_arns):
    """Test the agent runtime with streaming response"""
    endpoint_arn = agentcore_runtime_arns["agent_runtime_endpoint_arn"]

    if not endpoint_arn:
        pytest.skip("Agent runtime endpoint ARN not found in CDK stack outputs")

    response = bedrock_agentcore_client.invoke_agent_runtime_with_response_stream(
        runtimeEndpointArn=endpoint_arn,
        inputText="Can you provide a longer response to test streaming?",
        sessionId="test-agent-streaming-001",
    )

    assert response is not None
    assert "eventStream" in response

    # Verify we can iterate through the stream
    event_count = 0
    for event in response["eventStream"]:
        event_count += 1
        if event_count > 10:  # Limit to avoid long test runs
            break

    assert event_count > 0
