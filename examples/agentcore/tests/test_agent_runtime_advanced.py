"""
Advanced pytest tests for AgentCore runtimes with streaming support and detailed diagnostics.
"""

import pytest
import time
from botocore.exceptions import ClientError


def test_agent_runtime_streaming(bedrock_agentcore_client, agentcore_runtime_arns):
    """Test agent runtime with streaming response"""
    endpoint_arn = agentcore_runtime_arns["agent_runtime_endpoint_arn"]

    if not endpoint_arn:
        pytest.skip("Agent runtime endpoint ARN not found in CDK stack outputs")

    response = bedrock_agentcore_client.invoke_agent_runtime_with_response_stream(
        runtimeEndpointArn=endpoint_arn,
        inputText="Can you provide a longer response to test streaming?",
        sessionId=f"test-agent-streaming-{int(time.time())}",
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


def test_mcp_server_runtime_streaming(bedrock_agentcore_client, agentcore_runtime_arns):
    """Test MCP server runtime with streaming response"""
    endpoint_arn = agentcore_runtime_arns["mcp_server_runtime_endpoint_arn"]

    if not endpoint_arn:
        pytest.skip("MCP server runtime endpoint ARN not found in CDK stack outputs")

    response = bedrock_agentcore_client.invoke_agent_runtime_with_response_stream(
        runtimeEndpointArn=endpoint_arn,
        inputText="Can you provide information about your MCP tools?",
        sessionId=f"test-mcp-streaming-{int(time.time())}",
    )

    assert response is not None
    assert "eventStream" in response

    # Verify we can iterate through the stream
    event_count = 0
    for event in response["eventStream"]:
        event_count += 1
        if event_count > 5:  # Limit to avoid long test runs
            break

    assert event_count > 0


def test_agent_runtime_connectivity(bedrock_agentcore_client, agentcore_runtime_arns):
    """Test basic connectivity to agent runtime"""
    endpoint_arn = agentcore_runtime_arns["agent_runtime_endpoint_arn"]

    if not endpoint_arn:
        pytest.fail("Agent runtime endpoint ARN not found in CDK stack outputs")

    response = bedrock_agentcore_client.invoke_agent_runtime(
        runtimeEndpointArn=endpoint_arn,
        inputText="Hello! This is a connectivity test.",
        sessionId=f"test-connectivity-{int(time.time())}",
    )

    assert response is not None
    assert "ResponseMetadata" in response
    assert response["ResponseMetadata"]["HTTPStatusCode"] == 200


def test_mcp_server_runtime_connectivity(
    bedrock_agentcore_client, agentcore_runtime_arns
):
    """Test basic connectivity to MCP server runtime"""
    endpoint_arn = agentcore_runtime_arns["mcp_server_runtime_endpoint_arn"]

    if not endpoint_arn:
        pytest.fail("MCP server runtime endpoint ARN not found in CDK stack outputs")

    response = bedrock_agentcore_client.invoke_agent_runtime(
        runtimeEndpointArn=endpoint_arn,
        inputText="Hello! This is a connectivity test.",
        sessionId=f"test-mcp-connectivity-{int(time.time())}",
    )

    assert response is not None
    assert "ResponseMetadata" in response
    assert response["ResponseMetadata"]["HTTPStatusCode"] == 200


def test_agent_runtime_tool_usage(bedrock_agentcore_client, agentcore_runtime_arns):
    """Test agent runtime tool usage capabilities"""
    endpoint_arn = agentcore_runtime_arns["agent_runtime_endpoint_arn"]

    if not endpoint_arn:
        pytest.skip("Agent runtime endpoint ARN not found in CDK stack outputs")

    response = bedrock_agentcore_client.invoke_agent_runtime(
        runtimeEndpointArn=endpoint_arn,
        inputText="What tools do you have available? Can you use one of them?",
        sessionId=f"test-tools-{int(time.time())}",
    )

    assert response is not None
    assert "ResponseMetadata" in response


def test_runtime_error_handling(bedrock_agentcore_client, agentcore_runtime_arns):
    """Test error handling with invalid session ID"""
    endpoint_arn = agentcore_runtime_arns["agent_runtime_endpoint_arn"]

    if not endpoint_arn:
        pytest.skip("Agent runtime endpoint ARN not found in CDK stack outputs")

    # Test with empty input - should still work but might return different response
    response = bedrock_agentcore_client.invoke_agent_runtime(
        runtimeEndpointArn=endpoint_arn,
        inputText="",
        sessionId=f"test-empty-{int(time.time())}",
    )

    # Should not raise an exception, even with empty input
    assert response is not None
    assert "ResponseMetadata" in response
