"""Pytest configuration for AgentCore MCP server tests."""

from typing import Any

import boto3
import pytest


@pytest.fixture(scope="session")
def cdk_outputs() -> dict[str, Any]:
    """Get CDK stack outputs for testing deployed MCP server."""
    stack_name = "AgentCoreStack"

    try:
        # Create CloudFormation client (uses environment variables for region)
        cf_client = boto3.client("cloudformation")

        # Get stack outputs using AWS SDK
        response = cf_client.describe_stacks(StackName=stack_name)

        outputs = {}
        if "Stacks" in response and len(response["Stacks"]) > 0:
            stack_outputs = response["Stacks"][0].get("Outputs", [])
            for output in stack_outputs:
                outputs[output["OutputKey"]] = output["OutputValue"]

        return outputs

    except Exception as e:
        pytest.skip(f"Could not get CDK outputs for stack '{stack_name}': {e}")


@pytest.fixture(scope="session")
def bedrock_agentcore_control_client():
    """Create Bedrock AgentCore Control client for testing."""
    return boto3.client("bedrock-agentcore-control")


@pytest.fixture(scope="session")
def bedrock_agentcore_client():
    """Create Bedrock AgentCore client for testing MCP server invocation."""
    return boto3.client("bedrock-agentcore")


@pytest.fixture(scope="session")
def mcp_server_runtime_arn(cdk_outputs) -> str:
    """Get MCP server runtime ARN from CDK outputs."""
    # Look for the output key that contains "McpServerRuntimeArn"
    for key, value in cdk_outputs.items():
        if "McpServerRuntimeArn" in key:
            return value
    pytest.skip("McpServerRuntimeArn not found in CDK outputs")


@pytest.fixture(scope="session")
def mcp_server_runtime_endpoint_arn(cdk_outputs) -> str:
    """Get MCP server runtime endpoint ARN from CDK outputs."""
    # Look for the output key that contains "McpServerRuntimeEndpointArn"
    for key, value in cdk_outputs.items():
        if "McpServerRuntimeEndpointArn" in key:
            return value
    pytest.skip("McpServerRuntimeEndpointArn not found in CDK outputs")
