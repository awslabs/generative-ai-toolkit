"""Unified pytest configuration for AgentCore tests."""

import os
from pathlib import Path
from typing import Any

import boto3
import pytest


def get_cdk_stack_name() -> str:
    """Get CDK stack name from .env file, similar to config_loader.py."""

    # First try environment variable
    if stack_name := os.getenv("CDK_STACK_NAME"):
        return stack_name

    # Read from .env file relative to this script's location
    script_dir = Path(__file__).parent
    env_path = script_dir / "../.env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.startswith("CDK_STACK_NAME="):
                    return line.split("=", 1)[1].strip()

    raise RuntimeError("CDK_STACK_NAME not found in environment or .env file")


@pytest.fixture(scope="session")
def cdk_outputs() -> dict[str, Any]:
    """Get CDK stack outputs for testing deployed resources."""
    try:
        stack_name = get_cdk_stack_name()

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
        raise RuntimeError(
            f"Failed to get CDK outputs for stack '{stack_name}': {e}. "
            f"Ensure CDK stack is deployed with 'cdk deploy'."
        ) from e


@pytest.fixture(scope="session")
def bedrock_agentcore_control_client():
    """Create Bedrock AgentCore Control client for testing."""
    return boto3.client("bedrock-agentcore-control")


@pytest.fixture(scope="session")
def bedrock_agentcore_client():
    """Create Bedrock AgentCore client for testing runtime invocation."""
    return boto3.client("bedrock-agentcore")


# Agent-specific fixtures
@pytest.fixture(scope="session")
def agent_runtime_arn(cdk_outputs) -> str:
    """Get agent runtime ARN from CDK outputs."""
    # Look for the output key that contains "AgentRuntimeArn"
    for key, value in cdk_outputs.items():
        if "AgentRuntimeArn" in key:
            return value
    raise RuntimeError(
        "AgentRuntimeArn not found in CDK outputs. Ensure CDK stack is deployed correctly."
    )


@pytest.fixture(scope="session")
def agent_runtime_endpoint_arn(cdk_outputs) -> str:
    """Get agent runtime endpoint ARN from CDK outputs."""
    # Look for the output key that contains "AgentRuntimeEndpointArn"
    for key, value in cdk_outputs.items():
        if "AgentRuntimeEndpointArn" in key:
            return value
    raise RuntimeError(
        "AgentRuntimeEndpointArn not found in CDK outputs. Ensure CDK stack is deployed correctly."
    )


# MCP Server-specific fixtures
@pytest.fixture(scope="session")
def mcp_server_runtime_arn(cdk_outputs) -> str:
    """Get MCP server runtime ARN from CDK outputs."""
    # Look for the output key that contains "McpServerRuntimeArn"
    for key, value in cdk_outputs.items():
        if "McpServerRuntimeArn" in key:
            return value
    raise RuntimeError(
        "McpServerRuntimeArn not found in CDK outputs. Ensure CDK stack is deployed correctly."
    )


@pytest.fixture(scope="session")
def mcp_server_runtime_endpoint_arn(cdk_outputs) -> str:
    """Get MCP server runtime endpoint ARN from CDK outputs."""
    # Look for the output key that contains "McpServerRuntimeEndpointArn"
    for key, value in cdk_outputs.items():
        if "McpServerRuntimeEndpointArn" in key:
            return value
    raise RuntimeError(
        "McpServerRuntimeEndpointArn not found in CDK outputs. Ensure CDK stack is deployed correctly."
    )
