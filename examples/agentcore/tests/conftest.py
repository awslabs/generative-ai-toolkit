"""Unified pytest configuration for AgentCore tests."""

import os
import sys
from pathlib import Path
from typing import Any

import boto3
import pytest


def get_cdk_stack_name() -> str:
    """Get CDK stack name from .env file."""

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


# Set required environment variables to prevent SystemExit during test collection
# This must be done before any agent module imports
def _setup_required_environment_variables():
    """Set up required environment variables from CDK outputs or defaults."""

    # Set defaults for basic configuration
    os.environ["AWS_REGION"] = boto3.Session().region_name or "eu-central-1"
    os.environ["BEDROCK_MODEL_ID"] = "eu.anthropic.claude-sonnet-4-20250514-v1:0"

    # Get CDK stack name
    stack_name = get_cdk_stack_name()

    # Get CDK outputs
    cf_client = boto3.client("cloudformation")
    response = cf_client.describe_stacks(StackName=stack_name)
    stack_outputs = response["Stacks"][0]["Outputs"]

    # Set environment variables from CDK outputs
    for output in stack_outputs:
        key = output["OutputKey"]
        value = output["OutputValue"]

        if key.startswith("McpServerRuntimeArn"):
            os.environ["MCP_SERVER_RUNTIME_ARN"] = value
        elif key.startswith("AgentOAuthCredentialsSecretName"):
            os.environ["OAUTH_CREDENTIALS_SECRET_NAME"] = value
        elif key.startswith("OAuthAuthUserPoolId"):
            os.environ["OAUTH_USER_POOL_ID"] = value
        elif key.startswith("OAuthAuthUserPoolClientId"):
            os.environ["OAUTH_USER_POOL_CLIENT_ID"] = value
        elif key.startswith("AgentRuntimeArn"):
            os.environ["AGENT_RUNTIME_ARN"] = value
        elif key.startswith("AgentRuntimeEndpointArn"):
            os.environ["AGENT_RUNTIME_ENDPOINT_ARN"] = value
        elif key.startswith("McpServerRuntimeEndpointArn"):
            os.environ["MCP_SERVER_RUNTIME_ENDPOINT_ARN"] = value


# Set up environment variables before any imports
_setup_required_environment_variables()

# Add agent directory to Python path for imports
agent_dir = Path(__file__).parent.parent / "agent"
sys.path.insert(0, str(agent_dir))


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


# Agent-specific fixtures - now using environment variables


# MCP Server-specific fixtures - now using environment variables
