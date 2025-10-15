"""Pytest configuration for AgentCore agent tests."""

from typing import Any

import boto3
import pytest


@pytest.fixture(scope="session")
def cdk_outputs() -> dict[str, Any]:
    """Get CDK stack outputs for testing deployed agent."""
    stack_name = "AgentCoreIntegrationStack"

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
    """Create Bedrock AgentCore client for testing agent invocation."""
    return boto3.client("bedrock-agentcore")


@pytest.fixture(scope="session")
def agent_runtime_arn(cdk_outputs) -> str:
    """Get agent runtime ARN from CDK outputs."""
    # Look for the output key that contains "AgentRuntimeArn"
    for key, value in cdk_outputs.items():
        if "AgentRuntimeArn" in key:
            return value
    pytest.skip("AgentRuntimeArn not found in CDK outputs")


@pytest.fixture(scope="session")
def agent_runtime_endpoint_arn(cdk_outputs) -> str:
    """Get agent runtime endpoint ARN from CDK outputs."""
    # Look for the output key that contains "AgentRuntimeEndpointArn"
    for key, value in cdk_outputs.items():
        if "AgentRuntimeEndpointArn" in key:
            return value
    pytest.skip("AgentRuntimeEndpointArn not found in CDK outputs")
