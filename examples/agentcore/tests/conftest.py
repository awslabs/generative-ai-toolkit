"""
Pytest configuration and fixtures for AgentCore tests.
"""

import boto3
import pytest
from typing import Dict, Any


@pytest.fixture(scope="session")
def agentcore_runtime_arns() -> Dict[str, str]:
    """
    Get AgentCore runtime ARNs from CDK stack outputs.

    Returns:
        Dictionary with agent and mcp_server runtime endpoint ARNs
    """
    cloudformation = boto3.client("cloudformation")

    try:
        response = cloudformation.describe_stacks(StackName="AgentCoreIntegrationStack")

        stack = response["Stacks"][0]
        outputs = {
            output["OutputKey"]: output["OutputValue"]
            for output in stack.get("Outputs", [])
        }

        # Find outputs by prefix to handle hash postfixes
        agent_runtime_arn = None
        mcp_server_runtime_arn = None

        for key, value in outputs.items():
            if key.startswith("AgentCoreRuntimesAgentRuntimeArn"):
                agent_runtime_arn = value
            elif key.startswith("AgentCoreRuntimesMcpServerRuntimeArn"):
                mcp_server_runtime_arn = value

        return {
            "agent_runtime_arn": agent_runtime_arn,
            "mcp_server_runtime_arn": mcp_server_runtime_arn,
        }

    except Exception as e:
        pytest.fail(f"Failed to get CDK stack outputs: {e}")


@pytest.fixture(scope="session")
def bedrock_agentcore_client():
    """
    Create a Bedrock AgentCore client for testing.

    Returns:
        boto3 client for bedrock-agentcore service
    """
    return boto3.client("bedrock-agentcore")
