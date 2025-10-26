"""Basic deployment verification tests for the AgentCore agent."""

# nosec B101 - Assert statements are standard in pytest test files

import os

import pytest
from botocore.exceptions import ClientError


class TestAgentDeployment:
    """Basic tests to verify the agent is deployed and functional."""

    def test_agent_runtime_exists(self, bedrock_agentcore_control_client):
        """Test that the agent runtime exists and is accessible."""
        # Extract runtime ID from ARN
        agent_runtime_arn = os.environ["AGENT_RUNTIME_ARN"]
        runtime_id = agent_runtime_arn.split("/")[-1]

        try:
            response = bedrock_agentcore_control_client.get_agent_runtime(
                agentRuntimeId=runtime_id
            )
            assert response["agentRuntimeId"] == runtime_id  # nosec B101
            assert response["status"] in ["READY", "CREATING", "UPDATING"]  # nosec B101
        except ClientError as e:
            pytest.fail(f"Failed to get agent runtime: {e}")

    def test_agent_runtime_endpoint_exists(self, bedrock_agentcore_control_client):
        """Test that the agent runtime endpoint exists and is accessible."""
        # Extract runtime ID and endpoint name from ARN
        # ARN format: arn:aws:bedrock-agentcore:region:account:runtime/runtime-id/runtime-endpoint/endpoint-name
        agent_runtime_endpoint_arn = os.environ["AGENT_RUNTIME_ENDPOINT_ARN"]
        arn_parts = agent_runtime_endpoint_arn.split("/")
        runtime_id = arn_parts[-3]  # runtime-id
        endpoint_name = arn_parts[-1]  # endpoint-name

        try:
            response = bedrock_agentcore_control_client.get_agent_runtime_endpoint(
                agentRuntimeId=runtime_id, endpointName=endpoint_name
            )
            assert response["name"] == endpoint_name  # nosec B101
            assert runtime_id in response["agentRuntimeArn"]  # nosec B101
            assert response["status"] in ["READY", "CREATING", "UPDATING"]  # nosec B101
        except ClientError as e:
            pytest.fail(f"Failed to get agent runtime endpoint: {e}")
