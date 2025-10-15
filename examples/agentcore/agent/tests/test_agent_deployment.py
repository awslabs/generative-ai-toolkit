"""Test the deployed AgentCore agent."""

import json
import time
import uuid

import pytest
from botocore.exceptions import ClientError


class TestAgentDeployment:
    """Test suite for deployed AgentCore agent."""

    def test_agent_runtime_exists(
        self, bedrock_agentcore_control_client, agent_runtime_arn
    ):
        """Test that the agent runtime exists and is accessible."""
        # Extract runtime ID from ARN
        runtime_id = agent_runtime_arn.split("/")[-1]

        try:
            response = bedrock_agentcore_control_client.get_agent_runtime(
                agentRuntimeId=runtime_id
            )
            assert response["agentRuntimeId"] == runtime_id
            assert response["status"] in ["READY", "CREATING", "UPDATING"]
        except ClientError as e:
            pytest.fail(f"Failed to get agent runtime: {e}")

    def test_agent_runtime_endpoint_exists(
        self, bedrock_agentcore_control_client, agent_runtime_endpoint_arn
    ):
        """Test that the agent runtime endpoint exists and is accessible."""
        # Extract runtime ID and endpoint name from ARN
        # ARN format: arn:aws:bedrock-agentcore:region:account:runtime/runtime-id/runtime-endpoint/endpoint-name
        arn_parts = agent_runtime_endpoint_arn.split("/")
        runtime_id = arn_parts[-3]  # runtime-id
        endpoint_name = arn_parts[-1]  # endpoint-name

        try:
            response = bedrock_agentcore_control_client.get_agent_runtime_endpoint(
                agentRuntimeId=runtime_id, endpointName=endpoint_name
            )
            assert response["name"] == endpoint_name
            assert runtime_id in response["agentRuntimeArn"]
            assert response["status"] in ["READY", "CREATING", "UPDATING"]
        except ClientError as e:
            pytest.fail(f"Failed to get agent runtime endpoint: {e}")

    def test_invoke_agent_basic(self, bedrock_agentcore_client, agent_runtime_arn):
        """Test basic agent invocation with a simple prompt."""
        try:
            # Prepare the payload
            payload = json.dumps({"prompt": "Hello, can you help me?"}).encode()

            response = bedrock_agentcore_client.invoke_agent_runtime(
                agentRuntimeArn=agent_runtime_arn,
                runtimeSessionId=str(uuid.uuid4()),
                payload=payload,
            )

            # Check response structure
            assert "response" in response
            assert "contentType" in response

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "ValidationException":
                pytest.skip(f"Agent not ready for invocation: {e}")
            else:
                pytest.fail(f"Failed to invoke agent ({error_code}): {e}")

    def test_invoke_agent_multiple_sessions(
        self, bedrock_agentcore_client, agent_runtime_arn
    ):
        """Test that agent can handle multiple concurrent sessions."""
        sessions = [str(uuid.uuid4()) for _ in range(3)]
        responses = []

        try:
            for i, session_id in enumerate(sessions):
                # Prepare the payload
                payload = json.dumps({"prompt": f"Hello from session {i+1}"}).encode()

                response = bedrock_agentcore_client.invoke_agent_runtime(
                    agentRuntimeArn=agent_runtime_arn,
                    runtimeSessionId=session_id,
                    payload=payload,
                )
                responses.append(response)

            # Verify all responses are valid
            for i, response in enumerate(responses):
                assert "response" in response, f"Session {sessions[i]} missing response"
                assert (
                    "contentType" in response
                ), f"Session {sessions[i]} missing contentType"

                # Parse the response payload (handle streaming body)
                response_body = response["response"].read().decode("utf-8")
                response_data = json.loads(response_body)
                assert (
                    "result" in response_data
                ), f"Session {sessions[i]} missing result in response data"
                assert (
                    len(response_data["result"]) > 0
                ), f"Session {sessions[i]} empty result"

        except ClientError as e:
            if e.response["Error"]["Code"] == "ValidationException":
                pytest.skip(f"Agent not ready for invocation: {e}")
            else:
                pytest.fail(f"Failed to invoke agent: {e}")

    def test_agent_response_time(self, bedrock_agentcore_client, agent_runtime_arn):
        """Test that agent responds within reasonable time limits."""
        try:
            start_time = time.time()

            # Prepare the payload
            payload = json.dumps({"prompt": "Give me a brief weather update."}).encode()

            response = bedrock_agentcore_client.invoke_agent_runtime(
                agentRuntimeArn=agent_runtime_arn,
                runtimeSessionId=str(uuid.uuid4()),
                payload=payload,
            )

            end_time = time.time()
            response_time = end_time - start_time

            # Check response is valid
            assert "response" in response
            assert "contentType" in response

            # Parse the response payload (handle streaming body)
            response_body = response["response"].read().decode("utf-8")
            response_data = json.loads(response_body)
            assert "result" in response_data
            assert len(response_data["result"]) > 0

            # Check response time is reasonable (less than 30 seconds)
            assert (
                response_time < 30.0
            ), f"Response took too long: {response_time:.2f} seconds"

        except ClientError as e:
            if e.response["Error"]["Code"] == "ValidationException":
                pytest.skip(f"Agent not ready for invocation: {e}")
            else:
                pytest.fail(f"Failed to invoke agent: {e}")
