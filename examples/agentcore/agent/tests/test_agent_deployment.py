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

    def test_invoke_agent_weather_query(
        self, bedrock_agentcore_client, agent_runtime_endpoint_arn
    ):
        """Test agent invocation with a weather-related query."""
        # Extract endpoint ID from ARN
        endpoint_id = agent_runtime_endpoint_arn.split("/")[-1]

        try:
            response = bedrock_agentcore_client.invoke_agent_runtime(
                agentRuntimeEndpointId=endpoint_id,
                inputText="What's the weather like today?",
                sessionId="test-session-weather",
            )

            # Check response structure
            assert "completion" in response
            assert isinstance(response["completion"], str)
            assert len(response["completion"]) > 0

            # Check that response is weather-related (basic check)
            completion_lower = response["completion"].lower()
            weather_keywords = [
                "weather",
                "temperature",
                "forecast",
                "climate",
                "conditions",
            ]
            assert any(
                keyword in completion_lower for keyword in weather_keywords
            ), f"Response doesn't seem weather-related: {response['completion']}"

        except ClientError as e:
            if e.response["Error"]["Code"] == "ValidationException":
                pytest.skip(f"Agent not ready for invocation: {e}")
            else:
                pytest.fail(f"Failed to invoke agent: {e}")

    def test_invoke_agent_multiple_sessions(
        self, bedrock_agentcore_client, agent_runtime_endpoint_arn
    ):
        """Test that agent can handle multiple concurrent sessions."""
        # Extract endpoint ID from ARN
        endpoint_id = agent_runtime_endpoint_arn.split("/")[-1]

        sessions = ["test-session-1", "test-session-2", "test-session-3"]
        responses = []

        try:
            for session_id in sessions:
                response = bedrock_agentcore_client.invoke_agent_runtime(
                    agentRuntimeEndpointId=endpoint_id,
                    inputText=f"Hello from session {session_id}",
                    sessionId=session_id,
                )
                responses.append(response)

            # Verify all responses are valid
            for i, response in enumerate(responses):
                assert (
                    "completion" in response
                ), f"Session {sessions[i]} missing completion"
                assert isinstance(
                    response["completion"], str
                ), f"Session {sessions[i]} completion not string"
                assert (
                    len(response["completion"]) > 0
                ), f"Session {sessions[i]} empty completion"

        except ClientError as e:
            if e.response["Error"]["Code"] == "ValidationException":
                pytest.skip(f"Agent not ready for invocation: {e}")
            else:
                pytest.fail(f"Failed to invoke agent: {e}")

    def test_agent_response_time(
        self, bedrock_agentcore_client, agent_runtime_endpoint_arn
    ):
        """Test that agent responds within reasonable time limits."""

        # Extract endpoint ID from ARN
        endpoint_id = agent_runtime_endpoint_arn.split("/")[-1]

        try:
            start_time = time.time()

            response = bedrock_agentcore_client.invoke_agent_runtime(
                agentRuntimeEndpointId=endpoint_id,
                inputText="Give me a brief weather update.",
                sessionId="test-session-timing",
            )

            end_time = time.time()
            response_time = end_time - start_time

            # Check response is valid
            assert "completion" in response
            assert len(response["completion"]) > 0

            # Check response time is reasonable (less than 30 seconds)
            assert (
                response_time < 30.0
            ), f"Response took too long: {response_time:.2f} seconds"

        except ClientError as e:
            if e.response["Error"]["Code"] == "ValidationException":
                pytest.skip(f"Agent not ready for invocation: {e}")
            else:
                pytest.fail(f"Failed to invoke agent: {e}")
