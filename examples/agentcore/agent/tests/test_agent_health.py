"""Health and monitoring tests for the deployed AgentCore agent."""

import concurrent.futures

import pytest
from botocore.exceptions import ClientError


class TestAgentHealth:
    """Test suite for agent health and monitoring."""

    def test_agent_runtime_status(
        self, bedrock_agentcore_control_client, agent_runtime_arn
    ):
        """Test that the agent runtime is in a healthy state."""
        # Extract runtime ID from ARN
        runtime_id = agent_runtime_arn.split("/")[-1]

        try:
            response = bedrock_agentcore_control_client.get_agent_runtime(
                agentRuntimeId=runtime_id
            )

            # Check runtime is active
            assert (
                response["status"] == "READY"
            ), f"Runtime status is {response['status']}, expected READY"

            # Check basic runtime properties
            assert "agentRuntimeName" in response
            assert "roleArn" in response
            assert "createdAt" in response

        except ClientError as e:
            pytest.fail(f"Failed to get agent runtime status: {e}")

    def test_agent_endpoint_status(
        self, bedrock_agentcore_control_client, agent_runtime_endpoint_arn
    ):
        """Test that the agent endpoint is in a healthy state."""
        # Extract runtime ID and endpoint name from ARN
        # ARN format: arn:aws:bedrock-agentcore:region:account:runtime/runtime-id/runtime-endpoint/endpoint-name
        arn_parts = agent_runtime_endpoint_arn.split("/")
        runtime_id = arn_parts[-3]  # runtime-id
        endpoint_name = arn_parts[-1]  # endpoint-name

        try:
            response = bedrock_agentcore_control_client.get_agent_runtime_endpoint(
                agentRuntimeId=runtime_id, endpointName=endpoint_name
            )

            # Check endpoint is active
            assert (
                response["status"] == "READY"
            ), f"Endpoint status is {response['status']}, expected READY"

            # Check basic endpoint properties
            assert "name" in response
            assert "id" in response
            assert "createdAt" in response

        except ClientError as e:
            pytest.fail(f"Failed to get agent endpoint status: {e}")

    def test_agent_error_handling(
        self, bedrock_agentcore_client, agent_runtime_endpoint_arn
    ):
        """Test that agent handles invalid inputs gracefully."""
        # Extract endpoint ID from ARN
        endpoint_id = agent_runtime_endpoint_arn.split("/")[-1]

        # Test with empty input
        try:
            response = bedrock_agentcore_client.invoke_agent_runtime(
                agentRuntimeEndpointId=endpoint_id,
                inputText="",
                sessionId="test-session-empty",
            )

            # Should still get a response, even if input is empty
            assert "completion" in response

        except ClientError as e:
            # Some validation errors are acceptable for empty input
            if e.response["Error"]["Code"] not in ["ValidationException"]:
                pytest.fail(f"Unexpected error with empty input: {e}")

    def test_agent_session_isolation(
        self, bedrock_agentcore_client, agent_runtime_endpoint_arn
    ):
        """Test that different sessions are properly isolated."""
        # Extract endpoint ID from ARN
        endpoint_id = agent_runtime_endpoint_arn.split("/")[-1]

        try:
            # Send different messages to different sessions
            response1 = bedrock_agentcore_client.invoke_agent_runtime(
                agentRuntimeEndpointId=endpoint_id,
                inputText="My name is Alice",
                sessionId="test-session-alice",
            )

            response2 = bedrock_agentcore_client.invoke_agent_runtime(
                agentRuntimeEndpointId=endpoint_id,
                inputText="My name is Bob",
                sessionId="test-session-bob",
            )

            # Both should get valid responses
            assert "completion" in response1
            assert "completion" in response2
            assert len(response1["completion"]) > 0
            assert len(response2["completion"]) > 0

            # Now ask each session about their name
            response1_name = bedrock_agentcore_client.invoke_agent_runtime(
                agentRuntimeEndpointId=endpoint_id,
                inputText="What is my name?",
                sessionId="test-session-alice",
            )

            response2_name = bedrock_agentcore_client.invoke_agent_runtime(
                agentRuntimeEndpointId=endpoint_id,
                inputText="What is my name?",
                sessionId="test-session-bob",
            )

            # Responses should be different (basic check for session isolation)
            assert (
                response1_name["completion"] != response2_name["completion"]
            ), "Sessions may not be properly isolated"

        except ClientError as e:
            if e.response["Error"]["Code"] == "ValidationException":
                pytest.skip(f"Agent not ready for invocation: {e}")
            else:
                pytest.fail(f"Failed to test session isolation: {e}")

    def test_agent_concurrent_requests(
        self, bedrock_agentcore_client, agent_runtime_endpoint_arn
    ):
        """Test that agent can handle concurrent requests."""

        # Extract endpoint ID from ARN
        endpoint_id = agent_runtime_endpoint_arn.split("/")[-1]

        def invoke_agent(session_suffix):
            """Helper function to invoke agent."""
            return bedrock_agentcore_client.invoke_agent_runtime(
                agentRuntimeEndpointId=endpoint_id,
                inputText=f"Hello from concurrent request {session_suffix}",
                sessionId=f"test-concurrent-{session_suffix}",
            )

        try:
            # Submit multiple concurrent requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(invoke_agent, i) for i in range(3)]

                # Wait for all requests to complete
                responses = []
                for future in concurrent.futures.as_completed(futures, timeout=60):
                    try:
                        response = future.result()
                        responses.append(response)
                    except Exception as e:
                        pytest.fail(f"Concurrent request failed: {e}")

            # Verify all responses are valid
            assert len(responses) == 3, f"Expected 3 responses, got {len(responses)}"

            for i, response in enumerate(responses):
                assert "completion" in response, f"Response {i} missing completion"
                assert (
                    len(response["completion"]) > 0
                ), f"Response {i} has empty completion"

        except ClientError as e:
            if e.response["Error"]["Code"] == "ValidationException":
                pytest.skip(f"Agent not ready for invocation: {e}")
            else:
                pytest.fail(f"Failed to test concurrent requests: {e}")
        except concurrent.futures.TimeoutError:
            pytest.fail("Concurrent requests timed out")
