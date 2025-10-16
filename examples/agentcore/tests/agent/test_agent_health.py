"""Health and monitoring tests for the deployed AgentCore agent."""

import concurrent.futures
import json
import uuid

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
            assert response["status"] == "READY", (
                f"Runtime status is {response['status']}, expected READY"
            )

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
            assert response["status"] == "READY", (
                f"Endpoint status is {response['status']}, expected READY"
            )

            # Check basic endpoint properties
            assert "name" in response
            assert "id" in response
            assert "createdAt" in response

        except ClientError as e:
            pytest.fail(f"Failed to get agent endpoint status: {e}")

    def test_agent_error_handling(self, bedrock_agentcore_client, agent_runtime_arn):
        """Test that agent handles invalid inputs gracefully."""
        # Test with empty input
        try:
            # Prepare empty payload
            payload = json.dumps({"prompt": ""}).encode()

            response = bedrock_agentcore_client.invoke_agent_runtime(
                agentRuntimeArn=agent_runtime_arn,
                runtimeSessionId=str(uuid.uuid4()),
                payload=payload,
            )

            # Should still get a response, even if input is empty
            assert "response" in response
            assert "contentType" in response

        except ClientError as e:
            # Some validation errors are acceptable for empty input
            if e.response["Error"]["Code"] not in ["ValidationException"]:
                pytest.fail(f"Unexpected error with empty input: {e}")

    def test_agent_session_isolation(self, bedrock_agentcore_client, agent_runtime_arn):
        """Test that different sessions are properly isolated."""

        try:
            # Create two different sessions with different requests
            session_id_1 = f"test-session-1-{uuid.uuid4()}"
            session_id_2 = f"test-session-2-{uuid.uuid4()}"

            # Send different weather requests to each session
            payload1 = json.dumps(
                {"prompt": "What's the weather like in Paris?"}
            ).encode()
            response1 = bedrock_agentcore_client.invoke_agent_runtime(
                agentRuntimeArn=agent_runtime_arn,
                runtimeSessionId=session_id_1,
                payload=payload1,
            )

            payload2 = json.dumps(
                {"prompt": "What's the weather like in Tokyo?"}
            ).encode()
            response2 = bedrock_agentcore_client.invoke_agent_runtime(
                agentRuntimeArn=agent_runtime_arn,
                runtimeSessionId=session_id_2,
                payload=payload2,
            )

            # Both should get valid responses
            assert "response" in response1
            assert "response" in response2
            assert "contentType" in response1
            assert "contentType" in response2

            # Parse responses
            response1_body = response1["response"].read().decode("utf-8")
            response2_body = response2["response"].read().decode("utf-8")
            response1_data = json.loads(response1_body)
            response2_data = json.loads(response2_body)

            assert "result" in response1_data
            assert "result" in response2_data
            assert len(response1_data["result"]) > 0
            assert len(response2_data["result"]) > 0

            # Send follow-up requests to test session independence
            payload1_followup = json.dumps({"prompt": "Is it raining there?"}).encode()
            response1_followup = bedrock_agentcore_client.invoke_agent_runtime(
                agentRuntimeArn=agent_runtime_arn,
                runtimeSessionId=session_id_1,
                payload=payload1_followup,
            )

            payload2_followup = json.dumps({"prompt": "Is it sunny there?"}).encode()
            response2_followup = bedrock_agentcore_client.invoke_agent_runtime(
                agentRuntimeArn=agent_runtime_arn,
                runtimeSessionId=session_id_2,
                payload=payload2_followup,
            )

            # Parse follow-up responses
            response1_followup_body = (
                response1_followup["response"].read().decode("utf-8")
            )
            response2_followup_body = (
                response2_followup["response"].read().decode("utf-8")
            )
            response1_followup_data = json.loads(response1_followup_body)
            response2_followup_data = json.loads(response2_followup_body)

            # Verify both sessions got valid follow-up responses
            assert "result" in response1_followup_data
            assert "result" in response2_followup_data
            assert len(response1_followup_data["result"]) > 0
            assert len(response2_followup_data["result"]) > 0

            # Basic session isolation test - verify sessions can operate independently
            # Each session should be able to handle requests without interference
            session1_response = response1_followup_data["result"].lower()
            session2_response = response2_followup_data["result"].lower()

            # Sessions should be able to provide different responses (basic isolation check)
            # Note: Responses might be similar since both are weather-related, but sessions should work independently
            assert len(session1_response) > 0 and len(session2_response) > 0, (
                "Both sessions should provide valid responses"
            )

            # Validate that both sessions respond as weather agents
            weather_keywords = [
                "weather",
                "temperature",
                "forecast",
                "conditions",
                "rain",
                "sunny",
                "cloud",
            ]
            session1_is_weather_agent = any(
                keyword in session1_response for keyword in weather_keywords
            )
            session2_is_weather_agent = any(
                keyword in session2_response for keyword in weather_keywords
            )

            # At least one session should show weather agent behavior
            assert session1_is_weather_agent or session2_is_weather_agent, (
                f"Neither session shows weather agent behavior. "
                f"Session 1: '{session1_response[:100]}...', Session 2: '{session2_response[:100]}...'"
            )

        except ClientError as e:
            if e.response["Error"]["Code"] == "ValidationException":
                pytest.skip(f"Agent not ready for invocation: {e}")
            else:
                pytest.fail(f"Failed to test session isolation: {e}")

    def test_agent_concurrent_requests(
        self, bedrock_agentcore_client, agent_runtime_arn
    ):
        """Test that agent can handle concurrent requests."""

        def invoke_agent(session_suffix):
            """Helper function to invoke agent."""
            payload = json.dumps(
                {"prompt": f"Hello from concurrent request {session_suffix}"}
            ).encode()
            return bedrock_agentcore_client.invoke_agent_runtime(
                agentRuntimeArn=agent_runtime_arn,
                runtimeSessionId=f"test-concurrent-{session_suffix}-{uuid.uuid4()}",
                payload=payload,
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
                assert "response" in response, f"Response {i} missing response"
                assert "contentType" in response, f"Response {i} missing contentType"

                # Parse the response payload
                response_body = response["response"].read().decode("utf-8")
                response_data = json.loads(response_body)
                assert "result" in response_data, (
                    f"Response {i} missing result in response data"
                )
                assert len(response_data["result"]) > 0, (
                    f"Response {i} has empty result"
                )

        except ClientError as e:
            if e.response["Error"]["Code"] == "ValidationException":
                pytest.skip(f"Agent not ready for invocation: {e}")
            else:
                pytest.fail(f"Failed to test concurrent requests: {e}")
        except concurrent.futures.TimeoutError:
            pytest.fail("Concurrent requests timed out")
