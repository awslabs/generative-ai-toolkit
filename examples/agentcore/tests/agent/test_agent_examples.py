"""
Example tests demonstrating how to interact with a deployed AgentCore agent.

This test suite serves as documentation and examples for:
- Basic agent invocation patterns
- Weather tool usage via MCP integration
- Common payload structures
- Response handling
- Session management

These tests demonstrate the agent's capabilities rather than exhaustively testing edge cases.
"""

import json
import os
import time
import uuid

import pytest
from botocore.exceptions import ClientError


class TestAgentExamples:
    """Example interactions with the deployed AgentCore weather agent."""

    def test_basic_agent_invocation(self, bedrock_agentcore_client):
        """Example: Basic agent invocation with simple greeting."""
        try:
            # Prepare the payload
            payload = json.dumps(
                {"input": {"prompt": "Hello! Can you tell me what you can help with?"}}
            ).encode()

            # Invoke the agent
            response = bedrock_agentcore_client.invoke_agent_runtime(
                agentRuntimeArn=os.environ["AGENT_RUNTIME_ARN"],
                runtimeSessionId=str(uuid.uuid4()),
                payload=payload,
            )

            # Parse the response
            response_body = response["response"].read().decode("utf-8")
            response_data = json.loads(response_body)

            # Verify response structure
            assert "result" in response_data
            result = response_data["result"]
            assert len(result) > 0

            print(f"Agent response: {result}")

        except ClientError as e:
            pytest.fail(f"Failed basic invocation ({e.response['Error']['Code']}): {e}")

    def test_weather_query_example(self, bedrock_agentcore_client):
        """Example: Getting current weather for a city using MCP tools."""
        try:
            # Ask for weather in a specific city
            payload = json.dumps(
                {"input": {"prompt": "What's the current weather in Amsterdam?"}}
            ).encode()

            response = bedrock_agentcore_client.invoke_agent_runtime(
                agentRuntimeArn=os.environ["AGENT_RUNTIME_ARN"],
                runtimeSessionId=str(uuid.uuid4()),
                payload=payload,
            )

            # Parse response
            response_body = response["response"].read().decode("utf-8")
            response_data = json.loads(response_body)

            assert "result" in response_data
            result = response_data["result"]
            assert len(result) > 0

            # Verify weather information is provided
            result_lower = result.lower()
            assert "amsterdam" in result_lower

            # Should contain weather-related information
            weather_indicators = ["temperature", "weather", "degrees", "conditions"]
            assert any(indicator in result_lower for indicator in weather_indicators)

            print(f"Weather response: {result}")

        except ClientError as e:
            pytest.fail(f"Failed weather query ({e.response['Error']['Code']}): {e}")

    def test_weather_forecast_example(self, bedrock_agentcore_client):
        """Example: Getting weather forecast using MCP tools."""
        try:
            payload = json.dumps(
                {
                    "input": {
                        "prompt": "Can you give me a 3-day weather forecast for London?"
                    }
                }
            ).encode()

            response = bedrock_agentcore_client.invoke_agent_runtime(
                agentRuntimeArn=os.environ["AGENT_RUNTIME_ARN"],
                runtimeSessionId=str(uuid.uuid4()),
                payload=payload,
            )

            response_body = response["response"].read().decode("utf-8")
            response_data = json.loads(response_body)

            assert "result" in response_data
            result = response_data["result"]
            assert len(result) > 0

            result_lower = result.lower()
            assert "london" in result_lower

            # Should contain forecast information
            forecast_indicators = ["forecast", "days", "tomorrow", "weather"]
            assert any(indicator in result_lower for indicator in forecast_indicators)

            print(f"Forecast response: {result}")

        except ClientError as e:
            pytest.fail(f"Failed forecast query ({e.response['Error']['Code']}): {e}")

    def test_multiple_cities_comparison_example(self, bedrock_agentcore_client):
        """Example: Comparing weather across multiple cities."""
        try:
            payload = json.dumps(
                {
                    "input": {
                        "prompt": "Compare the current weather between Paris and Berlin. Which city has better weather today?"
                    }
                }
            ).encode()

            response = bedrock_agentcore_client.invoke_agent_runtime(
                agentRuntimeArn=os.environ["AGENT_RUNTIME_ARN"],
                runtimeSessionId=str(uuid.uuid4()),
                payload=payload,
            )

            response_body = response["response"].read().decode("utf-8")
            response_data = json.loads(response_body)

            assert "result" in response_data
            result = response_data["result"]
            assert len(result) > 0

            result_lower = result.lower()

            # Should mention both cities
            assert "paris" in result_lower
            assert "berlin" in result_lower

            # Should contain comparison language
            comparison_indicators = ["compare", "better", "warmer", "cooler", "both"]
            assert any(indicator in result_lower for indicator in comparison_indicators)

            print(f"Comparison response: {result}")

        except ClientError as e:
            pytest.fail(f"Failed comparison query ({e.response['Error']['Code']}): {e}")

    def test_session_continuity_example(self, bedrock_agentcore_client):
        """Example: Maintaining context within a session."""
        session_id = str(uuid.uuid4())

        try:
            # First message: Ask about weather in a city
            payload1 = json.dumps(
                {"input": {"prompt": "What's the weather like in Tokyo today?"}}
            ).encode()

            response1 = bedrock_agentcore_client.invoke_agent_runtime(
                agentRuntimeArn=os.environ["AGENT_RUNTIME_ARN"],
                runtimeSessionId=session_id,
                payload=payload1,
            )

            response_body1 = response1["response"].read().decode("utf-8")
            response_data1 = json.loads(response_body1)
            assert "result" in response_data1

            print(f"First response: {response_data1['result']}")

            # Second message: Follow-up question
            payload2 = json.dumps(
                {
                    "input": {
                        "prompt": "What about tomorrow's forecast for the same city?"
                    }
                }
            ).encode()

            response2 = bedrock_agentcore_client.invoke_agent_runtime(
                agentRuntimeArn=os.environ["AGENT_RUNTIME_ARN"],
                runtimeSessionId=session_id,
                payload=payload2,
            )

            response_body2 = response2["response"].read().decode("utf-8")
            response_data2 = json.loads(response_body2)
            assert "result" in response_data2

            result2 = response_data2["result"]
            result2_lower = result2.lower()

            # Should handle the follow-up appropriately
            context_indicators = ["tokyo", "forecast", "tomorrow", "same"]
            assert any(indicator in result2_lower for indicator in context_indicators)

            print(f"Follow-up response: {result2}")

        except ClientError as e:
            pytest.fail(
                f"Failed session continuity ({e.response['Error']['Code']}): {e}"
            )

    def test_agentcore_payload_structure_example(self, bedrock_agentcore_client):
        """Example: AgentCore payload structure with optional metadata."""
        test_cases = [
            # Standard AgentCore format
            {"input": {"prompt": "What's the weather in Rome?"}},
            # With session ID
            {
                "input": {"prompt": "What's the weather in Madrid?"},
                "sessionId": str(uuid.uuid4()),
            },
            # With additional metadata
            {
                "input": {"prompt": "What's the weather in Vienna?"},
                "sessionId": str(uuid.uuid4()),
                "metadata": {"source": "example_test"},
            },
        ]

        try:
            for i, test_payload in enumerate(test_cases):
                payload = json.dumps(test_payload).encode()

                response = bedrock_agentcore_client.invoke_agent_runtime(
                    agentRuntimeArn=os.environ["AGENT_RUNTIME_ARN"],
                    runtimeSessionId=str(uuid.uuid4()),
                    payload=payload,
                )

                response_body = response["response"].read().decode("utf-8")
                response_data = json.loads(response_body)

                assert "result" in response_data
                result = response_data["result"]
                assert len(result) > 0

                print(f"Payload structure {i+1} response: {result}")

        except ClientError as e:
            pytest.fail(
                f"Failed payload structure test ({e.response['Error']['Code']}): {e}"
            )

    def test_error_handling_example(self, bedrock_agentcore_client):
        """Example: How the agent handles invalid city names."""
        try:
            payload = json.dumps(
                {"input": {"prompt": "What's the weather in Nonexistentcity12345?"}}
            ).encode()

            response = bedrock_agentcore_client.invoke_agent_runtime(
                agentRuntimeArn=os.environ["AGENT_RUNTIME_ARN"],
                runtimeSessionId=str(uuid.uuid4()),
                payload=payload,
            )

            response_body = response["response"].read().decode("utf-8")
            response_data = json.loads(response_body)

            assert "result" in response_data
            result = response_data["result"]
            assert len(result) > 0

            # Agent should handle the error gracefully
            print(f"Error handling response: {result}")

        except ClientError as e:
            pytest.fail(
                f"Failed error handling test ({e.response['Error']['Code']}): {e}"
            )

    def test_performance_example(self, bedrock_agentcore_client):
        """Example: Measuring agent response time."""
        try:
            payload = json.dumps(
                {"input": {"prompt": "What's the current weather in New York City?"}}
            ).encode()

            start_time = time.time()
            response = bedrock_agentcore_client.invoke_agent_runtime(
                agentRuntimeArn=os.environ["AGENT_RUNTIME_ARN"],
                runtimeSessionId=str(uuid.uuid4()),
                payload=payload,
            )
            end_time = time.time()

            response_body = response["response"].read().decode("utf-8")
            response_data = json.loads(response_body)

            assert "result" in response_data
            result = response_data["result"]
            assert len(result) > 0

            response_time = end_time - start_time
            print(f"Response time: {response_time:.2f} seconds")
            print(f"Response: {result}")

            # Should respond within reasonable time
            assert response_time < 30.0

        except ClientError as e:
            pytest.fail(f"Failed performance test ({e.response['Error']['Code']}): {e}")

    def test_non_weather_query_example(self, bedrock_agentcore_client):
        """Example: How the agent handles non-weather queries."""
        try:
            payload = json.dumps(
                {"input": {"prompt": "Can you tell me about the history of computers?"}}
            ).encode()

            response = bedrock_agentcore_client.invoke_agent_runtime(
                agentRuntimeArn=os.environ["AGENT_RUNTIME_ARN"],
                runtimeSessionId=str(uuid.uuid4()),
                payload=payload,
            )

            response_body = response["response"].read().decode("utf-8")
            response_data = json.loads(response_body)

            assert "result" in response_data
            result = response_data["result"]
            assert len(result) > 0

            print(f"Non-weather query response: {result}")

        except ClientError as e:
            pytest.fail(
                f"Failed non-weather query ({e.response['Error']['Code']}): {e}"
            )

    def test_tool_capabilities_inquiry_example(self, bedrock_agentcore_client):
        """Example: Asking the agent about its capabilities."""
        try:
            payload = json.dumps(
                {
                    "input": {
                        "prompt": "What weather tools and capabilities do you have available?"
                    }
                }
            ).encode()

            response = bedrock_agentcore_client.invoke_agent_runtime(
                agentRuntimeArn=os.environ["AGENT_RUNTIME_ARN"],
                runtimeSessionId=str(uuid.uuid4()),
                payload=payload,
            )

            response_body = response["response"].read().decode("utf-8")
            response_data = json.loads(response_body)

            assert "result" in response_data
            result = response_data["result"]
            assert len(result) > 0

            # Should mention weather capabilities
            result_lower = result.lower()
            capability_indicators = [
                "weather",
                "tool",
                "capability",
                "provide",
                "check",
            ]
            assert any(indicator in result_lower for indicator in capability_indicators)

            print(f"Capabilities response: {result}")

        except ClientError as e:
            pytest.fail(
                f"Failed capabilities inquiry ({e.response['Error']['Code']}): {e}"
            )
