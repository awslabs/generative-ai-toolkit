"""
Example tests demonstrating how to interact with a deployed AgentCore agent using JWT authentication.

This test suite serves as documentation and examples for:
- Basic agent invocation patterns with JWT OAuth
- Weather tool usage via MCP integration
- Common payload structures
- Response handling
- Session management

Note: This file uses assert statements for test validation (B101 suppressed)  # nosec B101

These tests demonstrate the agent's capabilities rather than exhaustively testing edge cases.

Environment Variables Required:
    - AWS_REGION: AWS region where resources are deployed
    - CLIENT_USER_CREDENTIALS_SECRET_NAME: Secret name for client user credentials
    - OAUTH_USER_POOL_ID: Cognito User Pool ID
    - OAUTH_USER_POOL_CLIENT_ID: Cognito User Pool Client ID
    - AGENT_RUNTIME_ARN: Agent runtime ARN to invoke
"""

# nosec B101

import json
import os
import time
import urllib.parse
import uuid
from typing import Any

import pytest
import requests


class TestAgentExamples:
    """Example interactions with the deployed AgentCore weather agent using JWT authentication."""

    def _invoke_agent_with_jwt(
        self, access_token: str, prompt: str, session_id: str = None
    ) -> dict[str, Any]:
        """Helper method to invoke the agent runtime using JWT bearer token."""
        agent_runtime_arn = os.environ["AGENT_RUNTIME_ARN"]
        region = os.environ["AWS_REGION"]

        # URL encode the agent ARN
        escaped_agent_arn = urllib.parse.quote(agent_runtime_arn, safe="")

        # Construct the URL
        url = f"https://bedrock-agentcore.{region}.amazonaws.com/runtimes/{escaped_agent_arn}/invocations?qualifier=DEFAULT"

        # Set up headers
        if session_id is None:
            session_id = f"test-session-{uuid.uuid4().hex}"
        trace_id = f"test-trace-{uuid.uuid4().hex[:16]}"

        headers = {
            "Authorization": f"Bearer {access_token}",
            "X-Amzn-Trace-Id": trace_id,
            "Content-Type": "application/json",
            "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": session_id,
        }

        # Prepare payload in the correct format expected by AgentCore
        payload = {"input": {"prompt": prompt}}

        response = requests.post(
            url, headers=headers, data=json.dumps(payload), timeout=60
        )

        if response.status_code == 200:
            return response.json()
        else:
            # Try to get error details
            try:
                error_data = response.json()
                error_msg = f"Agent invocation failed with status {response.status_code}: {json.dumps(error_data)}"
            except json.JSONDecodeError:
                error_msg = f"Agent invocation failed with status {response.status_code}: {response.text}"

            raise Exception(error_msg)

    def test_basic_agent_invocation(self, jwt_token):
        """Example: Basic agent invocation with simple greeting."""
        try:
            # Invoke the agent
            runtime_session_id = str(uuid.uuid4())
            response_data = self._invoke_agent_with_jwt(
                jwt_token,
                "Hello! Can you tell me what you can help with?",
                runtime_session_id,
            )

            print(f"runtime_session_id: {runtime_session_id}")

            # Verify response structure
            assert "result" in response_data  # nosec B101
            result = response_data["result"]
            assert len(result) > 0  # nosec B101

            print(f"Agent response: {result}")

        except Exception as e:
            pytest.fail(f"Failed basic invocation: {e}")

    def test_weather_query_example(self, jwt_token):
        """Example: Getting current weather for a city using MCP tools."""
        try:
            # Ask for weather in a specific city
            response_data = self._invoke_agent_with_jwt(
                jwt_token, "What's the current weather in Amsterdam?"
            )

            assert "result" in response_data  # nosec B101
            result = response_data["result"]
            assert len(result) > 0  # nosec B101

            # Verify weather information is provided
            result_lower = result.lower()
            assert "amsterdam" in result_lower  # nosec B101

            # Should contain weather-related information
            weather_indicators = ["temperature", "weather", "degrees", "conditions"]
            assert any(
                indicator in result_lower for indicator in weather_indicators
            )  # nosec B101

            print(f"Weather response: {result}")

        except Exception as e:
            pytest.fail(f"Failed weather query: {e}")

    def test_weather_forecast_example(self, jwt_token):
        """Example: Getting weather forecast using MCP tools."""
        try:
            response_data = self._invoke_agent_with_jwt(
                jwt_token, "Can you give me a 3-day weather forecast for London?"
            )

            assert "result" in response_data  # nosec B101
            result = response_data["result"]
            assert len(result) > 0  # nosec B101

            result_lower = result.lower()
            assert "london" in result_lower  # nosec B101

            # Should contain forecast information
            forecast_indicators = ["forecast", "days", "tomorrow", "weather"]
            assert any(
                indicator in result_lower for indicator in forecast_indicators
            )  # nosec B101

            print(f"Forecast response: {result}")

        except Exception as e:
            pytest.fail(f"Failed forecast query: {e}")

    def test_multiple_cities_comparison_example(self, jwt_token):
        """Example: Comparing weather across multiple cities."""
        try:
            response_data = self._invoke_agent_with_jwt(
                jwt_token,
                "Compare the current weather between Paris and Berlin. Which city has better weather today?",
            )

            assert "result" in response_data  # nosec B101
            result = response_data["result"]
            assert len(result) > 0  # nosec B101

            result_lower = result.lower()

            # Should mention at least one of the cities (more flexible)
            city_indicators = ["paris", "berlin", "france", "germany"]
            assert any(city in result_lower for city in city_indicators)  # nosec B101

            # Should contain comparison or weather-related language (more flexible)
            comparison_indicators = [
                "compare",
                "comparison",
                "better",
                "warmer",
                "cooler",
                "both",
                "weather",
                "temperature",
                "versus",
                "vs",
                "between",
                "different",
            ]
            assert any(
                indicator in result_lower for indicator in comparison_indicators
            )  # nosec B101

            print(f"Comparison response: {result}")

        except Exception as e:
            pytest.fail(f"Failed comparison query: {e}")

    def test_session_continuity_example(self, jwt_token):
        """Example: Maintaining context within a session."""
        session_id = str(uuid.uuid4())

        try:
            # First message: Ask about weather in a city
            response_data1 = self._invoke_agent_with_jwt(
                jwt_token, "What's the weather like in Tokyo today?", session_id
            )
            assert "result" in response_data1  # nosec B101

            print(f"First response: {response_data1['result']}")

            # Second message: Follow-up question using the same session
            response_data2 = self._invoke_agent_with_jwt(
                jwt_token,
                "What about tomorrow's forecast for the same city?",
                session_id,
            )
            assert "result" in response_data2  # nosec B101

            result2 = response_data2["result"]
            result2_lower = result2.lower()

            # Should handle the follow-up appropriately
            context_indicators = ["tokyo", "forecast", "tomorrow", "same"]
            assert any(
                indicator in result2_lower for indicator in context_indicators
            )  # nosec B101

            print(f"Follow-up response: {result2}")

        except Exception as e:
            pytest.fail(f"Failed session continuity: {e}")

    def test_agentcore_payload_structure_example(self, jwt_token):
        """Example: AgentCore payload structure with optional metadata."""
        test_prompts = [
            "What's the weather in Rome?",
            "What's the weather in Madrid?",
            "What's the weather in Vienna?",
        ]

        try:
            for i, prompt in enumerate(test_prompts):
                response_data = self._invoke_agent_with_jwt(jwt_token, prompt)

                assert "result" in response_data  # nosec B101
                result = response_data["result"]
                assert len(result) > 0  # nosec B101

                print(f"Payload structure {i + 1} response: {result}")

        except Exception as e:
            pytest.fail(f"Failed payload structure test: {e}")

    def test_error_handling_example(self, jwt_token):
        """Example: How the agent handles invalid city names."""
        try:
            response_data = self._invoke_agent_with_jwt(
                jwt_token, "What's the weather in Nonexistentcity12345?"
            )

            assert "result" in response_data  # nosec B101
            result = response_data["result"]
            assert len(result) > 0  # nosec B101

            # Agent should handle the error gracefully
            print(f"Error handling response: {result}")

        except Exception as e:
            pytest.fail(f"Failed error handling test: {e}")

    def test_performance_example(self, jwt_token):
        """Example: Measuring agent response time."""
        try:
            start_time = time.time()
            response_data = self._invoke_agent_with_jwt(
                jwt_token, "What's the current weather in New York City?"
            )
            end_time = time.time()

            assert "result" in response_data  # nosec B101
            result = response_data["result"]
            assert len(result) > 0  # nosec B101

            response_time = end_time - start_time
            print(f"Response time: {response_time:.2f} seconds")
            print(f"Response: {result}")

            # Should respond within reasonable time
            assert response_time < 30.0  # nosec B101

        except Exception as e:
            pytest.fail(f"Failed performance test: {e}")

    def test_non_weather_query_example(self, jwt_token):
        """Example: How the agent handles non-weather queries."""
        try:
            response_data = self._invoke_agent_with_jwt(
                jwt_token, "Can you tell me about the history of computers?"
            )

            assert "result" in response_data  # nosec B101
            result = response_data["result"]
            assert len(result) > 0  # nosec B101

            print(f"Non-weather query response: {result}")

        except Exception as e:
            pytest.fail(f"Failed non-weather query: {e}")

    def test_tool_capabilities_inquiry_example(self, jwt_token):
        """Example: Asking the agent about its capabilities."""
        try:
            response_data = self._invoke_agent_with_jwt(
                jwt_token, "What weather tools and capabilities do you have available?"
            )

            assert "result" in response_data  # nosec B101
            result = response_data["result"]
            assert len(result) > 0  # nosec B101

            # Should mention weather capabilities
            result_lower = result.lower()
            capability_indicators = [
                "weather",
                "tool",
                "capability",
                "provide",
                "check",
            ]
            assert any(
                indicator in result_lower for indicator in capability_indicators
            )  # nosec B101

            print(f"Capabilities response: {result}")

        except Exception as e:
            pytest.fail(f"Failed capabilities inquiry: {e}")
