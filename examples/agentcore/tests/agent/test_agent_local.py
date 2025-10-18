"""
Local integration tests for the agent after CDK deployment.

These tests run the actual agent code locally using pytest, testing:
- Agent functionality with real Bedrock calls
- MCP integration with deployed MCP server
- Different payload structures
- Error handling scenarios

Prerequisites:
- CDK stack deployed (cdk deploy)
- AWS credentials configured
- Environment variables set (AWS_REGION, BEDROCK_MODEL_ID, MCP_SERVER_RUNTIME_ARN, OAUTH_CREDENTIALS_SECRET_NAME, OAUTH_USER_POOL_ID, OAUTH_USER_POOL_CLIENT_ID)

Run with: pytest examples/agentcore/tests/agent/test_agent_local.py -v
"""

import importlib
import os
import sys

import pytest


def import_agent_fresh():
    """Import agent module with fresh state."""
    if "agent" in sys.modules:
        del sys.modules["agent"]
    return importlib.import_module("agent")


class TestAgentLocalIntegration:
    """Integration tests that run the agent with real AWS calls after CDK deployment."""

    @pytest.mark.integration
    def test_agent_basic_functionality(self):
        """Test basic agent functionality with real Bedrock calls."""

        agent = import_agent_fresh()

        payload = {
            "input": {"prompt": "Hello! Please introduce yourself briefly."},
            "sessionId": "integration-test-basic",
        }

        try:
            result = agent.invoke(payload)

            assert "result" in result
            assert len(result["result"]) > 0
            assert isinstance(result["result"], str)

            # Should contain some indication it's a weather assistant
            result_lower = result["result"].lower()
            weather_indicators = ["weather", "assistant", "help", "information"]
            assert any(indicator in result_lower for indicator in weather_indicators)

        except Exception as e:
            pytest.fail(f"Agent invocation failed: {e}")

    @pytest.mark.integration
    def test_agent_with_mcp_weather_tools(self):
        """Test agent with MCP weather tools (requires deployed MCP server)."""

        agent = import_agent_fresh()

        payload = {
            "input": {
                "prompt": "What's the current weather in Amsterdam? Please use your weather tools."
            },
            "sessionId": "mcp-weather-test",
        }

        try:
            result = agent.invoke(payload)

            assert "result" in result
            assert len(result["result"]) > 0

            result_lower = result["result"].lower()

            # Should mention Amsterdam
            assert "amsterdam" in result_lower

            # Should contain weather information or appropriate error handling
            weather_indicators = [
                "temperature",
                "weather",
                "degrees",
                "conditions",
                "celsius",
                "fahrenheit",
                "sunny",
                "cloudy",
                "rain",
            ]
            error_indicators = ["sorry", "unable", "error", "unavailable"]

            has_weather_data = any(
                indicator in result_lower for indicator in weather_indicators
            )
            has_error_handling = any(
                indicator in result_lower for indicator in error_indicators
            )

            # Should either provide weather data or handle errors gracefully
            assert has_weather_data or has_error_handling

        except Exception as e:
            pytest.fail(f"Agent invocation failed: {e}")

    @pytest.mark.integration
    def test_agent_error_handling(self):
        """Test agent error handling with invalid model configuration.

        Verifies that the agent gracefully handles Bedrock model validation errors
        and returns a user-friendly error message instead of crashing.
        """

        # Test with invalid model ID
        original_model = os.environ.get("BEDROCK_MODEL_ID")
        os.environ["BEDROCK_MODEL_ID"] = "invalid-model-id"

        try:
            agent = import_agent_fresh()

            payload = {"input": {"prompt": "Test prompt"}, "sessionId": "error-test"}

            result = agent.invoke(payload)

            # Should handle error gracefully and return error message
            assert "result" in result
            result_text = result["result"].lower()

            # Check for the specific error message format returned by the agent
            assert (
                "model configuration error" in result_text
                or "is not available" in result_text
                or "invalid-model-id" in result_text
            ), f"Expected model configuration error message, got: {result['result']}"

        finally:
            # Restore original model ID
            if original_model:
                os.environ["BEDROCK_MODEL_ID"] = original_model

    @pytest.mark.integration
    def test_agent_session_handling(self):
        """Test agent with multiple calls in different sessions."""

        agent = import_agent_fresh()

        # Test multiple sessions
        sessions = [
            {"input": {"prompt": "Hello, I'm user 1"}, "sessionId": "session-1"},
            {"input": {"prompt": "Hello, I'm user 2"}, "sessionId": "session-2"},
            {"input": {"prompt": "Hello, I'm user 3"}, "sessionId": "session-3"},
        ]

        results = []
        for payload in sessions:
            try:
                result = agent.invoke(payload)
                results.append(result)
            except Exception as e:
                pytest.fail(f"Agent invocation failed for session: {e}")

        # All calls should succeed
        assert len(results) == len(sessions)
        for i, result in enumerate(results):
            assert "result" in result, f"Session {i + 1} missing result"
            assert len(result["result"]) > 0, f"Session {i + 1} empty result"
