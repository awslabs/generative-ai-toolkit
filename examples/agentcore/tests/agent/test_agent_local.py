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

# nosec B101

import os
import textwrap

import agent
import pytest

from generative_ai_toolkit.evaluate.interactive import GenerativeAIToolkit
from generative_ai_toolkit.metrics.modules.conversation import (
    ConversationExpectationMetric,
)
from generative_ai_toolkit.test import Case


class TestAgentLocalIntegration:
    """Integration tests that run the agent with real AWS calls after CDK deployment."""

    @pytest.mark.integration
    def test_agent_basic_functionality(self):
        """Test basic agent functionality with real Bedrock calls."""

        payload = {
            "input": {"prompt": "Hello! Please introduce yourself briefly."},
            "sessionId": "integration-test-basic",
        }

        try:
            result = agent.invoke(payload)

            assert "result" in result  # nosec B101
            assert len(result["result"]) > 0  # nosec B101
            assert isinstance(result["result"], str)  # nosec B101

            # Should contain some indication it's a weather assistant
            result_lower = result["result"].lower()
            weather_indicators = ["weather", "assistant", "help", "information"]
            assert any(
                indicator in result_lower for indicator in weather_indicators
            )  # nosec B101

        except Exception as e:
            pytest.fail(f"Agent invocation failed: {e}")

    @pytest.mark.integration
    def test_agent_with_mcp_weather_tools(self):
        """Test agent with MCP weather tools using Case with Overall Conversation Expectations."""

        # Create a Case with overall conversation expectations
        weather_case = Case(
            name="Weather query with MCP tools",
            user_inputs=[
                "What's the current weather in Amsterdam? Please use your weather tools."
            ],
            overall_expectations=textwrap.dedent(
                """
                The agent should:
                1. Recognize the user's request for weather information about Amsterdam
                2. Use the available MCP weather tools (get_weather) to retrieve current weather data
                3. Provide a helpful response that includes:
                   - Mention of Amsterdam as the requested city
                   - Weather information (temperature, conditions, etc.) obtained from the tools
                   - A natural, conversational response format
                4. If tools are unavailable, handle the error gracefully with an appropriate message

                The conversation should demonstrate successful integration between the agent and MCP server.
                """
            ),
        )

        try:
            # Run the case directly against the BedrockConverseAgent
            traces = weather_case.run(agent.bedrock_agent)

            # Evaluate using ConversationExpectationMetric
            results = GenerativeAIToolkit.eval(
                metrics=[ConversationExpectationMetric()], traces=[traces]
            )

            # Check evaluation results
            evaluation_results = list(results)
            assert (
                len(evaluation_results) > 0
            ), "Expected evaluation results from ConversationExpectationMetric"  # nosec B101
            conversation_result = evaluation_results[0]

            # Verify we have measurements
            assert (
                len(conversation_result.measurements) > 0
            ), "Expected conversation measurements"  # nosec B101

            # Check if the evaluation passed (score >= 7 is generally good)
            correctness_measurements = [
                m for m in conversation_result.measurements if "Correctness" in m.name
            ]

            if correctness_measurements:
                score = correctness_measurements[0].value
                print(f"\n✅ Correctness Score: {score}/10")
                assert (
                    score >= 7
                ), f"Expected correctness score >= 7, got {score}"  # nosec B101

            # Verify the agent actually used weather tools by checking traces
            tool_traces = [
                trace
                for trace in conversation_result.traces
                if hasattr(trace, "trace")
                and "get_weather" in str(trace.trace.attributes.get("ai.tool.name", ""))
            ]

            assert (
                len(tool_traces) > 0
            ), "Expected agent to use weather tools"  # nosec B101
            print(
                f"✅ Weather tool was used successfully ({len(tool_traces)} tool calls)"
            )

        except Exception as e:
            pytest.fail(f"Case-based evaluation failed: {e}")

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

            payload = {"input": {"prompt": "Test prompt"}, "sessionId": "error-test"}

            result = agent.invoke(payload)

            # Should handle error gracefully and return error message
            assert "result" in result  # nosec B101
            result_text = result["result"].lower()

            # Check for the specific error message format returned by the agent
            assert (  # nosec B101
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
        assert len(results) == len(sessions)  # nosec B101
        for i, result in enumerate(results):
            assert "result" in result, f"Session {i + 1} missing result"  # nosec B101
            assert (
                len(result["result"]) > 0
            ), f"Session {i + 1} empty result"  # nosec B101
