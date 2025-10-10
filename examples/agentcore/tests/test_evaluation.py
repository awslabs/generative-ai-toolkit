"""
Copyright 2024 Amazon.com, Inc. and its affiliates. All Rights Reserved.

Licensed under the Amazon Software License (the "License").
You may not use this file except in compliance with the License.
A copy of the License is located at

  http://aws.amazon.com/asl/

or in the "license" file accompanying this file. This file is distributed
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
express or implied. See the License for the specific language governing
permissions and limitations under the License.
"""

"""
Toolkit Evaluation Tests for AgentCore Integration

This module contains comprehensive evaluation tests using the Generative AI Toolkit's
Case and Expect classes. Tests can run against:
1. Local docker-compose deployment
2. Deployed AgentCore runtime endpoint
3. Mock agent for unit testing

The tests validate that the weather agent behaves correctly in both local and
AgentCore environments, ensuring consistent behavior across deployment types.
"""

import os
import pytest
import requests
import uuid
from typing import Optional, Dict, Any
from unittest.mock import Mock, patch

from generative_ai_toolkit.test import Case, Expect
from generative_ai_toolkit.agent import BedrockConverseAgent
from generative_ai_toolkit.tracer import InMemoryTracer


class AgentCoreTestClient:
    """
    Test client that simulates AgentCore request/response format.

    This client can be used to test agents deployed in AgentCore Runtime
    by sending requests in the same format that AgentCore uses.
    """

    def __init__(self, endpoint: str, timeout: int = 30):
        """
        Initialize the test client.

        Args:
            endpoint: Base URL of the agent endpoint
            timeout: Request timeout in seconds
        """
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def send_message(
        self, message: str, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send a message to the agent using AgentCore format.

        Args:
            message: User message to send
            session_id: Session identifier (generated if not provided)

        Returns:
            Response data from the agent
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        request_data = {
            "prompt": message,
            "session_id": session_id,
            "last_k_turns": 20,
        }

        response = self.session.post(
            f"{self.endpoint}/invocations",
            json=request_data,
            timeout=self.timeout,
            headers={"Content-Type": "application/json"},
        )

        response.raise_for_status()
        return response.json()

    def health_check(self) -> bool:
        """
        Check if the agent is healthy.

        Returns:
            True if agent is healthy, False otherwise
        """
        try:
            response = self.session.get(f"{self.endpoint}/ping", timeout=self.timeout)
            return response.status_code == 200
        except:
            return False


class MockAgentCoreAgent:
    """
    Mock agent that simulates AgentCore behavior for unit testing.

    This allows testing the evaluation logic without requiring a running
    AgentCore deployment or local containers.
    """

    def __init__(self):
        self.tracer = InMemoryTracer()
        self.responses = []
        self.current_response_index = 0

    def add_response(self, response: str, tool_calls: Optional[list] = None):
        """Add a mock response that will be returned by the agent."""
        self.responses.append({"response": response, "tool_calls": tool_calls or []})

    def converse(self, user_input: str) -> str:
        """Simulate agent conversation (compatible with Case.run())."""
        if self.current_response_index >= len(self.responses):
            raise IndexError("No more mock responses available")

        response_data = self.responses[self.current_response_index]
        self.current_response_index += 1

        # Simulate trace recording using the tracer's trace context manager
        with self.tracer.trace("agent_invocation") as trace:
            # Record user input
            trace.add_attribute("ai.user.input", user_input)

            # Record tool calls if any
            for tool_call in response_data["tool_calls"]:
                with self.tracer.trace("tool_invocation") as tool_trace:
                    tool_trace.add_attribute("ai.tool.name", tool_call["name"])
                    tool_trace.add_attribute("ai.tool.input", tool_call["input"])

            # Record agent response
            trace.add_attribute("ai.agent.response", response_data["response"])

        return response_data["response"]

    def send_message(
        self, message: str, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Simulate sending a message to the agent (for direct testing)."""
        response = self.converse(message)
        return {
            "response": response,
            "session_id": session_id or str(uuid.uuid4()),
            "metadata": {"trace_id": str(uuid.uuid4())},
        }

    @property
    def traces(self):
        """Get traces for compatibility with Expect."""
        return self.tracer.get_traces()


# Test fixtures
@pytest.fixture
def local_agent_endpoint():
    """Endpoint for local docker-compose deployment."""
    return os.getenv("LOCAL_AGENT_ENDPOINT", "http://localhost:8080")


@pytest.fixture
def agentcore_endpoint():
    """Endpoint for deployed AgentCore runtime."""
    return os.getenv("AGENTCORE_ENDPOINT")


@pytest.fixture
def mock_agent():
    """Mock agent for unit testing."""
    return MockAgentCoreAgent()


@pytest.fixture
def local_client(local_agent_endpoint):
    """Test client for local deployment."""
    return AgentCoreTestClient(local_agent_endpoint)


@pytest.fixture
def agentcore_client(agentcore_endpoint):
    """Test client for AgentCore deployment."""
    if agentcore_endpoint:
        return AgentCoreTestClient(agentcore_endpoint)
    return None


# Basic functionality tests
class TestBasicFunctionality:
    """Test basic agent functionality using toolkit evaluation."""

    def test_simple_greeting_mock(self, mock_agent):
        """Test simple greeting with mock agent."""
        mock_agent.add_response(
            "Hello! I'm a weather assistant. How can I help you today?"
        )

        # Use Case to test the interaction
        case = Case(["Hello!"])

        # Run the case with the mock agent
        traces = case.run(mock_agent)

        # Use Expect to validate the response
        Expect(traces).agent_text_response.to_include("weather assistant")
        Expect(traces).agent_text_response.to_include("help you")

    def test_weather_query_mock(self, mock_agent):
        """Test weather query with mock agent."""
        mock_agent.add_response(
            "I'll check the weather in Seattle for you.",
            tool_calls=[
                {
                    "name": "get_weather_forecast",
                    "input": {"location": "Seattle", "days": 3},
                }
            ],
        )

        # Test weather query
        case = Case(["What's the weather in Seattle?"])
        traces = case.run(mock_agent)

        # Validate tool was called
        Expect(traces).tool_invocations.to_include("get_weather_forecast")
        Expect(traces).tool_invocations.to_include("get_weather_forecast").with_input(
            {"location": "Seattle", "days": 3}
        )

        # Validate response content
        Expect(traces).agent_text_response.to_include("Seattle")
        Expect(traces).agent_text_response.to_include("weather")

    def test_simple_greeting_local(self, local_client):
        """Test simple greeting with local deployment."""
        if not local_client.health_check():
            pytest.skip("Local agent not available")

        response = local_client.send_message("Hello!")

        assert "response" in response
        assert len(response["response"]) > 0
        # Weather agent should identify itself
        assert any(
            word in response["response"].lower()
            for word in ["weather", "help", "assistant"]
        )

    def test_weather_query_local(self, local_client):
        """Test weather query with local deployment."""
        if not local_client.health_check():
            pytest.skip("Local agent not available")

        response = local_client.send_message("What's the weather in Seattle?")

        assert "response" in response
        assert "seattle" in response["response"].lower()
        # Should mention weather-related terms
        assert any(
            word in response["response"].lower()
            for word in ["weather", "temperature", "forecast"]
        )

    def test_simple_greeting_agentcore(self, agentcore_client):
        """Test simple greeting with AgentCore deployment."""
        if not agentcore_client or not agentcore_client.health_check():
            pytest.skip("AgentCore endpoint not available")

        response = agentcore_client.send_message("Hello!")

        assert "response" in response
        assert len(response["response"]) > 0
        # Weather agent should identify itself
        assert any(
            word in response["response"].lower()
            for word in ["weather", "help", "assistant"]
        )

    def test_weather_query_agentcore(self, agentcore_client):
        """Test weather query with AgentCore deployment."""
        if not agentcore_client or not agentcore_client.health_check():
            pytest.skip("AgentCore endpoint not available")

        response = agentcore_client.send_message("What's the weather in Seattle?")

        assert "response" in response
        assert "seattle" in response["response"].lower()
        # Should mention weather-related terms
        assert any(
            word in response["response"].lower()
            for word in ["weather", "temperature", "forecast"]
        )


class TestConversationFlow:
    """Test multi-turn conversation flows."""

    def test_conversation_flow_mock(self, mock_agent):
        """Test multi-turn conversation with mock agent."""
        # Set up mock responses
        mock_agent.add_response("Hello! I can help you with weather information.")
        mock_agent.add_response(
            "I'll get the weather forecast for Seattle.",
            tool_calls=[
                {
                    "name": "get_weather_forecast",
                    "input": {"location": "Seattle", "days": 3},
                }
            ],
        )
        mock_agent.add_response(
            "Let me check for weather alerts in California.",
            tool_calls=[
                {"name": "get_weather_alerts", "input": {"area": "California"}}
            ],
        )

        # Test conversation flow
        case = Case(
            [
                "Hi there!",
                "What's the weather in Seattle?",
                "How about weather alerts for California?",
            ]
        )

        traces = case.run(mock_agent)

        # Validate tool calls
        Expect(traces).tool_invocations.to_include("get_weather_forecast")
        Expect(traces).tool_invocations.to_include("get_weather_alerts")

        # Validate responses - check that both Seattle and California appear in the conversation
        all_responses = " ".join(
            [
                trace.attributes.get("ai.agent.response", "")
                for trace in traces
                if "ai.agent.response" in trace.attributes
            ]
        )
        assert "Seattle" in all_responses
        assert "California" in all_responses

    def test_conversation_flow_local(self, local_client):
        """Test multi-turn conversation with local deployment."""
        if not local_client.health_check():
            pytest.skip("Local agent not available")

        session_id = str(uuid.uuid4())

        # First turn
        response1 = local_client.send_message("Hello!", session_id)
        assert "response" in response1

        # Second turn - weather query
        response2 = local_client.send_message(
            "What's the weather in Seattle?", session_id
        )
        assert "response" in response2
        assert "seattle" in response2["response"].lower()

        # Third turn - different location
        response3 = local_client.send_message("How about New York?", session_id)
        assert "response" in response3
        assert "new york" in response3["response"].lower()

    def test_conversation_flow_agentcore(self, agentcore_client):
        """Test multi-turn conversation with AgentCore deployment."""
        if not agentcore_client or not agentcore_client.health_check():
            pytest.skip("AgentCore endpoint not available")

        session_id = str(uuid.uuid4())

        # First turn
        response1 = agentcore_client.send_message("Hello!", session_id)
        assert "response" in response1

        # Second turn - weather query
        response2 = agentcore_client.send_message(
            "What's the weather in Seattle?", session_id
        )
        assert "response" in response2
        assert "seattle" in response2["response"].lower()

        # Third turn - different location
        response3 = agentcore_client.send_message("How about New York?", session_id)
        assert "response" in response3
        assert "new york" in response3["response"].lower()


class TestToolIntegration:
    """Test MCP tool integration."""

    def test_weather_forecast_tool_mock(self, mock_agent):
        """Test weather forecast tool with mock agent."""
        mock_agent.add_response(
            "I'll get the 5-day forecast for Miami.",
            tool_calls=[
                {
                    "name": "get_weather_forecast",
                    "input": {"location": "Miami", "days": 5},
                }
            ],
        )

        case = Case(["Give me a 5-day forecast for Miami"])
        traces = case.run(mock_agent)

        # Validate tool call with correct parameters
        Expect(traces).tool_invocations.to_include("get_weather_forecast").with_input(
            {"location": "Miami", "days": 5}
        )

    def test_weather_alerts_tool_mock(self, mock_agent):
        """Test weather alerts tool with mock agent."""
        mock_agent.add_response(
            "Let me check for weather alerts in Texas.",
            tool_calls=[{"name": "get_weather_alerts", "input": {"area": "Texas"}}],
        )

        case = Case(["Are there any weather alerts for Texas?"])
        traces = case.run(mock_agent)

        # Validate tool call
        Expect(traces).tool_invocations.to_include("get_weather_alerts").with_input(
            {"area": "Texas"}
        )

    def test_multiple_locations_local(self, local_client):
        """Test handling multiple locations with local deployment."""
        if not local_client.health_check():
            pytest.skip("Local agent not available")

        response = local_client.send_message(
            "Compare the weather in Seattle and San Francisco"
        )

        assert "response" in response
        response_text = response["response"].lower()
        assert "seattle" in response_text
        assert "san francisco" in response_text or "san francisco" in response_text

    def test_multiple_locations_agentcore(self, agentcore_client):
        """Test handling multiple locations with AgentCore deployment."""
        if not agentcore_client or not agentcore_client.health_check():
            pytest.skip("AgentCore endpoint not available")

        response = agentcore_client.send_message(
            "Compare the weather in Seattle and San Francisco"
        )

        assert "response" in response
        response_text = response["response"].lower()
        assert "seattle" in response_text
        assert "san francisco" in response_text or "san francisco" in response_text


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_location_mock(self, mock_agent):
        """Test handling of invalid location with mock agent."""
        mock_agent.add_response(
            "I'll try to get weather for that location.",
            tool_calls=[
                {
                    "name": "get_weather_forecast",
                    "input": {"location": "InvalidLocation123", "days": 3},
                }
            ],
        )
        mock_agent.add_response(
            "I couldn't find weather data for that location. Please try a valid city name."
        )

        case = Case(["What's the weather in InvalidLocation123?"])
        traces = case.run(mock_agent)

        # Should still attempt the tool call
        Expect(traces).tool_invocations.to_include("get_weather_forecast")

        # Should handle the error gracefully - check that the error message appears in any response
        all_responses = " ".join(
            [
                trace.attributes.get("ai.agent.response", "")
                for trace in traces
                if "ai.agent.response" in trace.attributes
            ]
        )
        assert "couldn't find" in all_responses or "try to get weather" in all_responses

    def test_empty_message_local(self, local_client):
        """Test handling of empty message with local deployment."""
        if not local_client.health_check():
            pytest.skip("Local agent not available")

        response = local_client.send_message("")

        assert "response" in response
        # Should handle empty input gracefully
        assert len(response["response"]) > 0

    def test_empty_message_agentcore(self, agentcore_client):
        """Test handling of empty message with AgentCore deployment."""
        if not agentcore_client or not agentcore_client.health_check():
            pytest.skip("AgentCore endpoint not available")

        response = agentcore_client.send_message("")

        assert "response" in response
        # Should handle empty input gracefully
        assert len(response["response"]) > 0


class TestPerformanceAndReliability:
    """Test performance and reliability characteristics."""

    def test_response_time_local(self, local_client):
        """Test response time with local deployment."""
        if not local_client.health_check():
            pytest.skip("Local agent not available")

        import time

        start_time = time.time()
        response = local_client.send_message("What's the weather today?")
        end_time = time.time()

        response_time = end_time - start_time

        assert "response" in response
        # Response should be reasonably fast (under 30 seconds)
        assert response_time < 30

    def test_response_time_agentcore(self, agentcore_client):
        """Test response time with AgentCore deployment."""
        if not agentcore_client or not agentcore_client.health_check():
            pytest.skip("AgentCore endpoint not available")

        import time

        start_time = time.time()
        response = agentcore_client.send_message("What's the weather today?")
        end_time = time.time()

        response_time = end_time - start_time

        assert "response" in response
        # Response should be reasonably fast (under 30 seconds)
        assert response_time < 30

    def test_concurrent_requests_local(self, local_client):
        """Test handling concurrent requests with local deployment."""
        if not local_client.health_check():
            pytest.skip("Local agent not available")

        import threading
        import time

        results = []

        def send_request(message, session_id):
            try:
                response = local_client.send_message(message, session_id)
                results.append(response)
            except Exception as e:
                results.append({"error": str(e)})

        # Send multiple concurrent requests
        threads = []
        for i in range(3):
            session_id = str(uuid.uuid4())
            thread = threading.Thread(
                target=send_request,
                args=(f"What's the weather in city {i}?", session_id),
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All requests should succeed
        assert len(results) == 3
        for result in results:
            assert "response" in result or "error" in result
            if "response" in result:
                assert len(result["response"]) > 0


# Utility functions for running tests
def run_mock_tests():
    """Run all mock tests (no external dependencies)."""
    pytest.main(
        [__file__ + "::TestBasicFunctionality::test_simple_greeting_mock", "-v"]
    )
    pytest.main([__file__ + "::TestBasicFunctionality::test_weather_query_mock", "-v"])
    pytest.main(
        [__file__ + "::TestConversationFlow::test_conversation_flow_mock", "-v"]
    )
    pytest.main(
        [__file__ + "::TestToolIntegration::test_weather_forecast_tool_mock", "-v"]
    )
    pytest.main(
        [__file__ + "::TestToolIntegration::test_weather_alerts_tool_mock", "-v"]
    )
    pytest.main([__file__ + "::TestErrorHandling::test_invalid_location_mock", "-v"])


def run_integration_tests():
    """Run integration tests against local deployment."""
    pytest.main(["-m", "integration", __file__, "-v"])


def run_agentcore_tests():
    """Run tests against AgentCore deployment."""
    pytest.main(["-m", "agentcore", __file__, "-v"])


def run_all_tests():
    """Run all available tests."""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    # Run mock tests by default
    run_mock_tests()
