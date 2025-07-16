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
Example runner for Pydantic-based LLM tools with Generative AI Toolkit.

This module demonstrates how to:
1. Create a Generative AI Toolkit agent
2. Register a Pydantic-based tool (WeatherAlertsTool)
3. Process natural language requests through the agent
4. Handle tool responses in a conversational context

The examples show how the agent understands user intent and uses the tool appropriately.
"""

from generative_ai_toolkit.agent import BedrockConverseAgent
from generative_ai_toolkit.tracer import InMemoryTracer
from generative_ai_toolkit.conversation_history import InMemoryConversationHistory
from weather_tool import WeatherAlertsTool


def process_user_request(agent: BedrockConverseAgent, user_input: str) -> None:
    """
    Process a user request through the agent and print the response.

    This helper function demonstrates the conversation flow between user and agent.

    Args:
        agent: The Generative AI Toolkit agent
        user_input: The user's natural language request
    """
    print("\n" + "=" * 80)
    print("User Request:", user_input)

    # Get agent's response
    response = agent.converse(user_input)

    # Find any tool invocations in the conversation
    tool_traces = [
        trace
        for trace in agent.traces
        if trace.attributes.get("ai.trace.type") == "tool-invocation"
    ]

    # Print tool usage information if any tools were used
    if tool_traces:
        print("-" * 80)
        print("Tool Usage Details:")
        for trace in tool_traces:
            print(f"Tool: {trace.attributes.get('ai.tool.name')}")
            print(f"Input: {trace.attributes.get('ai.tool.input')}")

    print("-" * 80)
    print("Agent Response:", response)


def main():
    # Create an agent with appropriate system prompt
    agent = BedrockConverseAgent(
        model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        system_prompt="""You are a helpful weather assistant that can check for active weather alerts.
        When users ask about weather alerts or conditions, use the weather alerts tool to get accurate information.
        Always interpret the results and provide a natural, informative response.""",
        temperature=0.0,
        conversation_history=InMemoryConversationHistory(),
        tracer=InMemoryTracer,
    )

    # Create and register the weather alerts tool
    weather_tool = WeatherAlertsTool()
    agent.register_tool(weather_tool)

    # Example user requests demonstrating different tool capabilities
    user_requests = [
        # Request for specific location
        "What weather alerts are active in California right now?",
        # Basic request for all active alerts
        "Are there any active weather alerts I should know about?",
        # Request for specific type of alert
        "Are there any flood warnings or watches currently active?",
        # Complex request with multiple criteria
        "Tell me about any urgent tornado warnings that are in effect.",
        # Request using severity, certainty and message type parameters
        "Show me any extreme or severe weather alerts that are confirmed and not updates or cancellations.",
    ]

    # Process each request through the agent
    for request in user_requests:
        agent.reset()
        process_user_request(agent, request)


if __name__ == "__main__":
    main()
