"""
Examples for unit tests of an agent
"""

from generative_ai_toolkit.test import Case, Expect


def test_tool(city_agent):
    """
    tests if the defined tool works correctly
    """

    def sample_function():
        """This is a sample function"""
        return "Hello World"

    # register sample_function with agent and assign to variable tool
    tool = city_agent.register_tool(sample_function)

    print(tool.tool_spec)
    # assertion to check  if tool description is correct
    assert tool.tool_spec == {
        "name": "sample_function",
        "description": "This is a sample function",
        "inputSchema": {"json": {"type": "object", "properties": {}}},
    }
    # check if the return value equals expected.
    output = tool.invoke()

    assert isinstance(output, str)
    assert output == "Hello World"


# parameters: real agent and mocked agent, so we can switch between both
def test_agent_uses_in_memory_history(city_agent, mock_bedrock_converse):
    """
    tests if the agent has a _conversation_history
    """

    agent_test_response = "Hello, how can I help you?"
    # mocking makes agent output the message given as parameter
    mock_bedrock_converse.add_output(text_output=[agent_test_response])

    test_case = Case(name="Test chat history ", user_inputs=["Hi there !"])

    test_case.run(city_agent)  # run test case with the set up
    assert city_agent._conversation_history is not None
    # check if the first entry in conversation history equals “Hello, how can I help you?”
    assert (
        agent_test_response
        == city_agent._conversation_history.messages[1]["content"][0]["text"]
    )


def test_llm(unmocked_city_agent):
    """
    test the LLM with an non-mocked, real LLM
    """

    test_case = Case(
        user_inputs=["what is the capital of France?"],
    )
    traces = test_case.run(unmocked_city_agent)
    Expect(traces).agent_text_response.to_include("Paris")


def test_mocked_llm(mock_bedrock_converse, city_agent):
    """
    test the LLM with a mocked LLM to steer conversation
    """
    mock_bedrock_converse.add_output(
        text_output=["this is a sample agent. How can i help you?"]
    )
    mock_bedrock_converse.add_output(text_output=["the color of the sky is blue"])

    test_case = Case(
        user_inputs=["hello", "what color is the sky"],
    )

    traces = test_case.run(city_agent)
    Expect(traces).agent_text_response.to_include("blue")
