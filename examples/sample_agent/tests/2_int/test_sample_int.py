"""
Examples for integration tests of an agent
"""

import time

from generative_ai_toolkit.test import Case, Expect


def test_agent_calls_tool_search_events(mock_bedrock_converse, city_agent):
    """
    test if agent can use tool search_event.
    Force agent to call tool by mocking the orchestrator.
    """

    test_search_events_ok = Case(
        user_inputs=["<start conversation>", "Search me some events in Munich"]
    )
    mock_bedrock_converse.add_output(
        text_output=["This is the city agent, how can I help you?"]
    )
    mock_bedrock_converse.add_output(
        tool_use_output=[
            {"input": {"city_name": "Munich"}, "name": "search_events"},
            {"input": {"city_name": "Munich"}, "name": "get_news"},
        ]
    )
    mock_bedrock_converse.add_real_response()
    traces = test_search_events_ok.run(city_agent)
    # check in the traces if tool search_event was called. If not, raise error
    Expect(traces).tool_invocations.to_include("search_events")


def test_agent_calls_tool_negative_case(unmocked_city_agent):
    """
    negative test case for a tool: we call agent with a task he does not know...
    """
    city = "Munich"
    test_case = Case(user_inputs=[f"What kind of beer can you recommend in {city}?"])
    time.sleep(60)  # insert sleep to run integration test without throttling errror

    traces = test_case.run(unmocked_city_agent)
    Expect(traces).agent_text_response.to_include(city)
    Expect(traces).tool_invocations.to_have_length(0)


def test_conversation_happy_tool(mock_bedrock_converse, city_agent):
    """
    Integration test for a successful user interaction including a tool call.
    """
    # mock: have agent output the text and then mock that agent wants to call tool with
    # # parameter Munich as city.
    time.sleep(60)  # insert sleep to run integration test without throttling errror

    mock_bedrock_converse.add_output(
        text_output=["Okay, let me check the weather for you."]
    )
    mock_bedrock_converse.add_output(
        tool_use_output=[{"name": "get_weather", "input": {"city_name": "Munich"}}]
    )
    # agent will output text from above and call get_weather with parameter Munich.
    mock_bedrock_converse.add_real_response()

    test_case = Case(
        name="invoke weather tool",
        user_inputs=[
            "<Start conversation>",
            "I would really like to go to Munich, but only if the weather is good there!",
        ],
    )
    traces = test_case.run(city_agent)
    # check in traces if tool get_weather was invoked with parameter Munich
    Expect(traces).tool_invocations.to_include("get_weather").with_input(
        {"city_name": "Munich"}
    )
