# Copyright 2025 Amazon.com, Inc. and its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from generative_ai_toolkit.test import Expect


def test_multi_agent(mock_multi_agent):
    [
        supervisor,
        events_agent,
        weather_agent,
        supervisor_mock,
        events_agent_mock,
        weather_agent_mock,
    ] = mock_multi_agent

    supervisor_mock.add_output(
        tool_use_output=[
            {"name": "transfer_to_weather_agent", "input": {"user_input": "Amsterdam"}},
            {"name": "transfer_to_events_agent", "input": {"user_input": "Amsterdam"}},
        ]
    )
    weather_agent_mock.add_output(
        tool_use_output=[
            {"name": "get_weather", "input": {"city": "Amsterdam"}},
        ]
    )
    weather_agent_mock.add_output(
        text_output=["The weather in Amsterdam will be Sunny"],
    )
    events_agent_mock.add_output(
        tool_use_output=[
            {"name": "get_events", "input": {"city": "Amsterdam"}},
        ]
    )
    events_agent_mock.add_output(
        text_output=["These are the coming events in Amsterdam: bla bla bla"],
    )
    supervisor_mock.add_output(
        text_output=[
            "The weather in Amsterdam will be Sunny and the coming events are bla bla bla"
        ]
    )
    supervisor.converse("I want to go to Amsterdam, what is up there?")

    Expect(weather_agent.traces).tool_invocations.to_include("get_weather").with_input(
        {"city": "Amsterdam"}
    )
    Expect(weather_agent.traces).user_input.to_equal("Amsterdam")
    Expect(events_agent.traces).tool_invocations.to_include("get_events").with_input(
        {"city": "Amsterdam"}
    )
    Expect(events_agent.traces).user_input.to_equal("Amsterdam")
    Expect(supervisor.traces).tool_invocations.to_include("transfer_to_weather_agent")
    Expect(supervisor.traces).tool_invocations.to_include("transfer_to_events_agent")
    Expect(supervisor.traces).agent_text_response.to_equal(
        "The weather in Amsterdam will be Sunny and the coming events are bla bla bla"
    )

    trace_id = supervisor.traces[0].trace_id
    for trace in [*supervisor.traces, *weather_agent.traces, *events_agent.traces]:
        assert trace.trace_id == trace_id

    for supervisor_trace in supervisor.traces:
        if supervisor_trace.attributes["ai.trace.type"] == "tool-invocation":
            for subordinate_agent in [events_agent, weather_agent]:
                if (
                    supervisor_trace.attributes["ai.tool.name"]
                    == subordinate_agent.name
                ):
                    supervisor_span_id = supervisor_trace.span_id
                    subordinate_agent_root_trace = subordinate_agent.traces[0]
                    assert (
                        supervisor_span_id
                        == subordinate_agent_root_trace.parent_span.span_id
                    )
