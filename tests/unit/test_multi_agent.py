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

from generative_ai_toolkit.agent import BedrockConverseAgent
from generative_ai_toolkit.test import Expect
from generative_ai_toolkit.test.mock import MockBedrockConverse


def test_multi_agent(mock_multi_agent):
    supervisor = mock_multi_agent.supervisor
    weather = mock_multi_agent.weather
    events = mock_multi_agent.events

    supervisor.mock.add_output(
        tool_use_output=[
            {"name": "transfer_to_weather_agent", "input": {"user_input": "Amsterdam"}},
            {"name": "transfer_to_events_agent", "input": {"user_input": "Amsterdam"}},
        ]
    )
    weather.mock.add_output(
        tool_use_output=[
            {"name": "get_weather", "input": {"city": "Amsterdam"}},
        ]
    )
    weather.mock.add_output(
        text_output=["The weather in Amsterdam will be Sunny"],
    )
    events.mock.add_output(
        tool_use_output=[
            {"name": "get_events", "input": {"city": "Amsterdam"}},
        ]
    )
    events.mock.add_output(
        text_output=["These are the coming events in Amsterdam: bla bla bla"],
    )
    supervisor.mock.add_output(
        text_output=[
            "The weather in Amsterdam will be Sunny and the coming events are bla bla bla"
        ]
    )
    supervisor.agent.converse("I want to go to Amsterdam, what is up there?")

    # Get the traces explicitly through all tracers
    attribute_filter = {
        "ai.conversation.id": supervisor.agent.conversation_id,
        "ai.auth.context": supervisor.agent.auth_context,
    }
    all_traces = [
        *supervisor.agent.tracer.get_traces(attribute_filter=attribute_filter),
        *weather.agent.tracer.get_traces(attribute_filter=attribute_filter),
        *events.agent.tracer.get_traces(attribute_filter=attribute_filter),
    ]
    all_traces.sort(key=lambda trace: trace.started_at)

    # All traces should have the same trace id
    trace_id = supervisor.agent.traces[0].trace_id
    for trace in all_traces:
        assert trace.trace_id == trace_id

    # Traces should be accessible through the supervisor agent
    assert len(supervisor.agent.traces) == len(all_traces)
    assert supervisor.agent.traces == all_traces

    Expect(weather.agent.traces).tool_invocations.to_include("get_weather").with_input(
        {"city": "Amsterdam"}
    )
    Expect(weather.agent.traces).user_input.to_equal("Amsterdam")
    Expect(events.agent.traces).tool_invocations.to_include("get_events").with_input(
        {"city": "Amsterdam"}
    )
    Expect(events.agent.traces).user_input.to_equal("Amsterdam")
    Expect(supervisor.agent.traces).tool_invocations.to_include(
        "transfer_to_weather_agent"
    )
    Expect(supervisor.agent.traces).tool_invocations.to_include(
        "transfer_to_events_agent"
    )
    Expect(supervisor.agent.traces).agent_text_response.to_equal(
        "The weather in Amsterdam will be Sunny and the coming events are bla bla bla"
    )

    # The subordinate agent is called as tool
    # - The tool name should equal the subordinate agent name
    # - The supervisor tool span id should be the parent span of the subordinate agent
    for supervisor_trace in supervisor.agent.traces:
        if supervisor_trace.attributes["ai.trace.type"] == "tool-invocation":
            for subordinate in [events, weather]:
                if (
                    supervisor_trace.attributes["ai.tool.name"]
                    == subordinate.agent.name
                ):
                    supervisor_span_id = supervisor_trace.span_id
                    subordinate_agent_root_trace = subordinate.agent.traces[0]
                    assert (
                        supervisor_span_id
                        == subordinate_agent_root_trace.parent_span.span_id
                    )


def test_multi_agent_with_reused_subagent():
    supervisor_mock = MockBedrockConverse()
    supervisor = BedrockConverseAgent(
        model_id="dummy", session=supervisor_mock.session()
    )
    subagent_l1_mock = MockBedrockConverse()
    subagent_l1 = BedrockConverseAgent(
        model_id="dummy",
        name="subagent_l1",
        description="transfer_to_subagent_l1",
        session=subagent_l1_mock.session(),
    )
    subagent_l2_mock = MockBedrockConverse()
    subagent_l2 = BedrockConverseAgent(
        model_id="dummy",
        name="subagent_l2",
        description="transfer_to_subagent_l2",
        session=subagent_l2_mock.session(),
    )
    subagent_l3_mock = MockBedrockConverse()
    subagent_l3 = BedrockConverseAgent(
        model_id="dummy",
        name="subagent_l3",
        description="transfer_to_subagent_l3",
        session=subagent_l3_mock.session(),
    )

    supervisor.register_tool(subagent_l1)
    supervisor.register_tool(subagent_l2)
    subagent_l1.register_tool(subagent_l3)
    subagent_l2.register_tool(subagent_l3)

    supervisor_mock.add_output(
        tool_use_output=[
            {"name": "subagent_l1", "input": {"user_input": "Hello L1 from L0"}},
        ]
    )
    supervisor_mock.add_output(
        tool_use_output=[
            {"name": "subagent_l2", "input": {"user_input": "Hello L2 from L0"}},
        ]
    )
    subagent_l1_mock.add_output(
        tool_use_output=[
            {"name": "subagent_l3", "input": {"user_input": "Hello L3 from L1"}},
        ]
    )
    subagent_l1_mock.add_output(
        text_output=["Hello back to L0 from L1"],
    )

    subagent_l2_mock.add_output(
        tool_use_output=[
            {"name": "subagent_l3", "input": {"user_input": "Hello L3 from L2"}},
        ]
    )
    subagent_l2_mock.add_output(
        text_output=["Hello back to L0 from L2"],
    )

    subagent_l3_mock.add_output(
        text_output=["Hello back to L1 from L3"],
    )
    subagent_l3_mock.add_output(
        text_output=["Hello back to L2 from L3"],
    )

    supervisor_mock.add_output(
        text_output=[
            "Hello back to User!",
        ]
    )

    supervisor.converse("Hello L0!")

    Expect(supervisor.traces).tool_invocations.to_include("subagent_l1").with_input(
        {"user_input": "Hello L1 from L0"}
    )
    Expect(supervisor.traces).tool_invocations.to_include("subagent_l2").with_input(
        {"user_input": "Hello L2 from L0"}
    )
    Expect(supervisor.traces).agent_text_response.to_equal("Hello back to User!")

    Expect(subagent_l1.traces).tool_invocations.to_include("subagent_l3").with_input(
        {"user_input": "Hello L3 from L1"}
    )
    Expect(subagent_l1.traces).agent_text_response.to_equal("Hello back to L0 from L1")

    Expect(subagent_l2.traces).tool_invocations.to_include("subagent_l3").with_input(
        {"user_input": "Hello L3 from L2"}
    )

    Expect(subagent_l2.traces).agent_text_response.to_equal("Hello back to L0 from L2")

    Expect(subagent_l3.traces).at(0).agent_text_response.to_equal(
        "Hello back to L1 from L3"
    )
    Expect(subagent_l3.traces).at(1).agent_text_response.to_equal(
        "Hello back to L2 from L3"
    )

    l0_span_ids = {trace.span_id for trace in supervisor.traces}
    l1_span_ids = {trace.span_id for trace in subagent_l1.traces}
    l2_span_ids = {trace.span_id for trace in subagent_l2.traces}
    l3_span_ids = {trace.span_id for trace in subagent_l3.traces}

    # l0 combines all
    assert l0_span_ids & (l1_span_ids | l2_span_ids | l3_span_ids)
    assert len((l1_span_ids | l2_span_ids | l3_span_ids) - l0_span_ids) == 0

    # There should be NO overlap between L1 and L2, although they both have L3 traces
    assert not l1_span_ids & l2_span_ids

    # There should be overlap between L1 traces and L3, and L2 traces and L3
    assert l1_span_ids & l3_span_ids
    assert l2_span_ids & l3_span_ids
