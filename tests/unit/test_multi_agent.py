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

from collections import defaultdict
from collections.abc import Sequence

from generative_ai_toolkit.agent import BedrockConverseAgent
from generative_ai_toolkit.test import Expect
from generative_ai_toolkit.test.mock import MockBedrockConverse
from generative_ai_toolkit.tracer import Trace


def get_all_subagent_subcontext_ids(all_traces: Sequence[Trace]):
    result: dict[str, dict[str | None, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )

    for trace in all_traces:
        if (
            trace.attributes.get("ai.trace.type") == "tool-invocation"
            and "ai.tool.subagent.subcontext.id" in trace.attributes
            and "ai.tool.name" in trace.attributes
        ):
            subagent_name = trace.attributes["ai.tool.name"]
            supervisor_name = trace.attributes.get("ai.agent.name")
            subcontext_id = trace.attributes["ai.tool.subagent.subcontext.id"]
            result[subagent_name][supervisor_name].add(subcontext_id)

    return result


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
    conversation_id = supervisor.agent.conversation_id

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

    # Get traces for each subagent by finding out the subcontext_id used
    subcontext_ids = get_all_subagent_subcontext_ids(supervisor.agent.traces)
    weather_traces_subcontext_id = subcontext_ids["transfer_to_weather_agent"][
        None
    ].pop()
    weather.agent.set_conversation_id(
        conversation_id, subcontext_id=weather_traces_subcontext_id
    )
    weather_traces = weather.agent.traces
    events_traces_subcontext_id = subcontext_ids["transfer_to_events_agent"][None].pop()
    events.agent.set_conversation_id(
        conversation_id, subcontext_id=events_traces_subcontext_id
    )
    events_traces = events.agent.traces

    Expect(weather_traces).tool_invocations.to_include("get_weather").with_input(
        {"city": "Amsterdam"}
    )
    Expect(weather_traces).user_input.to_equal("Amsterdam")
    Expect(events_traces).tool_invocations.to_include("get_events").with_input(
        {"city": "Amsterdam"}
    )
    Expect(events_traces).user_input.to_equal("Amsterdam")
    Expect(supervisor.agent.traces).tool_invocations.to_include(
        "transfer_to_weather_agent"
    )
    Expect(supervisor.agent.traces).tool_invocations.to_include(
        "transfer_to_events_agent"
    )
    Expect(supervisor.agent.traces).agent_text_response.to_equal(
        "The weather in Amsterdam will be Sunny and the coming events are bla bla bla"
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
    conversation_id = supervisor.conversation_id
    subcontext_ids = get_all_subagent_subcontext_ids(supervisor.traces)

    Expect(supervisor.traces).tool_invocations.to_include("subagent_l1").with_input(
        {"user_input": "Hello L1 from L0"}
    )
    Expect(supervisor.traces).tool_invocations.to_include("subagent_l2").with_input(
        {"user_input": "Hello L2 from L0"}
    )
    Expect(supervisor.traces).agent_text_response.to_equal("Hello back to User!")

    subagent_l1_subcontext_id = subcontext_ids["subagent_l1"][None].pop()
    subagent_l2_subcontext_id = subcontext_ids["subagent_l2"][None].pop()

    subagent_l1.set_conversation_id(
        conversation_id, subcontext_id=subagent_l1_subcontext_id
    )
    subagent_l1_traces = subagent_l1.traces

    Expect(subagent_l1_traces).tool_invocations.to_include("subagent_l3").with_input(
        {"user_input": "Hello L3 from L1"}
    )
    Expect(subagent_l1_traces).agent_text_response.to_equal("Hello back to L0 from L1")

    subagent_l2.set_conversation_id(
        conversation_id, subcontext_id=subagent_l2_subcontext_id
    )
    subagent_l2_traces = subagent_l2.traces

    Expect(subagent_l2_traces).tool_invocations.to_include("subagent_l3").with_input(
        {"user_input": "Hello L3 from L2"}
    )

    Expect(subagent_l2_traces).agent_text_response.to_equal("Hello back to L0 from L2")

    subagent_l3_from_l1_subcontext_id = subcontext_ids["subagent_l3"][
        "subagent_l1"
    ].pop()
    subagent_l3.set_conversation_id(
        conversation_id, subcontext_id=subagent_l3_from_l1_subcontext_id
    )
    subagent_l3_from_l1_traces = subagent_l3.traces
    Expect(subagent_l3_from_l1_traces).agent_text_response.to_equal(
        "Hello back to L1 from L3"
    )

    subagent_l3_from_l2_subcontext_id = subcontext_ids["subagent_l3"][
        "subagent_l2"
    ].pop()
    subagent_l3.set_conversation_id(
        conversation_id, subcontext_id=subagent_l3_from_l2_subcontext_id
    )
    subagent_l3_from_l2_traces = subagent_l3.traces
    Expect(subagent_l3_from_l2_traces).agent_text_response.to_equal(
        "Hello back to L2 from L3"
    )

    l0_span_ids = {trace.span_id for trace in supervisor.traces}
    l1_span_ids = {trace.span_id for trace in subagent_l1_traces}
    l2_span_ids = {trace.span_id for trace in subagent_l2_traces}
    l3_from_l1_span_ids = {trace.span_id for trace in subagent_l3_from_l1_traces}
    l3_from_l2_span_ids = {trace.span_id for trace in subagent_l3_from_l2_traces}
    l3_span_ids = l3_from_l1_span_ids | l3_from_l2_span_ids

    # L0 (supervisor) should contain all traces from all subagents
    assert (l1_span_ids | l2_span_ids | l3_span_ids).issubset(l0_span_ids)

    # Verify all subagent traces are accounted for in L0
    # (L0 may have additional traces like converse, cycle, etc.)
    assert len((l1_span_ids | l2_span_ids | l3_span_ids) - l0_span_ids) == 0

    assert not l1_span_ids & l2_span_ids
    assert not l3_from_l1_span_ids & l3_from_l2_span_ids
