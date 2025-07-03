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
from generative_ai_toolkit.context import AgentContext, AuthContext
from generative_ai_toolkit.tracer.trace import Trace


def test_tools_can_access_agent_context(mock_bedrock_converse):

    agent = BedrockConverseAgent(
        model_id="dummy", session=mock_bedrock_converse.session()
    )
    test_span_name = "test_tools_can_access_agent_context"
    test_auth_context: AuthContext = {
        "principal_id": "test-principal-id",
        "extra": {"some_other_attribute": "some-other-attribute"},
    }
    test_conversation_id = "test-conversation-id"
    current_trace_during_tool_invocation: Trace | None = None

    def dummy_tool() -> str:
        """
        Returns a static string
        """
        context = AgentContext.current()

        nonlocal current_trace_during_tool_invocation
        current_trace_during_tool_invocation = context.tracer.current_trace
        current_trace_during_tool_invocation.add_attribute("foo", "bar")

        with context.tracer.trace(test_span_name) as trace:
            trace.add_attribute("test.auth.context", context.auth_context)
            trace.add_attribute(
                "test.auth.principal.id", context.auth_context["principal_id"]
            )
            trace.add_attribute(
                "test.auth.some_other_attribute",
                context.auth_context.get("extra", {})["some_other_attribute"],
            )
            trace.add_attribute("test.conversation.id", context.conversation_id)

        return "static string"

    agent.register_tool(dummy_tool)

    mock_bedrock_converse.add_output(
        tool_use_output={"name": "dummy_tool", "input": {}}
    )
    mock_bedrock_converse.add_output("Agent response")

    agent.set_auth_context(**test_auth_context)
    agent.set_conversation_id(test_conversation_id)
    agent.converse("Test message")

    tool_invocation_trace = next(
        trace
        for trace in agent.traces
        if trace.attributes.get("ai.trace.type") == "tool-invocation"
    )

    assert tool_invocation_trace == current_trace_during_tool_invocation

    assert "foo" in tool_invocation_trace.attributes
    assert tool_invocation_trace.attributes["foo"] == "bar"

    for trace in agent.traces:
        if trace.span_name == test_span_name:
            assert trace.parent_span == tool_invocation_trace
            assert trace.attributes["test.auth.context"] == test_auth_context
            assert (
                trace.attributes["test.auth.principal.id"]
                == test_auth_context["principal_id"]
            )
            assert (
                trace.attributes["test.auth.some_other_attribute"]
                == test_auth_context.get("extra", {})["some_other_attribute"]
            )
            assert trace.attributes["test.conversation.id"] == test_conversation_id
            break
    else:
        raise Exception("Did not find test span in traces!")


def test_tools_can_access_agent_context_multi_tool(mock_bedrock_converse):
    """
    test that the AgentContext mechanism also works when multiple tools are used
    (as they would be run in threads, this is different from single tool case)
    """

    agent = BedrockConverseAgent(
        model_id="dummy", session=mock_bedrock_converse.session()
    )
    test_span_name = "test_tools_can_access_agent_context"
    test_auth_context_principal_id = "test-principal-id"
    test_conversation_id = "test-conversation-id"

    def dummy_tool() -> str:
        """
        Returns a static string
        """
        context = AgentContext.current()

        with context.tracer.trace(test_span_name + "_1") as trace:
            trace.add_attribute("test.auth.context", context.auth_context)
            trace.add_attribute("test.conversation.id", context.conversation_id)

        return "static string"

    def dummy_tool2() -> str:
        """
        Returns a static string
        """
        context = AgentContext.current()

        with context.tracer.trace(test_span_name + "_2") as trace:
            trace.add_attribute("test.auth.context", context.auth_context)
            trace.add_attribute("test.conversation.id", context.conversation_id)

        return "static string"

    agent.register_tool(dummy_tool)
    agent.register_tool(dummy_tool2)

    mock_bedrock_converse.add_output(
        tool_use_output=[
            {"name": "dummy_tool", "input": {}},
            {"name": "dummy_tool2", "input": {}},
        ]
    )
    mock_bedrock_converse.add_output("Agent response")

    agent.set_auth_context(principal_id=test_auth_context_principal_id)
    agent.set_conversation_id(test_conversation_id)
    agent.converse("Test message")

    for span_name_suffix in ["_1", "_2"]:
        span_name = test_span_name + span_name_suffix
        for trace in agent.traces:
            if trace.span_name == span_name:
                assert (
                    trace.attributes["test.auth.context"]["principal_id"]
                    == test_auth_context_principal_id
                )
                assert trace.attributes["test.conversation.id"] == test_conversation_id
                break
        else:
            raise Exception(f"Did not find test span {span_name} in traces!")
