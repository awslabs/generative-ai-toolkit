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

import time
from threading import Event

from generative_ai_toolkit.agent import BedrockConverseAgent


def test_stop_event(mock_bedrock_converse):
    agent = BedrockConverseAgent(
        model_id="dummy", session=mock_bedrock_converse.session()
    )

    def dummy_tool() -> str:
        """
        Returns a static string
        """

        # Sleep the thread for 30 seconds.
        # Real tool implementations should check the AgentContext.current().stop_event
        # and abort early if it is set!
        time.sleep(30)
        return "Tool invocation is cancelled before this return value is ever read"

    agent.register_tool(dummy_tool)

    mock_bedrock_converse.add_output(
        tool_use_output={"name": "dummy_tool", "input": {}}
    )

    stop_event = Event()
    for trace in agent.converse_stream(
        "Go do it", stop_event=stop_event, stream="traces"
    ):
        if trace.attributes.get("ai.trace.type") == "tool-invocation":
            if not trace.ended_at:
                time.sleep(0.2)
                stop_event.set()
            else:
                assert "ai.tool.error" in trace.attributes
                assert trace.attributes["ai.tool.error"] == "StopEventAbortError()"

    assert agent.traces[0].attributes.get("ai.conversation.aborted") is True


def test_stop_event_multi_agent(mock_bedrock_converse):
    supervisor_mock = mock_bedrock_converse.__class__()
    subagent_mock = mock_bedrock_converse.__class__()
    supervisor = BedrockConverseAgent(
        model_id="dummy", session=supervisor_mock.session()
    )
    subagent = BedrockConverseAgent(
        model_id="dummy",
        session=subagent_mock.session(),
        name="subagent",
        description="Subagent",
    )

    supervisor.register_tool(subagent)

    def dummy_tool() -> str:
        """
        Returns a static string
        """

        # Sleep the thread for 30 seconds.
        # Real tool implementations should check the AgentContext.current().stop_event
        # and abort early if it is set!
        time.sleep(30)
        return "Tool invocation is cancelled before this return value is ever read"

    subagent.register_tool(dummy_tool)

    supervisor_mock.add_output(
        tool_use_output=[
            {"name": "subagent", "input": {"user_input": "Go do it subagent"}}
        ]
    )
    subagent_mock.add_output(tool_use_output={"name": "dummy_tool", "input": {}})

    stop_event = Event()
    for trace in supervisor.converse_stream(
        "Go do it supervisor", stop_event=stop_event, stream="traces"
    ):
        if (
            trace.attributes.get("ai.trace.type") == "tool-invocation"
            and trace.attributes["ai.tool.name"] == "dummy_tool"
        ):
            if not trace.ended_at:
                time.sleep(0.2)
                stop_event.set()
            else:
                assert "ai.tool.error" in trace.attributes
                assert trace.attributes["ai.tool.error"] == "StopEventAbortError()"

    assert supervisor.traces[0].attributes.get("ai.conversation.aborted") is True
