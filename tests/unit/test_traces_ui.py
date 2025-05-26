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

from generative_ai_toolkit.ui import chat_messages_from_traces


def test_chat_messages_from_traces_converse(mock_multi_agent):
    [
        supervisor,
        events_agent,
        weather_agent,
        supervisor_mock,
        events_agent_mock,
        weather_agent_mock,
    ] = mock_multi_agent

    def populate_mock():
        supervisor_mock.add_output(
            tool_use_output=[
                {
                    "name": "transfer_to_weather_agent",
                    "input": {"user_input": "Amsterdam"},
                },
                {
                    "name": "transfer_to_events_agent",
                    "input": {"user_input": "Amsterdam"},
                },
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

    for method in [supervisor.converse, supervisor.converse_stream]:
        supervisor.reset()
        populate_mock()
        list(method("I want to go to Amsterdam, what is up there?"))

        *_, messages = chat_messages_from_traces(
            supervisor.traces,
        )

        assert len(messages) == 6
        text = messages[-1].content
        assert (
            type(text) is str
            and text.strip()
            == "The weather in Amsterdam will be Sunny and the coming events are bla bla bla"
        )
        assert messages[-1].role == "assistant"
