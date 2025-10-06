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

from generative_ai_toolkit.ui.lib import chat_messages_from_traces


def test_chat_messages_from_traces_converse(mock_multi_agent):
    supervisor = mock_multi_agent.supervisor
    weather = mock_multi_agent.weather
    events = mock_multi_agent.events

    def populate_mock():
        supervisor.mock.add_output(
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

    for method in [supervisor.agent.converse, supervisor.agent.converse_stream]:
        supervisor.agent.reset()
        populate_mock()
        list(method("I want to go to Amsterdam, what is up there?"))

        chat_messages = chat_messages_from_traces(
            supervisor.agent.traces,
        )

        assert len(chat_messages.messages) == 16
        text = chat_messages.messages[-1].content
        assert (
            type(text) is str
            and text.strip()
            == "The weather in Amsterdam will be Sunny and the coming events are bla bla bla"
        )
        assert chat_messages.messages[-1].role == "assistant"
