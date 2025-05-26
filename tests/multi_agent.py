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
from generative_ai_toolkit.test.mock import MockBedrockConverse


def multi_agent():
    supervisor_mock = MockBedrockConverse()
    weather_agent_mock = MockBedrockConverse()
    events_agent_mock = MockBedrockConverse()

    def get_weather(city: str):
        """
        Gets the weather forecast for the provided city

        Parameters
        ---
        city : string
          The city to get the weather forecast for
        """
        return "Sunny"

    weather_agent = BedrockConverseAgent(
        model_id="dummy",
        session=weather_agent_mock.session(),
        system_prompt="You provide the weather forecast for the specified city.",
        name="transfer_to_weather_agent",
        description="Get the weather forecast for a city.",
    )
    weather_agent.register_tool(get_weather)

    def get_events(city: str):
        """
        Gets upcoming events (concerts, festivals, etc) in the provided city

        Parameters
        ---
        city : string
          The city to get the upcoming events for
        """
        return [
            {"name": "Nirvana tribute band concert", "when": "Tomorrow"},
            {"name": "Food truck festival", "when": "Coming Saturday"},
            {
                "name": "Open Museum day -- all museums have free entry",
                "when": "Coming Sunday",
            },
        ]

    events_agent = BedrockConverseAgent(
        model_id="dummy",
        session=events_agent_mock.session(),
        system_prompt="You provide the upcoming events for the specified city.",
        name="transfer_to_events_agent",
        description="Get the upcoming events for a city.",
    )
    events_agent.register_tool(get_events)

    supervisor = BedrockConverseAgent(
        model_id="dummy",
        session=supervisor_mock.session(),
        system_prompt="You provide users current information on cities that they may want to visit",
    )
    supervisor.register_tool(weather_agent)
    supervisor.register_tool(events_agent)

    return [
        supervisor,
        events_agent,
        weather_agent,
        supervisor_mock,
        events_agent_mock,
        weather_agent_mock,
    ]
