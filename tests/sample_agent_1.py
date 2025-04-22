# Copyright 2024 Amazon.com, Inc. and its affiliates. All Rights Reserved.
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

import boto3.session

from generative_ai_toolkit.agent import BedrockConverseAgent


def sample_agent_1(session: boto3.session.Session | None = None):
    agent = BedrockConverseAgent(model_id="amazon.nova-lite-v1:0", session=session)

    def weather_tool(city: str, unit: str = "celsius") -> str:
        """
        Get the weather report for a city

        Parameters
        ---
        city: str
          The city
        unit: str
          The unit of degrees (e.g. celsius)
        """
        return f"The weather in {city} is 20 degrees {unit}."

    agent.register_tool(weather_tool)
    return agent
