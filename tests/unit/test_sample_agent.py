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

from generative_ai_toolkit.test import Case, Expect


def test_mock_agent_1(mock_agent_1, mock_bedrock_converse):
    sample_response1 = "Hello, how can I help you today"
    mock_bedrock_converse.add_output(text_output=[sample_response1])
    sample_response2 = "I don't have a name"
    mock_bedrock_converse.add_output(text_output=[sample_response2])

    # run conversation through:
    Case(["Hi there!", "What's your name?"]).run(mock_agent_1)

    # check agent responses:
    Expect(mock_agent_1.traces).agent_text_response.at(0).to_equal(sample_response1)
    Expect(mock_agent_1.traces).agent_text_response.to_equal(sample_response2)


def test_mock_agent_2(mock_agent_2, mock_bedrock_converse):
    mock_bedrock_converse.add_output(
        text_output=[
            "Okay, let's find some interesting museums for you within a 30 minute drive. "
        ],
        tool_use_output=[{"name": "get_current_location", "input": {}}],
    )
    mock_bedrock_converse.add_output(
        text_output=[
            "Alright, I've got your current location. Now let me check what interesting museums are nearby that you can get to in under 30 minutes."
        ],
        tool_use_output=[
            {
                "name": "get_interesting_things_to_do",
                "input": {
                    "current_location": [52.00667, 4.35556],
                    "max_drive_time_minutes": 30,
                },
            }
        ],
    )
    mock_bedrock_converse.add_output(
        text_output=[
            "Based on the results, here are 5 interesting museum options within a 30 minute drive:\n\n1. Museum of Modern Art (10 minute drive)\n2. Aquarium of the Pacific (20 minute drive) \n3. Science Museum (14 minute drive)\n4. Performing Arts Center (16 minute drive)\n5. Historic Mansion Tour (17 minute drive)\n\nFor the outdoor activities like the Aquarium and Performing Arts Center, let me also check the weather forecast:"
        ],
        tool_use_output=[
            {
                "name": "weather_inquiry",
                "input": {
                    "latitude_longitude_list": [
                        [52.00767, 4.35656],
                        [52.00467, 4.35856],
                        [52.00767, 4.35156],
                    ]
                },
            }
        ],
    )
    mock_bedrock_converse.add_output(
        text_output=["Based on the information I have, the top 5 museum options ..."],
    )
    Case(["I want to go to a museum. Max 30 min driving please."]).run(mock_agent_2)
    Expect(mock_agent_2.traces).agent_text_response.to_include(
        "Based on the information I have, the top 5 museum options ..."
    )
    Expect(mock_agent_2.traces).tool_invocations.to_include(
        "get_current_location"
    ).with_input({})
    Expect(mock_agent_2.traces).tool_invocations.to_include(
        "get_interesting_things_to_do"
    ).with_input(
        {
            "current_location": [52.00667, 4.35556],
            "max_drive_time_minutes": 30,
        }
    )
    Expect(mock_agent_2.traces).tool_invocations.to_include(
        "weather_inquiry"
    ).with_input(
        {
            "latitude_longitude_list": [
                [52.00767, 4.35656],
                [52.00467, 4.35856],
                [52.00767, 4.35156],
            ]
        }
    )
