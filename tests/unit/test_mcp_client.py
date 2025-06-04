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

from pathlib import Path

import pytest

from generative_ai_toolkit.agent import Agent, BedrockConverseAgent
from generative_ai_toolkit.mcp.client import McpClient
from generative_ai_toolkit.test import Expect

HERE = Path(__file__).parent


def test_mcp_client(mock_bedrock_converse):
    agent = BedrockConverseAgent(
        model_id="dummy", session=mock_bedrock_converse.session()
    )

    mcp_client = McpClient(agent, client_config_path=str(HERE / "mcp.json"))
    assert mcp_client.config.model_dump() == {
        "mcpServers": {
            "WeatherForecasts": {
                "args": [
                    "mcp_server_get_weather.py",
                ],
                "command": "python",
                "env": {
                    "WEATHER": "Sunny",
                    "FASTMCP_LOG_LEVEL": "ERROR",
                },
            },
        },
    }

    def chat_loop(agent: Agent):
        agent.converse("What is the weather currently?")

    mock_bedrock_converse.add_output(tool_use_output=[{"name": "current_weather"}])
    mock_bedrock_converse.add_output(text_output=["Sunny"])

    mcp_client.chat(chat_loop=chat_loop)

    Expect(agent.traces).tool_invocations.to_include("current_weather")
    Expect(agent.traces).agent_text_response.to_equal("Sunny")


def test_invalid_json(mock_bedrock_converse):
    agent = BedrockConverseAgent(
        model_id="dummy", session=mock_bedrock_converse.session()
    )
    with pytest.raises(RuntimeError, match="invalid.json"):
        McpClient(agent, client_config_path=str(HERE / "invalid.json"))
