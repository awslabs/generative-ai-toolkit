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

import asyncio
import time
from pathlib import Path
from threading import Event, Thread

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
                "expectation": "I will use this tool to check the weather",
            },
        },
    }

    def chat_fn(agent: Agent, stop_event: Event):
        agent.converse("What is the weather currently?")

    mock_bedrock_converse.add_output(tool_use_output=[{"name": "current_weather"}])
    mock_bedrock_converse.add_output(text_output=["Sunny"])

    # Run chat() in thread, as it uses asyncio.run() which wants to create its own event loop
    run_until_complete_in_thread(mcp_client.chat, chat_fn)

    Expect(agent.traces).tool_invocations.to_include("current_weather")
    Expect(agent.traces).agent_text_response.to_equal("Sunny")


def test_invalid_json(mock_bedrock_converse):
    agent = BedrockConverseAgent(
        model_id="dummy", session=mock_bedrock_converse.session()
    )
    with pytest.raises(RuntimeError, match="invalid.json"):
        McpClient(agent, client_config_path=str(HERE / "invalid.json"))


def test_mcp_server_verification(mock_bedrock_converse):
    agent = BedrockConverseAgent(
        model_id="dummy", session=mock_bedrock_converse.session()
    )

    def pass_verification(*, mcp_server_config, tool_spec):
        assert mcp_server_config.expectation
        time.sleep(0.1)

    async def pass_verification_async(*, mcp_server_config, tool_spec):
        assert mcp_server_config.expectation
        await asyncio.sleep(0.1)

    def fail_verification(*, mcp_server_config, tool_spec):
        assert mcp_server_config.expectation
        time.sleep(0.1)
        raise RuntimeError(f"MCP server tool verification failed: {mcp_server_config}")

    async def fail_verification_async(*, mcp_server_config, tool_spec):
        assert mcp_server_config.expectation
        await asyncio.sleep(0.1)
        raise RuntimeError(f"MCP server tool verification failed: {mcp_server_config}")

    def chat_fn(agent: Agent, stop_event: Event):
        pass

    # Sync - success
    # Run chat() in thread, as it uses asyncio.run() which wants to create its own event loop
    run_until_complete_in_thread(
        McpClient(
            agent,
            client_config_path=str(HERE / "mcp.json"),
            verify_mcp_server_tool=pass_verification,
        ).chat,
        chat_fn=chat_fn,
    )

    # Async - success
    # Run chat() in thread, as it uses asyncio.run() which wants to create its own event loop
    run_until_complete_in_thread(
        McpClient(
            agent,
            client_config_path=str(HERE / "mcp.json"),
            verify_mcp_server_tool=pass_verification_async,
        ).chat,
        chat_fn=chat_fn,
    )

    # Sync - fail
    # Run chat() in thread, as it uses asyncio.run() which wants to create its own event loop
    with pytest.raises(RuntimeError, match="MCP server tool verification failed"):
        run_until_complete_in_thread(
            McpClient(
                agent,
                client_config_path=str(HERE / "mcp.json"),
                verify_mcp_server_tool=fail_verification_async,
            ).chat,
            chat_fn=chat_fn,
        )

    # Async - fail
    # Run chat() in thread, as it uses asyncio.run() which wants to create its own event loop
    with pytest.raises(RuntimeError, match="MCP server tool verification failed"):
        run_until_complete_in_thread(
            McpClient(
                agent,
                client_config_path=str(HERE / "mcp.json"),
                verify_mcp_server_tool=fail_verification,
            ).chat,
            chat_fn=chat_fn,
        )


def run_until_complete_in_thread(target, *args, **kwargs):
    res = []
    err = []

    def capture(*args, **kwargs):
        try:
            res.append(target(*args, **kwargs))
        except Exception as e:
            err.append(e)

    thread = Thread(target=capture, args=args, kwargs=kwargs, daemon=True)
    thread.start()
    thread.join()
    if err:
        raise err[0]
    return res[0]
