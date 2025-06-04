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
import contextlib
import json
import os
from collections.abc import Callable, Sequence
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, Field, PrivateAttr

from generative_ai_toolkit.agent.agent import Agent

if TYPE_CHECKING:
    from generative_ai_toolkit.agent.tool import ToolSpecificationTypeDef

CONFIG_PATHS = [
    Path(os.curdir) / "mcp.json",
    os.path.expanduser("~/.aws/amazonq/mcp.json"),
]

DIM = "\033[2m"
RESET = "\033[0m"


class McpServerConfig(BaseModel):
    command: str
    env: dict[str, str] = Field({})
    args: list[str] = Field([])


class McpClientConfig(BaseModel):
    mcpServers: dict[str, McpServerConfig]  # noqa: N815
    _path: str = PrivateAttr(default="")

    @property
    def path(self) -> str:
        return self._path

    @classmethod
    def from_file(cls, path: str | os.PathLike) -> "McpClientConfig":
        with open(path) as f:
            data = json.load(f)
        instance = cls(**data)
        instance._path = str(path)
        return instance


class McpClient:

    def __init__(
        self, agent: Agent, client_config_path: os.PathLike | str | None = None
    ):
        self.agent = agent
        self.config = self.load_client_config(
            [client_config_path] if client_config_path else None
        )
        # Enable the MCP config to have relative paths:
        config_dir = os.path.dirname(self.config.path)
        if config_dir:
            os.chdir(config_dir)

    async def connect_mcp_servers(
        self,
        loop: asyncio.AbstractEventLoop,
        exit_stack: contextlib.AsyncExitStack,
    ):
        def make_tool_func(session: ClientSession, tool_name: str):
            def func(**kwargs):
                fut = asyncio.run_coroutine_threadsafe(
                    session.call_tool(
                        tool_name,
                        arguments=kwargs,
                        read_timeout_seconds=timedelta(seconds=30),
                    ),
                    loop,
                )

                res = fut.result().model_dump()
                self.agent.tracer.current_trace.add_attribute("ai.mcp.response", res)
                return res["content"][0]["text"]

            return func

        for params in self.config.mcpServers.values():
            session, tools = await self.connect_to_mcp_server(
                exit_stack=exit_stack,
                command=params.command,
                args=params.args,
                env=params.env,
            )
            for tool in tools:
                tool_spec: ToolSpecificationTypeDef = {
                    "name": tool.name,
                    "description": tool.description or "",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": tool.inputSchema["properties"],
                        }
                    },
                }
                self.agent.register_tool(
                    make_tool_func(session, tool.name),
                    tool_spec=tool_spec,
                )

    async def connect_to_mcp_server(
        self,
        exit_stack: contextlib.AsyncExitStack,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ):
        server_params = StdioServerParameters(command=command, args=args or [], env=env)

        stdio_transport = await exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        stdio, write = stdio_transport
        session = await exit_stack.enter_async_context(ClientSession(stdio, write))

        await session.initialize()
        response = await session.list_tools()
        tools = response.tools
        return session, tools

    def chat(
        self,
        chat_loop: Callable[[Agent], Any] | None = None,
    ):
        asyncio.run(self._chat(chat_loop))

    async def _chat(
        self,
        chat_loop: Callable[[Agent], Any] | None = None,
    ):
        async with contextlib.AsyncExitStack() as exit_stack:
            loop = asyncio.get_running_loop()
            await self.connect_mcp_servers(loop, exit_stack)
            await loop.run_in_executor(
                None, chat_loop or self._default_chat_loop, self.agent
            )

    def _default_chat_loop(self, agent: Agent):
        """
        Chat with the MCP client

        This is meant as a testing utility. Any serious MCP client would likely customize this implementation.
        """

        print(f"MCP server configuration loaded: {self.config.path}")
        print(
            "\nMCP client ready. Type /q to quit. Type /t to list the available tools.\n"
        )
        while True:
            user_input = input("You: ").strip()
            if not user_input or user_input == "/q":
                print("Assistant: Goodbye!")
                break

            if user_input.strip() == "/t":
                print("\nListing available tools:\n")
                for tool in agent.tools.values():
                    print(f"  {tool.tool_spec["name"]}")
                    print(f"  {"_" * len(tool.tool_spec["name"])}\n")
                    for line in (
                        tool.tool_spec.get("description", "").strip().splitlines()
                    ):
                        print(f"    {line}")
                    print()
                print()
                continue

            for index, fragment in enumerate(agent.converse_stream(user_input)):
                if not index:
                    print("Assistant: ", end="", flush=True)
                if fragment.strip() == "<thinking>":
                    print(DIM, end="", flush=True)
                print(fragment, end="", flush=True)
                if fragment.strip() == "</thinking>":
                    print(RESET, end="", flush=True)
            print()

    @staticmethod
    def load_client_config(
        paths: Sequence[str | os.PathLike] | None = None,
    ) -> McpClientConfig:
        for path in paths or CONFIG_PATHS:
            try:
                cfg = McpClientConfig.from_file(path)
            except FileNotFoundError:
                if not paths:
                    # Using default locations
                    continue
            except json.decoder.JSONDecodeError as err:
                raise RuntimeError(f"Failed to parse {path}: {err}") from err
            else:
                break
        else:
            return McpClientConfig(mcpServers={})
        return cfg
