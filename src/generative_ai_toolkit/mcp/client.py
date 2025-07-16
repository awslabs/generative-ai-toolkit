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
import signal
import threading
from collections.abc import Awaitable, Callable, Sequence
from datetime import timedelta
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, Any, Protocol

from mcp import ClientSession, StdioServerParameters, Tool
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
    model_config = {"extra": "allow"}
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


class VerifyMcpServerToolType(Protocol):
    def __call__(
        self,
        *,
        mcp_server_config: McpServerConfig,
        tool_spec: "ToolSpecificationTypeDef",
    ) -> Any | Awaitable[Any]: ...


class McpClient:

    def __init__(
        self,
        agent: Agent,
        client_config_path: os.PathLike | str | None = None,
        verify_mcp_server_tool: VerifyMcpServerToolType | None = None,
    ):
        self.agent = agent
        self.config = self.load_client_config(
            [client_config_path] if client_config_path else None
        )
        self.verify_mcp_server_tool = verify_mcp_server_tool
        # Enable the MCP config to have relative paths:
        config_dir = os.path.dirname(self.config.path)
        if config_dir:
            os.chdir(config_dir)

    async def connect_mcp_servers(
        self,
        loop: asyncio.AbstractEventLoop,
    ):
        exit_stacks: list[contextlib.AsyncExitStack] = []

        async def cleanup(exit_stack: contextlib.AsyncExitStack):
            try:
                await exit_stack.aclose()
            except Exception:
                pass

        async def cleanup_all():
            try:
                await asyncio.gather(*map(cleanup, exit_stacks))
            except Exception:
                pass

        try:
            connected_mcp_servers = asyncio.as_completed(
                self.connect_to_mcp_server(mcp_server_config)
                for mcp_server_config in self.config.mcpServers.values()
            )

            tasks = []
            for connected_mcp_server in connected_mcp_servers:
                exit_stack, mcp_server_config, sessions, tools = (
                    await connected_mcp_server
                )
                exit_stacks.append(exit_stack)
                for tool in tools:
                    tasks.append(
                        asyncio.create_task(
                            self.register_mcp_tool(
                                loop,
                                mcp_server_config,
                                sessions,
                                tool,
                            )
                        )
                    )
            await asyncio.gather(*tasks)

        except Exception:
            await cleanup_all()
            raise

        return cleanup_all

    async def register_mcp_tool(
        self,
        loop: asyncio.AbstractEventLoop,
        mcp_server_config: McpServerConfig,
        session: ClientSession,
        tool: Tool,
    ):
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
        if self.verify_mcp_server_tool:
            if asyncio.iscoroutinefunction(self.verify_mcp_server_tool):
                await self.verify_mcp_server_tool(
                    mcp_server_config=mcp_server_config,
                    tool_spec=tool_spec,
                )
            else:
                await asyncio.to_thread(
                    self.verify_mcp_server_tool,
                    mcp_server_config=mcp_server_config,
                    tool_spec=tool_spec,
                )

        def func(**kwargs):
            fut = asyncio.run_coroutine_threadsafe(
                session.call_tool(
                    tool.name,
                    arguments=kwargs,
                    read_timeout_seconds=timedelta(seconds=30),
                ),
                loop,
            )

            res = fut.result().model_dump()
            self.agent.tracer.current_trace.add_attribute("ai.mcp.response", res)
            return res["content"][0]["text"]

        self.agent.register_tool(
            func,
            tool_spec=tool_spec,
        )

    async def connect_to_mcp_server(self, mcp_server_config: McpServerConfig):
        exit_stack = contextlib.AsyncExitStack()
        server_params = StdioServerParameters(
            command=mcp_server_config.command,
            args=mcp_server_config.args,
            env=mcp_server_config.env,
        )

        stdio_transport = await exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        stdio, write = stdio_transport
        session = await exit_stack.enter_async_context(ClientSession(stdio, write))

        await session.initialize()
        response = await session.list_tools()
        tools = response.tools
        return exit_stack, mcp_server_config, session, tools

    def chat(
        self,
        chat_fn: Callable[[Agent, Event], Any] | None = None,
        *,
        stop_event: Event | None = None,
    ):
        asyncio.run(self._chat(chat_fn, stop_event=stop_event))

    async def _chat(
        self,
        chat_fn: Callable[[Agent, Event], Any] | None = None,
        *,
        stop_event: Event | None = None,
    ):
        loop = asyncio.get_running_loop()
        cleanup = await self.connect_mcp_servers(loop)
        stop_event = stop_event or Event()

        if threading.current_thread() is threading.main_thread():

            def handler(signum, frame):
                stop_event.set()

            signal.signal(signal.SIGINT, handler)

        try:
            await loop.run_in_executor(
                None, chat_fn or self._default_chat_fn, self.agent, stop_event
            )
        finally:
            await cleanup()

    def _default_chat_fn(self, agent: Agent, stop_event: Event):
        """
        Chat with the MCP client

        This is meant as a testing utility. Any serious MCP client would likely customize this implementation.
        """
        from generative_ai_toolkit.ui import chat_ui  # noqa: PLC0415

        demo = chat_ui(agent)
        demo.launch(prevent_thread_lock=True, quiet=True, inbrowser=True)
        print(f"\nMCP server configuration loaded: {self.config.path or 'None'}")
        print("\nRegistered tools:\n")
        for tool in agent.tools.values():
            print(f"  {tool.tool_spec["name"]}")
            print(f"  {"_" * len(tool.tool_spec["name"])}\n")
            for line in tool.tool_spec.get("description", "").strip().splitlines():
                print(f"    {line}")
            print()
        print()
        print(f"Running MCP client at {demo.local_url}\n")
        print("Press CTRL-C to quit.\n")

        stop_event.wait()

        print("\n\nGoodbye!")

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
