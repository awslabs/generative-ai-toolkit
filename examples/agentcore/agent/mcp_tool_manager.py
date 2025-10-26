#!/usr/bin/env python3
"""MCP Tool Manager for integrating MCP servers with Generative AI Toolkit agents."""

import asyncio
import logging
import os

from simple_mcp_client import SimpleMcpClient

from generative_ai_toolkit.agent import BedrockConverseAgent
from generative_ai_toolkit.agent.tool import Tool

logger = logging.getLogger(__name__)


class McpTool(Tool):
    """A Tool implementation that wraps MCP tool calls."""

    def __init__(self, mcp_tool, tool_manager: "McpToolManager"):
        self.mcp_tool = mcp_tool
        self.tool_manager = tool_manager
        self._tool_spec = {
            "name": mcp_tool.name,
            "description": mcp_tool.description,
            "inputSchema": {"json": mcp_tool.inputSchema},
        }

    @property
    def tool_spec(self):
        return self._tool_spec

    def invoke(self, **kwargs):
        """Invoke the MCP tool synchronously."""
        try:
            # Run the async operation synchronously to match Tool protocol
            return asyncio.run(self._async_invoke(**kwargs))
        except Exception as e:
            logger.error(
                f"Error calling MCP tool {self.mcp_tool.name}: {e}", exc_info=True
            )
            return f"Error calling {self.mcp_tool.name}: {str(e)}"

    async def _async_invoke(self, **kwargs):
        """Internal async implementation of MCP tool invocation."""
        try:
            # Create a dedicated client for this tool call to avoid concurrency issues
            client = await self.tool_manager.get_dedicated_client()

            logger.info(f"Calling MCP tool '{self.mcp_tool.name}' with args: {kwargs}")

            # Use the client in a context manager to ensure proper cleanup
            async with client:
                result = await client.call_tool(self.mcp_tool.name, kwargs)
                logger.info(f"MCP tool '{self.mcp_tool.name}' completed successfully")

                # Extract text content from MCP result
                if hasattr(result, "content") and result.content:
                    if hasattr(result.content[0], "text"):
                        return result.content[0].text
                    else:
                        return str(result.content[0])
                else:
                    return str(result)
        except Exception as e:
            logger.error(
                f"Error in _async_invoke for {self.mcp_tool.name}: {e}", exc_info=True
            )
            raise


class McpToolManager:
    """Manages MCP client and tool registration."""

    def __init__(self):
        self.mcp_client: SimpleMcpClient | None = None
        self.tools_registered = False
        self.current_jwt_token: str | None = None

    def set_jwt_token(self, jwt_token: str) -> None:
        """Set the JWT token for MCP authentication."""
        self.current_jwt_token = jwt_token
        # If we have an existing client, update its token
        if self.mcp_client:
            self.mcp_client.set_jwt_token(jwt_token)

    def _get_or_create_client(self) -> SimpleMcpClient:
        """Get or create MCP client instance (without connecting)."""
        if self.mcp_client is None:
            mcp_arn = os.environ[
                "MCP_SERVER_RUNTIME_ARN"
            ]  # Required env var, validated at startup

            logger.info(f"Creating MCP client for runtime: {mcp_arn}")
            self.mcp_client = SimpleMcpClient(
                runtime_arn=mcp_arn, jwt_token=self.current_jwt_token
            )

        # Ensure the client has the current JWT token
        if self.current_jwt_token:
            self.mcp_client.set_jwt_token(self.current_jwt_token)

        return self.mcp_client

    async def get_connected_client(self) -> SimpleMcpClient:
        """Get MCP client and ensure it's connected (for tool registration only)."""
        client = self._get_or_create_client()

        # Simple connection check without testing - avoid connection interference
        try:
            if not client.is_connected():
                logger.info("Connecting to MCP server")
                await client.connect()
                logger.info("MCP client connected successfully")
            else:
                logger.debug("Reusing existing MCP connection")
        except Exception as e:
            logger.info(f"Connection invalid ({e}), reconnecting...")
            # Force reconnection
            try:
                await client.disconnect()
            except Exception:  # nosec B110
                pass  # Ignore disconnect errors during reconnection

            await client.connect()
            logger.info("MCP client reconnected successfully")

        return client

    async def get_dedicated_client(self) -> SimpleMcpClient:
        """Get a dedicated MCP client for a single tool call to avoid concurrency issues."""
        mcp_arn = os.environ["MCP_SERVER_RUNTIME_ARN"]

        # Create a new client instance for this specific call with current JWT token
        logger.debug("Creating dedicated MCP client for tool call")
        return SimpleMcpClient(runtime_arn=mcp_arn, jwt_token=self.current_jwt_token)

    async def register_mcp_tools(self, agent: BedrockConverseAgent) -> bool:
        """Register MCP tools with the Generative AI Toolkit agent."""
        if self.tools_registered:
            logger.info("MCP tools already registered, skipping registration")
            return True

        try:
            # Get connected MCP client
            mcp_client = await self.get_connected_client()

            # List available tools from MCP server
            tools_result = await mcp_client.list_tools()

            if not tools_result.tools:
                logger.warning("No tools available from MCP server")
                return False

            logger.info(f"Registering {len(tools_result.tools)} MCP tools with agent")

            # Register each MCP tool with the Generative AI Toolkit
            for mcp_tool in tools_result.tools:
                logger.info(f"Registering tool: {mcp_tool.name}")

                # Create MCP tool wrapper
                tool = McpTool(mcp_tool, self)

                # Register with the agent
                agent.register_tool(tool)

            self.tools_registered = True
            logger.info("MCP tools registered successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to register MCP tools: {e}")
            return False

    async def cleanup(self):
        """Clean up MCP client resources."""
        if self.mcp_client:
            try:
                await self.mcp_client.disconnect()
            except Exception as e:
                logger.warning(f"Error during MCP client cleanup: {e}")
            finally:
                self.mcp_client = None
                self.tools_registered = False
