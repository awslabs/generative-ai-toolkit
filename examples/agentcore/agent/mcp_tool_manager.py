#!/usr/bin/env python3
"""MCP Tool Manager for integrating MCP servers with Generative AI Toolkit agents."""

import logging
import os

from simple_mcp_client import SimpleMcpClient

from generative_ai_toolkit.agent import BedrockConverseAgent

logger = logging.getLogger(__name__)


class McpToolManager:
    """Manages MCP client and tool registration."""

    def __init__(self):
        self.mcp_client: SimpleMcpClient | None = None
        self.tools_registered = False
        self._connection_valid = False

    async def get_mcp_client(self) -> SimpleMcpClient:
        """Get or create MCP client with lazy initialization."""
        mcp_arn = os.environ[
            "MCP_SERVER_RUNTIME_ARN"
        ]  # Required env var, validated at startup

        # Always create a fresh client for each event loop to avoid connection issues
        try:
            logger.info(f"Creating fresh MCP client for runtime: {mcp_arn}")
            client = SimpleMcpClient(runtime_arn=mcp_arn)
            await client.connect()
            logger.info("MCP client connected successfully")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize MCP client: {e}")
            raise

    async def register_mcp_tools(self, agent: BedrockConverseAgent) -> bool:
        """Register MCP tools with the Generative AI Toolkit agent."""
        if self.tools_registered:
            return True

        try:
            mcp_client = await self.get_mcp_client()

            # List available tools from MCP server
            tools_result = await mcp_client.list_tools()

            if not tools_result.tools:
                logger.warning("No tools available from MCP server")
                return False

            logger.info(f"Registering {len(tools_result.tools)} MCP tools with agent")

            # Register each MCP tool with the Generative AI Toolkit
            for mcp_tool in tools_result.tools:
                logger.info(f"Registering tool: {mcp_tool.name}")

                # Create a wrapper function for the MCP tool
                def create_tool_wrapper(tool_name: str):
                    async def tool_wrapper(**kwargs) -> str:
                        """Wrapper function to call MCP tool."""
                        try:
                            # Create a fresh client for each tool call to avoid event loop issues
                            client = await self.get_mcp_client()

                            result = await client.call_tool(tool_name, kwargs)

                            # Extract text content from MCP result
                            if hasattr(result, "content") and result.content:
                                if hasattr(result.content[0], "text"):
                                    return result.content[0].text
                                else:
                                    return str(result.content[0])
                            else:
                                return str(result)

                        except Exception as e:
                            logger.error(f"Error calling MCP tool {tool_name}: {e}")
                            return f"Error calling {tool_name}: {str(e)}"

                    return tool_wrapper

                # Create the wrapper and register it
                tool_func = create_tool_wrapper(mcp_tool.name)
                tool_func.__name__ = mcp_tool.name
                tool_func.__doc__ = mcp_tool.description

                # Create tool_spec in Bedrock format from MCP tool schema
                tool_spec = {
                    "name": mcp_tool.name,
                    "description": mcp_tool.description,
                    "inputSchema": {"json": mcp_tool.inputSchema},
                }

                # Register with the agent
                agent.register_tool(tool_func, tool_spec=tool_spec)

            self.tools_registered = True
            logger.info("MCP tools registered successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to register MCP tools: {e}")
            return False
