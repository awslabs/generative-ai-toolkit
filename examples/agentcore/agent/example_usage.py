"""
Example usage of AgentCoreAuthenticatedMcpClient with AgentCore Runtime.

This example shows how to connect to an AgentCore Runtime MCP server
using OAuth Bearer token authentication.
"""

import asyncio
import logging
import os

from authenticated_mcp_client import AgentCoreAuthenticatedMcpClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Example usage of AgentCore authenticated MCP client."""

    # Your AgentCore Runtime ARNs
    runtime_arn = "arn:aws:bedrock-agentcore:eu-central-1:060337561279:runtime/steditt_agentcore_stack_mcp_server-1qQZnm2z5G"

    # Cognito configuration (you'll need to provide these)
    user_pool_id = os.getenv("COGNITO_USER_POOL_ID", "eu-central-1_XXXXXXXXX")
    client_id = os.getenv("COGNITO_CLIENT_ID", "your_client_id")
    username = os.getenv("COGNITO_USERNAME", "testuser")
    password = os.getenv("COGNITO_PASSWORD", "your_password")

    # Create the authenticated MCP client
    client = AgentCoreAuthenticatedMcpClient(
        runtime_arn=runtime_arn,
        user_pool_id=user_pool_id,
        client_id=client_id,
        username=username,
        password=password,
        auto_refresh=True,
        timeout=120,
    )

    try:
        # Connect to the AgentCore Runtime MCP server
        logger.info("Connecting to AgentCore Runtime MCP server...")
        await client.connect()

        # Get connection info
        connection_info = client.get_connection_info()
        logger.info(f"Connection info: {connection_info}")

        # Perform health check
        health = await client.health_check()
        logger.info(f"Health check: {health}")

        # List available tools
        logger.info("Listing available tools...")
        tools_result = await client.list_tools()

        logger.info(f"Found {len(tools_result.tools)} tools:")
        for tool in tools_result.tools:
            logger.info(f"  - {tool.name}: {tool.description}")

        # Example: Call a tool (assuming there's a weather tool)
        if tools_result.tools:
            first_tool = tools_result.tools[0]
            logger.info(f"Calling tool: {first_tool.name}")

            # You'll need to provide appropriate arguments based on the tool's schema
            try:
                result = await client.call_tool(first_tool.name, {})
                logger.info(f"Tool result: {result}")
            except Exception as e:
                logger.warning(f"Tool call failed (may need arguments): {e}")

    except Exception as e:
        logger.error(f"Error: {e}")

    finally:
        # Disconnect
        await client.disconnect()
        logger.info("Disconnected from AgentCore Runtime MCP server")


async def context_manager_example():
    """Example using the client as an async context manager."""

    runtime_arn = "arn:aws:bedrock-agentcore:eu-central-1:060337561279:runtime/steditt_agentcore_stack_mcp_server-1qQZnm2z5G"

    # Using environment variables for configuration
    user_pool_id = os.getenv("COGNITO_USER_POOL_ID")
    client_id = os.getenv("COGNITO_CLIENT_ID")
    username = os.getenv("COGNITO_USERNAME")
    password = os.getenv("COGNITO_PASSWORD")

    if not all([user_pool_id, client_id, username, password]):
        logger.error(
            "Please set COGNITO_USER_POOL_ID, COGNITO_CLIENT_ID, COGNITO_USERNAME, and COGNITO_PASSWORD environment variables"
        )
        return

    # Use as async context manager for automatic cleanup
    async with AgentCoreAuthenticatedMcpClient(
        runtime_arn=runtime_arn,
        user_pool_id=user_pool_id,
        client_id=client_id,
        username=username,
        password=password,
    ) as client:

        # List tools
        tools_result = await client.list_tools()
        logger.info(f"Available tools: {[tool.name for tool in tools_result.tools]}")


if __name__ == "__main__":
    # Run the basic example
    asyncio.run(main())

    # Uncomment to run the context manager example
    # asyncio.run(context_manager_example())
