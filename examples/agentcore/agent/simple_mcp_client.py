"""
Simple MCP client for AgentCore Runtime with OAuth Bearer token authentication.

Provides basic MCP client functionality for short-running examples.
No complex session management or token refresh needed.
"""

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import boto3
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# Add the agent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from simple_auth import SimpleAuth

logger = logging.getLogger(__name__)

# Suppress expected warnings during cleanup
logging.getLogger("httpx").setLevel(logging.WARNING)  # HTTP 404s during cleanup
logging.getLogger("asyncio").setLevel(
    logging.CRITICAL
)  # Library-level async cleanup issues


class SimpleMcpClient:
    """Simple MCP client for AgentCore Runtime with OAuth authentication."""

    def __init__(self, runtime_arn: str):
        """
        Initialize simple MCP client.

        Args:
            runtime_arn: AgentCore Runtime ARN for the MCP server
        """
        self.runtime_arn = runtime_arn
        self.region = runtime_arn.split(":")[3]  # Extract region from ARN
        self.mcp_url = self._construct_mcp_url()

        # Initialize components
        self.auth = SimpleAuth(region=self.region)

        # Connection state
        self._session = None
        self._transport_context = None
        self._session_context = None
        self._connected = False

    def _construct_mcp_url(self) -> str:
        """Construct the MCP endpoint URL from the runtime ARN."""
        encoded_arn = quote(self.runtime_arn, safe="")
        return f"https://bedrock-agentcore.{self.region}.amazonaws.com/runtimes/{encoded_arn}/invocations?qualifier=DEFAULT"

    def _get_credentials_from_env(self):
        """Get OAuth credentials from environment variables and Secrets Manager."""

        @dataclass
        class UserCredentials:
            username: str
            password: str

        secret_name = os.environ["OAUTH_CREDENTIALS_SECRET_NAME"]
        client = boto3.client("secretsmanager", region_name=self.region)
        response = client.get_secret_value(SecretId=secret_name)
        credentials_data = json.loads(response["SecretString"])

        return UserCredentials(
            username=credentials_data["username"], password=credentials_data["password"]
        )

    async def connect(self) -> None:
        """
        Connect to AgentCore Runtime MCP server with authentication.

        Raises:
            Exception: When authentication or connection fails
        """
        try:

            logger.info(
                f"Connecting to AgentCore Runtime MCP server: {self.runtime_arn}"
            )

            # Get configuration from environment variables
            user_pool_id = os.environ["OAUTH_USER_POOL_ID"]
            client_id = os.environ["OAUTH_USER_POOL_CLIENT_ID"]
            credentials = self._get_credentials_from_env()

            # Get Bearer token
            bearer_token = self.auth.get_bearer_token(
                user_pool_id=user_pool_id,
                client_id=client_id,
                username=credentials.username,
                password=credentials.password,
            )

            # Create authenticated headers
            headers = {
                "Authorization": f"Bearer {bearer_token}",
                "Content-Type": "application/json",
            }

            # Connect using streamable HTTP client
            self._transport_context = streamablehttp_client(
                self.mcp_url, headers, timeout=30
            )
            read_stream, write_stream, _ = await self._transport_context.__aenter__()

            # Create MCP session
            self._session_context = ClientSession(read_stream, write_stream)
            self._session = await self._session_context.__aenter__()

            # Initialize the MCP session
            await asyncio.wait_for(self._session.initialize(), timeout=10.0)

            self._connected = True
            logger.info("Successfully connected to AgentCore Runtime MCP server")

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            await self._cleanup()
            raise

    async def disconnect(self) -> None:
        """Disconnect from AgentCore Runtime MCP server."""
        logger.info("Disconnecting from AgentCore Runtime MCP server")
        await self._cleanup()

    async def _cleanup(self) -> None:
        """Clean up connection resources."""
        self._connected = False

        # Close MCP session
        if self._session_context:
            try:
                await self._session_context.__aexit__(None, None, None)
            except Exception as e:
                # Only log unexpected errors (404s and cancel scope errors are expected during cleanup)
                error_msg = str(e).lower()
                if "404" not in error_msg and "cancel scope" not in error_msg:
                    logger.warning(f"Error closing MCP session: {e}")
            finally:
                self._session = None
                self._session_context = None

        # Close transport
        if self._transport_context:
            try:
                await self._transport_context.__aexit__(None, None, None)
            except Exception as e:
                # Only log unexpected errors (404s and cancel scope errors are expected during cleanup)
                error_msg = str(e).lower()
                if "404" not in error_msg and "cancel scope" not in error_msg:
                    logger.warning(f"Error closing transport: {e}")
            finally:
                self._transport_context = None

    async def list_tools(self):
        """
        List available tools from the MCP server.

        Returns:
            MCP tools list result

        Raises:
            Exception: When not connected or request fails
        """
        if not self._connected or not self._session:
            raise Exception("Not connected to MCP server. Call connect() first.")

        logger.info("Listing tools from MCP server")
        result = await self._session.list_tools()
        logger.info(f"Found {len(result.tools)} tools")
        return result

    async def call_tool(self, name: str, arguments: dict[str, Any]):
        """
        Call a tool on the MCP server.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool execution result

        Raises:
            Exception: When not connected or tool call fails
        """
        if not self._connected or not self._session:
            raise Exception("Not connected to MCP server. Call connect() first.")

        logger.info(f"Calling tool '{name}' with arguments: {arguments}")
        result = await self._session.call_tool(name, arguments=arguments)
        logger.info(f"Tool '{name}' executed successfully")
        return result

    def is_connected(self) -> bool:
        """Check if connected to MCP server."""
        return self._connected and self._session is not None

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
