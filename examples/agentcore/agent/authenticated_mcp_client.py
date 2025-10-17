"""
Authenticated MCP client for AgentCore Runtime with OAuth Bearer token authentication.

This module provides an MCP client that connects to AgentCore Runtime MCP servers
using HTTP transport and OAuth Bearer token authentication.
"""

import asyncio
import logging
from typing import Any
from urllib.parse import quote

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from .auth_helper import AuthenticationError, McpAuthHelper
from .config_loader import ConfigLoader
from .token_manager import SessionManager, TokenManager

logger = logging.getLogger(__name__)


class AgentCoreAuthenticatedMcpClient:
    """
    MCP client for AgentCore Runtime with OAuth Bearer token authentication.

    Connects to AgentCore Runtime MCP servers using HTTP transport and
    automatic Bearer token injection for authenticated requests.
    """

    def __init__(
        self,
        runtime_arn: str,
        user_pool_id: str,
        client_id: str,
        username: str | None = None,
        password: str | None = None,
        auth_helper: McpAuthHelper | None = None,
        config_loader: ConfigLoader | None = None,
        auto_refresh: bool = True,
        timeout: int = 120,
    ) -> None:
        """
        Initialize AgentCore authenticated MCP client.

        Args:
            runtime_arn: AgentCore Runtime ARN for the MCP server
            user_pool_id: Cognito User Pool ID
            client_id: Cognito App Client ID
            username: Username for authentication (loaded from config if None)
            password: Password for authentication (loaded from config if None)
            auth_helper: Authentication helper instance (created if None)
            config_loader: Configuration loader instance (created if None)
            auto_refresh: Whether to automatically refresh tokens
            timeout: Request timeout in seconds
        """
        self.runtime_arn = runtime_arn
        self.user_pool_id = user_pool_id
        self.client_id = client_id
        self.username = username
        self.password = password
        self.timeout = timeout

        # Extract region from ARN: arn:aws:bedrock-agentcore:region:account:runtime/runtime-id
        self.region = runtime_arn.split(":")[3]

        # Initialize authentication components
        self.auth_helper = auth_helper or McpAuthHelper(region=self.region)
        self.config_loader = config_loader or ConfigLoader(region=self.region)
        self.auto_refresh = auto_refresh

        self._token_manager: TokenManager | None = None
        self._session_manager: SessionManager | None = None
        self._session: ClientSession | None = None
        self._connected = False

        # Construct MCP endpoint URL
        self.mcp_url = self._construct_mcp_url()

    def _construct_mcp_url(self) -> str:
        """
        Construct the MCP endpoint URL from the runtime ARN.

        Returns:
            MCP endpoint URL for AgentCore Runtime
        """
        # URL encode the ARN
        encoded_arn = quote(self.runtime_arn, safe="")

        # Construct the MCP endpoint URL
        return f"https://bedrock-agentcore.{self.region}.amazonaws.com/runtimes/{encoded_arn}/invocations?qualifier=DEFAULT"

    async def connect(self) -> None:
        """
        Establish authenticated connection to AgentCore Runtime MCP server.

        Raises:
            AuthenticationError: When authentication fails
            Exception: When connection fails
        """
        try:
            logger.info(
                f"Connecting to AgentCore Runtime MCP server: {self.runtime_arn}"
            )

            # Set up authentication
            await self._setup_authentication()

            # Get Bearer token
            bearer_token = self._token_manager.get_valid_token()

            # Create authenticated headers
            headers = {
                "authorization": f"Bearer {bearer_token}",
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            }

            logger.info(f"Connecting to MCP URL: {self.mcp_url}")

            # Connect using streamable HTTP client
            self._transport_context = streamablehttp_client(
                self.mcp_url, headers, timeout=self.timeout, terminate_on_close=False
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
            logger.error(f"Failed to connect to AgentCore Runtime MCP server: {e}")
            raise

    async def disconnect(self) -> None:
        """
        Disconnect from AgentCore Runtime MCP server and cleanup resources.
        """
        logger.info("Disconnecting from AgentCore Runtime MCP server")

        self._connected = False

        # Close MCP session
        if hasattr(self, "_session_context") and self._session_context:
            try:
                await self._session_context.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing MCP session: {e}")
            finally:
                self._session = None
                self._session_context = None

        # Close transport
        if hasattr(self, "_transport_context") and self._transport_context:
            try:
                await self._transport_context.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing transport: {e}")
            finally:
                self._transport_context = None

        # Cleanup authentication session
        if self._session_manager:
            try:
                self._session_manager.__exit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error cleaning up session manager: {e}")
            finally:
                self._session_manager = None
                self._token_manager = None

    async def list_tools(self):
        """
        List available tools from the AgentCore Runtime MCP server.

        Returns:
            MCP tools list result

        Raises:
            Exception: When not connected or request fails
        """
        if not self._connected or not self._session:
            raise Exception("Not connected to MCP server. Call connect() first.")

        try:
            # Refresh token if needed
            await self._refresh_token_if_needed()

            logger.info("Listing tools from AgentCore Runtime MCP server")
            result = await self._session.list_tools()
            logger.info(f"Retrieved {len(result.tools)} tools from MCP server")
            return result

        except Exception as e:
            logger.error(f"Failed to list tools: {e}")
            raise

    async def call_tool(self, name: str, arguments: dict[str, Any]):
        """
        Call a specific tool on the AgentCore Runtime MCP server.

        Args:
            name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Tool execution result

        Raises:
            Exception: When not connected or tool call fails
        """
        if not self._connected or not self._session:
            raise Exception("Not connected to MCP server. Call connect() first.")

        try:
            # Refresh token if needed
            await self._refresh_token_if_needed()

            logger.info(f"Calling tool '{name}' with arguments: {arguments}")
            result = await self._session.call_tool(name, arguments=arguments)
            logger.info(f"Tool '{name}' executed successfully")
            return result

        except Exception as e:
            logger.error(f"Failed to call tool '{name}': {e}")
            raise

    async def _setup_authentication(self) -> None:
        """
        Set up OAuth authentication for AgentCore Runtime MCP server.

        Raises:
            AuthenticationError: When authentication setup fails
        """
        try:
            logger.info("Setting up OAuth authentication for AgentCore Runtime")

            # Load credentials from config if not provided during initialization
            if self.username is None or self.password is None:
                credentials = self.config_loader.get_credentials()
                username = self.username or credentials.username
                password = self.password or credentials.password
            else:
                username = self.username
                password = self.password

            # Create token manager with provided Cognito configuration
            self._token_manager = TokenManager(
                auth_helper=self.auth_helper,
                user_pool_id=self.user_pool_id,
                client_id=self.client_id,
                username=username,
                password=password,
                auto_refresh=self.auto_refresh,
            )

            # Create session manager for cleanup
            self._session_manager = SessionManager(self._token_manager)

            # Get initial Bearer token to validate authentication
            _ = self._token_manager.get_valid_token()

            logger.info("OAuth authentication setup completed successfully")

        except Exception as e:
            logger.error(f"Authentication setup failed: {e}")
            raise AuthenticationError(
                f"Failed to setup OAuth authentication: {e}"
            ) from e

    async def _refresh_token_if_needed(self) -> None:
        """
        Refresh Bearer token if needed and update connection headers.

        Raises:
            AuthenticationError: When token refresh fails
        """
        if not self._token_manager:
            return

        try:
            # Get current valid token (will refresh if needed)
            _ = self._token_manager.get_valid_token()

            # Note: With streamable HTTP client, we can't easily update headers mid-connection
            # If token refresh is needed, we would need to reconnect
            # For now, we rely on the token manager's auto-refresh functionality

        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            raise AuthenticationError(
                f"Failed to refresh authentication token: {e}"
            ) from e

    def is_connected(self) -> bool:
        """
        Check if client is connected to AgentCore Runtime MCP server.

        Returns:
            True if connected, False otherwise
        """
        return self._connected and self._session is not None

    def get_connection_info(self) -> dict[str, Any]:
        """
        Get information about current connection.

        Returns:
            Dictionary containing connection information
        """
        token_info = None
        if self._token_manager:
            token_info = self._token_manager.get_token_info()

        return {
            "runtime_arn": self.runtime_arn,
            "mcp_url": self.mcp_url,
            "region": self.region,
            "connected": self.is_connected(),
            "auto_refresh": self.auto_refresh,
            "token_info": token_info,
        }

    async def health_check(self) -> dict[str, Any]:
        """
        Perform health check on AgentCore Runtime MCP connection.

        Returns:
            Dictionary containing health status information
        """
        health_info = {
            "connection_status": "disconnected",
            "authentication_status": "unknown",
            "tools_available": 0,
            "last_error": None,
        }

        try:
            # Check connection status
            if not self.is_connected():
                health_info["connection_status"] = "disconnected"
                return health_info

            health_info["connection_status"] = "connected"

            # Check authentication by trying to list tools
            tools_result = await self.list_tools()
            health_info["authentication_status"] = "valid"
            health_info["tools_available"] = len(tools_result.tools)

        except AuthenticationError as e:
            health_info["authentication_status"] = "invalid"
            health_info["last_error"] = f"Authentication error: {str(e)}"

        except Exception as e:
            health_info["last_error"] = f"Unexpected error: {str(e)}"

        return health_info

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
