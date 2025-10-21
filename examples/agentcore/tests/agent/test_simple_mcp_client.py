"""
Integration tests for SimpleMcpClient with AgentCore Runtime.

This module tests the simplified MCP client with OAuth Bearer token authentication
using real AgentCore Runtime MCP servers and deployed infrastructure.
"""

# nosec B101

import os

import pytest
from simple_mcp_client import SimpleMcpClient


class TestSimpleMcpClientIntegration:
    """Integration tests with real AgentCore Runtime MCP server."""

    @pytest.fixture(scope="class")
    def simple_client(self):
        """Create simple MCP client with real configuration."""
        return SimpleMcpClient(runtime_arn=os.environ["MCP_SERVER_RUNTIME_ARN"])

    def test_client_initialization_with_real_arn(self, simple_client):
        """Test client initialization with real AgentCore Runtime ARN."""
        mcp_server_runtime_arn = os.environ["MCP_SERVER_RUNTIME_ARN"]
        assert simple_client.runtime_arn == mcp_server_runtime_arn  # nosec B101
        assert simple_client.region in mcp_server_runtime_arn  # nosec B101
        assert not simple_client.is_connected()  # nosec B101

        # Verify MCP URL construction
        assert "bedrock-agentcore" in simple_client.mcp_url  # nosec B101
        assert simple_client.region in simple_client.mcp_url  # nosec B101
        assert "invocations?qualifier=DEFAULT" in simple_client.mcp_url  # nosec B101

    @pytest.mark.asyncio
    async def test_full_connection_to_agentcore_mcp_server(self, simple_client):
        """Test full connection to deployed AgentCore Runtime MCP server."""
        try:
            # Connect to the MCP server
            await simple_client.connect()

            # Verify connection
            assert simple_client.is_connected()  # nosec B101

        except Exception as e:
            # Log the error details for debugging
            print(f"\nConnection failed with error: {e}")
            print(f"Error type: {type(e)}")
            print(f"MCP URL: {simple_client.mcp_url}")

            # Don't fail the test immediately - let's see what the error is
            pytest.fail(f"Failed to connect to AgentCore Runtime MCP server: {e}")

        finally:
            # Always disconnect
            try:
                await simple_client.disconnect()
            except Exception as disconnect_error:
                print(f"Disconnect error: {disconnect_error}")

    @pytest.mark.asyncio
    async def test_list_tools_from_deployed_mcp_server(self, simple_client):
        """Test listing tools from deployed AgentCore Runtime MCP server."""
        try:
            # Connect and list tools
            await simple_client.connect()

            tools_result = await simple_client.list_tools()

            # Verify we got tools
            assert hasattr(
                tools_result, "tools"
            ), "Expected tools in response"  # nosec B101
            assert (
                len(tools_result.tools) > 0
            ), "Expected at least one tool"  # nosec B101

            # Log available tools for debugging
            print(f"\nFound {len(tools_result.tools)} tools:")
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {tool.description}")

            # Verify tool structure
            first_tool = tools_result.tools[0]
            assert hasattr(first_tool, "name"), "Tool should have name"  # nosec B101
            assert hasattr(
                first_tool, "description"
            ), "Tool should have description"  # nosec B101
            assert hasattr(
                first_tool, "inputSchema"
            ), "Tool should have input schema"  # nosec B101

        except Exception as e:
            pytest.fail(f"Failed to list tools from MCP server: {e}")

        finally:
            await simple_client.disconnect()

    @pytest.mark.asyncio
    async def test_call_tool_on_deployed_mcp_server(self, simple_client):
        """Test calling weather tools on deployed AgentCore Runtime MCP server."""
        try:
            # Connect and get tools
            await simple_client.connect()
            tools_result = await simple_client.list_tools()

            # Test must fail if no tools are available
            assert (
                tools_result.tools
            ), "MCP server must have tools available"  # nosec B101
            assert (
                len(tools_result.tools) > 0
            ), "Expected at least one tool from MCP server"  # nosec B101

            # Find and test get_weather tool
            weather_tool = None
            forecast_tool = None

            for tool in tools_result.tools:
                if tool.name == "get_weather":
                    weather_tool = tool
                elif tool.name == "get_forecast":
                    forecast_tool = tool

            # Test get_weather tool
            if weather_tool:
                print("\nTesting get_weather tool:")
                print(f"Description: {weather_tool.description}")

                result = await simple_client.call_tool(
                    "get_weather", {"city": "Amsterdam"}
                )

                assert (
                    result is not None
                ), "get_weather tool should return a result"  # nosec B101
                print(f"Weather result: {result}")

                # Verify the result contains expected weather information
                result_str = str(result).lower()
                assert (
                    "amsterdam" in result_str
                ), "Result should mention Amsterdam"  # nosec B101
                assert any(
                    indicator in result_str
                    for indicator in ["weather", "temperature", "sunny", "°c"]
                ), "Result should contain weather information"  # nosec B101

            # Test get_forecast tool
            if forecast_tool:
                print("\nTesting get_forecast tool:")
                print(f"Description: {forecast_tool.description}")

                result = await simple_client.call_tool(
                    "get_forecast", {"city": "London", "days": 5}
                )

                assert (
                    result is not None
                ), "get_forecast tool should return a result"  # nosec B101
                print(f"Forecast result: {result}")

                # Verify the result contains expected forecast information
                result_str = str(result).lower()
                assert (
                    "london" in result_str
                ), "Result should mention London"  # nosec B101
                assert any(
                    indicator in result_str
                    for indicator in ["forecast", "days", "weather", "°c"]
                ), "Result should contain forecast information"  # nosec B101

            # Ensure we tested at least one tool
            if not weather_tool and not forecast_tool:
                pytest.fail(
                    "Expected to find get_weather or get_forecast tools in MCP server"
                )

            print(
                f"\nSuccessfully tested {len([t for t in [weather_tool, forecast_tool] if t])} weather tools"
            )

        except Exception as e:
            pytest.fail(f"Failed to call tools on MCP server: {e}")

        finally:
            await simple_client.disconnect()

    @pytest.mark.asyncio
    async def test_context_manager_with_real_server(self, simple_client):
        """Test async context manager with real AgentCore Runtime MCP server."""
        try:
            # Use as context manager
            async with simple_client as client:
                # Verify connection
                assert client.is_connected()  # nosec B101

                # List tools to verify functionality
                tools_result = await client.list_tools()
                assert len(tools_result.tools) > 0  # nosec B101

            # Verify disconnection after context exit
            assert not simple_client.is_connected()  # nosec B101

        except Exception as e:
            pytest.fail(f"Context manager test failed: {e}")

    @pytest.mark.asyncio
    async def test_automatic_configuration_loading(self, simple_client):
        """Test that configuration is loaded automatically from CDK stack."""
        try:
            # The client should be able to load configuration automatically from environment variables
            await simple_client.connect()

            # If we get here, configuration was loaded successfully
            assert simple_client.auth is not None  # nosec B101

            # Verify we can list tools (proves authentication worked)
            tools_result = await simple_client.list_tools()
            assert len(tools_result.tools) > 0  # nosec B101

        except Exception as e:
            pytest.fail(f"Automatic configuration loading failed: {e}")

        finally:
            await simple_client.disconnect()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
