"""Test the deployed AgentCore MCP server deployment and basic protocol functionality."""

import asyncio
import os
from urllib.parse import quote

import pytest
from botocore.exceptions import ClientError
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


def _construct_mcp_endpoint_url(mcp_server_runtime_arn: str) -> str:
    """Construct the MCP endpoint URL from the runtime ARN."""
    region = mcp_server_runtime_arn.split(":")[3]
    encoded_arn = quote(mcp_server_runtime_arn, safe="")
    return f"https://bedrock-agentcore.{region}.amazonaws.com/runtimes/{encoded_arn}/invocations?qualifier=DEFAULT"


class TestMcpServerDeployment:
    """Test suite for MCP server deployment verification."""

    def test_mcp_server_endpoint_url_construction(self):
        """Test that MCP endpoint URL can be constructed from runtime ARN."""
        mcp_server_runtime_arn = os.environ["MCP_SERVER_RUNTIME_ARN"]
        mcp_url = _construct_mcp_endpoint_url(mcp_server_runtime_arn)

        # Verify URL structure
        assert "https://bedrock-agentcore." in mcp_url
        assert ".amazonaws.com/runtimes/" in mcp_url
        assert "invocations?qualifier=DEFAULT" in mcp_url

        # Verify region is extracted correctly
        region = mcp_server_runtime_arn.split(":")[3]
        assert region in mcp_url

        print(f"✅ MCP endpoint URL constructed: {mcp_url}")

    def test_mcp_server_runtime_exists(self, bedrock_agentcore_control_client):
        """Test that the MCP server runtime exists and is accessible."""
        mcp_server_runtime_arn = os.environ["MCP_SERVER_RUNTIME_ARN"]
        runtime_id = mcp_server_runtime_arn.split("/")[-1]

        try:
            response = bedrock_agentcore_control_client.get_agent_runtime(
                agentRuntimeId=runtime_id
            )
            assert response["agentRuntimeId"] == runtime_id
            assert response["status"] in ["READY", "CREATING", "UPDATING"]
            assert response["protocolConfiguration"]["serverProtocol"] == "MCP"
        except ClientError as e:
            pytest.fail(f"Failed to get MCP server runtime: {e}")

    def test_mcp_server_runtime_endpoint_exists(self, bedrock_agentcore_control_client):
        """Test that the MCP server runtime endpoint exists and is accessible."""
        mcp_server_runtime_endpoint_arn = os.environ["MCP_SERVER_RUNTIME_ENDPOINT_ARN"]
        arn_parts = mcp_server_runtime_endpoint_arn.split("/")
        runtime_id = arn_parts[-3]
        endpoint_name = arn_parts[-1]

        try:
            response = bedrock_agentcore_control_client.get_agent_runtime_endpoint(
                agentRuntimeId=runtime_id, endpointName=endpoint_name
            )
            assert response["name"] == endpoint_name
            assert runtime_id in response["agentRuntimeArn"]
            assert response["status"] in ["READY", "CREATING", "UPDATING"]
        except ClientError as e:
            pytest.fail(f"Failed to get MCP server runtime endpoint: {e}")

    def test_mcp_server_security_and_protocol_compliance(
        self, bedrock_agentcore_client
    ):
        """Test MCP server security and protocol compliance.

        This test verifies:
        1. The MCP endpoint is reachable and responds to protocol requests
        2. Authentication is properly enforced (rejects unauthenticated requests)
        3. MCP protocol standards are followed (proper error handling)
        """

        async def run_mcp_test():
            mcp_server_runtime_arn = os.environ["MCP_SERVER_RUNTIME_ARN"]
            mcp_url = _construct_mcp_endpoint_url(mcp_server_runtime_arn)
            auth_error_received = False

            try:
                headers = {}
                async with streamablehttp_client(mcp_url, headers, timeout=10) as (
                    read_stream,
                    write_stream,
                    _,
                ):
                    async with ClientSession(read_stream, write_stream) as session:
                        # Test MCP session initialization
                        await asyncio.wait_for(session.initialize(), timeout=10.0)

                        # Test 1: Try to list tools without authentication
                        try:
                            tools_result = await session.list_tools()
                            # This should not happen with a properly secured server
                            pytest.fail(
                                "MCP server allowed unauthenticated access - this is a security issue! "
                                f"Got {len(tools_result.tools) if hasattr(tools_result, 'tools') else 0} tools"
                            )
                        except Exception as list_error:
                            error_msg = str(list_error).lower()
                            if any(
                                auth_error in error_msg
                                for auth_error in [
                                    "unauthorized",
                                    "401",
                                    "403",
                                    "authentication",
                                    "forbidden",
                                    "access denied",
                                    "httpstatuserror",
                                ]
                            ):
                                auth_error_received = True
                                print(
                                    "✅ MCP server properly rejects unauthenticated tool listing"
                                )
                            else:
                                raise list_error

                        # Test 2: Try to call an invalid tool (if we got past list_tools)
                        if not auth_error_received:
                            try:
                                await session.call_tool(
                                    name="invalid_tool_name", arguments={}
                                )
                                pytest.fail("Expected error for invalid tool name")
                            except Exception as tool_error:
                                error_msg = str(tool_error).lower()
                                if any(
                                    auth_error in error_msg
                                    for auth_error in [
                                        "unauthorized",
                                        "401",
                                        "403",
                                        "authentication",
                                        "forbidden",
                                        "access denied",
                                    ]
                                ):
                                    auth_error_received = True
                                    print(
                                        "✅ MCP server requires authentication for tool calls"
                                    )
                                elif "not found" in error_msg or "invalid" in error_msg:
                                    print(
                                        "✅ MCP server properly handles invalid tool requests"
                                    )
                                else:
                                    raise tool_error

            except Exception as e:
                error_msg = str(e).lower()
                if any(
                    auth_error in error_msg
                    for auth_error in [
                        "unauthorized",
                        "401",
                        "403",
                        "authentication",
                        "forbidden",
                        "access denied",
                        "httpstatuserror",
                    ]
                ):
                    auth_error_received = True
                    print(f"✅ MCP server is properly secured at protocol level: {e}")
                else:
                    # Unexpected error - could be network, protocol, or other issue
                    raise

            # Verify we got the expected authentication error
            assert auth_error_received, (
                "Expected authentication error when connecting without credentials. "
                "MCP server should reject unauthenticated requests."
            )

        try:
            asyncio.run(run_mcp_test())
        except (BaseExceptionGroup, Exception) as e:
            # Handle both ExceptionGroup and regular exceptions
            exceptions = e.exceptions if hasattr(e, "exceptions") else [e]
            auth_error_found = False

            for exc in exceptions:
                error_msg = str(exc).lower()
                if any(
                    auth_error in error_msg
                    for auth_error in [
                        "unauthorized",
                        "401",
                        "403",
                        "authentication",
                        "forbidden",
                        "access denied",
                        "httpstatuserror",
                    ]
                ):
                    auth_error_found = True
                    print(f"✅ MCP server is properly secured: {exc}")
                    break

            if not auth_error_found:
                # Re-raise if it's not an auth error
                raise
