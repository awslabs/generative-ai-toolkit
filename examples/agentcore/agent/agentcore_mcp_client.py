"""
Copyright 2024 Amazon.com, Inc. and its affiliates. All Rights Reserved.

Licensed under the Amazon Software License (the "License").
You may not use this file except in compliance with the License.
A copy of the License is located at

  http://aws.amazon.com/asl/

or in the "license" file accompanying this file. This file is distributed
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
express or implied. See the License for the specific language governing
permissions and limitations under the License.
"""

"""
AgentCore MCP Client

This module provides an MCP client specifically designed for AgentCore Runtime
environments. It communicates with MCP servers running in other AgentCore
Runtimes using the AWS Bedrock AgentCore InvokeAgentRuntime API.
"""

import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

import boto3
from botocore.exceptions import ClientError, BotoCoreError


logger = logging.getLogger(__name__)


class AgentCoreMCPClient:
    """
    MCP client for AgentCore Runtime environments.

    This client communicates with MCP servers running in AgentCore Runtimes
    using the AWS Bedrock AgentCore InvokeAgentRuntime API instead of direct
    HTTP connections.
    """

    def __init__(
        self,
        mcp_server_endpoint_arn: str,
        region: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        """
        Initialize the AgentCore MCP client.

        Args:
            mcp_server_endpoint_arn: ARN of the MCP server runtime endpoint
            region: AWS region (defaults to AWS_REGION env var)
            session_id: Session ID for MCP communication (auto-generated if not provided)
        """
        self.mcp_server_endpoint_arn = mcp_server_endpoint_arn
        self.region = region or "us-east-1"
        import uuid

        self.session_id = session_id or str(uuid.uuid4())

        # Initialize AWS client
        self.bedrock_client = boto3.client("bedrock-agentcore", region_name=self.region)

        # Cache for available tools
        self._tools_cache: Optional[List[Dict[str, Any]]] = None
        self._connected = False

        logger.info(
            f"AgentCoreMCPClient initialized for endpoint: {mcp_server_endpoint_arn}"
        )

    async def connect(self) -> None:
        """
        Connect to the MCP server and initialize the session.

        This performs the MCP initialization handshake with the server.
        """
        try:
            logger.info("Connecting to MCP server...")

            # Send MCP initialize request
            initialize_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "clientInfo": {"name": "agentcore-mcp-client", "version": "1.0.0"},
                },
            }

            response = await self._invoke_mcp_server(initialize_request)

            if "error" in response:
                raise RuntimeError(f"MCP initialization failed: {response['error']}")

            # Send initialized notification
            initialized_notification = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            }

            await self._invoke_mcp_server(initialized_notification)

            self._connected = True
            logger.info("Successfully connected to MCP server")

        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {str(e)}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        try:
            if self._connected:
                logger.info("Disconnecting from MCP server...")
                self._connected = False
                self._tools_cache = None
                logger.info("Disconnected from MCP server")
        except Exception as e:
            logger.error(f"Error during disconnect: {str(e)}")

    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        List available tools from the MCP server.

        Returns:
            List of tool definitions with name, description, and input schema
        """
        if not self._connected:
            await self.connect()

        if self._tools_cache is not None:
            return self._tools_cache

        try:
            logger.debug("Listing tools from MCP server...")

            # Send tools/list request
            list_tools_request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}

            response = await self._invoke_mcp_server(list_tools_request)

            if "error" in response:
                raise RuntimeError(f"Failed to list tools: {response['error']}")

            tools = response.get("result", {}).get("tools", [])

            # Cache the tools
            self._tools_cache = tools

            logger.info(f"Retrieved {len(tools)} tools from MCP server")
            return tools

        except Exception as e:
            logger.error(f"Failed to list tools: {str(e)}")
            raise

    async def call_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Tool execution result
        """
        if not self._connected:
            await self.connect()

        try:
            logger.debug(f"Calling tool '{tool_name}' with arguments: {arguments}")

            # Send tools/call request
            call_tool_request = {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            }

            response = await self._invoke_mcp_server(call_tool_request)

            if "error" in response:
                error_msg = response["error"].get("message", "Unknown error")
                logger.error(f"Tool call failed: {error_msg}")
                raise RuntimeError(f"Tool call failed: {error_msg}")

            result = response.get("result", {})
            logger.debug(f"Tool call successful: {result}")

            return result

        except Exception as e:
            logger.error(f"Failed to call tool '{tool_name}': {str(e)}")
            raise

    async def _invoke_mcp_server(self, mcp_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke the MCP server using AgentCore Runtime API.

        Args:
            mcp_request: MCP JSON-RPC request

        Returns:
            MCP JSON-RPC response
        """
        try:
            # Convert MCP request to JSON
            request_payload = json.dumps(mcp_request)

            logger.debug(f"Invoking MCP server with payload: {request_payload}")

            # Call the MCP server runtime using AWS SDK
            response = await asyncio.to_thread(
                self.bedrock_client.invoke_agent_runtime,
                agentRuntimeArn=self.mcp_server_endpoint_arn,
                runtimeSessionId=self.session_id,
                payload=request_payload.encode(),
            )

            # Parse the response
            response_body = response.get("output", {})

            # Handle streaming response if present
            if "eventStream" in response:
                # Collect all chunks from the event stream
                response_text = ""
                for event in response["eventStream"]:
                    if "chunk" in event:
                        chunk_data = event["chunk"].get("bytes", b"")
                        response_text += chunk_data.decode("utf-8")
            else:
                # Handle direct response
                response_text = response_body.get("text", "{}")

            # Parse JSON response
            try:
                mcp_response = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse MCP response as JSON: {response_text}")
                raise RuntimeError(f"Invalid JSON response from MCP server: {str(e)}")

            logger.debug(f"Received MCP response: {mcp_response}")
            return mcp_response

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            error_message = e.response["Error"]["Message"]
            logger.error(f"AWS API error ({error_code}): {error_message}")
            raise RuntimeError(f"AWS API error: {error_message}")

        except BotoCoreError as e:
            logger.error(f"AWS SDK error: {str(e)}")
            raise RuntimeError(f"AWS SDK error: {str(e)}")

        except Exception as e:
            logger.error(f"Unexpected error invoking MCP server: {str(e)}")
            raise

    def is_connected(self) -> bool:
        """Check if the client is connected to the MCP server."""
        return self._connected

    def get_session_id(self) -> str:
        """Get the current session ID."""
        return self.session_id

    def get_endpoint_arn(self) -> str:
        """Get the MCP server endpoint ARN."""
        return self.mcp_server_endpoint_arn


class AgentCoreMCPToolWrapper:
    """
    Wrapper for MCP tools that integrates with BedrockConverseAgent.

    This class wraps MCP tool calls to provide a synchronous interface
    that can be registered with the agent's tool registry.
    """

    def __init__(
        self,
        mcp_client: AgentCoreMCPClient,
        tool_name: str,
        tool_description: str,
        tool_schema: Dict[str, Any],
    ):
        """
        Initialize the tool wrapper.

        Args:
            mcp_client: The AgentCore MCP client
            tool_name: Name of the tool
            tool_description: Description of the tool
            tool_schema: JSON schema for tool input
        """
        self.mcp_client = mcp_client
        self.tool_name = tool_name
        self.tool_description = tool_description
        self.tool_schema = tool_schema

    def __call__(self, **kwargs) -> Any:
        """
        Call the MCP tool with the provided arguments.

        Args:
            **kwargs: Tool arguments

        Returns:
            Tool execution result
        """
        try:
            # Run the async tool call in a new event loop
            result = asyncio.run(self.mcp_client.call_tool(self.tool_name, kwargs))

            # Extract text content from MCP response
            if isinstance(result, dict) and "content" in result:
                content = result["content"]
                if isinstance(content, list) and len(content) > 0:
                    first_content = content[0]
                    if isinstance(first_content, dict) and "text" in first_content:
                        return first_content["text"]

            # Return the raw result if we can't extract text
            return result

        except Exception as e:
            logger.error(f"MCP tool call failed for {self.tool_name}: {str(e)}")
            return {"error": f"Tool call failed: {str(e)}"}

    @property
    def __name__(self):
        """Return the tool name for agent registration."""
        return self.tool_name

    @property
    def __doc__(self):
        """Return the tool description for agent registration."""
        return self.tool_description
