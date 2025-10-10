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
Standalone MCP Server for Weather Tools

This module implements an MCP (Model Context Protocol) server that provides
weather-related tools using Pydantic models for validation. It demonstrates:

1. MCP server implementation using the mcp library
2. Integration with Pydantic-based tools
3. HTTP endpoint for MCP communication
4. Proper error handling and logging

The server provides weather forecast and alert tools that can be consumed
by MCP clients, including the Generative AI Toolkit.
"""

import logging
import json
from typing import List

from mcp.server import Server
from mcp.types import Tool, TextContent
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import StreamingResponse, JSONResponse
from starlette.requests import Request
import uvicorn

from weather_tools import WeatherForecastTool, WeatherAlertTool


# JSON-RPC 2.0 Error Codes
class JSONRPCErrorCode:
    """Standard JSON-RPC 2.0 error codes."""

    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize tool instances
_weather_forecast_tool = WeatherForecastTool()
_weather_alert_tool = WeatherAlertTool()

# Tool registry mapping
TOOL_MAP = {
    "get_weather_forecast": _weather_forecast_tool,
    "get_weather_alerts": _weather_alert_tool,
}


def get_available_tool_names():
    """Get list of available tool names dynamically."""
    return list(TOOL_MAP.keys())


# Create MCP server
server = Server("weather-tools")


@server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """Return available weather tools with their Pydantic schemas."""
    tools = []

    # Dynamically build tools list from tool registry
    for tool_instance in TOOL_MAP.values():
        spec = tool_instance.tool_spec
        tools.append(
            Tool(
                name=spec["name"],
                description=spec["description"],
                inputSchema=spec["inputSchema"]["json"],
            )
        )

    logger.info(f"Listed {len(tools)} weather tools")
    return tools


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> List[TextContent]:
    """Handle tool calls for weather tools."""
    if name not in TOOL_MAP:
        error_msg = (
            f"Unknown tool: {name}. Available tools: {get_available_tool_names()}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        logger.info(f"Calling tool '{name}' with arguments: {arguments}")
        result = TOOL_MAP[name].invoke(**arguments)
        logger.info(f"Tool '{name}' completed successfully")

        # Convert result to JSON string for MCP response
        result_json = json.dumps(result, indent=2)
        return [TextContent(type="text", text=result_json)]

    except Exception as e:
        error_msg = f"Error calling tool '{name}': {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)


async def handle_mcp(request: Request):
    """Handle MCP requests via AgentCore Runtime protocol."""
    try:
        if request.method == "POST":
            # Handle AgentCore runtime payload - could be JSON or binary
            content_type = request.headers.get("content-type", "")

            if "application/json" in content_type:
                # Direct JSON request (for testing)
                body = await request.json()
            else:
                # AgentCore runtime sends payload as text/binary
                raw_body = await request.body()
                try:
                    # Try to decode as JSON string
                    body_text = raw_body.decode("utf-8")
                    body = json.loads(body_text)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    # If that fails, try to parse as JSON directly
                    try:
                        body = json.loads(raw_body)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse request body: {raw_body}")
                        return JSONResponse(
                            {"error": "Invalid JSON in request body"}, status_code=400
                        )

            method = body.get("method")
            request_id = body.get("id")

            logger.info(f"Received MCP request: {method}")

            # Handle list_tools
            if method == "tools/list":
                tools = await handle_list_tools()
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": [
                            {
                                "name": tool.name,
                                "description": tool.description,
                                "inputSchema": tool.inputSchema,
                            }
                            for tool in tools
                        ]
                    },
                }
                return JSONResponse(response)

            # Handle call_tool
            elif method == "tools/call":
                params = body.get("params", {})
                name = params.get("name")
                arguments = params.get("arguments", {})

                try:
                    result = await handle_call_tool(name, arguments)
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": getattr(content, "type", "text"),
                                    "text": getattr(content, "text", str(content)),
                                }
                                for content in result
                            ]
                        },
                    }
                except Exception as e:
                    logger.error(f"Tool call error: {str(e)}")
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": JSONRPCErrorCode.INTERNAL_ERROR,
                            "message": str(e),
                        },
                    }

                return JSONResponse(response)

            # Handle initialize
            elif method == "initialize":
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": {"name": "weather-tools", "version": "1.0.0"},
                    },
                }
                return JSONResponse(response)

            # Handle ping/health check
            elif method == "ping":
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "status": "ok",
                        "message": "Weather MCP server is running",
                    },
                }
                return JSONResponse(response)

            else:
                logger.warning(f"Unsupported method: {method}")
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": JSONRPCErrorCode.METHOD_NOT_FOUND,
                        "message": f"Method not found: {method}",
                    },
                }
                return JSONResponse(response)

        elif request.method == "GET":
            # Health check endpoint
            return JSONResponse(
                {
                    "status": "ok",
                    "server": "weather-tools",
                    "version": "1.0.0",
                    "tools": get_available_tool_names(),
                }
            )

        else:
            return JSONResponse({"error": "Method not allowed"}, status_code=405)

    except Exception as e:
        logger.error(f"Error handling MCP request: {str(e)}")
        return JSONResponse(
            {"error": f"Internal server error: {str(e)}"}, status_code=500
        )


async def handle_health(request: Request):
    """Health check endpoint."""
    import time

    return JSONResponse(
        {
            "status": "healthy",
            "server": "weather-tools-mcp-server",
            "version": "1.0.0",
            "timestamp": str(int(time.time())),
            "tools_available": get_available_tool_names(),
        }
    )


# Create Starlette application
app = Starlette(
    routes=[
        Route("/mcp", handle_mcp, methods=["GET", "POST"]),
        Route("/health", handle_health, methods=["GET"]),
        Route("/", handle_health, methods=["GET"]),  # Root endpoint for health check
    ]
)


if __name__ == "__main__":
    import argparse
    import os

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Weather Tools MCP Server")
    parser.add_argument(
        "--host", default=os.getenv("HOST", "0.0.0.0"), help="Host to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8000")),
        help="Port to bind to",
    )
    args = parser.parse_args()

    logger.info("Starting Weather Tools MCP Server")
    logger.info(f"Available tools: {', '.join(get_available_tool_names())}")
    logger.info("Endpoints:")
    logger.info("  - POST /mcp - MCP protocol endpoint")
    logger.info("  - GET /health - Health check")
    logger.info("  - GET / - Root health check")
    logger.info(f"Server will start on {args.host}:{args.port}")

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        raise
