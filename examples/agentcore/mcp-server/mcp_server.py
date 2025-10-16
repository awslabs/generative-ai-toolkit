#!/usr/bin/env python3
"""MCP Server for AgentCore Runtime."""

import logging

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("MCP Server starting")

# Create FastMCP server
mcp = FastMCP("Weather MCP Server")


@mcp.tool()
def get_weather(city: str) -> str:
    """Get current weather for a city.

    Args:
        city: Name of the city

    Returns:
        Weather information for the city
    """
    logger.info(f"Getting weather for {city}")
    return f"The weather in {city} is sunny with 22°C"


@mcp.tool()
def get_forecast(city: str, days: int = 3) -> str:
    """Get weather forecast for a city.

    Args:
        city: Name of the city
        days: Number of days to forecast (default: 3)

    Returns:
        Weather forecast for the specified days
    """
    logger.info(f"Getting {days}-day forecast for {city}")
    return f"{days}-day forecast for {city}: Sunny, 22-25°C"


app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload: dict) -> dict:
    """Handle MCP server invocation from AgentCore Runtime."""
    logger.info(f"MCP Server received: {payload}")

    try:
        # Extract MCP request from payload
        method = payload.get("method", "")
        params = payload.get("params", {})

        if method == "tools/list":
            tools = []
            for tool_name, tool_func in mcp._tools.items():
                tools.append(
                    {
                        "name": tool_name,
                        "description": tool_func.__doc__ or "",
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        },
                    }
                )
            return {"tools": tools}

        elif method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})

            if tool_name in mcp._tools:
                result = mcp._tools[tool_name](**arguments)
                return {"content": [{"type": "text", "text": result}]}
            else:
                return {"error": f"Tool {tool_name} not found"}

        else:
            return {"error": f"Unknown method: {method}"}

    except Exception as e:
        logger.error(f"Error processing MCP request: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    app.run()
