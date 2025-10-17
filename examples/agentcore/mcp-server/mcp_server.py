#!/usr/bin/env python3
"""MCP Server for AgentCore Runtime."""

import logging

from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("MCP Server starting")

# Create FastMCP server with correct configuration for AgentCore Runtime
mcp = FastMCP(host="0.0.0.0", stateless_http=True)


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


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
