#!/usr/bin/env python3
"""MCP Server for AgentCore Runtime using modular weather tools with FastMCP."""

import logging

from get_forecast_tool import register_get_forecast_tool
from get_weather_tool import register_get_weather_tool
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logger.info("MCP Server starting")

# Create FastMCP server with correct configuration for AgentCore Runtime
mcp = FastMCP(host="0.0.0.0", stateless_http=True)

# Register weather tools from separate modules
register_get_weather_tool(mcp)
register_get_forecast_tool(mcp)

if __name__ == "__main__":
    logger.info("Starting Weather Tools MCP Server with FastMCP")
    mcp.run(transport="streamable-http")
