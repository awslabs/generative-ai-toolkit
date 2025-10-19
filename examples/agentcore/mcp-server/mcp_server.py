#!/usr/bin/env python3
"""MCP Server for AgentCore Runtime using Pydantic models with FastMCP."""

import logging

from mcp.server.fastmcp import FastMCP
from weather_tools import register_weather_tools

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logger.info("MCP Server starting")

# Create FastMCP server with correct configuration for AgentCore Runtime
mcp = FastMCP(host="0.0.0.0", stateless_http=True)

# Register all weather tools
register_weather_tools(mcp)

if __name__ == "__main__":
    logger.info("Starting Weather Tools MCP Server with FastMCP")
    mcp.run(transport="streamable-http")
