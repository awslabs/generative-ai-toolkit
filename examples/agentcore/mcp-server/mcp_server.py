#!/usr/bin/env python3
"""MCP Server for AgentCore Runtime using Pydantic models with FastMCP."""

import json
import logging

from mcp.server.fastmcp import FastMCP
from weather_models import WeatherForecastRequest, WeatherRequest
from weather_tools import WeatherForecastTool, WeatherTool

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logger.info("MCP Server starting")

# Initialize tool instances
_weather_tool = WeatherTool()
_forecast_tool = WeatherForecastTool()

# Create FastMCP server with correct configuration for AgentCore Runtime
mcp = FastMCP(host="0.0.0.0", stateless_http=True)


# Register weather tools using Pydantic models as parameters
@mcp.tool()
def get_weather(request: WeatherRequest) -> str:
    """Get current weather for a city.

    Use this tool when users ask about current weather conditions in a specific city.

    Examples:
    - "What's the weather like in New York?"
    - "How's the weather in London today?"
    - "Tell me the current weather in Tokyo"

    Args:
        request: Weather request containing city information

    Returns:
        JSON string containing weather information
    """
    logger.info(f"get_weather called with request: {request}")
    result = _weather_tool.invoke(city=request.city)
    logger.info("get_weather completed successfully")
    return json.dumps(result, indent=2)


@mcp.tool()
def get_forecast(request: WeatherForecastRequest) -> str:
    """Get weather forecast for a city.

    Use this tool when users ask about weather forecasts for multiple days.

    Examples:
    - "What's the 5-day forecast for Paris?"
    - "Give me the weather forecast for Seattle"
    - "What will the weather be like in Miami this week?"

    Args:
        request: Weather forecast request containing city and days information

    Returns:
        JSON string containing forecast information
    """
    logger.info(f"get_forecast called with request: {request}")
    result = _forecast_tool.invoke(city=request.city, days=request.days)
    logger.info("get_forecast completed successfully")
    return json.dumps(result, indent=2)


if __name__ == "__main__":
    logger.info("Starting Weather Tools MCP Server with FastMCP")
    mcp.run(transport="streamable-http")
