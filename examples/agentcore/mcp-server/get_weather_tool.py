"""
Get Weather Tool - Current weather information for cities.

This module provides current weather conditions for a specified city via MCP protocol.
It demonstrates using Pydantic models for input/output validation and FastMCP tool registration.
"""

import json
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WeatherRequest(BaseModel):
    """
    Request parameters for getting current weather information.

    Use this tool when users ask about current weather conditions in a specific city.

    Examples:
    - "What's the weather like in New York?"
    - "How's the weather in London today?"
    - "Tell me the current weather in Tokyo"
    """

    city: str = Field(
        description="Name of the city to get weather information for",
        min_length=1,
        max_length=100,
    )


class WeatherResponse(BaseModel):
    """
    Response structure for weather information.

    This model represents the structured response from weather tools,
    containing weather information and metadata.
    """

    success: bool = Field(
        description="Whether the weather request completed successfully"
    )

    city: str = Field(description="Name of the city the weather information is for")

    weather_info: str = Field(description="Detailed weather information")

    temperature: str | None = Field(default=None, description="Temperature information")

    conditions: str | None = Field(
        default=None, description="Weather conditions (sunny, cloudy, rainy, etc.)"
    )

    message: str | None = Field(
        default=None, description="Additional information about the request"
    )

    error: str | None = Field(
        default=None, description="Error message if the request failed"
    )


class WeatherTool:
    """
    Tool for getting current weather information for a city.

    This tool provides current weather conditions for a specified city.
    """

    def invoke(self, **kwargs) -> dict[str, Any]:
        """
        Invoke the weather tool.

        Args:
            **kwargs: Keyword arguments matching WeatherRequest fields.

        Returns:
            Dictionary containing the weather results.
        """
        try:
            # Create request from kwargs
            request = WeatherRequest(**kwargs)
            response = self._get_weather(request)
            return response.model_dump()
        except Exception as e:
            # Handle validation errors and other exceptions gracefully
            error_message = f"Invalid request parameters: {str(e)}"
            response = WeatherResponse(
                success=False,
                city=kwargs.get("city", "Unknown"),
                weather_info="",
                error=error_message,
            )
            return response.model_dump()

    def _get_weather(self, request: WeatherRequest) -> WeatherResponse:
        """
        Get current weather for a city.

        Args:
            request: The validated weather request.

        Returns:
            A WeatherResponse containing the weather information.
        """
        logger.info(f"Getting weather for {request.city}")

        # Mock weather data - in a real implementation, this would call a weather API
        weather_info = f"The weather in {request.city} is sunny with 22°C"

        return WeatherResponse(
            success=True,
            city=request.city,
            weather_info=weather_info,
            temperature="22°C",
            conditions="sunny",
            message=f"Successfully retrieved weather for {request.city}",
        )


# Initialize tool instance
_weather_tool = WeatherTool()


def register_get_weather_tool(mcp: FastMCP) -> None:
    """
    Register the get_weather tool with the FastMCP server.

    Note: The LLM learns about this tool from the WeatherRequest Pydantic model:
    - Model docstring provides tool description and usage examples
    - Field descriptions define parameter types and validation rules

    FastMCP automatically transforms the Pydantic model into JSON schemas
    that help the LLM understand when and how to use this tool.

    Args:
        mcp: The FastMCP server instance to register the tool with.
    """

    @mcp.tool()
    def get_weather(request: WeatherRequest) -> str:
        """Get current weather for a city.

        Args:
            request: Weather request parameters

        Returns:
            JSON string containing weather information
        """
        logger.info(f"get_weather called with request: {request}")
        result = _weather_tool.invoke(city=request.city)
        logger.info("get_weather completed successfully")
        return json.dumps(result, indent=2)
