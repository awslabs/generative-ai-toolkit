"""
Implementation of weather tools for MCP server using Pydantic models.

This module implements weather tools for the FastMCP server and demonstrates:

1. Using Pydantic models for input/output validation
2. Tool registration with FastMCP decorators
3. Clear tool specifications for LLM consumption
4. Proper error handling and response structuring

The tools provide weather information and forecasts for cities via MCP protocol.
"""

import json
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP
from weather_models import (
    WeatherForecastRequest,
    WeatherForecastResponse,
    WeatherRequest,
    WeatherResponse,
)

logger = logging.getLogger(__name__)


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


class WeatherForecastTool:
    """
    Tool for getting weather forecast information for a city.

    This tool provides weather forecasts for a specified number of days.
    """

    def invoke(self, **kwargs) -> dict[str, Any]:
        """
        Invoke the weather forecast tool.

        Args:
            **kwargs: Keyword arguments matching WeatherForecastRequest fields.

        Returns:
            Dictionary containing the forecast results.
        """
        try:
            # Create request from kwargs
            request = WeatherForecastRequest(**kwargs)
            response = self._get_forecast(request)
            return response.model_dump()
        except Exception as e:
            # Handle validation errors and other exceptions gracefully
            error_message = f"Invalid request parameters: {str(e)}"
            response = WeatherForecastResponse(
                success=False,
                city=kwargs.get("city", "Unknown"),
                days=kwargs.get("days", 3),
                forecast_info="",
                error=error_message,
            )
            return response.model_dump()

    def _get_forecast(self, request: WeatherForecastRequest) -> WeatherForecastResponse:
        """
        Get weather forecast for a city.

        Args:
            request: The validated weather forecast request.

        Returns:
            A WeatherForecastResponse containing the forecast information.
        """
        logger.info(f"Getting {request.days}-day forecast for {request.city}")

        # Mock forecast data - in a real implementation, this would call a weather API
        forecast_info = (
            f"{request.days}-day forecast for {request.city}: Sunny, 22-25°C"
        )

        return WeatherForecastResponse(
            success=True,
            city=request.city,
            days=request.days,
            forecast_info=forecast_info,
            temperature_range="22-25°C",
            conditions="sunny",
            message=f"Successfully retrieved {request.days}-day forecast for {request.city}",
        )


# Initialize tool instances
_weather_tool = WeatherTool()
_forecast_tool = WeatherForecastTool()


def register_weather_tools(mcp: FastMCP) -> None:
    """
    Register weather tools with the FastMCP server.

    Note: The LLM learns about available tools from the **Request Pydantic models**:

    - **Model docstrings** (like WeatherRequest.__doc__): Provide the tool
      description and usage guidance that help the LLM decide WHEN to use each tool.
      These contain examples of user queries that should trigger the tool.

    - **Field descriptions**: Define parameter types, validation rules, and
      constraints that tell the LLM HOW to structure the tool parameters correctly.

    FastMCP automatically transforms these Pydantic models into JSON schemas that
    the LLM consumes. The Request model docstrings are the primary source for
    tool selection guidance - make sure they clearly describe when to use each
    tool and include realistic usage examples.

    Args:
        mcp: The FastMCP server instance to register tools with.
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

    @mcp.tool()
    def get_forecast(request: WeatherForecastRequest) -> str:
        """Get weather forecast for a city.

        Args:
            request: Weather forecast request parameters

        Returns:
            JSON string containing forecast information
        """
        logger.info(f"get_forecast called with request: {request}")
        result = _forecast_tool.invoke(city=request.city, days=request.days)
        logger.info("get_forecast completed successfully")
        return json.dumps(result, indent=2)
