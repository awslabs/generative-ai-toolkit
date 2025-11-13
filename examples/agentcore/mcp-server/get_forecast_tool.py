"""
Get Forecast Tool - Weather forecast information for cities.

This module provides weather forecasts for a specified number of days via MCP protocol.
It demonstrates using Pydantic models for input/output validation and FastMCP tool registration.
"""

import json
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WeatherForecastRequest(BaseModel):
    """
    Request parameters for getting weather forecast information.

    Use this tool when users ask about weather forecasts for multiple days.

    Examples:
    - "What's the 5-day forecast for Paris?"
    - "Give me the weather forecast for Seattle"
    - "What will the weather be like in Miami this week?"
    """

    city: str = Field(
        description="Name of the city to get weather forecast for",
        min_length=1,
        max_length=100,
    )

    days: int = Field(
        default=3, description="Number of days to forecast (1-7 days)", ge=1, le=7
    )


class WeatherForecastResponse(BaseModel):
    """
    Response structure for weather forecast information.

    This model represents the structured response from weather forecast tools,
    containing forecast information and metadata.
    """

    success: bool = Field(
        description="Whether the weather forecast request completed successfully"
    )

    city: str = Field(description="Name of the city the forecast is for")

    days: int = Field(description="Number of days in the forecast")

    forecast_info: str = Field(description="Detailed forecast information")

    temperature_range: str | None = Field(
        default=None, description="Temperature range for the forecast period"
    )

    conditions: str | None = Field(
        default=None, description="General weather conditions for the forecast period"
    )

    message: str | None = Field(
        default=None, description="Additional information about the forecast"
    )

    error: str | None = Field(
        default=None, description="Error message if the request failed"
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


# Initialize tool instance
_forecast_tool = WeatherForecastTool()


def register_get_forecast_tool(mcp: FastMCP) -> None:
    """
    Register the get_forecast tool with the FastMCP server.

    Note: The LLM learns about this tool from the WeatherForecastRequest Pydantic model:
    - Model docstring provides tool description and usage examples
    - Field descriptions define parameter types and validation rules

    FastMCP automatically transforms the Pydantic model into JSON schemas
    that help the LLM understand when and how to use this tool.

    Args:
        mcp: The FastMCP server instance to register the tool with.
    """

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
