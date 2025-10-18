"""
Implementation of weather tools using Pydantic models.

This module implements weather tools that follow the Generative AI Toolkit interface
and demonstrate:

1. Using Pydantic models for input/output validation
2. Implementing tools with proper tool specifications
3. Clear tool specifications for LLM consumption
4. Proper error handling and response structuring

The tools provide weather information and forecasts for cities.
"""

import logging
from typing import Any

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

    @property
    def tool_spec(self) -> dict[str, Any]:
        """
        Get the tool specification for the weather tool.

        Returns:
            Dictionary containing the tool specification.
        """
        schema = WeatherRequest.model_json_schema()

        return {
            "name": "get_weather",
            "description": WeatherRequest.__doc__,
            "inputSchema": schema,
        }

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

    @property
    def tool_spec(self) -> dict[str, Any]:
        """
        Get the tool specification for the weather forecast tool.

        Returns:
            Dictionary containing the tool specification.
        """
        schema = WeatherForecastRequest.model_json_schema()

        return {
            "name": "get_forecast",
            "description": WeatherForecastRequest.__doc__,
            "inputSchema": schema,
        }

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
