"""
Pydantic data models for weather tools.

This module defines the Pydantic models used for input validation and response
structuring in the weather tools. These models provide strong typing,
validation rules, and documentation for the tool's inputs and outputs.
"""

from pydantic import BaseModel, Field


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
