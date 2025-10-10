"""
Copyright 2024 Amazon.com, Inc. and its affiliates. All Rights Reserved.

Licensed under the Amazon Software License (the "License").
You may not use this file except in compliance with the License.
A copy of the License is located at

  http://aws.amazon.com/asl/

or in the "license" file accompanying this file. This file is distributed
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
express or implied. See the License for the specific language governing
permissions and limitations under the License.
"""

"""
Pydantic data models for weather tools.

This module defines the Pydantic models used for input validation and response
structuring in the weather forecast tools. These models provide strong typing,
validation rules, and documentation for the tool's inputs and outputs.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# Supporting models used by response classes
class DailyForecast(BaseModel):
    """
    Model representing a single day's weather forecast.
    """

    date: str = Field(description="Date of the forecast (YYYY-MM-DD format)")
    temperature_high: int = Field(description="High temperature in Fahrenheit")
    temperature_low: int = Field(description="Low temperature in Fahrenheit")
    conditions: str = Field(
        description="Weather conditions (e.g., 'Sunny', 'Partly Cloudy', 'Rain')"
    )
    precipitation_chance: float = Field(
        ge=0.0, le=1.0, description="Probability of precipitation (0.0 to 1.0)"
    )
    humidity: Optional[int] = Field(
        default=None, ge=0, le=100, description="Relative humidity percentage"
    )
    wind_speed: Optional[int] = Field(
        default=None, ge=0, description="Wind speed in mph"
    )


class WeatherAlert(BaseModel):
    """
    Model representing a single weather alert.
    """

    id: str = Field(description="Unique identifier for the alert")
    event: str = Field(description="Type of weather event")
    headline: Optional[str] = Field(default=None, description="Alert headline")
    description: str = Field(description="Detailed description of the alert")
    severity: str = Field(description="Severity level of the alert")
    urgency: str = Field(description="Urgency level of the alert")
    area_desc: str = Field(description="Description of the affected area")
    effective: Optional[str] = Field(
        default=None, description="When the alert becomes effective"
    )
    expires: str = Field(description="When the alert expires")


# Weather Forecast Request/Response pair
class WeatherForecastRequest(BaseModel):
    """
    Request parameters for the weather forecast tool.

    IMPORTANT: Always use this tool when processing user requests that:
    1. Need weather forecast information for a specific location
    2. Ask about upcoming weather conditions
    3. Need temperature, precipitation, or general weather outlook
    4. Ask about weather for planning activities or travel

    This tool provides weather forecast data for a specified location and number of days.

    Examples:
    - Get 3-day forecast: WeatherForecastRequest(location="Seattle, WA")
    - Get 7-day forecast: WeatherForecastRequest(location="New York, NY", days=7)
    - Get forecast by coordinates: WeatherForecastRequest(location="47.6062,-122.3321", days=5)
    """

    location: str = Field(
        description="City name, state (e.g., 'Seattle, WA') or coordinates (e.g., '47.6062,-122.3321')"
    )

    days: int = Field(
        default=3,
        ge=1,
        le=7,
        description="Number of forecast days to retrieve (1-7 days)",
    )


class WeatherForecastResponse(BaseModel):
    """
    Response structure for the weather forecast tool.

    This model represents the structured response from the weather forecast tool,
    containing forecast information and any processing metadata.

    The success field indicates whether the forecast request completed successfully.
    Additional fields provide details about the forecast and any processing metadata.

    Examples of returned values:
    - Successful request: {"success": True, "location": "Seattle, WA", "forecast": [...], "forecast_days": 3}
    - Failed request: {"success": False, "error": "Invalid location", "message": "Unable to retrieve forecast"}
    """

    success: bool = Field(
        description="Whether the weather forecast request completed successfully"
    )

    location: Optional[str] = Field(
        default=None, description="The location for which the forecast was retrieved"
    )

    forecast: Optional[List[DailyForecast]] = Field(
        default=None, description="List of daily weather forecasts"
    )

    forecast_days: Optional[int] = Field(
        default=None, description="Number of forecast days returned"
    )

    processing_time_ms: Optional[int] = Field(
        default=None, description="Time taken to process the request in milliseconds"
    )

    message: Optional[str] = Field(
        default=None, description="Additional information about the request results"
    )

    error: Optional[str] = Field(
        default=None, description="Error message if the request failed"
    )


# Weather Alert Request/Response pair
class WeatherAlertRequest(BaseModel):
    """
    Request parameters for the weather alerts tool.

    IMPORTANT: Always use this tool when processing user requests that:
    1. Need information about active weather alerts or warnings
    2. Ask about current weather hazards in a specific area
    3. Need to check if there are any severe weather conditions
    4. Ask about emergency weather situations

    This tool provides active weather alerts for a specified area.

    Examples:
    - Get alerts for a state: WeatherAlertRequest(area="CA")
    - Get alerts for a specific region: WeatherAlertRequest(area="King County, WA")
    """

    area: str = Field(
        description="State code (e.g., 'CA', 'TX') or area name (e.g., 'King County, WA')"
    )

    severity: Optional[str] = Field(
        default=None,
        description="Filter by severity level: 'Extreme', 'Severe', 'Moderate', 'Minor'",
    )


class WeatherAlertResponse(BaseModel):
    """
    Response structure for the weather alerts tool.

    This model represents the structured response from the weather alerts tool,
    containing information about active weather alerts and any processing metadata.
    """

    success: bool = Field(
        description="Whether the weather alerts request completed successfully"
    )

    area: Optional[str] = Field(
        default=None, description="The area for which alerts were retrieved"
    )

    alerts: Optional[List[WeatherAlert]] = Field(
        default=None, description="List of active weather alerts"
    )

    alert_count: Optional[int] = Field(
        default=None, description="Number of alerts returned"
    )

    processing_time_ms: Optional[int] = Field(
        default=None, description="Time taken to process the request in milliseconds"
    )

    message: Optional[str] = Field(
        default=None, description="Additional information about the request results"
    )

    error: Optional[str] = Field(
        default=None, description="Error message if the request failed"
    )
