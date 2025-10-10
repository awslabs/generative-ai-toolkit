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
Implementation of weather tools using Pydantic models.

This module implements weather forecast and alert tools that demonstrate:
1. Using Pydantic models for input/output validation
2. Implementing tools that follow the Generative AI Toolkit interface
3. Mock data generation for demonstration purposes
4. Proper error handling and validation

The tools provide mock weather data for demonstration purposes.
"""

import time
import random
from datetime import datetime, timedelta
from typing import Any, Dict

from weather_models import (
    WeatherForecastRequest,
    WeatherForecastResponse,
    DailyForecast,
    WeatherAlertRequest,
    WeatherAlertResponse,
    WeatherAlert,
)


class WeatherForecastTool:
    """
    Tool for providing weather forecast information.

    This tool generates mock weather forecast data for demonstration purposes.
    In a production environment, this would integrate with a real weather API.
    """

    def __init__(self):
        """Initialize the weather forecast tool."""
        self.mock_locations = {
            "seattle, wa": {"lat": 47.6062, "lon": -122.3321, "name": "Seattle, WA"},
            "new york, ny": {"lat": 40.7128, "lon": -74.0060, "name": "New York, NY"},
            "los angeles, ca": {
                "lat": 34.0522,
                "lon": -118.2437,
                "name": "Los Angeles, CA",
            },
            "chicago, il": {"lat": 41.8781, "lon": -87.6298, "name": "Chicago, IL"},
            "miami, fl": {"lat": 25.7617, "lon": -80.1918, "name": "Miami, FL"},
            "denver, co": {"lat": 39.7392, "lon": -104.9903, "name": "Denver, CO"},
        }

        self.weather_conditions = [
            "Sunny",
            "Partly Cloudy",
            "Cloudy",
            "Light Rain",
            "Rain",
            "Heavy Rain",
            "Snow",
            "Light Snow",
            "Thunderstorms",
            "Fog",
        ]

    @property
    def tool_spec(self) -> Dict[str, Any]:
        """
        Get the tool specification for the weather forecast tool.

        Returns:
            Dictionary containing the tool specification.
        """
        schema = WeatherForecastRequest.model_json_schema()

        return {
            "name": "get_weather_forecast",
            "description": WeatherForecastRequest.__doc__,
            "inputSchema": {"json": schema},
        }

    def invoke(self, **kwargs) -> Dict[str, Any]:
        """
        Invoke the weather forecast tool.

        Args:
            **kwargs: Keyword arguments matching WeatherForecastRequest fields.

        Returns:
            Dictionary containing the weather forecast results.
        """
        try:
            # Create request from kwargs
            request = WeatherForecastRequest(**kwargs)
            response = self._get_weather_forecast(request)
            return response.model_dump()
        except Exception as e:
            # Handle validation errors and other exceptions gracefully
            error_message = f"Invalid request parameters: {str(e)}"
            response = WeatherForecastResponse(
                success=False, error=error_message, processing_time_ms=0
            )
            return response.model_dump()

    def _get_weather_forecast(
        self, request: WeatherForecastRequest
    ) -> WeatherForecastResponse:
        """
        Generate mock weather forecast data.

        Args:
            request: The validated weather forecast request.

        Returns:
            A WeatherForecastResponse containing the forecast data.
        """
        start_time = time.time()

        try:
            # Normalize location for lookup
            location_key = request.location.lower().strip()

            # Check if it's a known location
            if location_key in self.mock_locations:
                location_info = self.mock_locations[location_key]
                location_name = location_info["name"]
            else:
                # For unknown locations, use the input as-is
                location_name = request.location

            # Generate mock forecast data
            forecast = []
            base_date = datetime.now().date()

            for i in range(request.days):
                forecast_date = base_date + timedelta(days=i)

                # Generate realistic but random weather data
                temp_high = random.randint(45, 85)
                temp_low = temp_high - random.randint(10, 25)
                conditions = random.choice(self.weather_conditions)
                precipitation = (
                    random.uniform(0.0, 0.8)
                    if "Rain" in conditions or "Snow" in conditions
                    else random.uniform(0.0, 0.3)
                )
                humidity = random.randint(30, 90)
                wind_speed = random.randint(5, 25)

                daily_forecast = DailyForecast(
                    date=forecast_date.strftime("%Y-%m-%d"),
                    temperature_high=temp_high,
                    temperature_low=temp_low,
                    conditions=conditions,
                    precipitation_chance=round(precipitation, 2),
                    humidity=humidity,
                    wind_speed=wind_speed,
                )
                forecast.append(daily_forecast)

            processing_time = int((time.time() - start_time) * 1000)

            return WeatherForecastResponse(
                success=True,
                location=location_name,
                forecast=forecast,
                forecast_days=len(forecast),
                processing_time_ms=processing_time,
                message=f"Successfully retrieved {len(forecast)}-day forecast for {location_name}",
            )

        except Exception as e:
            return WeatherForecastResponse(
                success=False,
                error=f"Unexpected error: {str(e)}",
                processing_time_ms=int((time.time() - start_time) * 1000),
                message="Failed to retrieve weather forecast",
            )


class WeatherAlertTool:
    """
    Tool for providing weather alert information.

    This tool generates mock weather alert data for demonstration purposes.
    In a production environment, this would integrate with a real weather alert API.
    """

    def __init__(self):
        """Initialize the weather alert tool."""
        self.mock_alerts = {
            "ca": [
                {
                    "id": "NWS-CA-001",
                    "event": "Heat Advisory",
                    "headline": "Excessive Heat Warning in effect",
                    "description": "Dangerously hot conditions with temperatures up to 105°F expected.",
                    "severity": "Moderate",
                    "urgency": "Expected",
                    "area_desc": "Southern California",
                    "effective": "2024-01-15T12:00:00Z",
                    "expires": "2024-01-17T20:00:00Z",
                }
            ],
            "tx": [
                {
                    "id": "NWS-TX-001",
                    "event": "Severe Thunderstorm Warning",
                    "headline": "Severe thunderstorms approaching",
                    "description": "Severe thunderstorms with damaging winds and large hail possible.",
                    "severity": "Severe",
                    "urgency": "Immediate",
                    "area_desc": "Central Texas",
                    "effective": "2024-01-15T14:00:00Z",
                    "expires": "2024-01-15T18:00:00Z",
                }
            ],
            "fl": [
                {
                    "id": "NWS-FL-001",
                    "event": "Hurricane Watch",
                    "headline": "Hurricane conditions possible",
                    "description": "Hurricane conditions possible within 48 hours. Prepare now.",
                    "severity": "Extreme",
                    "urgency": "Expected",
                    "area_desc": "South Florida",
                    "effective": "2024-01-15T06:00:00Z",
                    "expires": "2024-01-18T06:00:00Z",
                }
            ],
        }

    @property
    def tool_spec(self) -> Dict[str, Any]:
        """
        Get the tool specification for the weather alert tool.

        Returns:
            Dictionary containing the tool specification.
        """
        schema = WeatherAlertRequest.model_json_schema()

        return {
            "name": "get_weather_alerts",
            "description": WeatherAlertRequest.__doc__,
            "inputSchema": {"json": schema},
        }

    def invoke(self, **kwargs) -> Dict[str, Any]:
        """
        Invoke the weather alert tool.

        Args:
            **kwargs: Keyword arguments matching WeatherAlertRequest fields.

        Returns:
            Dictionary containing the weather alert results.
        """
        try:
            # Create request from kwargs
            request = WeatherAlertRequest(**kwargs)
            response = self._get_weather_alerts(request)
            return response.model_dump()
        except Exception as e:
            # Handle validation errors and other exceptions gracefully
            error_message = f"Invalid request parameters: {str(e)}"
            response = WeatherAlertResponse(
                success=False, error=error_message, processing_time_ms=0
            )
            return response.model_dump()

    def _get_weather_alerts(self, request: WeatherAlertRequest) -> WeatherAlertResponse:
        """
        Generate mock weather alert data.

        Args:
            request: The validated weather alert request.

        Returns:
            A WeatherAlertResponse containing the alert data.
        """
        start_time = time.time()

        try:
            # Normalize area for lookup
            area_key = request.area.lower().strip()

            # Extract state code if it's a state abbreviation
            if len(area_key) == 2:
                state_code = area_key
            else:
                # Try to extract state code from area name
                if "," in area_key:
                    state_part = area_key.split(",")[-1].strip()
                    if len(state_part) == 2:
                        state_code = state_part
                    else:
                        state_code = None
                else:
                    state_code = None

            # Get mock alerts for the area
            alerts = []
            if state_code and state_code in self.mock_alerts:
                mock_alert_data = self.mock_alerts[state_code]

                for alert_data in mock_alert_data:
                    # Filter by severity if specified
                    if request.severity and alert_data["severity"] != request.severity:
                        continue

                    alert = WeatherAlert(
                        id=alert_data["id"],
                        event=alert_data["event"],
                        headline=alert_data["headline"],
                        description=alert_data["description"],
                        severity=alert_data["severity"],
                        urgency=alert_data["urgency"],
                        area_desc=alert_data["area_desc"],
                        effective=alert_data["effective"],
                        expires=alert_data["expires"],
                    )
                    alerts.append(alert)

            processing_time = int((time.time() - start_time) * 1000)

            # Create appropriate message based on results
            if alerts:
                message = f"Successfully retrieved {len(alerts)} active weather alert(s) for {request.area}"
            else:
                message = f"No active alerts found for {request.area}"

            return WeatherAlertResponse(
                success=True,
                area=request.area,
                alerts=alerts,
                alert_count=len(alerts),
                processing_time_ms=processing_time,
                message=message,
            )

        except Exception as e:
            return WeatherAlertResponse(
                success=False,
                area=request.area,
                error=f"Unexpected error: {str(e)}",
                processing_time_ms=int((time.time() - start_time) * 1000),
                message="Failed to retrieve weather alerts",
            )
