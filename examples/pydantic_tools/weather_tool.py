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
Implementation of a weather alerts tool using Pydantic models.

This module implements a tool for fetching active weather alerts from the National
Weather Service API. It demonstrates:

1. Using Pydantic models for input/output validation
2. Implementing a tool that follows the Generative AI Toolkit interface
3. Proper API integration with error handling
4. Clear tool specifications for LLM consumption

The tool can filter alerts by various criteria such as area, event type, urgency,
and severity, making it useful for weather-related queries from LLM agents.
"""

import time
import requests
from typing import Any, Dict

# Import models from models.py
from models import WeatherAlertRequest, WeatherAlertResponse, Alert


class WeatherAlertsTool:
    """
    Tool for fetching active weather alerts from the National Weather Service API.

    This tool can retrieve active weather alerts and filter them by various criteria
    such as area, event type, urgency, and severity.
    """

    def __init__(self, base_url: str = "https://api.weather.gov"):
        """
        Initialize the weather alerts tool.

        Args:
            base_url: Base URL for the National Weather Service API
        """
        self.base_url = base_url
        self.user_agent = (
            "GenerativeAIToolkit/1.0 (https://github.com/aws/generative-ai-toolkit)"
        )

    @property
    def tool_spec(self) -> Dict[str, Any]:
        """
        Get the tool specification for the weather alerts tool.

        Returns:
            Dictionary containing the tool specification.
        """
        # Use the WeatherAlertRequest schema for the input schema
        schema = WeatherAlertRequest.model_json_schema()

        return {
            "name": "get_weather_alerts",
            "description": WeatherAlertRequest.__doc__,
            "inputSchema": {"json": schema},
        }

    def invoke(self, **kwargs) -> Dict[str, Any]:
        """
        Invoke the weather alerts tool.

        Args:
            **kwargs: Keyword arguments matching WeatherAlertRequest fields.

        Returns:
            Dictionary containing the weather alerts results.
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
        Fetch active weather alerts from the National Weather Service API.

        Args:
            request: The validated weather alert request containing filter criteria.

        Returns:
            A WeatherAlertResponse containing the alerts and request metadata.
        """
        start_time = time.time()

        try:
            # Construct the API endpoint URL based on request parameters
            endpoint = "/alerts/active"

            # Add area filtering if specified
            if request.area:
                if request.region_type == "area":
                    endpoint = f"/alerts/active/area/{request.area}"
                elif request.region_type == "zone":
                    endpoint = f"/alerts/active/zone/{request.area}"
                elif request.region_type == "county":
                    endpoint = f"/alerts/active/county/{request.area}"

            # Construct query parameters
            params = {}
            if request.status:
                params["status"] = request.status
            if request.message_type:
                params["message_type"] = request.message_type
            if request.event:
                params["event"] = request.event
            if request.code:
                params["code"] = request.code
            if request.urgency:
                params["urgency"] = request.urgency
            if request.severity:
                params["severity"] = request.severity
            if request.certainty:
                params["certainty"] = request.certainty

            # Make the API request
            url = f"{self.base_url}{endpoint}"
            headers = {"User-Agent": self.user_agent, "Accept": "application/geo+json"}

            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()  # Raise exception for HTTP errors

            # Parse the response
            data = response.json()

            # Extract alerts from the GeoJSON response
            alerts = []
            if "features" in data:
                for feature in data["features"]:
                    if "properties" in feature:
                        props = feature["properties"]

                        # Extract the relevant fields for each alert
                        alert = Alert(
                            id=props.get("id", ""),
                            area_desc=props.get("areaDesc", ""),
                            event=props.get("event", ""),
                            headline=props.get("headline"),
                            description=props.get("description", ""),
                            instruction=props.get("instruction"),
                            severity=props.get("severity", ""),
                            urgency=props.get("urgency", ""),
                            certainty=props.get("certainty", ""),
                            effective=props.get("effective"),
                            onset=props.get("onset"),
                            expires=props.get("expires", ""),
                            status=props.get("status", ""),
                            message_type=props.get("messageType", ""),
                            sender_name=props.get("senderName", ""),
                        )
                        alerts.append(alert)

            processing_time = int((time.time() - start_time) * 1000)

            # Create appropriate message based on results
            if alerts:
                message = (
                    f"Successfully retrieved {len(alerts)} active weather alert(s)."
                )
            else:
                message = "No active alerts found for the specified criteria."

            return WeatherAlertResponse(
                success=True,
                alerts=alerts,
                alert_count=len(alerts),
                processing_time_ms=processing_time,
                message=message,
            )

        except requests.exceptions.HTTPError as e:
            return WeatherAlertResponse(
                success=False,
                error=f"HTTP error: {str(e)}",
                processing_time_ms=int((time.time() - start_time) * 1000),
                message="Failed to retrieve weather alerts",
            )
        except requests.exceptions.ConnectionError:
            return WeatherAlertResponse(
                success=False,
                error="Connection error: Unable to connect to the National Weather Service API",
                processing_time_ms=int((time.time() - start_time) * 1000),
                message="Failed to retrieve weather alerts",
            )
        except requests.exceptions.Timeout:
            return WeatherAlertResponse(
                success=False,
                error="Timeout error: The request to the National Weather Service API timed out",
                processing_time_ms=int((time.time() - start_time) * 1000),
                message="Failed to retrieve weather alerts",
            )
        except requests.exceptions.RequestException as e:
            return WeatherAlertResponse(
                success=False,
                error=f"Request error: {str(e)}",
                processing_time_ms=int((time.time() - start_time) * 1000),
                message="Failed to retrieve weather alerts",
            )
        except Exception as e:
            return WeatherAlertResponse(
                success=False,
                error=f"Unexpected error: {str(e)}",
                processing_time_ms=int((time.time() - start_time) * 1000),
                message="Failed to retrieve weather alerts",
            )
