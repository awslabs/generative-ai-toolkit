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

###
# Pydantic data models for the weather alerts tool.

# This module defines the Pydantic models used for input validation and response
# structuring in the weather alerts tool. These models provide strong typing,
# validation rules, and documentation for the tool's inputs and outputs.
###


from pydantic import BaseModel, Field, model_validator


class WeatherAlertRequest(BaseModel):
    """
    Request parameters for the weather alerts tool.

    IMPORTANT: Always use this tool when processing user requests that:
    1. Need information about active weather alerts or warnings
    2. Ask about current weather hazards in a specific area
    3. Need to check if there are any severe weather conditions
    4. Ask about emergency weather situations
    5. Need information about weather advisories
    6. Ask about evacuation orders due to weather

    This tool fetches active weather alerts from the National Weather Service API.
    It can filter alerts by area (state, zone, county), event type, status, and urgency.

    Examples:
    - Get alerts for a state: WeatherAlertRequest(area="CA")
    - Get alerts by event type: WeatherAlertRequest(event="Flood")
    - Get alerts for a specific county: WeatherAlertRequest(area="CA", region_type="county")
    - Get severe alerts only: WeatherAlertRequest(urgency="Immediate")
    - Get confirmed severe alerts: WeatherAlertRequest(severity="Severe", certainty="Observed", message_type="alert")
    """

    area: str | None = Field(
        default=None,
        description="State code (e.g., 'CA', 'TX') or zone/county code to filter alerts by area.",
    )

    region_type: str | None = Field(
        default=None,
        description="Type of region to filter by: 'area' (state), 'zone', or 'county'.",
        pattern="^(area|zone|county)$",
    )

    status: str | None = Field(
        default=None,
        description="Filter by alert status: 'actual', 'exercise', 'system', 'test', or 'draft'.",
        pattern="^(actual|exercise|system|test|draft)$",
    )

    message_type: str | None = Field(
        default=None,
        description="Filter by message type: 'alert', 'update', or 'cancel'.",
        pattern="^(alert|update|cancel)$",
    )

    event: str | None = Field(
        default=None,
        description="Filter by event type (e.g., 'Tornado', 'Flood', 'Severe Thunderstorm').",
    )

    code: str | None = Field(
        default=None,
        description="Filter by specific event code.",
    )

    urgency: str | None = Field(
        default=None,
        description="Filter by urgency level: 'Immediate', 'Expected', 'Future', 'Past', or 'Unknown'.",
        pattern="^(Immediate|Expected|Future|Past|Unknown)$",
    )

    severity: str | None = Field(
        default=None,
        description="Filter by severity level: 'Extreme', 'Severe', 'Moderate', 'Minor', or 'Unknown'.",
        pattern="^(Extreme|Severe|Moderate|Minor|Unknown)$",
    )

    certainty: str | None = Field(
        default=None,
        description="Filter by certainty level: 'Observed', 'Likely', 'Possible', 'Unlikely', or 'Unknown'.",
        pattern="^(Observed|Likely|Possible|Unlikely|Unknown)$",
    )

    # Note: limit parameter removed as it is not supported by /alerts/active endpoint

    # Add model validation to ensure region_type is provided if area is specified
    @model_validator(mode="after")
    def validate_region_type(self) -> "WeatherAlertRequest":
        """Validate that region_type is provided if area is specified."""
        if self.area and not self.region_type:
            # Default to 'area' (state) if not specified
            self.region_type = "area"
        return self


class Alert(BaseModel):
    """
    Model representing a single weather alert.
    """

    id: str = Field(description="Unique identifier for the alert.")
    area_desc: str = Field(description="Description of the affected area.")
    event: str = Field(description="Type of weather event.")
    headline: str | None = Field(default=None, description="Alert headline.")
    description: str = Field(description="Detailed description of the alert.")
    instruction: str | None = Field(
        default=None, description="Instructions for the public."
    )
    severity: str = Field(description="Severity level of the alert.")
    urgency: str = Field(description="Urgency level of the alert.")
    certainty: str = Field(description="Certainty level of the alert.")
    effective: str | None = Field(
        default=None, description="When the alert becomes effective."
    )
    onset: str | None = Field(
        default=None, description="When the event is expected to begin."
    )
    expires: str = Field(description="When the alert expires.")
    status: str = Field(
        description="Status of the alert (actual, exercise, test, etc.)."
    )
    message_type: str = Field(description="Type of message (alert, update, cancel).")
    sender_name: str = Field(description="Name of the alert sender.")


class WeatherAlertResponse(BaseModel):
    """
    Response structure for the weather alerts tool.

    This model represents the structured response from the weather alerts tool,
    containing information about active weather alerts and any processing metadata.

    The success field indicates whether the API request completed successfully.
    Additional fields provide details about the alerts and any processing metadata.

    Examples of returned values:
    - Successful request: {"success": True, "alerts": [{"id": "NWS-IDP-PROD-123", "event": "Flood Warning", ...}], "alert_count": 1}
    - No alerts found: {"success": True, "alerts": [], "alert_count": 0, "message": "No active alerts found for the specified criteria."}
    - Failed request: {"success": False, "error": "Invalid area code", "message": "Failed to retrieve weather alerts"}
    """

    success: bool = Field(
        description="Whether the weather alerts request completed successfully."
    )

    alerts: list[Alert] | None = Field(
        default=None,
        description="List of active weather alerts matching the request criteria.",
    )

    alert_count: int | None = Field(
        default=None,
        description="Number of alerts returned.",
    )

    processing_time_ms: int | None = Field(
        default=None, description="Time taken to process the request in milliseconds."
    )

    message: str | None = Field(
        default=None,
        description="Additional information about the request results.",
    )

    error: str | None = Field(
        default=None, description="Error message if the request failed."
    )
