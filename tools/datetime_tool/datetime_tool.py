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
DateTime tool for the Siemens P2P Agent.

This module implements a tool for parsing, validating, and comparing dates.
It can determine if dates are valid, in the past or future, and compare dates.
Uses Pydantic models for structured input and output validation.
"""

import time
from datetime import datetime, date
from typing import Optional, Any, Dict, List
from pydantic import BaseModel, Field
from dateutil import parser
from dateutil.relativedelta import relativedelta


class DateTimeRequest(BaseModel):
    """
    Request parameters for the datetime tool.

    IMPORTANT: Always use this tool when processing user requests that:
    1. Need to parse date strings into structured date objects
    2. Need to validate if a date is valid (e.g., February 29 in non-leap years)
    3. Need to determine if a date is in the past or future
    4. Need to compare two dates to see which is earlier/later
    5. Need to calculate differences between dates
    6. Ask about "date validation", "is this date valid", "when is this date"
    7. Need to check invoice dates, due dates, or payment dates in P2P workflows
    8. Ask about "today", "current date", "now", or need current date/time information

    This tool handles comprehensive date operations including parsing various date formats,
    validation, comparison, and temporal analysis. It's optimized for P2P/AP processing
    where date validation and comparison are critical for invoice processing.

    Examples:
    - Parse date: DateTimeRequest(operation="PARSE", date_string="2024-02-29")
    - Parse current date: DateTimeRequest(operation="PARSE", date_string="today")
    - Validate date: DateTimeRequest(operation="VALIDATE", date_string="2024-02-30")
    - Check if past: DateTimeRequest(operation="IS_PAST", date_string="2023-01-01")
    - Compare with today: DateTimeRequest(operation="COMPARE", date_string="2024-01-01", compare_date="today")
    - Calculate difference from now: DateTimeRequest(operation="DIFFERENCE", date_string="2024-01-01", compare_date="now")
    """

    operation: str = Field(
        description="The datetime operation to perform: 'PARSE', 'VALIDATE', 'IS_PAST', 'IS_FUTURE', 'COMPARE', or 'DIFFERENCE'.",
        pattern="^(PARSE|VALIDATE|IS_PAST|IS_FUTURE|COMPARE|DIFFERENCE)$",
    )

    date_string: str = Field(
        description="The date string to process. Supports various formats like '2024-01-01', 'January 1, 2024', '01/01/2024', 'today', 'now', 'current date', etc."
    )

    compare_date: Optional[str] = Field(
        default=None,
        description="Second date string for comparison operations (COMPARE, DIFFERENCE). If not provided, uses current date.",
    )

    reference_date: Optional[str] = Field(
        default=None,
        description="Reference date for IS_PAST/IS_FUTURE operations. If not provided, uses current date.",
    )

    purpose: Optional[str] = Field(
        default=None,
        description="A brief explanation of why this date operation is being performed.",
    )


class DateTimeResponse(BaseModel):
    """
    Response structure for the datetime tool.

    This model represents the structured response from the datetime tool,
    containing parsed dates, validation results, comparison outcomes, and metadata.

    Examples of returned values:
    - Parse success: {"parsed_date": "2024-01-01", "is_valid": True, "operation": "PARSE"}
    - Invalid date: {"is_valid": False, "error": "February 30 is not a valid date", "operation": "VALIDATE"}
    - Past date: {"is_past": True, "days_difference": -365, "operation": "IS_PAST"}
    - Date comparison: {"comparison_result": "earlier", "days_difference": -31, "operation": "COMPARE"}
    """

    operation: str = Field(description="The datetime operation that was performed.")

    parsed_date: Optional[str] = Field(
        default=None,
        description="The parsed date in ISO format (YYYY-MM-DD) if parsing was successful.",
    )

    is_valid: Optional[bool] = Field(
        default=None, description="Whether the date string represents a valid date."
    )

    is_past: Optional[bool] = Field(
        default=None,
        description="Whether the date is in the past (for IS_PAST operation).",
    )

    is_future: Optional[bool] = Field(
        default=None,
        description="Whether the date is in the future (for IS_FUTURE operation).",
    )

    comparison_result: Optional[str] = Field(
        default=None,
        description="Result of date comparison: 'earlier', 'later', or 'same' (for COMPARE operation).",
    )

    days_difference: Optional[int] = Field(
        default=None,
        description="Difference in days between dates. Negative means first date is earlier.",
    )

    formatted_date: Optional[str] = Field(
        default=None, description="Human-readable formatted date string."
    )

    processing_time_ms: Optional[int] = Field(
        default=None,
        description="Time taken to process the datetime operation in milliseconds.",
    )

    message: Optional[str] = Field(
        default=None, description="Additional information about the operation results."
    )

    error: Optional[str] = Field(
        default=None, description="Error message if the datetime operation failed."
    )


class DateTimeTool:
    """
    Tool for parsing, validating, and comparing dates.

    This tool can handle various date formats, validate date correctness,
    and perform temporal comparisons and calculations.
    """

    def __init__(self):
        """Initialize the datetime tool."""
        pass

    @property
    def tool_spec(self) -> Dict[str, Any]:
        """
        Get the tool specification for the datetime tool.

        Returns:
            Dictionary containing the tool specification.
        """
        schema = DateTimeRequest.model_json_schema()

        return {
            "name": "datetime_operation",
            "description": DateTimeRequest.__doc__,
            "inputSchema": {"json": schema},
        }

    def invoke(self, **kwargs) -> Dict[str, Any]:
        """
        Invoke the datetime tool.

        Args:
            **kwargs: Keyword arguments matching DateTimeRequest fields.

        Returns:
            Dictionary containing the datetime operation results.
        """
        try:
            request = DateTimeRequest(**kwargs)
            response = self._process_datetime(request)
            return response.model_dump()
        except Exception as e:
            error_message = f"Invalid request parameters: {str(e)}"
            response = DateTimeResponse(
                operation=kwargs.get("operation", "UNKNOWN"),
                error=error_message,
                processing_time_ms=0,
            )
            return response.model_dump()

    def _process_datetime(self, request: DateTimeRequest) -> DateTimeResponse:
        """
        Process the datetime operation.

        Args:
            request: The validated datetime request.

        Returns:
            A DateTimeResponse containing the operation results.
        """
        start_time = time.time()

        operation_handlers = {
            "PARSE": self._parse_date,
            "VALIDATE": self._validate_date,
            "IS_PAST": self._check_is_past,
            "IS_FUTURE": self._check_is_future,
            "COMPARE": self._compare_dates,
            "DIFFERENCE": self._calculate_difference,
        }

        try:
            handler = operation_handlers.get(request.operation)
            if handler:
                return handler(request, start_time)
            else:
                return DateTimeResponse(
                    operation=request.operation,
                    error=f"Invalid operation: {request.operation}",
                    processing_time_ms=int((time.time() - start_time) * 1000),
                )

        except Exception as e:
            return DateTimeResponse(
                operation=request.operation,
                error=f"Error processing datetime operation: {str(e)}",
                processing_time_ms=int((time.time() - start_time) * 1000),
            )

    def _parse_date_string(self, date_string: str) -> datetime:
        """Parse a date string into a datetime object."""
        try:
            # Handle current date/time keywords
            normalized_string = self._normalize_current_date_keywords(date_string)

            # Use dateutil parser for flexible date parsing
            return parser.parse(normalized_string)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Unable to parse date string '{date_string}': {str(e)}")

    def _normalize_current_date_keywords(self, date_string: str) -> str:
        """Convert current date keywords to actual current date/time."""
        normalized = date_string.lower().strip()

        # Keywords that refer to current date
        current_date_keywords = [
            "today",
            "current date",
            "current day",
            "this date",
            "todays date",
            "today's date",
        ]

        # Keywords that refer to current date and time
        current_datetime_keywords = [
            "now",
            "current time",
            "current datetime",
            "right now",
            "this moment",
            "current timestamp",
        ]

        # Check for current date keywords
        if normalized in current_date_keywords:
            return datetime.now().strftime("%Y-%m-%d")

        # Check for current datetime keywords
        if normalized in current_datetime_keywords:
            return datetime.now().isoformat()

        # Return original string if no keywords matched
        return date_string

    def _parse_date(
        self, request: DateTimeRequest, start_time: float
    ) -> DateTimeResponse:
        """Parse a date string."""
        try:
            parsed_dt = self._parse_date_string(request.date_string)

            return DateTimeResponse(
                operation="PARSE",
                parsed_date=parsed_dt.date().isoformat(),
                is_valid=True,
                formatted_date=parsed_dt.strftime("%B %d, %Y"),
                processing_time_ms=int((time.time() - start_time) * 1000),
                message=f"Successfully parsed date: {request.date_string}",
            )
        except ValueError as e:
            return DateTimeResponse(
                operation="PARSE",
                is_valid=False,
                error=str(e),
                processing_time_ms=int((time.time() - start_time) * 1000),
            )

    def _validate_date(
        self, request: DateTimeRequest, start_time: float
    ) -> DateTimeResponse:
        """Validate if a date string represents a valid date."""
        try:
            parsed_dt = self._parse_date_string(request.date_string)

            # Additional validation for edge cases
            original_date = parsed_dt.date()

            return DateTimeResponse(
                operation="VALIDATE",
                parsed_date=original_date.isoformat(),
                is_valid=True,
                formatted_date=parsed_dt.strftime("%B %d, %Y"),
                processing_time_ms=int((time.time() - start_time) * 1000),
                message=f"Date '{request.date_string}' is valid",
            )
        except ValueError as e:
            return DateTimeResponse(
                operation="VALIDATE",
                is_valid=False,
                error=str(e),
                processing_time_ms=int((time.time() - start_time) * 1000),
            )

    def _check_is_past(
        self, request: DateTimeRequest, start_time: float
    ) -> DateTimeResponse:
        """Check if a date is in the past."""
        try:
            parsed_dt = self._parse_date_string(request.date_string)

            # Get reference date (current date or provided reference)
            if request.reference_date:
                reference_dt = self._parse_date_string(request.reference_date)
            else:
                reference_dt = datetime.now()

            parsed_date = parsed_dt.date()
            reference_date = reference_dt.date()

            is_past = parsed_date < reference_date
            days_diff = (parsed_date - reference_date).days

            return DateTimeResponse(
                operation="IS_PAST",
                parsed_date=parsed_date.isoformat(),
                is_valid=True,
                is_past=is_past,
                days_difference=days_diff,
                formatted_date=parsed_dt.strftime("%B %d, %Y"),
                processing_time_ms=int((time.time() - start_time) * 1000),
                message=f"Date {parsed_date.isoformat()} is {'in the past' if is_past else 'not in the past'}",
            )
        except ValueError as e:
            return DateTimeResponse(
                operation="IS_PAST",
                is_valid=False,
                error=str(e),
                processing_time_ms=int((time.time() - start_time) * 1000),
            )

    def _check_is_future(
        self, request: DateTimeRequest, start_time: float
    ) -> DateTimeResponse:
        """Check if a date is in the future."""
        try:
            parsed_dt = self._parse_date_string(request.date_string)

            # Get reference date (current date or provided reference)
            if request.reference_date:
                reference_dt = self._parse_date_string(request.reference_date)
            else:
                reference_dt = datetime.now()

            parsed_date = parsed_dt.date()
            reference_date = reference_dt.date()

            is_future = parsed_date > reference_date
            days_diff = (parsed_date - reference_date).days

            return DateTimeResponse(
                operation="IS_FUTURE",
                parsed_date=parsed_date.isoformat(),
                is_valid=True,
                is_future=is_future,
                days_difference=days_diff,
                formatted_date=parsed_dt.strftime("%B %d, %Y"),
                processing_time_ms=int((time.time() - start_time) * 1000),
                message=f"Date {parsed_date.isoformat()} is {'in the future' if is_future else 'not in the future'}",
            )
        except ValueError as e:
            return DateTimeResponse(
                operation="IS_FUTURE",
                is_valid=False,
                error=str(e),
                processing_time_ms=int((time.time() - start_time) * 1000),
            )

    def _compare_dates(
        self, request: DateTimeRequest, start_time: float
    ) -> DateTimeResponse:
        """Compare two dates."""
        try:
            parsed_dt1 = self._parse_date_string(request.date_string)

            if request.compare_date:
                parsed_dt2 = self._parse_date_string(request.compare_date)
            else:
                parsed_dt2 = datetime.now()

            date1 = parsed_dt1.date()
            date2 = parsed_dt2.date()

            days_diff = (date1 - date2).days

            if date1 < date2:
                comparison_result = "earlier"
            elif date1 > date2:
                comparison_result = "later"
            else:
                comparison_result = "same"

            return DateTimeResponse(
                operation="COMPARE",
                parsed_date=date1.isoformat(),
                is_valid=True,
                comparison_result=comparison_result,
                days_difference=days_diff,
                formatted_date=parsed_dt1.strftime("%B %d, %Y"),
                processing_time_ms=int((time.time() - start_time) * 1000),
                message=f"Date {date1.isoformat()} is {comparison_result} than {date2.isoformat()}",
            )
        except ValueError as e:
            return DateTimeResponse(
                operation="COMPARE",
                is_valid=False,
                error=str(e),
                processing_time_ms=int((time.time() - start_time) * 1000),
            )

    def _calculate_difference(
        self, request: DateTimeRequest, start_time: float
    ) -> DateTimeResponse:
        """Calculate the difference between two dates."""
        try:
            parsed_dt1 = self._parse_date_string(request.date_string)

            if request.compare_date:
                parsed_dt2 = self._parse_date_string(request.compare_date)
            else:
                parsed_dt2 = datetime.now()

            date1 = parsed_dt1.date()
            date2 = parsed_dt2.date()

            days_diff = (date1 - date2).days
            abs_days = abs(days_diff)

            # Create human-readable message
            if days_diff == 0:
                message = "The dates are the same"
            elif days_diff > 0:
                message = f"First date is {abs_days} day{'s' if abs_days != 1 else ''} after the second date"
            else:
                message = f"First date is {abs_days} day{'s' if abs_days != 1 else ''} before the second date"

            return DateTimeResponse(
                operation="DIFFERENCE",
                parsed_date=date1.isoformat(),
                is_valid=True,
                days_difference=days_diff,
                formatted_date=parsed_dt1.strftime("%B %d, %Y"),
                processing_time_ms=int((time.time() - start_time) * 1000),
                message=message,
            )
        except ValueError as e:
            return DateTimeResponse(
                operation="DIFFERENCE",
                is_valid=False,
                error=str(e),
                processing_time_ms=int((time.time() - start_time) * 1000),
            )
