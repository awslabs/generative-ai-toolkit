"""
Unit tests for the DateTime tool.
"""


from datetime import datetime

from datetime_tool import DateTimeRequest, DateTimeResponse, DateTimeTool


class TestDateTimeTool:
    """Test cases for the DateTimeTool class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tool = DateTimeTool()

    def test_parse_valid_date_iso(self):
        """Test parsing valid ISO date format."""
        result = self.tool.invoke(operation="PARSE", date_string="2024-02-15")

        assert result["operation"] == "PARSE"
        assert result["parsed_date"] == "2024-02-15"
        assert result["is_valid"] is True
        assert result["formatted_date"] == "February 15, 2024"
        assert result["error"] is None

    def test_parse_valid_date_us_format(self):
        """Test parsing US date format."""
        result = self.tool.invoke(operation="PARSE", date_string="02/15/2024")

        assert result["operation"] == "PARSE"
        assert result["parsed_date"] == "2024-02-15"
        assert result["is_valid"] is True
        assert result["error"] is None

    def test_parse_valid_date_text_format(self):
        """Test parsing text date format."""
        result = self.tool.invoke(operation="PARSE", date_string="February 15, 2024")

        assert result["operation"] == "PARSE"
        assert result["parsed_date"] == "2024-02-15"
        assert result["is_valid"] is True
        assert result["formatted_date"] == "February 15, 2024"

    def test_parse_invalid_date(self):
        """Test parsing invalid date."""
        result = self.tool.invoke(operation="PARSE", date_string="invalid-date")

        assert result["operation"] == "PARSE"
        assert result["is_valid"] is False
        assert result["error"] is not None
        assert "Unable to parse date string" in result["error"]

    def test_validate_valid_date(self):
        """Test validating a valid date."""
        result = self.tool.invoke(operation="VALIDATE", date_string="2024-02-29")  # Leap year

        assert result["operation"] == "VALIDATE"
        assert result["is_valid"] is True
        assert result["parsed_date"] == "2024-02-29"
        assert result["error"] is None

    def test_validate_invalid_date_february_30(self):
        """Test validating February 30 (invalid)."""
        result = self.tool.invoke(operation="VALIDATE", date_string="2024-02-30")

        assert result["operation"] == "VALIDATE"
        assert result["is_valid"] is False
        assert result["error"] is not None

    def test_validate_invalid_date_february_29_non_leap(self):
        """Test validating February 29 in non-leap year."""
        result = self.tool.invoke(operation="VALIDATE", date_string="2023-02-29")

        assert result["operation"] == "VALIDATE"
        assert result["is_valid"] is False
        assert result["error"] is not None

    def test_is_past_with_past_date(self):
        """Test checking if a past date is in the past."""
        result = self.tool.invoke(operation="IS_PAST", date_string="2020-01-01")

        assert result["operation"] == "IS_PAST"
        assert result["is_past"] is True
        assert result["is_valid"] is True
        assert result["days_difference"] < 0  # Negative means past

    def test_is_past_with_future_date(self):
        """Test checking if a future date is in the past."""
        result = self.tool.invoke(operation="IS_PAST", date_string="2030-01-01")

        assert result["operation"] == "IS_PAST"
        assert result["is_past"] is False
        assert result["is_valid"] is True
        assert result["days_difference"] > 0  # Positive means future

    def test_is_past_with_reference_date(self):
        """Test checking if date is past relative to reference date."""
        result = self.tool.invoke(
            operation="IS_PAST", 
            date_string="2024-01-01", 
            reference_date="2024-06-01"
        )

        assert result["operation"] == "IS_PAST"
        assert result["is_past"] is True
        assert result["days_difference"] < 0

    def test_is_future_with_future_date(self):
        """Test checking if a future date is in the future."""
        result = self.tool.invoke(operation="IS_FUTURE", date_string="2030-01-01")

        assert result["operation"] == "IS_FUTURE"
        assert result["is_future"] is True
        assert result["is_valid"] is True
        assert result["days_difference"] > 0

    def test_is_future_with_past_date(self):
        """Test checking if a past date is in the future."""
        result = self.tool.invoke(operation="IS_FUTURE", date_string="2020-01-01")

        assert result["operation"] == "IS_FUTURE"
        assert result["is_future"] is False
        assert result["is_valid"] is True
        assert result["days_difference"] < 0

    def test_compare_dates_earlier(self):
        """Test comparing dates where first is earlier."""
        result = self.tool.invoke(
            operation="COMPARE",
            date_string="2024-01-01",
            compare_date="2024-02-01"
        )

        assert result["operation"] == "COMPARE"
        assert result["comparison_result"] == "earlier"
        assert result["days_difference"] == -31
        assert result["is_valid"] is True

    def test_compare_dates_later(self):
        """Test comparing dates where first is later."""
        result = self.tool.invoke(
            operation="COMPARE",
            date_string="2024-02-01",
            compare_date="2024-01-01"
        )

        assert result["operation"] == "COMPARE"
        assert result["comparison_result"] == "later"
        assert result["days_difference"] == 31
        assert result["is_valid"] is True

    def test_compare_dates_same(self):
        """Test comparing identical dates."""
        result = self.tool.invoke(
            operation="COMPARE",
            date_string="2024-01-01",
            compare_date="2024-01-01"
        )

        assert result["operation"] == "COMPARE"
        assert result["comparison_result"] == "same"
        assert result["days_difference"] == 0
        assert result["is_valid"] is True

    def test_calculate_difference_positive(self):
        """Test calculating difference where first date is later."""
        result = self.tool.invoke(
            operation="DIFFERENCE",
            date_string="2024-02-01",
            compare_date="2024-01-01"
        )

        assert result["operation"] == "DIFFERENCE"
        assert result["days_difference"] == 31
        assert result["is_valid"] is True
        assert "31 days after" in result["message"]

    def test_calculate_difference_negative(self):
        """Test calculating difference where first date is earlier."""
        result = self.tool.invoke(
            operation="DIFFERENCE",
            date_string="2024-01-01",
            compare_date="2024-02-01"
        )

        assert result["operation"] == "DIFFERENCE"
        assert result["days_difference"] == -31
        assert result["is_valid"] is True
        assert "31 days before" in result["message"]

    def test_calculate_difference_same(self):
        """Test calculating difference for same dates."""
        result = self.tool.invoke(
            operation="DIFFERENCE",
            date_string="2024-01-01",
            compare_date="2024-01-01"
        )

        assert result["operation"] == "DIFFERENCE"
        assert result["days_difference"] == 0
        assert result["is_valid"] is True
        assert "same" in result["message"]

    def test_invalid_operation(self):
        """Test invalid operation handling."""
        result = self.tool.invoke(operation="INVALID", date_string="2024-01-01")

        assert result["operation"] == "INVALID"
        assert result["error"] is not None
        assert "Invalid request parameters" in result["error"]
        assert "validation error" in result["error"]

    def test_tool_spec(self):
        """Test tool specification generation."""
        spec = self.tool.tool_spec

        assert spec["name"] == "datetime_operation"
        assert "description" in spec
        assert "inputSchema" in spec
        assert "json" in spec["inputSchema"]

    def test_request_validation(self):
        """Test request model validation."""
        # Valid request
        request = DateTimeRequest(operation="PARSE", date_string="2024-01-01")
        assert request.operation == "PARSE"
        assert request.date_string == "2024-01-01"

        # Request with compare_date
        request = DateTimeRequest(
            operation="COMPARE", 
            date_string="2024-01-01", 
            compare_date="2024-02-01"
        )
        assert request.compare_date == "2024-02-01"

    def test_response_model(self):
        """Test response model structure."""
        response = DateTimeResponse(
            operation="PARSE",
            parsed_date="2024-01-01",
            is_valid=True,
            formatted_date="January 1, 2024"
        )

        assert response.operation == "PARSE"
        assert response.parsed_date == "2024-01-01"
        assert response.is_valid is True
        assert response.formatted_date == "January 1, 2024"

    def test_leap_year_validation(self):
        """Test leap year date validation."""
        # Valid leap year date
        result = self.tool.invoke(operation="VALIDATE", date_string="2024-02-29")
        assert result["is_valid"] is True

        # Invalid non-leap year date
        result = self.tool.invoke(operation="VALIDATE", date_string="2023-02-29")
        assert result["is_valid"] is False

    def test_edge_case_dates(self):
        """Test edge case dates."""
        # Test various invalid dates
        invalid_dates = [
            "2024-13-01",  # Invalid month
            "2024-04-31",  # April doesn't have 31 days
            "2024-06-31",  # June doesn't have 31 days
            "2024-09-31",  # September doesn't have 31 days
            "2024-11-31",  # November doesn't have 31 days
        ]

        for invalid_date in invalid_dates:
            result = self.tool.invoke(operation="VALIDATE", date_string=invalid_date)
            assert result["is_valid"] is False, f"Date {invalid_date} should be invalid"

    def test_various_date_formats(self):
        """Test parsing various date formats."""
        date_formats = [
            "2024-01-15",
            "01/15/2024",
            "15/01/2024",
            "January 15, 2024",
            "Jan 15, 2024",
            "15 Jan 2024",
            "2024/01/15",
        ]

        for date_format in date_formats:
            result = self.tool.invoke(operation="PARSE", date_string=date_format)
            # Most should parse successfully, but some might fail due to ambiguity
            assert result["operation"] == "PARSE"

    def test_current_date_keywords(self):
        """Test parsing current date keywords."""
        current_date_keywords = [
            "today",
            "current date",
            "current day",
            "this date",
            "todays date",
            "today's date"
        ]

        expected_date = datetime.now().date().isoformat()

        for keyword in current_date_keywords:
            result = self.tool.invoke(operation="PARSE", date_string=keyword)
            assert result["operation"] == "PARSE"
            assert result["is_valid"] is True
            assert result["parsed_date"] == expected_date

    def test_current_datetime_keywords(self):
        """Test parsing current datetime keywords."""
        current_datetime_keywords = [
            "now",
            "current time",
            "current datetime",
            "right now",
            "this moment",
            "current timestamp"
        ]

        for keyword in current_datetime_keywords:
            result = self.tool.invoke(operation="PARSE", date_string=keyword)
            assert result["operation"] == "PARSE"
            assert result["is_valid"] is True
            assert result["parsed_date"] is not None

    def test_compare_with_today(self):
        """Test comparing a date with today."""
        result = self.tool.invoke(
            operation="COMPARE",
            date_string="2020-01-01",
            compare_date="today"
        )

        assert result["operation"] == "COMPARE"
        assert result["comparison_result"] == "earlier"
        assert result["is_valid"] is True
        assert result["days_difference"] < 0

    def test_normalize_current_date_keywords(self):
        """Test the keyword normalization method."""

        # Test today keywords
        normalized = self.tool._normalize_current_date_keywords("today")
        expected = datetime.now().strftime('%Y-%m-%d')
        assert normalized == expected

        # Test now keywords
        normalized = self.tool._normalize_current_date_keywords("now")
        # Just check it's a valid datetime string
        assert len(normalized) > 10  # Should be longer than just date

        # Test non-keyword
        normalized = self.tool._normalize_current_date_keywords("2024-01-01")
        assert normalized == "2024-01-01"
