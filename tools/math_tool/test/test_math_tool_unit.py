"""
Unit tests for the Mathematical Operations tool.
"""

from ..math_tool import MathRequest, MathResponse, MathTool


class TestMathTool:
    """Test cases for the MathTool class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tool = MathTool()

    def test_simple_addition(self):
        """Test simple addition operation."""
        result = self.tool.invoke(expression="1+2")

        assert result["result"] == 3.0
        assert result["expression"] == "1+2"
        assert result["is_integer"] is True
        assert result["formatted_result"] == "3"
        assert result["error"] is None

    def test_complex_expression(self):
        """Test complex mathematical expression."""
        result = self.tool.invoke(expression="(10 + 5) * 2 - 3")

        assert result["result"] == 27.0
        assert result["expression"] == "(10 + 5) * 2 - 3"
        assert result["is_integer"] is True
        assert result["formatted_result"] == "27"

    def test_decimal_result(self):
        """Test expression that results in decimal."""
        result = self.tool.invoke(expression="10 / 3")

        assert abs(result["result"] - 3.3333333333333335) < 0.0000001
        assert result["is_integer"] is False
        assert result["error"] is None

    def test_percentage_calculation(self):
        """Test percentage calculation."""
        result = self.tool.invoke(
            expression="100 * 0.15", description="Calculate 15% of 100"
        )

        assert result["result"] == 15.0
        assert result["is_integer"] is True
        assert result["formatted_result"] == "15"
        assert result["description"] == "Calculate 15% of 100"

    def test_power_operation(self):
        """Test power operation."""
        result = self.tool.invoke(expression="2**3")

        assert result["result"] == 8.0
        assert result["is_integer"] is True
        assert result["formatted_result"] == "8"

    def test_modulo_operation(self):
        """Test modulo operation."""
        result = self.tool.invoke(expression="10 % 3")

        assert result["result"] == 1.0
        assert result["is_integer"] is True
        assert result["formatted_result"] == "1"

    def test_sqrt_function(self):
        """Test square root function with SymPy."""
        result = self.tool.invoke(expression="sqrt(16)")

        assert result["result"] == 4.0
        assert result["is_integer"] is True
        assert result["formatted_result"] == "4"

    def test_trigonometric_function(self):
        """Test trigonometric function with SymPy."""
        result = self.tool.invoke(expression="sin(pi/2)")

        assert result["result"] == 1.0
        assert result["is_integer"] is True
        assert result["formatted_result"] == "1"

    def test_pi_constant(self):
        """Test pi constant with SymPy."""
        result = self.tool.invoke(expression="pi")

        assert abs(result["result"] - 3.14159265358979) < 0.0000001
        assert result["is_integer"] is False

    def test_exponential_function(self):
        """Test exponential function with SymPy."""
        result = self.tool.invoke(expression="exp(0)")

        assert result["result"] == 1.0
        assert result["is_integer"] is True
        assert result["formatted_result"] == "1"

    def test_spoken_math_simple(self):
        """Test simple spoken math."""
        result = self.tool.invoke(expression="one plus two")

        assert result["result"] == 3.0
        assert result["is_integer"] is True
        assert result["formatted_result"] == "3"

    def test_spoken_math_statement(self):
        """Test spoken math statement with equals."""
        result = self.tool.invoke(expression="five times three equals fifteen")

        assert result["result"] == 15.0
        assert result["is_integer"] is True
        assert result["formatted_result"] == "15"

    def test_spoken_math_complex(self):
        """Test complex spoken math."""
        result = self.tool.invoke(expression="ten minus four plus two")

        assert result["result"] == 8.0
        assert result["is_integer"] is True
        assert result["formatted_result"] == "8"

    def test_empty_expression(self):
        """Test empty expression handling."""
        result = self.tool.invoke(expression="")

        assert result["result"] is None
        assert result["error"] == "Mathematical expression is empty."

    def test_invalid_expression(self):
        """Test invalid mathematical expression."""
        result = self.tool.invoke(expression="1 + +")

        assert result["result"] is None
        assert "invalid mathematical expression" in result["error"].lower()

    def test_division_by_zero(self):
        """Test division by zero handling."""
        result = self.tool.invoke(expression="10 / 0")

        assert result["result"] is None
        assert "Expression did not evaluate to a number" in result["error"]

    def test_invalid_syntax(self):
        """Test invalid mathematical syntax."""
        result = self.tool.invoke(expression="1 + + 2")

        # SymPy actually evaluates "1 + + 2" as "1 + 2" = 3
        assert result["result"] == 3.0
        assert result["error"] is None

    def test_tool_spec(self):
        """Test tool specification generation."""
        spec = self.tool.tool_spec

        assert spec["name"] == "calculate_math"
        assert "description" in spec
        assert "inputSchema" in spec
        assert "json" in spec["inputSchema"]

    def test_request_validation(self):
        """Test request model validation."""
        # Valid request
        request = MathRequest(expression="1+1")
        assert request.expression == "1+1"
        assert request.description is None

        # Request with description
        request = MathRequest(expression="2*3", description="Test calculation")
        assert request.expression == "2*3"
        assert request.description == "Test calculation"

    def test_response_model(self):
        """Test response model structure."""
        response = MathResponse(
            result=5.0, expression="2+3", is_integer=True, formatted_result="5"
        )

        assert response.result == 5.0
        assert response.expression == "2+3"
        assert response.is_integer is True
        assert response.formatted_result == "5"
        assert response.error is None

    def test_convert_spoken_math(self):
        """Test spoken math conversion method."""
        # Test direct conversion
        converted = self.tool._convert_spoken_math("one plus two")
        assert converted == "1 + 2"

        # Test with equals
        converted = self.tool._convert_spoken_math("three times four equals twelve")
        assert converted == "3 * 4"

        # Test mixed case
        converted = self.tool._convert_spoken_math("Five MINUS Two")
        assert converted == "5 - 2"
