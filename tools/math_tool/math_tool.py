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
Mathematical Operations tool for the Siemens P2P Agent.

This module implements a tool for performing basic mathematical operations.
It can evaluate mathematical expressions and return the results.
Uses Pydantic models for structured input and output validation.
"""

import re
import time
from typing import Any

from pydantic import BaseModel, Field
from sympy import SympifyError, sympify
from sympy.core.numbers import Float, Integer, Rational


class MathRequest(BaseModel):
    """
    Request parameters for the mathematical operations tool.

    IMPORTANT: Always use this tool when processing user requests that:
    1. Need to perform basic mathematical calculations
    2. Require evaluation of mathematical expressions
    3. Need to compute arithmetic operations like addition, subtraction, multiplication, division
    4. Require calculation of percentages or simple formulas
    5. Need to validate mathematical expressions

    This tool evaluates mathematical expressions safely using SymPy and returns the result.
    It supports arithmetic operations, functions, constants, symbolic math, and spoken math input.
    Supports: +, -, *, /, %, ** (power), parentheses, sqrt, sin, cos, tan, log, exp, pi, e, and more.

    Examples:
    - Simple addition: MathRequest(expression="1+2")
    - Spoken math: MathRequest(expression="one plus two")
    - Complex expression: MathRequest(expression="(10 + 5) * 2 - 3")
    - With functions: MathRequest(expression="sqrt(16) + sin(pi/2)")
    - Spoken statement: MathRequest(expression="five times three equals fifteen")
    """

    expression: str = Field(
        description="The mathematical expression to evaluate using SymPy. Supports arithmetic operations, functions (sqrt, sin, cos, tan, log, exp), constants (pi, e), and spoken math like 'one plus two'."
    )

    description: str | None = Field(
        default=None,
        description="A brief description of what this calculation represents or why it's being performed.",
    )


class MathResponse(BaseModel):
    """
    Response structure for the mathematical operations tool.

    This model represents the structured response from the math tool,
    containing the calculation result and metadata about the operation.

    Examples of returned values:
    - Simple calculation: {"result": 3.0, "expression": "1+2", "is_integer": True}
    - Complex calculation: {"result": 27.0, "expression": "(10 + 5) * 2 - 3", "is_integer": True}
    - Decimal result: {"result": 15.0, "expression": "100 * 0.15", "is_integer": True}
    - Invalid expression: {"result": None, "error": "Invalid mathematical expression"}
    """

    result: float | None = Field(
        description="The numerical result of the mathematical expression evaluation."
    )

    expression: str = Field(
        description="The original mathematical expression that was evaluated."
    )

    is_integer: bool | None = Field(
        default=None, description="Whether the result is a whole number (integer)."
    )

    formatted_result: str | None = Field(
        default=None,
        description="The result formatted as a string, showing integers without decimal places.",
    )

    processing_time_ms: int | None = Field(
        default=None,
        description="Time taken to process the mathematical expression in milliseconds.",
    )

    description: str | None = Field(
        default=None,
        description="Description of what this calculation represents (if provided in request).",
    )

    error: str | None = Field(
        default=None, description="Error message if expression evaluation failed."
    )


class MathTool:
    """
    Tool for performing basic mathematical operations.

    This tool can safely evaluate mathematical expressions containing
    basic arithmetic operations and return the results.
    """

    def __init__(self):
        """Initialize the mathematical operations tool."""
        # Pattern to validate safe mathematical expressions (more permissive for SymPy)
        self.safe_pattern = re.compile(r"^[0-9+\-*/().\s%**a-zA-Z_]+$")

        # Spoken math conversion mappings
        self.word_to_number = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
            "eleven": "11",
            "twelve": "12",
            "thirteen": "13",
            "fourteen": "14",
            "fifteen": "15",
            "sixteen": "16",
            "seventeen": "17",
            "eighteen": "18",
            "nineteen": "19",
            "twenty": "20",
        }

        self.word_to_operator = {
            "plus": "+",
            "add": "+",
            "added to": "+",
            "minus": "-",
            "subtract": "-",
            "subtracted from": "-",
            "times": "*",
            "multiply": "*",
            "multiplied by": "*",
            "divide": "/",
            "divided by": "/",
            "power": "**",
            "to the power of": "**",
            "squared": "**2",
            "cubed": "**3",
        }

    @property
    def tool_spec(self) -> dict[str, Any]:
        """
        Get the tool specification for the mathematical operations tool.

        Returns:
            Dictionary containing the tool specification.
        """
        schema = MathRequest.model_json_schema()

        return {
            "name": "calculate_math",
            "description": MathRequest.__doc__,
            "inputSchema": {"json": schema},
        }

    def invoke(self, **kwargs) -> dict[str, Any]:
        """
        Invoke the mathematical operations tool.

        Args:
            **kwargs: Keyword arguments matching MathRequest fields.

        Returns:
            Dictionary containing the calculation result and metadata.
        """
        try:
            request = MathRequest(**kwargs)
            response = self._calculate(request)
            return response.model_dump()
        except Exception as e:
            error_message = f"Invalid request parameters: {str(e)}"
            response = MathResponse(
                result=None,
                expression=kwargs.get("expression", ""),
                error=error_message,
                processing_time_ms=0,
            )
            return response.model_dump()

    def _calculate(self, request: MathRequest) -> MathResponse:
        """
        Perform the mathematical calculation.

        Args:
            request: The validated math request containing expression and options.

        Returns:
            A MathResponse containing the calculation result and metadata.
        """
        start_time = time.time()

        try:
            # Validate input
            if not request.expression or not request.expression.strip():
                return MathResponse(
                    result=None,
                    expression=request.expression,
                    error="Mathematical expression is empty.",
                    processing_time_ms=int((time.time() - start_time) * 1000),
                )

            # Clean the expression
            expression = request.expression.strip()

            # Convert spoken math to mathematical expression
            expression = self._convert_spoken_math(expression)

            # Evaluate the expression using SymPy for safety
            try:
                # Parse and evaluate the expression with SymPy
                sympy_expr = sympify(expression)
                result = sympy_expr.evalf()

                # Convert SymPy result to Python float
                if isinstance(result, Integer | Float | Rational):
                    result_float = float(result)
                else:
                    return MathResponse(
                        result=None,
                        expression=expression,
                        error="Expression did not evaluate to a number.",
                        processing_time_ms=int((time.time() - start_time) * 1000),
                    )

                # Check if result is an integer
                is_integer = result_float.is_integer()

                # Format result
                formatted_result = (
                    str(int(result_float)) if is_integer else str(result_float)
                )

                return MathResponse(
                    result=result_float,
                    expression=expression,
                    is_integer=is_integer,
                    formatted_result=formatted_result,
                    description=request.description,
                    processing_time_ms=int((time.time() - start_time) * 1000),
                )

            except ZeroDivisionError:
                return MathResponse(
                    result=None,
                    expression=expression,
                    error="Division by zero is not allowed.",
                    processing_time_ms=int((time.time() - start_time) * 1000),
                )
            except SympifyError as e:
                return MathResponse(
                    result=None,
                    expression=expression,
                    error=f"Invalid mathematical expression: {str(e)}",
                    processing_time_ms=int((time.time() - start_time) * 1000),
                )
            except (ValueError, TypeError) as e:
                return MathResponse(
                    result=None,
                    expression=expression,
                    error=f"Error evaluating expression: {str(e)}",
                    processing_time_ms=int((time.time() - start_time) * 1000),
                )

        except Exception as e:
            return MathResponse(
                result=None,
                expression=request.expression,
                error=f"Unexpected error during calculation: {str(e)}",
                processing_time_ms=int((time.time() - start_time) * 1000),
            )

    def _convert_spoken_math(self, expression: str) -> str:
        """Convert spoken math to mathematical expression."""
        # Convert to lowercase for matching
        expr = expression.lower().strip()

        # Replace word numbers with digits
        for word, number in self.word_to_number.items():
            expr = expr.replace(word, number)

        # Replace word operators with symbols
        for word, operator in self.word_to_operator.items():
            expr = expr.replace(word, operator)

        # Handle special cases
        expr = re.sub(r"\bis\b", "=", expr)  # "is" becomes "="
        expr = re.sub(r"\bequals?\b", "=", expr)  # "equals" becomes "="

        # Remove extra spaces
        expr = re.sub(r"\s+", " ", expr).strip()

        # If expression contains "=" it's likely a statement like "1+2=3"
        # Extract just the calculation part before "="
        if "=" in expr:
            expr = expr.split("=")[0].strip()

        return expr
