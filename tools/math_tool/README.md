# Math Tool

A tool for performing mathematical calculations with support for arithmetic operations, functions, constants, and spoken math input using SymPy for safe evaluation.

## Features

- **Basic arithmetic operations** (+, -, *, /, %, **)
- **Mathematical functions** (sqrt, sin, cos, tan, log, exp)
- **Constants** (pi, e)
- **Spoken math support** ("one plus two", "five times three")
- **Safe expression evaluation** using SymPy
- Built-in error handling and validation
- Processing time tracking
- Integer detection and formatting

## Agent Integration

```python
from generative_ai_toolkit.agent import BedrockConverseAgent
from math_tool import MathTool

# Create agent with Math tool
agent = BedrockConverseAgent(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
)
agent.register_tool(MathTool())

# Chat with agent about calculations
response = agent.converse("What is 15% of 200?")
response = agent.converse("Calculate the square root of 144")
response = agent.converse("What is two plus three times four?")
```

## Usage

### Basic Operations

```python
from math_tool import MathTool

tool = MathTool()

# Simple arithmetic
result = tool.invoke(expression="1+2")

# Complex expressions
result = tool.invoke(expression="(10 + 5) * 2 - 3")

# Mathematical functions
result = tool.invoke(expression="sqrt(16) + sin(pi/2)")

# Spoken math
result = tool.invoke(expression="one plus two")

# With description
result = tool.invoke(expression="100 * 0.15", description="Calculate 15% of 100")
```

### Response Format

All calculations return a structured response:

```python
{
    "result": float,              # numerical result
    "expression": str,            # original expression
    "is_integer": bool,           # whether result is whole number
    "formatted_result": str,      # clean string representation
    "processing_time_ms": int,    # calculation time
    "description": str,           # optional description
    "error": str                  # error message if failed
}
```

## Supported Operations

### Arithmetic Operations
- Addition: `+`
- Subtraction: `-`
- Multiplication: `*`
- Division: `/`
- Modulo: `%`
- Power: `**`

### Mathematical Functions
- Square root: `sqrt(x)`
- Trigonometric: `sin(x)`, `cos(x)`, `tan(x)`
- Logarithmic: `log(x)`
- Exponential: `exp(x)`

### Constants
- Pi: `pi`
- Euler's number: `e`

### Spoken Math
The tool converts natural language to mathematical expressions:

```python
# Word numbers
"one plus two" → "1 + 2"
"fifteen minus seven" → "15 - 7"

# Operations
"times" → "*"
"divided by" → "/"
"to the power of" → "**"

# Statements
"five times three equals fifteen" → "5 * 3"
```

## Examples

### Basic Calculations
```python
tool.invoke(expression="2 + 3")
# {"result": 5.0, "is_integer": True, "formatted_result": "5"}

tool.invoke(expression="10 / 3")
# {"result": 3.333..., "is_integer": False, "formatted_result": "3.333..."}
```

### Advanced Math
```python
tool.invoke(expression="sqrt(144)")
# {"result": 12.0, "is_integer": True, "formatted_result": "12"}

tool.invoke(expression="sin(pi/2)")
# {"result": 1.0, "is_integer": True, "formatted_result": "1"}
```

### Spoken Math
```python
tool.invoke(expression="twenty minus five")
# {"result": 15.0, "is_integer": True, "formatted_result": "15"}

tool.invoke(expression="three squared")
# {"result": 9.0, "is_integer": True, "formatted_result": "9"}
```

## Tool Specification

The tool implements the standard tool interface with:

- **Name**: `calculate_math`
- **Input Schema**: Validates expression and optional description
- **Safe Evaluation**: Uses SymPy for secure mathematical parsing

## Error Handling

The tool handles various error conditions:

- Empty expressions
- Invalid mathematical syntax
- Expressions that don't evaluate to numbers
- Calculation errors and exceptions

## Testing

Run the unit tests:

```bash
pytest test/test_math_tool_unit.py
```

Tests cover all operations, spoken math conversion, error handling, and edge cases including:
- Basic arithmetic operations
- Complex mathematical expressions
- Mathematical functions and constants
- Spoken math parsing
- Error conditions and validation