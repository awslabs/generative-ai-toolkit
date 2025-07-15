# DateTime Tool

A tool for parsing, validating, and comparing dates with support for various date formats and temporal operations.

## Features

- **Parse dates** from various formats (ISO, US, text formats)
- **Validate dates** for correctness and edge cases
- **Check temporal relationships** (past/future relative to reference dates)
- **Compare dates** and calculate differences
- **Handle natural language** date keywords ("today", "now", etc.)
- Built-in error handling and validation
- Processing time tracking
- Flexible date format support

## Agent Integration

```python
from generative_ai_toolkit.agent import BedrockConverseAgent
from datetime_tool import DateTimeTool

# Create agent with DateTime tool
agent = BedrockConverseAgent(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
)
agent.register_tool(DateTimeTool())

# Chat with agent about date operations
response = agent.converse("Is February 29, 2024 a valid date?")
response = agent.converse("How many days until Christmas 2024?")
response = agent.converse("Parse the date '02/15/2024' and format it nicely")
```

## Usage

### Basic Operations

```python
from datetime_tool import DateTimeTool

tool = DateTimeTool()

# Parse a date
result = tool.invoke(operation="PARSE", date_string="2024-02-15")

# Validate a date
result = tool.invoke(operation="VALIDATE", date_string="2024-02-29")

# Check if date is in the past
result = tool.invoke(operation="IS_PAST", date_string="2020-01-01")

# Check if date is in the future
result = tool.invoke(operation="IS_FUTURE", date_string="2030-01-01")

# Compare two dates
result = tool.invoke(
    operation="COMPARE",
    date_string="2024-01-01",
    compare_date="2024-02-01"
)

# Calculate difference between dates
result = tool.invoke(
    operation="DIFFERENCE",
    date_string="2024-02-01",
    compare_date="2024-01-01"
)
```

### Supported Date Formats

- **ISO format**: `2024-02-15`
- **US format**: `02/15/2024`
- **Text format**: `February 15, 2024`
- **Natural language**: `today`, `now`, `current date`

### Response Format

All operations return a structured response:

```python
{
    "operation": str,
    "parsed_date": str,        # ISO format (YYYY-MM-DD)
    "is_valid": bool,
    "is_past": bool,           # for IS_PAST operation
    "is_future": bool,         # for IS_FUTURE operation
    "comparison_result": str,   # "earlier", "later", "same"
    "days_difference": int,     # negative = earlier, positive = later
    "formatted_date": str,      # human-readable format
    "processing_time_ms": int,
    "message": str,
    "error": str               # if operation failed
}
```

## Operations

### PARSE
Parses a date string and returns the standardized ISO format.

### VALIDATE
Validates if a date string represents a valid date (handles leap years, month boundaries, etc.).

### IS_PAST
Checks if a date is in the past relative to current date or a reference date.

### IS_FUTURE
Checks if a date is in the future relative to current date or a reference date.

### COMPARE
Compares two dates and returns whether the first is "earlier", "later", or "same".

### DIFFERENCE
Calculates the difference in days between two dates.

## Tool Specification

The tool implements the standard tool interface with:

- **Name**: `datetime_operation`
- **Input Schema**: Validates operation, date_string, compare_date, and reference_date parameters
- **Operations**: `PARSE`, `VALIDATE`, `IS_PAST`, `IS_FUTURE`, `COMPARE`, `DIFFERENCE`

## Error Handling

The tool handles various error conditions:

- Invalid date formats
- Invalid operations
- Missing required parameters
- Edge cases (leap years, invalid dates like February 30)
- Parsing errors for malformed date strings

## Testing

Run the unit tests:

```bash
pytest test/test_datetime_tool_unit.py
```

Tests cover all operations, date formats, edge cases, and error conditions.