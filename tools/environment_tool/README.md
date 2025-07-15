# Environment Tool

A tool for managing environment variables at runtime with security protections for sensitive system variables.

## Features

- **List environment variables** with optional prefix filtering
- **Get specific variable values** with security masking
- **Set new variables** or update existing ones
- **Delete variables** with protection for critical system variables
- **Security protections** for sensitive variables (API keys, secrets, tokens)
- Built-in error handling and validation
- Processing time tracking
- Force override for protected operations

## Agent Integration

```python
from generative_ai_toolkit.agent import BedrockConverseAgent
from environment_tool import EnvironmentTool

# Create agent with Environment tool
agent = BedrockConverseAgent(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
)
agent.register_tool(EnvironmentTool())

# Chat with agent about environment variables
response = agent.converse("List all AWS environment variables")
response = agent.converse("What is the value of PATH?")
response = agent.converse("Set DEBUG_MODE to true")
```

## Usage

### Basic Operations

```python
from environment_tool import EnvironmentTool

tool = EnvironmentTool()

# List all environment variables
result = tool.invoke(action="list")

# List with prefix filter
result = tool.invoke(action="list", prefix="AWS_")

# Get specific variable
result = tool.invoke(action="get", name="PATH")

# Set new variable
result = tool.invoke(action="set", name="DEBUG_MODE", value="true")

# Delete variable
result = tool.invoke(action="delete", name="TEMP_VAR")

# Force operations on protected variables
result = tool.invoke(action="set", name="PATH", value="/new/path", force=True)
```

### Response Format

All operations return a structured response:

```python
{
    "success": bool,
    "action": str,
    "variables": dict,        # for list operations
    "value": str,            # for get operations
    "protected_count": int,   # number of protected variables
    "processing_time_ms": int,
    "message": str,
    "error": str             # if operation failed
}
```

## Operations

### LIST
Lists all environment variables or those matching a prefix filter. Sensitive variables are masked with `***PROTECTED***`.

### GET
Retrieves the value of a specific environment variable. Protected variables show masked values.

### SET
Sets or updates an environment variable. Protected variables require `force=True`.

### DELETE
Removes an environment variable. Protected variables require `force=True`.

## Security Features

### Protected Variables
The following variables are automatically protected:
- System variables: `PATH`, `HOME`, `USER`, `SHELL`, `PWD`, `TERM`
- AWS credentials: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`
- API keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`

### Sensitive Variable Detection
Variables containing these keywords are automatically masked:
- `KEY`
- `SECRET`
- `TOKEN`

### Force Override
Use `force=True` to bypass protection for critical operations:

```python
# Set a protected variable
result = tool.invoke(action="set", name="PATH", value="/new/path", force=True)

# Delete a protected variable
result = tool.invoke(action="delete", name="HOME", force=True)
```

## Tool Specification

The tool implements the standard tool interface with:

- **Name**: `environment_operation`
- **Input Schema**: Validates action, name, value, prefix, and force parameters
- **Actions**: `list`, `get`, `set`, `delete`

## Error Handling

The tool handles various error conditions:

- Invalid actions
- Missing required parameters (name for get/set/delete, value for set)
- Non-existent variables
- Protected variable operations without force flag
- System-level environment access errors

## Testing

Run the unit tests:

```bash
pytest test/test_environment_tool_unit.py
```

Tests cover all operations, security features, edge cases, and error conditions including:
- Unicode and special character handling
- Case sensitivity
- Empty environment scenarios
- Protected variable enforcement