# S3 Tool

A tool for performing S3 operations including listing, reading, writing, and deleting objects in Amazon S3 buckets.

## Features

- **List objects** in S3 bucket with optional prefix filtering
- **Read content** from S3 objects
- **Write content** to S3 objects
- **Delete objects** from S3 bucket
- Built-in error handling and validation
- Processing time tracking
- OpenTelemetry compatible tracing support

## Configuration

Set the `S3_BUCKET_NAME` environment variable to configure the target bucket:

```bash
export S3_BUCKET_NAME=your-bucket-name
```

## Agent Integration

```python
from generative_ai_toolkit.agent import BedrockConverseAgent
from s3_tool import S3Tool

# Create agent with S3 tool
agent = BedrockConverseAgent(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
)
agent.register_tool(S3Tool())

# Chat with agent about S3 operations
response = agent.converse("List all files in the bucket")
response = agent.converse("Read the content of document.txt")
response = agent.converse("Save this text to notes.txt: Hello World")
```

## Usage

### Basic Operations

```python
from s3_tool import S3Tool

tool = S3Tool()

# List all objects
result = tool.invoke(action="list")

# List with prefix filter
result = tool.invoke(action="list", prefix="documents/")

# Read object content
result = tool.invoke(action="read", key="file.txt")

# Write content to object
result = tool.invoke(action="write", key="new-file.txt", content="Hello World")

# Delete object
result = tool.invoke(action="delete", key="old-file.txt")
```

### Response Format

All operations return a structured response:

```python
{
    "success": bool,
    "action": str,
    "objects": list,  # for list operations
    "content": str,   # for read operations
    "key": str,
    "bucket": str,
    "processing_time_ms": int,
    "message": str,
    "error": str      # if operation failed
}
```

## Tool Specification

The tool implements the standard tool interface with:

- **Name**: `s3_operation`
- **Input Schema**: Validates action, key, content, prefix, and max_keys parameters
- **Actions**: `list`, `read`, `write`, `delete`

## Error Handling

The tool handles various error conditions:

- Missing bucket configuration
- Invalid actions
- Missing required parameters
- AWS service errors
- Network connectivity issues

## Testing

Run the unit tests:

```bash
pytest test/test_s3_tool_unit.py
```

Tests use moto for S3 mocking and cover all operations and error conditions.
