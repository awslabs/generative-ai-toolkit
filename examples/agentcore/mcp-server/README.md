# Weather Tools MCP Server

This directory contains a Model Context Protocol (MCP) server that provides weather-related tools using Pydantic models for validation. The server demonstrates how to create reusable tools that can be consumed by MCP clients, including the Generative AI Toolkit.

## Features

- **Weather Forecast Tool**: Provides multi-day weather forecasts for specified locations
- **Weather Alerts Tool**: Retrieves active weather alerts for specified areas
- **Pydantic Validation**: Strong typing and validation for all tool inputs and outputs
- **MCP Protocol**: Standard MCP interface for tool discovery and execution
- **Containerized**: Docker support for easy deployment
- **Health Checks**: Built-in health monitoring endpoints

## Available Tools

### get_weather_forecast
Provides weather forecast information for a specified location.

**Parameters:**
- `location` (string): City name, state (e.g., "Seattle, WA") or coordinates
- `days` (integer, 1-7): Number of forecast days (default: 3)

**Example:**
```json
{
  "location": "Seattle, WA",
  "days": 5
}
```

### get_weather_alerts
Retrieves active weather alerts for a specified area.

**Parameters:**
- `area` (string): State code (e.g., "CA") or area name
- `severity` (string, optional): Filter by severity level

**Example:**
```json
{
  "area": "CA",
  "severity": "Severe"
}
```

## Quick Start

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the server:**
   ```bash
   python mcp_server.py
   ```

3. **Test the server:**
   ```bash
   python test_server.py
   ```

The server will be available at `http://localhost:8000`.

### Docker Deployment

1. **Build and run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

2. **Run tests:**
   ```bash
   docker-compose --profile test up --build
   ```

3. **Build Docker image manually:**
   ```bash
   docker build -t weather-mcp-server .
   docker run -p 8000:8000 weather-mcp-server
   ```

## API Endpoints

- `POST /mcp` - MCP protocol endpoint for tool discovery and execution
- `GET /health` - Health check endpoint
- `GET /` - Root health check endpoint

## MCP Protocol Examples

### List Available Tools
```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list",
    "params": {}
  }'
```

### Call Weather Forecast Tool
```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
      "name": "get_weather_forecast",
      "arguments": {
        "location": "Seattle, WA",
        "days": 3
      }
    }
  }'
```

## Integration with Generative AI Toolkit

This MCP server can be used with the Generative AI Toolkit's MCP client:

```python
from generative_ai_toolkit.mcp.client import MCPClient

# Connect to the MCP server
client = MCPClient("http://localhost:8000/mcp")

# List available tools
tools = await client.list_tools()

# Call a tool
result = await client.call_tool(
    "get_weather_forecast",
    {"location": "Seattle, WA", "days": 3}
)
```

## File Structure

```
mcp-server/
├── weather_models.py      # Pydantic models for weather data
├── weather_tools.py       # Weather tool implementations
├── mcp_server.py         # MCP server implementation
├── test_server.py        # Test suite for the server
├── Dockerfile            # Container configuration
├── docker-compose.yml    # Docker Compose configuration
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Development Notes

- The server uses mock data for demonstration purposes
- In production, replace mock data with real weather API integration
- All tools use Pydantic models for input validation and output structuring
- The server follows MCP protocol version 2024-11-05
- Health checks are implemented for container orchestration

## Troubleshooting

### Server Won't Start
- Check that port 8000 is available
- Verify all dependencies are installed
- Check logs for specific error messages

### Tools Not Working
- Verify the MCP protocol requests are properly formatted
- Check that tool names match exactly: `get_weather_forecast`, `get_weather_alerts`
- Ensure input parameters match the Pydantic model requirements

### Docker Issues
- Ensure Docker is running and has sufficient resources
- Check that no other services are using port 8000
- Verify the Dockerfile builds successfully

