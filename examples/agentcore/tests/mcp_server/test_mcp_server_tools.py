"""Tests for deployed MCP server tool schema validation."""

import json
import os
import sys
from pathlib import Path

import pytest

# Add agent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "agent"))

from simple_mcp_client import SimpleMcpClient


class TestMcpServerDeployment:
    """Test suite for deployed MCP server tool schema validation."""

    async def _get_tools_via_mcp_client(self):
        """Helper method to get tools using SimpleMcpClient."""
        mcp_server_runtime_arn = os.environ["MCP_SERVER_RUNTIME_ARN"]

        async with SimpleMcpClient(runtime_arn=mcp_server_runtime_arn) as mcp_client:
            tools_result = await mcp_client.list_tools()
            return tools_result.tools

    @pytest.mark.asyncio
    async def test_mcp_server_tools_list_endpoint(self, cdk_outputs):
        """Test that the deployed MCP server responds to tools/list requests using MCP client."""
        mcp_server_runtime_arn = os.environ["MCP_SERVER_RUNTIME_ARN"]

        # Use SimpleMcpClient to connect and list tools
        async with SimpleMcpClient(runtime_arn=mcp_server_runtime_arn) as mcp_client:
            # List available tools from MCP server
            tools_result = await mcp_client.list_tools()

            # Verify we got a valid response
            assert tools_result is not None
            assert hasattr(tools_result, "tools")
            assert isinstance(tools_result.tools, list)

            # Verify we have the expected weather tools
            tool_names = [tool.name for tool in tools_result.tools]
            assert "get_weather" in tool_names
            assert "get_forecast" in tool_names
            assert len(tools_result.tools) == 2

    @pytest.mark.asyncio
    async def test_mcp_server_returns_weather_tools(self, cdk_outputs):
        """Test that the MCP server returns the expected weather tools."""
        tools = await self._get_tools_via_mcp_client()

        # Verify we have the expected tools
        tool_names = [tool.name for tool in tools]
        assert "get_weather" in tool_names
        assert "get_forecast" in tool_names

        # Verify we have exactly 2 tools
        assert len(tools) == 2

    @pytest.mark.asyncio
    async def test_weather_tool_json_schema_structure(self, cdk_outputs):
        """Test that the get_weather tool returns proper JSON schema with Pydantic model structure."""
        tools = await self._get_tools_via_mcp_client()

        # Find the get_weather tool
        weather_tool = None
        for tool in tools:
            if tool.name == "get_weather":
                weather_tool = tool
                break

        assert weather_tool is not None, "get_weather tool not found"

        # Verify tool structure
        assert hasattr(weather_tool, "name")
        assert hasattr(weather_tool, "description")
        assert hasattr(weather_tool, "inputSchema")

        # Verify the inputSchema is a proper JSON schema
        schema = weather_tool.inputSchema
        assert isinstance(schema, dict)
        assert "type" in schema
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

        # Verify request parameter exists (Pydantic model parameter)
        assert "request" in schema["properties"]
        request_prop = schema["properties"]["request"]

        # The request should reference a Pydantic model definition
        if "$ref" in request_prop:
            # Schema uses $ref to reference model definition
            assert "$defs" in schema or "definitions" in schema
            # Find the referenced model in definitions
            ref_path = request_prop["$ref"]
            if ref_path.startswith("#/$defs/"):
                model_name = ref_path.split("/")[-1]
                model_def = schema["$defs"][model_name]
            elif ref_path.startswith("#/definitions/"):
                model_name = ref_path.split("/")[-1]
                model_def = schema["definitions"][model_name]
            else:
                model_def = request_prop
        else:
            # Schema has model definition inline
            model_def = request_prop

        assert model_def["type"] == "object"
        assert "properties" in model_def
        assert "required" in model_def

        # Verify city property exists within the model
        assert "city" in model_def["properties"]
        city_prop = model_def["properties"]["city"]
        assert city_prop["type"] == "string"
        assert "description" in city_prop
        assert "minLength" in city_prop
        assert "maxLength" in city_prop

        # Verify required fields
        assert "request" in schema["required"]
        assert "city" in model_def["required"]

    @pytest.mark.asyncio
    async def test_forecast_tool_json_schema_structure(self, cdk_outputs):
        """Test that the get_forecast tool returns proper JSON schema with Pydantic model structure."""
        tools = await self._get_tools_via_mcp_client()

        # Find the get_forecast tool
        forecast_tool = None
        for tool in tools:
            if tool.name == "get_forecast":
                forecast_tool = tool
                break

        assert forecast_tool is not None, "get_forecast tool not found"

        # Verify tool structure
        assert hasattr(forecast_tool, "name")
        assert hasattr(forecast_tool, "description")
        assert hasattr(forecast_tool, "inputSchema")

        # Verify the inputSchema is a proper JSON schema
        schema = forecast_tool.inputSchema
        assert isinstance(schema, dict)
        assert "type" in schema
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

        # Verify request parameter exists (Pydantic model parameter)
        assert "request" in schema["properties"]
        request_prop = schema["properties"]["request"]

        # The request should reference a Pydantic model definition
        if "$ref" in request_prop:
            # Schema uses $ref to reference model definition
            assert "$defs" in schema or "definitions" in schema
            # Find the referenced model in definitions
            ref_path = request_prop["$ref"]
            if ref_path.startswith("#/$defs/"):
                model_name = ref_path.split("/")[-1]
                model_def = schema["$defs"][model_name]
            elif ref_path.startswith("#/definitions/"):
                model_name = ref_path.split("/")[-1]
                model_def = schema["definitions"][model_name]
            else:
                model_def = request_prop
        else:
            # Schema has model definition inline
            model_def = request_prop

        assert model_def["type"] == "object"
        assert "properties" in model_def
        assert "required" in model_def

        # Verify city property within the model
        assert "city" in model_def["properties"]
        city_prop = model_def["properties"]["city"]
        assert city_prop["type"] == "string"
        assert "description" in city_prop

        # Verify days property within the model
        assert "days" in model_def["properties"]
        days_prop = model_def["properties"]["days"]
        assert days_prop["type"] == "integer"
        assert "default" in days_prop
        assert days_prop["default"] == 3
        assert "minimum" in days_prop
        assert "maximum" in days_prop

        # Verify required fields (request is required, within model only city is required)
        assert "request" in schema["required"]
        assert "city" in model_def["required"]
        assert "days" not in model_def["required"]

    @pytest.mark.asyncio
    async def test_tool_schemas_are_valid_json(self, cdk_outputs):
        """Test that all tool schemas are valid JSON and can be serialized."""
        tools = await self._get_tools_via_mcp_client()

        # Test that each tool's schema can be serialized and deserialized
        for tool in tools:
            schema = tool.inputSchema

            # Test JSON serialization/deserialization
            schema_json = json.dumps(schema)
            parsed_schema = json.loads(schema_json)
            assert parsed_schema == schema

            # Verify schema has required JSON Schema fields
            assert "type" in schema
            assert (
                "properties" in schema
                or "items" in schema
                or schema["type"] in ["string", "number", "boolean"]
            )

    @pytest.mark.asyncio
    async def test_tool_descriptions_are_present(self, cdk_outputs):
        """Test that all tools have meaningful descriptions."""
        tools = await self._get_tools_via_mcp_client()

        # Verify each tool has a meaningful description
        for tool in tools:
            assert hasattr(tool, "description")
            description = tool.description
            assert isinstance(description, str)
            assert len(description.strip()) > 10  # Meaningful description

            # Verify descriptions contain usage guidance
            description_lower = description.lower()
            assert any(
                keyword in description_lower
                for keyword in ["use", "tool", "when", "example", "get"]
            )

    @pytest.mark.asyncio
    async def test_pydantic_model_validation_rules_preserved(self, cdk_outputs):
        """Test that Pydantic model validation rules are preserved in the JSON schema."""
        tools = await self._get_tools_via_mcp_client()

        # Test weather tool validation rules
        weather_tool = next(
            (tool for tool in tools if tool.name == "get_weather"), None
        )
        assert weather_tool is not None

        # Get the model definition (handle $ref or inline)
        weather_request_schema = weather_tool.inputSchema["properties"]["request"]
        if "$ref" in weather_request_schema:
            ref_path = weather_request_schema["$ref"]
            if ref_path.startswith("#/$defs/"):
                model_name = ref_path.split("/")[-1]
                weather_model_def = weather_tool.inputSchema["$defs"][model_name]
            elif ref_path.startswith("#/definitions/"):
                model_name = ref_path.split("/")[-1]
                weather_model_def = weather_tool.inputSchema["definitions"][model_name]
        else:
            weather_model_def = weather_request_schema

        city_prop = weather_model_def["properties"]["city"]

        # Verify Pydantic Field validation rules are preserved
        assert city_prop["minLength"] == 1
        assert city_prop["maxLength"] == 100
        assert "description" in city_prop
        assert len(city_prop["description"]) > 0

        # Test forecast tool validation rules
        forecast_tool = next(
            (tool for tool in tools if tool.name == "get_forecast"), None
        )
        assert forecast_tool is not None

        # Get the model definition (handle $ref or inline)
        forecast_request_schema = forecast_tool.inputSchema["properties"]["request"]
        if "$ref" in forecast_request_schema:
            ref_path = forecast_request_schema["$ref"]
            if ref_path.startswith("#/$defs/"):
                model_name = ref_path.split("/")[-1]
                forecast_model_def = forecast_tool.inputSchema["$defs"][model_name]
            elif ref_path.startswith("#/definitions/"):
                model_name = ref_path.split("/")[-1]
                forecast_model_def = forecast_tool.inputSchema["definitions"][
                    model_name
                ]
        else:
            forecast_model_def = forecast_request_schema

        days_prop = forecast_model_def["properties"]["days"]

        # Verify Pydantic Field validation rules for days
        assert days_prop["minimum"] == 1
        assert days_prop["maximum"] == 7
        assert days_prop["default"] == 3
        assert "description" in days_prop
        assert len(days_prop["description"]) > 0

    @pytest.mark.asyncio
    async def test_pydantic_model_titles_and_descriptions(self, cdk_outputs):
        """Test that Pydantic model titles and descriptions are included in schemas."""
        tools = await self._get_tools_via_mcp_client()

        for tool in tools:
            request_schema = tool.inputSchema["properties"]["request"]

            # Get the model definition (handle $ref or inline)
            if "$ref" in request_schema:
                ref_path = request_schema["$ref"]
                if ref_path.startswith("#/$defs/"):
                    model_name = ref_path.split("/")[-1]
                    model_def = tool.inputSchema["$defs"][model_name]
                elif ref_path.startswith("#/definitions/"):
                    model_name = ref_path.split("/")[-1]
                    model_def = tool.inputSchema["definitions"][model_name]
                else:
                    model_def = request_schema
            else:
                model_def = request_schema

            # Verify Pydantic model has title
            assert "title" in model_def
            assert isinstance(model_def["title"], str)
            assert len(model_def["title"]) > 0

            # Verify Pydantic model has description with examples
            if "description" in model_def:
                description = model_def["description"]
                assert isinstance(description, str)
                # Should contain usage examples from our Pydantic model docstrings
                assert any(
                    keyword in description.lower()
                    for keyword in ["example", "use", "when"]
                )
