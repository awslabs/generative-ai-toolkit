"""Tests for deployed MCP server tool schema validation."""

# nosec B101

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
            assert tools_result is not None  # nosec B101
            assert hasattr(tools_result, "tools")  # nosec B101
            assert isinstance(tools_result.tools, list)  # nosec B101

            # Verify we have the expected weather tools
            tool_names = [tool.name for tool in tools_result.tools]
            assert "get_weather" in tool_names  # nosec B101
            assert "get_forecast" in tool_names  # nosec B101
            assert len(tools_result.tools) == 2  # nosec B101

    @pytest.mark.asyncio
    async def test_mcp_server_returns_weather_tools(self, cdk_outputs):
        """Test that the MCP server returns the expected weather tools."""
        tools = await self._get_tools_via_mcp_client()

        # Verify we have the expected tools
        tool_names = [tool.name for tool in tools]
        assert "get_weather" in tool_names  # nosec B101
        assert "get_forecast" in tool_names  # nosec B101

        # Verify we have exactly 2 tools
        assert len(tools) == 2  # nosec B101

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

        assert weather_tool is not None, "get_weather tool not found"  # nosec B101

        # Verify tool structure
        assert hasattr(weather_tool, "name")  # nosec B101
        assert hasattr(weather_tool, "description")  # nosec B101
        assert hasattr(weather_tool, "inputSchema")  # nosec B101

        # Verify the inputSchema is a proper JSON schema
        schema = weather_tool.inputSchema
        assert isinstance(schema, dict)  # nosec B101
        assert "type" in schema  # nosec B101
        assert schema["type"] == "object"  # nosec B101
        assert "properties" in schema  # nosec B101
        assert "required" in schema  # nosec B101

        # Verify request parameter exists (Pydantic model parameter)
        assert "request" in schema["properties"]  # nosec B101
        request_prop = schema["properties"]["request"]

        # The request should reference a Pydantic model definition
        if "$ref" in request_prop:
            # Schema uses $ref to reference model definition
            assert "$defs" in schema or "definitions" in schema  # nosec B101
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

        assert model_def["type"] == "object"  # nosec B101
        assert "properties" in model_def  # nosec B101
        assert "required" in model_def  # nosec B101

        # Verify city property exists within the model
        assert "city" in model_def["properties"]  # nosec B101
        city_prop = model_def["properties"]["city"]
        assert city_prop["type"] == "string"  # nosec B101
        assert "description" in city_prop  # nosec B101
        assert "minLength" in city_prop  # nosec B101
        assert "maxLength" in city_prop  # nosec B101

        # Verify required fields
        assert "request" in schema["required"]  # nosec B101
        assert "city" in model_def["required"]  # nosec B101

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

        assert forecast_tool is not None, "get_forecast tool not found"  # nosec B101

        # Verify tool structure
        assert hasattr(forecast_tool, "name")  # nosec B101
        assert hasattr(forecast_tool, "description")  # nosec B101
        assert hasattr(forecast_tool, "inputSchema")  # nosec B101

        # Verify the inputSchema is a proper JSON schema
        schema = forecast_tool.inputSchema
        assert isinstance(schema, dict)  # nosec B101
        assert "type" in schema  # nosec B101
        assert schema["type"] == "object"  # nosec B101
        assert "properties" in schema  # nosec B101
        assert "required" in schema  # nosec B101

        # Verify request parameter exists (Pydantic model parameter)
        assert "request" in schema["properties"]  # nosec B101
        request_prop = schema["properties"]["request"]

        # The request should reference a Pydantic model definition
        if "$ref" in request_prop:
            # Schema uses $ref to reference model definition
            assert "$defs" in schema or "definitions" in schema  # nosec B101
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

        assert model_def["type"] == "object"  # nosec B101
        assert "properties" in model_def  # nosec B101
        assert "required" in model_def  # nosec B101

        # Verify city property within the model
        assert "city" in model_def["properties"]  # nosec B101
        city_prop = model_def["properties"]["city"]
        assert city_prop["type"] == "string"  # nosec B101
        assert "description" in city_prop  # nosec B101

        # Verify days property within the model
        assert "days" in model_def["properties"]  # nosec B101
        days_prop = model_def["properties"]["days"]
        assert days_prop["type"] == "integer"  # nosec B101
        assert "default" in days_prop  # nosec B101
        assert days_prop["default"] == 3  # nosec B101
        assert "minimum" in days_prop  # nosec B101
        assert "maximum" in days_prop  # nosec B101

        # Verify required fields (request is required, within model only city is required)
        assert "request" in schema["required"]  # nosec B101
        assert "city" in model_def["required"]  # nosec B101
        assert "days" not in model_def["required"]  # nosec B101

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
            assert parsed_schema == schema  # nosec B101

            # Verify schema has required JSON Schema fields
            assert "type" in schema  # nosec B101
            assert (  # nosec B101
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
            assert hasattr(tool, "description")  # nosec B101
            description = tool.description
            assert isinstance(description, str)  # nosec B101
            assert len(description.strip()) > 10  # Meaningful description  # nosec B101

            # Verify descriptions contain usage guidance
            description_lower = description.lower()
            assert any(  # nosec B101
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
        assert weather_tool is not None  # nosec B101

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
        assert city_prop["minLength"] == 1  # nosec B101
        assert city_prop["maxLength"] == 100  # nosec B101
        assert "description" in city_prop  # nosec B101
        assert len(city_prop["description"]) > 0  # nosec B101

        # Test forecast tool validation rules
        forecast_tool = next(
            (tool for tool in tools if tool.name == "get_forecast"), None
        )
        assert forecast_tool is not None  # nosec B101

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
        assert days_prop["minimum"] == 1  # nosec B101
        assert days_prop["maximum"] == 7  # nosec B101
        assert days_prop["default"] == 3  # nosec B101
        assert "description" in days_prop  # nosec B101
        assert len(days_prop["description"]) > 0  # nosec B101

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
            assert "title" in model_def  # nosec B101
            assert isinstance(model_def["title"], str)  # nosec B101
            assert len(model_def["title"]) > 0  # nosec B101

            # Verify Pydantic model has description with examples
            if "description" in model_def:
                description = model_def["description"]
                assert isinstance(description, str)  # nosec B101
                # Should contain usage examples from our Pydantic model docstrings
                assert any(  # nosec B101
                    keyword in description.lower()
                    for keyword in ["example", "use", "when"]
                )
