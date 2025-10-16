"""Test the deployed AgentCore MCP server."""

import json
import time
import uuid

import pytest
from botocore.exceptions import ClientError


class TestMcpServerDeployment:
    """Test suite for deployed AgentCore MCP server."""

    def test_mcp_server_runtime_exists(
        self, bedrock_agentcore_control_client, mcp_server_runtime_arn
    ):
        """Test that the MCP server runtime exists and is accessible."""
        # Extract runtime ID from ARN
        runtime_id = mcp_server_runtime_arn.split("/")[-1]

        try:
            response = bedrock_agentcore_control_client.get_agent_runtime(
                agentRuntimeId=runtime_id
            )
            assert response["agentRuntimeId"] == runtime_id
            assert response["status"] in ["READY", "CREATING", "UPDATING"]
            assert response["protocolConfiguration"] == "MCP"
        except ClientError as e:
            pytest.fail(f"Failed to get MCP server runtime: {e}")

    def test_mcp_server_runtime_endpoint_exists(
        self, bedrock_agentcore_control_client, mcp_server_runtime_endpoint_arn
    ):
        """Test that the MCP server runtime endpoint exists and is accessible."""
        # Extract runtime ID and endpoint name from ARN
        # ARN format: arn:aws:bedrock-agentcore:region:account:runtime/runtime-id/runtime-endpoint/endpoint-name
        arn_parts = mcp_server_runtime_endpoint_arn.split("/")
        runtime_id = arn_parts[-3]  # runtime-id
        endpoint_name = arn_parts[-1]  # endpoint-name

        try:
            response = bedrock_agentcore_control_client.get_agent_runtime_endpoint(
                agentRuntimeId=runtime_id, endpointName=endpoint_name
            )
            assert response["name"] == endpoint_name
            assert runtime_id in response["agentRuntimeArn"]
            assert response["status"] in ["READY", "CREATING", "UPDATING"]
        except ClientError as e:
            pytest.fail(f"Failed to get MCP server runtime endpoint: {e}")

    def test_invoke_mcp_server_tools_list(self, bedrock_agentcore_client, mcp_server_runtime_arn):
        """Test MCP server tools/list method."""
        try:
            # Prepare MCP tools/list payload
            payload = json.dumps({
                "method": "tools/list",
                "params": {}
            }).encode()

            response = bedrock_agentcore_client.invoke_agent_runtime(
                agentRuntimeArn=mcp_server_runtime_arn,
                runtimeSessionId=str(uuid.uuid4()),
                payload=payload,
            )

            # Check response structure
            assert "response" in response
            assert "contentType" in response

            # Parse the response payload
            response_body = response["response"].read().decode("utf-8")
            response_data = json.loads(response_body)
            
            # Should contain tools list
            assert "tools" in response_data
            assert isinstance(response_data["tools"], list)

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "ValidationException":
                pytest.skip(f"MCP server not ready for invocation: {e}")
            else:
                pytest.fail(f"Failed to invoke MCP server ({error_code}): {e}")

    def test_invoke_mcp_server_weather_tool(self, bedrock_agentcore_client, mcp_server_runtime_arn):
        """Test MCP server weather tool invocation."""
        try:
            # Prepare MCP tools/call payload for weather tool
            payload = json.dumps({
                "method": "tools/call",
                "params": {
                    "name": "get_weather",
                    "arguments": {"city": "Amsterdam"}
                }
            }).encode()

            response = bedrock_agentcore_client.invoke_agent_runtime(
                agentRuntimeArn=mcp_server_runtime_arn,
                runtimeSessionId=str(uuid.uuid4()),
                payload=payload,
            )

            # Check response structure
            assert "response" in response
            assert "contentType" in response

            # Parse the response payload
            response_body = response["response"].read().decode("utf-8")
            response_data = json.loads(response_body)
            
            # Should contain content with weather information
            assert "content" in response_data
            assert isinstance(response_data["content"], list)
            assert len(response_data["content"]) > 0
            assert "text" in response_data["content"][0]
            assert "Amsterdam" in response_data["content"][0]["text"]

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "ValidationException":
                pytest.skip(f"MCP server not ready for invocation: {e}")
            else:
                pytest.fail(f"Failed to invoke MCP server weather tool ({error_code}): {e}")

    def test_invoke_mcp_server_forecast_tool(self, bedrock_agentcore_client, mcp_server_runtime_arn):
        """Test MCP server forecast tool invocation."""
        try:
            # Prepare MCP tools/call payload for forecast tool
            payload = json.dumps({
                "method": "tools/call",
                "params": {
                    "name": "get_forecast",
                    "arguments": {"city": "Berlin", "days": 5}
                }
            }).encode()

            response = bedrock_agentcore_client.invoke_agent_runtime(
                agentRuntimeArn=mcp_server_runtime_arn,
                runtimeSessionId=str(uuid.uuid4()),
                payload=payload,
            )

            # Check response structure
            assert "response" in response
            assert "contentType" in response

            # Parse the response payload
            response_body = response["response"].read().decode("utf-8")
            response_data = json.loads(response_body)
            
            # Should contain content with forecast information
            assert "content" in response_data
            assert isinstance(response_data["content"], list)
            assert len(response_data["content"]) > 0
            assert "text" in response_data["content"][0]
            assert "Berlin" in response_data["content"][0]["text"]
            assert "5-day" in response_data["content"][0]["text"]

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "ValidationException":
                pytest.skip(f"MCP server not ready for invocation: {e}")
            else:
                pytest.fail(f"Failed to invoke MCP server forecast tool ({error_code}): {e}")

    def test_mcp_server_response_time(self, bedrock_agentcore_client, mcp_server_runtime_arn):
        """Test that MCP server responds within reasonable time limits."""
        try:
            start_time = time.time()

            # Prepare MCP tools/list payload
            payload = json.dumps({
                "method": "tools/list",
                "params": {}
            }).encode()

            response = bedrock_agentcore_client.invoke_agent_runtime(
                agentRuntimeArn=mcp_server_runtime_arn,
                runtimeSessionId=str(uuid.uuid4()),
                payload=payload,
            )

            end_time = time.time()
            response_time = end_time - start_time

            # Check response is valid
            assert "response" in response
            assert "contentType" in response

            # Parse the response payload
            response_body = response["response"].read().decode("utf-8")
            response_data = json.loads(response_body)
            assert "tools" in response_data

            # Check response time is reasonable (less than 10 seconds for MCP)
            assert (
                response_time < 10.0
            ), f"Response took too long: {response_time:.2f} seconds"

        except ClientError as e:
            if e.response["Error"]["Code"] == "ValidationException":
                pytest.skip(f"MCP server not ready for invocation: {e}")
            else:
                pytest.fail(f"Failed to invoke MCP server: {e}")

    def test_invalid_mcp_method(self, bedrock_agentcore_client, mcp_server_runtime_arn):
        """Test MCP server handles invalid methods gracefully."""
        try:
            # Prepare invalid MCP method payload
            payload = json.dumps({
                "method": "invalid/method",
                "params": {}
            }).encode()

            response = bedrock_agentcore_client.invoke_agent_runtime(
                agentRuntimeArn=mcp_server_runtime_arn,
                runtimeSessionId=str(uuid.uuid4()),
                payload=payload,
            )

            # Check response structure
            assert "response" in response
            assert "contentType" in response

            # Parse the response payload
            response_body = response["response"].read().decode("utf-8")
            response_data = json.loads(response_body)
            
            # Should contain error for invalid method
            assert "error" in response_data
            assert "invalid/method" in response_data["error"]

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "ValidationException":
                pytest.skip(f"MCP server not ready for invocation: {e}")
            else:
                pytest.fail(f"Failed to invoke MCP server ({error_code}): {e}")
