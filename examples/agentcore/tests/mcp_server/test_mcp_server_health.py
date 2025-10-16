"""Health check tests for AgentCore MCP server."""

import pytest
from botocore.exceptions import ClientError


class TestMcpServerHealth:
    """Health check tests for MCP server runtime."""

    def test_mcp_server_runtime_status(
        self, bedrock_agentcore_control_client, mcp_server_runtime_arn
    ):
        """Test that MCP server runtime is in a healthy state."""
        runtime_id = mcp_server_runtime_arn.split("/")[-1]

        try:
            response = bedrock_agentcore_control_client.get_agent_runtime(
                agentRuntimeId=runtime_id
            )

            # Check runtime is in a healthy state
            status = response.get("status")
            assert status in [
                "READY",
                "CREATING",
                "UPDATING",
            ], f"Runtime status is {status}"

            # Check runtime configuration
            assert (
                response.get("protocolConfiguration", {}).get("serverProtocol") == "MCP"
            )
            assert response.get("agentRuntimeName") == "agentcore_mcp_server"

        except ClientError as e:
            pytest.fail(f"Failed to check MCP server runtime health: {e}")

    def test_mcp_server_endpoint_status(
        self, bedrock_agentcore_control_client, mcp_server_runtime_endpoint_arn
    ):
        """Test that MCP server endpoint is in a healthy state."""
        arn_parts = mcp_server_runtime_endpoint_arn.split("/")
        runtime_id = arn_parts[-3]
        endpoint_name = arn_parts[-1]

        try:
            response = bedrock_agentcore_control_client.get_agent_runtime_endpoint(
                agentRuntimeId=runtime_id, endpointName=endpoint_name
            )

            # Check endpoint is in a healthy state
            status = response.get("status")
            assert status in [
                "READY",
                "CREATING",
                "UPDATING",
            ], f"Endpoint status is {status}"

            # Check endpoint configuration
            assert response.get("name") == "agentcore_mcp_server_endpoint"

        except ClientError as e:
            pytest.fail(f"Failed to check MCP server endpoint health: {e}")

    def test_mcp_server_runtime_logs_exist(
        self, bedrock_agentcore_control_client, mcp_server_runtime_arn
    ):
        """Test that MCP server runtime has associated CloudWatch logs."""
        import boto3

        runtime_id = mcp_server_runtime_arn.split("/")[-1]
        logs_client = boto3.client("logs")

        try:
            # Check if log group exists for the runtime
            log_group_name = f"/aws/bedrock-agentcore/runtimes/{runtime_id}-DEFAULT"

            response = logs_client.describe_log_groups(
                logGroupNamePrefix=log_group_name, limit=1
            )

            # Log group should exist (may be created after first invocation)
            log_groups = response.get("logGroups", [])
            if log_groups:
                assert len(log_groups) >= 1
                assert log_groups[0]["logGroupName"] == log_group_name
            else:
                # Log group may not exist yet if runtime hasn't been invoked
                pytest.skip(
                    "Log group not yet created (runtime may not have been invoked)"
                )

        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                pytest.skip(
                    "Log group not yet created (runtime may not have been invoked)"
                )
            else:
                pytest.fail(f"Failed to check MCP server logs: {e}")

    def test_mcp_server_runtime_metrics_available(self, mcp_server_runtime_arn):
        """Test that CloudWatch metrics are available for the MCP server runtime."""
        import boto3
        from datetime import datetime, timedelta

        runtime_id = mcp_server_runtime_arn.split("/")[-1]
        cloudwatch = boto3.client("cloudwatch")

        try:
            # Check for basic runtime metrics
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=1)

            response = cloudwatch.get_metric_statistics(
                Namespace="bedrock-agentcore",
                MetricName="Invocations",
                Dimensions=[{"Name": "RuntimeId", "Value": runtime_id}],
                StartTime=start_time,
                EndTime=end_time,
                Period=300,  # 5 minutes
                Statistics=["Sum"],
            )

            # Metrics may not be available immediately
            datapoints = response.get("Datapoints", [])
            # Just check that the metric query doesn't fail
            assert isinstance(datapoints, list)

        except ClientError as e:
            if e.response["Error"]["Code"] in [
                "InvalidParameterValue",
                "ResourceNotFound",
            ]:
                pytest.skip(
                    "Metrics not yet available (runtime may not have been invoked)"
                )
            else:
                pytest.fail(f"Failed to check MCP server metrics: {e}")
