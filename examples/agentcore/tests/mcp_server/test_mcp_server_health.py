"""Health check tests for AgentCore MCP server monitoring and observability."""

from datetime import UTC, datetime, timedelta

import boto3


class TestMcpServerHealth:
    """Health check tests for MCP server monitoring and observability."""

    def test_mcp_server_runtime_logs_exist(
        self, bedrock_agentcore_control_client, mcp_server_runtime_arn
    ):
        """Test that MCP server runtime has associated CloudWatch logs."""
        runtime_id = mcp_server_runtime_arn.split("/")[-1]
        logs_client = boto3.client("logs")

        log_group_name = f"/aws/bedrock-agentcore/runtimes/{runtime_id}-DEFAULT"

        response = logs_client.describe_log_groups(
            logGroupNamePrefix=log_group_name, limit=1
        )

        log_groups = response.get("logGroups", [])
        assert len(log_groups) == 1, f"Expected log group {log_group_name} to exist"
        assert log_groups[0]["logGroupName"] == log_group_name

    def test_mcp_server_runtime_metrics_queryable(self, mcp_server_runtime_arn):
        """Test that CloudWatch metrics can be queried for the MCP server runtime."""
        runtime_id = mcp_server_runtime_arn.split("/")[-1]
        cloudwatch = boto3.client("cloudwatch")

        end_time = datetime.now(UTC)
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

        # The query should succeed and return a list (may be empty for new deployments)
        datapoints = response.get("Datapoints", [])
        assert isinstance(datapoints, list)
