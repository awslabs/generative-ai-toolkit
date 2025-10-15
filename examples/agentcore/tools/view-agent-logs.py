#!/usr/bin/env python3
"""
Script to view CloudWatch logs for an AgentCore agent.
Usage: python view-agent-logs.py [AGENT_NAME] [REGION] [HOURS_BACK]
"""

import argparse
import re
import sys
from datetime import datetime, timedelta

import boto3


def find_agent_runtime(agent_name: str, region: str) -> str | None:
    """Find the latest agent runtime matching the name."""
    try:
        client = boto3.client("bedrock-agentcore-control", region_name=region)
        response = client.list_agent_runtimes()

        # Filter runtimes that start with the agent name
        matching_runtimes = [
            runtime
            for runtime in response.get("agentRuntimes", [])
            if runtime["agentRuntimeName"].startswith(agent_name)
        ]

        if not matching_runtimes:
            return None

        # Sort by lastUpdatedAt and get the latest
        latest_runtime = sorted(matching_runtimes, key=lambda x: x["lastUpdatedAt"])[-1]

        return latest_runtime["agentRuntimeId"]

    except Exception as e:
        print(f"❌ Error finding agent runtime: {e}")
        return None


def list_available_runtimes(region: str):
    """List all available agent runtimes."""
    try:
        client = boto3.client("bedrock-agentcore-control", region_name=region)
        response = client.list_agent_runtimes()

        print("\nAvailable agent runtimes:")
        print(f"{'Name':<30} {'ID':<40} {'Status':<15}")
        print("-" * 85)

        for runtime in response.get("agentRuntimes", []):
            print(
                f"{runtime['agentRuntimeName']:<30} {runtime['agentRuntimeId']:<40} {runtime['status']:<15}"
            )

    except Exception as e:
        print(f"❌ Error listing runtimes: {e}")


def check_log_group_exists(log_group: str, region: str) -> bool:
    """Check if the log group exists."""
    try:
        client = boto3.client("logs", region_name=region)
        response = client.describe_log_groups(logGroupNamePrefix=log_group)
        return len(response.get("logGroups", [])) > 0
    except Exception:
        return False


def list_available_log_groups(region: str):
    """List available AgentCore log groups."""
    try:
        client = boto3.client("logs", region_name=region)
        response = client.describe_log_groups(
            logGroupNamePrefix="/aws/bedrock-agentcore"
        )

        print("\nAvailable AgentCore log groups:")
        for log_group in response.get("logGroups", []):
            print(f"  {log_group['logGroupName']}")

    except Exception as e:
        print(f"❌ Error listing log groups: {e}")


def colorize_log_message(message: str) -> str:
    """Add color coding to log messages based on level."""
    if any(level in message.upper() for level in ["ERROR", "FATAL"]):
        return f"\033[31m{message}\033[0m"  # Red
    elif "WARN" in message.upper():
        return f"\033[33m{message}\033[0m"  # Yellow
    elif "INFO" in message.upper():
        return f"\033[32m{message}\033[0m"  # Green
    return message


def view_logs(agent_name: str, region: str, hours_back: int):
    """View CloudWatch logs for the agent."""
    print(f"Looking for agent: {agent_name}")
    print(f"Region: {region}")
    print(f"Hours back: {hours_back}")
    print("=" * 40)

    # Find the agent runtime
    runtime_id = find_agent_runtime(agent_name, region)
    if not runtime_id:
        print(f"❌ No agent runtimes found starting with '{agent_name}'")
        list_available_runtimes(region)
        return False

    log_group = f"/aws/bedrock-agentcore/runtimes/{runtime_id}-DEFAULT"

    print(f"✅ Found agent runtime: {runtime_id}")
    print(f"📋 Log Group: {log_group}")
    print("=" * 40)

    # Check if log group exists
    if not check_log_group_exists(log_group, region):
        print(f"❌ Log group '{log_group}' not found")
        list_available_log_groups(region)
        return False

    # Calculate start time
    start_time = datetime.now() - timedelta(hours=hours_back)
    start_timestamp = int(start_time.timestamp() * 1000)

    print(f"🔍 Fetching logs from the last {hours_back} hour(s)...")
    print()

    try:
        client = boto3.client("logs", region_name=region)

        # Get all log streams in the log group
        print("🔍 Discovering log streams...")
        log_streams = []
        paginator = client.get_paginator("describe_log_streams")

        for page in paginator.paginate(logGroupName=log_group):
            log_streams.extend(page.get("logStreams", []))

        print(f"📋 Found {len(log_streams)} log streams")

        # Filter log events from ALL log streams with pagination
        all_events = []
        next_token = None

        while True:
            filter_params = {"logGroupName": log_group, "startTime": start_timestamp}

            if next_token:
                filter_params["nextToken"] = next_token

            response = client.filter_log_events(**filter_params)

            events = response.get("events", [])
            all_events.extend(events)

            next_token = response.get("nextToken")
            if not next_token:
                break

            print(f"📄 Retrieved {len(events)} events (total: {len(all_events)})")

        if not all_events:
            print("📝 No logs found in the specified time range.")
            print(
                "💡 Tip: Try increasing hours back or check if the agent has been invoked recently"
            )
            return True

        # Sort events by timestamp
        all_events.sort(key=lambda x: x["timestamp"])
        print(
            f"📊 Processing {len(all_events)} total log entries from {len(log_streams)} streams"
        )
        print()

        for event in all_events:
            timestamp = event["timestamp"]
            log_stream = event.get("logStreamName", "unknown")
            message = event["message"].strip()

            # Convert timestamp to readable format
            readable_time = datetime.fromtimestamp(timestamp / 1000).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            # Extract UUID from stream name if present
            uuid_match = re.search(r"[a-f0-9-]{36}", log_stream)
            stream_id = uuid_match.group(0) if uuid_match else "unknown"

            # Apply color coding
            colored_message = colorize_log_message(message)

            print(f"[{readable_time}] [{stream_id}] {colored_message}")

        print()
        print(f"📊 Displayed {len(all_events)} log entries")
        return True

    except Exception as e:
        print(f"❌ Error fetching logs: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="View CloudWatch logs for an AgentCore agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python view-agent-logs.py                                    # Use defaults
  python view-agent-logs.py my_agent                          # Specific agent
  python view-agent-logs.py my_agent us-east-1                # Different region
  python view-agent-logs.py my_agent us-east-1 24             # Last 24 hours
        """,
    )

    parser.add_argument(
        "agent_name",
        nargs="?",
        default="agentcore_integration_agent",
        help="Agent name to search for (default: agentcore_integration_agent)",
    )

    parser.add_argument(
        "region",
        nargs="?",
        default="eu-central-1",
        help="AWS region (default: eu-central-1)",
    )

    parser.add_argument(
        "hours_back",
        nargs="?",
        type=int,
        default=1,
        help="Hours back to search (default: 1)",
    )

    args = parser.parse_args()

    success = view_logs(args.agent_name, args.region, args.hours_back)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
