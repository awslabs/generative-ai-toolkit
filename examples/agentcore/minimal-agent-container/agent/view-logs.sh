#!/bin/bash
set -e

REGION="eu-central-1"
BASE_AGENT_NAME="minimal_agent"

# Find the latest agent runtime
LATEST_RUNTIME=$(aws bedrock-agentcore-control list-agent-runtimes \
  --region "$REGION" \
  --query "agentRuntimes[?starts_with(agentRuntimeName, '$BASE_AGENT_NAME')] | sort_by(@, &lastUpdatedAt) | [-1].agentRuntimeId" \
  --output text)

if [ "$LATEST_RUNTIME" = "None" ] || [ -z "$LATEST_RUNTIME" ]; then
  echo "No agent runtimes found starting with '$BASE_AGENT_NAME'"
  exit 1
fi

LOG_GROUP="/aws/bedrock-agentcore/runtimes/$LATEST_RUNTIME-DEFAULT"

echo "Viewing logs for: $LATEST_RUNTIME"
echo "Log Group: $LOG_GROUP"
echo "=========================================="

# Get logs from runtime-logs streams only, sorted by time (macOS compatible)
aws logs filter-log-events \
  --log-group-name "$LOG_GROUP" \
  --region "$REGION" \
  --start-time $(date -v-1H +%s)000 \
  --log-stream-name-prefix "$(date +%Y/%m/%d)/[runtime-logs]" \
  --query 'events[*].[timestamp,logStreamName,message]' \
  --output text | \
  sort -n | \
  while IFS=$'\t' read -r timestamp stream message; do
    # Convert timestamp to human readable (macOS compatible)
    readable_time=$(date -r $(echo $timestamp | cut -c1-10) '+%Y-%m-%d %H:%M:%S')
    # Extract UUID from stream name
    stream_id=$(echo "$stream" | grep -o '[a-f0-9-]\{36\}' | head -1)
    echo "[$readable_time] [$stream_id] $message"
  done
