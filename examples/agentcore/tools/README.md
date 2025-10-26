# AgentCore Tools

This directory contains utility tools for working with AgentCore agents.

## view-agent-logs.py

Cross-platform Python tool to view CloudWatch logs for AgentCore agents.

### Usage

```bash
# View logs for default agent (agentcore_integration_agent) in eu-central-1, last 1 hour
python view-agent-logs.py

# Specify agent name
python view-agent-logs.py my_agent_name

# Specify agent name and region
python view-agent-logs.py my_agent_name us-east-1

# Specify agent name, region, and hours back
python view-agent-logs.py my_agent_name us-east-1 24

# Show help
python view-agent-logs.py --help
```

### Features

- **Automatic Agent Discovery**: Finds the latest agent runtime matching the provided name
- **Error Handling**: Shows helpful error messages and lists available agents/log groups
- **Color Coding**: Highlights ERROR (red), WARN (yellow), and INFO (green) messages
- **Cross-platform**: Works on Windows, macOS, and Linux
- **Flexible Time Range**: Specify how many hours back to search for logs
- **Stream Identification**: Shows session IDs from log stream names

### Prerequisites

- AWS CLI configured with appropriate permissions
- For AgentCore logs, you need:
  - `bedrock-agentcore:ListAgentRuntimes`
  - `logs:DescribeLogGroups`
  - `logs:FilterLogEvents`

### Examples

#### Investigating a 500 Error
```bash
# Check recent logs for errors
python view-agent-logs.py agentcore_integration_agent eu-central-1 2

# Look for specific error patterns
python view-agent-logs.py my_agent | grep -i error
```

#### Monitoring Agent Activity
```bash
# Check logs over a longer period
python view-agent-logs.py my_agent us-east-1 24

# Monitor recent activity
python view-agent-logs.py my_agent eu-central-1 1
```

### Troubleshooting

#### No Agent Found
If you get "No agent runtimes found", the script will list all available agents. Make sure:
- The agent name prefix is correct
- You're using the right AWS region
- Your AWS credentials have the necessary permissions

#### No Log Group Found
If the log group doesn't exist:
- The agent may not have been deployed yet
- The agent may not have been invoked (logs are created on first invocation)
- Check the agent runtime status with `aws bedrock-agentcore-control get-agent-runtime`

#### No Logs in Time Range
If no logs appear:
- Increase the hours back parameter
- Check if the agent has been invoked recently
- Verify the agent runtime is in READY status