import json

import boto3

# Get the latest agent runtime starting with "minimal_agent"
client = boto3.client("bedrock-agentcore", region_name="eu-central-1")

# List agent runtimes and find the latest one starting with "minimal_agent"
control_client = boto3.client("bedrock-agentcore-control", region_name="eu-central-1")
response = control_client.list_agent_runtimes()

minimal_agent_runtimes = [
    runtime for runtime in response["agentRuntimes"]
    if runtime["agentRuntimeName"].startswith("minimal_agent")
]

if not minimal_agent_runtimes:
    raise Exception("No agent runtimes found starting with 'minimal_agent'")

# Get the most recently updated runtime
latest_runtime = max(minimal_agent_runtimes, key=lambda x: x["lastUpdatedAt"])
runtime_arn = latest_runtime["agentRuntimeArn"]

print(f"Using latest agent runtime: {latest_runtime['agentRuntimeName']} ({latest_runtime['agentRuntimeId']})")

payload = json.dumps({"input": {"prompt": "Test version reporting"}})

response = client.invoke_agent_runtime(
    agentRuntimeArn=runtime_arn,
    runtimeSessionId="dfmeoagmreaklgmrkleafremoigrmtesogmtrskhmtkrlshmt",  # Must be 33+ chars
    payload=payload,
    qualifier="DEFAULT",  # Optional
)
response_body = response["response"].read()
response_data = json.loads(response_body)
print("Agent Response:", response_data)
