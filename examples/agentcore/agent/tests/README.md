# AgentCore Agent Tests

This directory contains pytest tests for the deployed AgentCore agent.

## Overview

The tests validate that the deployed agent is working correctly by:
- Checking that the AgentCore runtime and endpoint are active
- Testing basic agent invocation functionality
- Validating weather-specific responses
- Testing session isolation and concurrent requests
- Monitoring agent health and error handling

## Prerequisites

1. **Deployed Infrastructure**: The CDK stack must be deployed with the agent runtime
2. **AWS Credentials**: Valid AWS credentials configured for the target account/region
3. **Dependencies**: Install test dependencies with `pip install -r requirements.txt`

## Running Tests

### All Tests
```bash
cd examples/agentcore/agent
pytest tests/
```

### Specific Test Files
```bash
# Test deployment status
pytest tests/test_agent_deployment.py

# Test health and monitoring
pytest tests/test_agent_health.py
```

### Verbose Output
```bash
pytest tests/ -v
```

## Test Structure

### `test_agent_deployment.py`
- **test_agent_runtime_exists**: Verifies the agent runtime is accessible
- **test_agent_runtime_endpoint_exists**: Verifies the endpoint is accessible
- **test_invoke_agent_basic**: Tests basic agent invocation
- **test_invoke_agent_weather_query**: Tests weather-specific functionality
- **test_invoke_agent_multiple_sessions**: Tests multiple session handling
- **test_agent_response_time**: Validates response time performance

### `test_agent_health.py`
- **test_agent_runtime_status**: Checks runtime health status
- **test_agent_endpoint_status**: Checks endpoint health status
- **test_agent_error_handling**: Tests error handling with invalid inputs
- **test_agent_session_isolation**: Validates session isolation
- **test_agent_concurrent_requests**: Tests concurrent request handling

## Configuration

The tests automatically:
1. Read CDK stack outputs to get runtime and endpoint ARNs
2. Create Bedrock client for API calls
3. Skip tests if infrastructure is not deployed or not ready

## Troubleshooting

### Tests Skipped
If tests are skipped, it usually means:
- CDK stack is not deployed
- Agent runtime is not in ACTIVE state
- AWS credentials are not configured

### Validation Errors
If you get `ValidationException` errors:
- The agent runtime may still be starting up
- Wait a few minutes and retry the tests

### Timeout Errors
If tests timeout:
- The agent may be under heavy load
- Check CloudWatch logs for the agent runtime
- Verify the agent container is healthy

## Example Output

```bash
$ pytest tests/ -v
========================= test session starts =========================
tests/test_agent_deployment.py::TestAgentDeployment::test_agent_runtime_exists PASSED
tests/test_agent_deployment.py::TestAgentDeployment::test_agent_runtime_endpoint_exists PASSED
tests/test_agent_deployment.py::TestAgentDeployment::test_invoke_agent_basic PASSED
tests/test_agent_deployment.py::TestAgentDeployment::test_invoke_agent_weather_query PASSED
tests/test_agent_health.py::TestAgentHealth::test_agent_runtime_status PASSED
tests/test_agent_health.py::TestAgentHealth::test_agent_endpoint_status PASSED
========================= 6 passed in 12.34s =========================
```