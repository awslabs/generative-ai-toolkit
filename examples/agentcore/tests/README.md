# AgentCore Integration Tests

This directory contains comprehensive evaluation tests for the AgentCore integration example using the Generative AI Toolkit's testing framework.

## Test Types

### Mock Tests
- **Purpose**: Unit tests that don't require external services
- **Dependencies**: None (uses mock agent)
- **Run with**: `python run_tests.py --type mock`
- **Coverage**: Basic functionality, tool integration, conversation flow, error handling

### Integration Tests
- **Purpose**: Test against local docker-compose deployment
- **Dependencies**: Local agent running on http://localhost:8080
- **Run with**: `python run_tests.py --type integration`
- **Setup**: `./run_local.sh start` (from parent directory)

### AgentCore Tests
- **Purpose**: Test against deployed AgentCore runtime
- **Dependencies**: Deployed AgentCore endpoint
- **Run with**: `python run_tests.py --type agentcore --agentcore-endpoint https://your-endpoint`
- **Setup**: Deploy using CDK infrastructure

## Quick Start

1. **Run mock tests** (no setup required):
   ```bash
   python run_tests.py --type mock
   ```

2. **Run integration tests** (requires local deployment):
   ```bash
   # Start local environment
   ../run_local.sh start
   
   # Run tests
   python ../run_tests.py --type integration
   
   # Or run the standalone test script
   python test_local.py --test-type basic
   ```

3. **Run all available tests**:
   ```bash
   python run_tests.py --type all
   ```

## Test Files

- `test_evaluation.py` - Comprehensive test suite using Generative AI Toolkit's Case and Expect classes
- `test_local.py` - Standalone testing script with AgentCore request format (can be run independently)
- `conftest.py` - Pytest configuration and fixtures
- `requirements.txt` - Test dependencies

## Test Structure

### TestBasicFunctionality
- Simple greeting responses
- Weather query handling
- Basic agent behavior validation

### TestConversationFlow
- Multi-turn conversations
- Session management
- Context preservation

### TestToolIntegration
- MCP tool invocation
- Parameter validation
- Tool response handling

### TestErrorHandling
- Invalid input handling
- Error recovery
- Graceful degradation

### TestPerformanceAndReliability
- Response time validation
- Concurrent request handling
- Load testing

## Using the Toolkit's Testing Framework

The tests demonstrate how to use the Generative AI Toolkit's `Case` and `Expect` classes:

```python
# Create a test case
case = Case(["What's the weather in Seattle?"])

# Run against an agent
traces = case.run(agent)

# Validate the results
Expect(traces).tool_invocations.to_include("get_weather_forecast")
Expect(traces).agent_text_response.to_include("Seattle")
```

## Configuration

Tests can be configured via environment variables:

- `LOCAL_AGENT_ENDPOINT`: Local agent URL (default: http://localhost:8080)
- `AGENTCORE_ENDPOINT`: AgentCore runtime URL
- `TEST_TIMEOUT`: Request timeout in seconds (default: 30)

## Requirements

Install test dependencies:

```bash
pip install -r requirements.txt
```

Or install the toolkit with evaluation features:

```bash
pip install -e "../../..[all]"
```