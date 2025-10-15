# OpenTelemetry Configuration Guide for AgentCore

## Understanding OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED

The `OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED` environment variable controls whether OpenTelemetry automatically intercepts Python's standard logging calls and redirects them to the OpenTelemetry Logs Protocol (OTLP) endpoint.

## The Issue You Experienced

When you set `OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED=true`, you only saw the "Agent starting - logger configured" message but not the logs from the `invoke()` function. Here's why:

### What Happens with `OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED=true`

1. **Logging Interception**: OpenTelemetry intercepts all Python `logging` calls
2. **OTLP Redirection**: Instead of going to CloudWatch logs, logs are sent to the OTLP logs endpoint
3. **Incomplete Configuration**: Without proper OTLP logs configuration, the logs get lost

### What Happens with `OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED=false`

1. **Normal Logging**: Python logging works normally
2. **CloudWatch Integration**: Logs go directly to CloudWatch logs as expected
3. **AgentCore Compatibility**: Works seamlessly with AgentCore's built-in logging

## Configuration Options

### Option 1: Recommended for AgentCore (Current Configuration)

```dockerfile
ENV OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED=false
CMD ["opentelemetry-instrument", "python", "agent.py"]
```

**Benefits:**
- ✅ Normal Python logging works as expected
- ✅ Logs appear in CloudWatch logs immediately
- ✅ Compatible with AgentCore's built-in observability
- ✅ Still gets OpenTelemetry tracing and metrics
- ✅ Simpler configuration

### Option 2: Full OTLP Logging (Advanced)

```dockerfile
ENV OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED=true
ENV OTEL_LOGS_EXPORTER=otlp
ENV OTEL_EXPORTER_OTLP_LOGS_PROTOCOL=http/protobuf
ENV OTEL_EXPORTER_OTLP_LOGS_ENDPOINT=https://logs.eu-central-1.amazonaws.com/v1/logs
ENV OTEL_EXPORTER_OTLP_LOGS_HEADERS=x-aws-log-group=/aws/bedrock-agentcore/runtimes/YOUR_AGENT_ID,x-aws-log-stream=runtime-logs
CMD ["opentelemetry-instrument", "python", "agent.py"]
```

**Requirements:**
- Must create the log group and log stream beforehand
- Must configure proper IAM permissions for CloudWatch Logs
- More complex setup and troubleshooting

## AWS Documentation References

According to AWS documentation:

1. **For AgentCore**: The recommended approach is to use `opentelemetry-instrument` with normal logging
2. **For OTLP Logs**: You must set all required environment variables for the complete pipeline
3. **Python Specific**: `OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED=true` requires `OTEL_PYTHON_DISTRO=aws_distro` and `OTEL_PYTHON_CONFIGURATOR=aws_configurator`

## Current Working Configuration

The current Dockerfile uses Option 1, which provides:

- ✅ **Tracing**: OpenTelemetry traces for distributed tracing
- ✅ **Metrics**: OpenTelemetry metrics for performance monitoring  
- ✅ **Logging**: Standard Python logging to CloudWatch logs
- ✅ **AgentCore Integration**: Full compatibility with AgentCore observability

## Troubleshooting

### If you want to enable OTLP logging:

1. **Enable the environment variables** in the Dockerfile (uncomment Option 2)
2. **Update the log group name** to match your actual AgentCore runtime ID
3. **Ensure IAM permissions** include CloudWatch Logs access
4. **Redeploy the agent** with CDK

### If logs are missing:

1. **Check the environment variables** are set correctly
2. **Verify the log group exists** in CloudWatch
3. **Check IAM permissions** for the execution role
4. **Use the log viewing tool** to see all log streams

## Best Practices

1. **Start Simple**: Use `OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED=false` for initial development
2. **Enable OTLP Gradually**: Only enable OTLP logging if you need advanced log correlation features
3. **Monitor Both**: Use both CloudWatch logs and OpenTelemetry traces for comprehensive observability
4. **Test Thoroughly**: Always test logging configuration changes in a development environment first

## Example: Testing Your Configuration

```bash
# Deploy the agent
cd examples/agentcore/infrastructure
cdk deploy --require-approval never

# Test the agent
cd ../agent/tests
python -m pytest test_agent_deployment.py::TestAgentDeployment::test_invoke_agent_basic -v

# Check the logs
cd ..
python tools/view-agent-logs.py
```

This will show you exactly what logs are being generated and where they're going.