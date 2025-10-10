#!/bin/bash

# Run the agent container locally with ADOT for development
# This simulates the AgentCore runtime environment

echo "Running agent container locally with ADOT..."

docker run --rm -it \
  -p 8080:8080 \
  -e AGENT_VERSION=6 \
  -e AWS_REGION=eu-central-1 \
  -e OTEL_PYTHON_DISTRO=aws_distro \
  -e OTEL_PYTHON_CONFIGURATOR=aws_configurator \
  -e OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf \
  -e OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED=true \
  -e OTEL_SERVICE_NAME=minimal_agent.local \
  -e OTEL_RESOURCE_ATTRIBUTES="service.name=minimal_agent.local,deployment.environment=local" \
  --name agent-local \
  minimal-agentcore-agent:latest
