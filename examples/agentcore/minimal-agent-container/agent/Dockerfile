# syntax=docker/dockerfile:1
FROM python:3.13-slim

# Build argument for version
ARG AGENT_VERSION=unknown
ENV AGENT_VERSION=${AGENT_VERSION}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user
RUN useradd -m -u 1000 agentcore
USER agentcore

# Copy application files
COPY --chown=agentcore:agentcore *.py .

# Set OpenTelemetry environment variables
ENV OTEL_PYTHON_DISTRO=aws_distro
ENV OTEL_PYTHON_CONFIGURATOR=aws_configurator
ENV OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
ENV OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED=true
ENV OTEL_EXPORTER_OTLP_LOGS_HEADERS=x-aws-log-stream=runtime-logs,x-aws-metric-namespace=bedrock-agentcore

EXPOSE 8080

# Health check endpoint - AgentCore runtime health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Use OpenTelemetry instrumentation
CMD ["opentelemetry-instrument", "python", "agent.py"]
