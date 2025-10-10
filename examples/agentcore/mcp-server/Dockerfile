# Copyright 2024 Amazon.com, Inc. and its affiliates. All Rights Reserved.
#
# Licensed under the Amazon Software License (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#   http://aws.amazon.com/asl/
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# Weather Tools MCP Server for AgentCore Runtime
# Use ARM64 platform and Amazon Linux 2023 as required by AgentCore Runtime
FROM --platform=linux/arm64 public.ecr.aws/amazonlinux/amazonlinux:2023

# Update system and install Python 3.11 and build dependencies
RUN dnf update -y && \
    dnf install -y python3.11 python3.11-pip python3.11-devel gcc gcc-c++ && \
    dnf clean all

# Create application directory
RUN mkdir -p /opt/amazon

# Set working directory
WORKDIR /opt/amazon

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Install required OpenTelemetry and application dependencies
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY weather_models.py .
COPY weather_tools.py .
COPY mcp_server.py .
COPY test_server.py .

# Create non-root user for security (Amazon Linux commands)
RUN /usr/sbin/groupadd -r mcpuser && /usr/sbin/useradd -r -g mcpuser mcpuser && \
    chown -R mcpuser:mcpuser /opt/amazon

USER mcpuser

# Set up runtime environment for AgentCore
ENV PYTHONPATH=/opt/amazon
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8000
ENV HOST=0.0.0.0

# Expose port 8000 (required for MCP protocol)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3.11 -c "import requests; requests.get('http://localhost:8000/health').raise_for_status()" || exit 1

# AgentCore Runtime entrypoint with OpenTelemetry instrumentation
ENTRYPOINT ["opentelemetry-instrument", "python3.11", "mcp_server.py"]

# Default command arguments for AgentCore Runtime
CMD ["--host", "0.0.0.0", "--port", "8000"]