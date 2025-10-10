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

# Weather Agent Dockerfile for AgentCore Runtime
#
# This Dockerfile creates a container image for the weather agent that is
# compatible with Amazon Bedrock AgentCore Runtime requirements.

# Use Amazon Linux 2023 as required by AgentCore Runtime
FROM public.ecr.aws/amazonlinux/amazonlinux:2023

# Update system and install Python 3.12 and build dependencies
RUN dnf update -y && \
    dnf install -y python3.12 python3.12-pip python3.12-devel gcc gcc-c++ shadow-utils && \
    dnf install -y --allowerasing curl && \
    dnf clean all

# Create application directory
RUN mkdir -p /opt/amazon

# Set working directory
WORKDIR /opt/amazon

# Set environment variables for AgentCore Runtime
ENV PYTHONPATH=/opt/amazon
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV FLASK_APP=app.py
ENV PORT=8080
ENV HOST=0.0.0.0

# Weather Agent Configuration
ENV BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
ENV MCP_SERVER_ENDPOINT=""
ENV AWS_REGION=us-east-1
ENV MAX_TOKENS=4096
ENV TEMPERATURE=0.1
ENV LOG_LEVEL=INFO

# Copy requirements first for better caching
COPY requirements.txt .

# Install required OpenTelemetry and application dependencies
RUN python3.12 -m pip install --no-cache-dir aws_opentelemetry_distro_genai_beta>=0.1.4
RUN python3.12 -m pip install --no-cache-dir --upgrade pip && \
    python3.12 -m pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security (Amazon Linux commands)
RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    chown -R appuser:appuser /opt/amazon

USER appuser

# Expose port 8080 (required by AgentCore Runtime)
EXPOSE 8080

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/ping || exit 1

# AgentCore Runtime entrypoint with OpenTelemetry instrumentation
ENTRYPOINT ["opentelemetry-instrument", "python3.12", "app.py"]

# Default command arguments for AgentCore Runtime
CMD ["--host", "0.0.0.0", "--port", "8080"]