# syntax=docker/dockerfile:1
FROM python:3.13-slim

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
COPY --chown=agentcore:agentcore agent.py .

EXPOSE 8080

# Health check endpoint (AgentCore requires /ping endpoint)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/ping || exit 1

CMD ["python", "agent.py"]
