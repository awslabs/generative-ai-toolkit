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

# Copy application files (copy all Python files for the module)
COPY --chown=agentcore:agentcore *.py ./

# Switch to non-root user
USER agentcore

EXPOSE 8080

# Health check endpoint (AgentCore provides /ping and /health endpoints)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/ping || exit 1

CMD ["python", "agent.py"]
