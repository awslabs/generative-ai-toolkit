# AgentCore Built-in Observability Features Guide

## Overview

Amazon Bedrock AgentCore provides comprehensive built-in observability features that automatically capture metrics, traces, and logs for your agents without requiring code modifications. This guide explains what's available by default and how to enhance observability in your agent code.

## What AgentCore Provides by Default

### 1. **Runtime Metrics** (Automatic)
AgentCore automatically provides these metrics at 1-minute intervals:

- **Invocations**: Total number of requests to your agent
- **Latency**: End-to-end processing time
- **System Errors**: Server-side errors (500s)
- **User Errors**: Client-side errors (400s)
- **Throttles**: Rate limiting events (429s)
- **Session Count**: Number of active sessions
- **Total Errors**: Combined error percentage

### 2. **Resource Usage Metrics** (Automatic)
- **CPU Usage**: vCPU-Hours consumed
- **Memory Usage**: GB-Hours consumed
- Available at account, agent runtime, and endpoint levels

### 3. **Application Logs** (Automatic)
- Standard output and error messages from your agent
- Located in: `/aws/bedrock-agentcore/runtimes/<agent_id>-<endpoint_name>/[runtime-logs]`
- Includes request/response payloads when enabled

### 4. **Built-in Spans** (When Observability Enabled)
- **InvokeAgentRuntime** spans with attributes:
  - Operation name, resource ARN, request ID
  - Agent ID, endpoint name, session ID
  - Latency, error type, account ID, region

## Current Agent.py Analysis

Your current `agent.py` is already well-positioned to work with AgentCore's observability:

```python
# ✅ Good: Standard Python logging works with AgentCore
logger.info("Agent starting - logger configured")
logger.info(f"Received invocation: {payload}")
logger.info(f"Processing message: {user_message}")
logger.info(f"Sending response: {response}")
```

**What's Working:**
- ✅ Python logging goes directly to CloudWatch logs
- ✅ AgentCore automatically captures runtime metrics
- ✅ Request/response logging provides audit trail
- ✅ Error handling is visible in logs

## Enhanced Observability Options

### Option 1: Add Custom Spans (Recommended)

To add detailed tracing without interfering with basic logging:

```python
#!/usr/bin/env python3
"""Weather Agent for AgentCore Runtime using Generative AI Toolkit."""

import logging
import os
from typing import Dict, Any

import boto3
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from generative_ai_toolkit.agent import BedrockConverseAgent

# Optional: Add OpenTelemetry for custom spans
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    tracer = trace.get_tracer(__name__)
    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False
    tracer = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Agent starting - logger configured")

app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload: Dict[str, Any]) -> Dict[str, str]:
    """Process agent invocation from AgentCore Runtime."""
    
    # Start custom span if tracing is available
    if TRACING_AVAILABLE and tracer:
        with tracer.start_as_current_span("agent.invoke") as span:
            return _process_invocation(payload, span)
    else:
        return _process_invocation(payload, None)


def _process_invocation(payload: Dict[str, Any], span=None) -> Dict[str, str]:
    """Process the agent invocation with optional tracing."""
    logger.info(f"Received invocation: {payload}")
    
    # Add span attributes if available
    if span:
        span.set_attribute("agent.payload_keys", list(payload.keys()))
    
    # Handle different payload structures
    if "input" in payload and "prompt" in payload["input"]:
        user_message = str(payload["input"]["prompt"])
    elif "prompt" in payload:
        user_message = str(payload["prompt"])
    else:
        user_message = "No prompt provided"
    
    logger.info(f"Processing message: {user_message}")
    
    if span:
        span.set_attribute("agent.user_message_length", len(user_message))
        span.add_event("message_extracted", {"message_preview": user_message[:50]})
    
    try:
        # Create Generative AI Toolkit agent
        with tracer.start_as_current_span("agent.create_bedrock_agent") if span else nullcontext():
            region_name = os.environ.get("AWS_REGION", "us-east-1")
            session = boto3.Session(region_name=region_name)
            
            agent = BedrockConverseAgent(
                model_id=os.environ.get(
                    "BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0"
                ),
                session=session,
                system_prompt="You are a helpful weather assistant. Provide accurate weather information when asked.",
            )
        
        # Get response from agent
        with tracer.start_as_current_span("agent.converse") if span else nullcontext():
            response = agent.converse(user_message)
        
        logger.info(f"Sending response: {response}")
        
        if span:
            span.set_attribute("agent.response_length", len(response))
            span.set_attribute("agent.success", True)
            span.set_status(Status(StatusCode.OK))
        
        return {"result": response}
        
    except Exception as e:
        logger.error(f"Error processing invocation: {e}", exc_info=True)
        
        if span:
            span.set_attribute("agent.error", str(e))
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
        
        return {"result": f"Error: {str(e)}"}


# Null context manager for when tracing is not available
class nullcontext:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


if __name__ == "__main__":
    app.run()
```

### Option 2: Add Custom Metrics

```python
# Optional: Add custom metrics
try:
    from opentelemetry import metrics
    meter = metrics.get_meter(__name__)
    
    # Create custom metrics
    invocation_counter = meter.create_counter(
        "agent.invocations.custom",
        description="Custom invocation counter"
    )
    
    processing_time = meter.create_histogram(
        "agent.processing_time",
        description="Time spent processing requests"
    )
    
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

# In your invoke function:
if METRICS_AVAILABLE:
    start_time = time.time()
    invocation_counter.add(1, {"agent_type": "weather"})
    
    # ... process request ...
    
    processing_time.record(time.time() - start_time)
```

### Option 3: Session ID Propagation

```python
# Add session ID support for better tracing correlation
def invoke(payload: Dict[str, Any]) -> Dict[str, str]:
    """Process agent invocation with session tracking."""
    
    # Extract session ID if available
    session_id = payload.get("sessionId") or payload.get("session_id")
    
    if session_id:
        logger.info(f"Processing request for session: {session_id}")
        
        # Add to span if tracing is available
        if span:
            span.set_attribute("session.id", session_id)
    
    # ... rest of processing
```

## Minimal Enhancement (Recommended for Your Use Case)

For your current agent, I recommend this minimal enhancement that adds observability without complexity:

```python
#!/usr/bin/env python3
"""Weather Agent for AgentCore Runtime using Generative AI Toolkit."""

import logging
import os
import time
from typing import Dict, Any

import boto3
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from generative_ai_toolkit.agent import BedrockConverseAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Agent starting - logger configured")

app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload: Dict[str, Any]) -> Dict[str, str]:
    """Process agent invocation from AgentCore Runtime."""
    start_time = time.time()
    
    logger.info(f"Received invocation: {payload}")
    
    # Extract session ID for correlation
    session_id = payload.get("sessionId") or payload.get("session_id", "unknown")
    logger.info(f"Processing request for session: {session_id}")
    
    # Handle different payload structures
    if "input" in payload and "prompt" in payload["input"]:
        user_message = str(payload["input"]["prompt"])
    elif "prompt" in payload:
        user_message = str(payload["prompt"])
    else:
        user_message = "No prompt provided"
    
    logger.info(f"Processing message: {user_message[:100]}...")  # Truncate for logs
    
    try:
        # Create Generative AI Toolkit agent
        region_name = os.environ.get("AWS_REGION", "us-east-1")
        session = boto3.Session(region_name=region_name)
        
        agent = BedrockConverseAgent(
            model_id=os.environ.get(
                "BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0"
            ),
            session=session,
            system_prompt="You are a helpful weather assistant. Provide accurate weather information when asked.",
        )
        
        # Get response from agent
        logger.info("Calling Bedrock Converse API")
        response = agent.converse(user_message)
        
        processing_time = time.time() - start_time
        logger.info(f"Sending response (processed in {processing_time:.2f}s): {response[:100]}...")
        
        return {"result": response}
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error processing invocation after {processing_time:.2f}s: {e}", exc_info=True)
        return {"result": f"Error: {str(e)}"}


if __name__ == "__main__":
    app.run()
```

## What You Get with AgentCore Observability

### 1. **CloudWatch Dashboards**
- Access via CloudWatch → GenAI Observability → Bedrock AgentCore
- Automatic graphs for invocations, latency, errors
- Resource usage visualization

### 2. **Log Correlation**
- All logs automatically tagged with request IDs
- Session-level correlation available
- Error tracking across requests

### 3. **Distributed Tracing**
- Automatic spans for AgentCore operations
- Integration with AWS X-Ray when enabled
- Custom spans if you add OpenTelemetry

### 4. **Alerting**
- Set up CloudWatch alarms on any metric
- Monitor error rates, latency spikes
- Resource usage alerts

## Best Practices

### ✅ Do:
1. **Use structured logging** with consistent formats
2. **Include session IDs** for correlation
3. **Log processing times** for performance monitoring
4. **Truncate large payloads** in logs to avoid noise
5. **Use appropriate log levels** (INFO for normal flow, ERROR for exceptions)

### ❌ Don't:
1. **Log sensitive data** (credentials, PII)
2. **Over-instrument** - start simple and add as needed
3. **Use `opentelemetry-instrument`** - it conflicts with AgentCore
4. **Ignore built-in metrics** - they provide most of what you need

## Viewing Your Observability Data

### CloudWatch Console:
1. Go to CloudWatch → GenAI Observability
2. Select "Bedrock AgentCore" tab
3. Find your agent runtime
4. View metrics, traces, and logs

### Log Groups:
- **Application Logs**: `/aws/bedrock-agentcore/runtimes/<agent-id>-DEFAULT/[runtime-logs]`
- **Spans**: `/aws/spans/default` (when tracing enabled)
- **OTEL Logs**: `/aws/bedrock-agentcore/runtimes/<agent-id>-DEFAULT/otel-rt-logs`

## Conclusion

Your current `agent.py` already works well with AgentCore's built-in observability. The automatic metrics, logging, and resource tracking provide comprehensive monitoring without any code changes. 

For enhanced observability, consider the minimal enhancement approach above, which adds session correlation and timing information while maintaining simplicity and reliability.