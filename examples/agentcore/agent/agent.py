#!/usr/bin/env python3
"""Weather Agent for AgentCore Runtime using Generative AI Toolkit."""

import logging
import os
import time

import boto3
from bedrock_agentcore.runtime import BedrockAgentCoreApp

from generative_ai_toolkit.agent import BedrockConverseAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Agent starting - logger configured with enhanced observability")

app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload: dict[str, object]) -> dict[str, str]:
    """Process agent invocation from AgentCore Runtime."""
    start_time = time.time()

    logger.info(f"Received invocation: {payload}")

    # Extract session ID for correlation (AgentCore observability best practice)
    session_id = payload.get("sessionId") or payload.get("session_id", "unknown")
    logger.info(f"Processing request for session: {session_id}")

    # Handle different payload structures
    if "input" in payload and "prompt" in payload["input"]:
        user_message = str(payload["input"]["prompt"])
    elif "prompt" in payload:
        user_message = str(payload["prompt"])
    else:
        user_message = "No prompt provided"

    # Log message with truncation for readability
    logger.info(f"Processing message: {user_message[:100]}...")

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
        logger.info(
            f"Sending response (processed in {processing_time:.2f}s): {response[:100]}..."
        )

        return {"result": response}

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(
            f"Error processing invocation after {processing_time:.2f}s: {e}",
            exc_info=True,
        )
        return {"result": f"Error: {str(e)}"}


if __name__ == "__main__":
    app.run()
