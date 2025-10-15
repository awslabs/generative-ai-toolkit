#!/usr/bin/env python3
"""Weather Agent for AgentCore Runtime using Generative AI Toolkit."""

import logging
import os
import sys

import boto3
from bedrock_agentcore.runtime import BedrockAgentCoreApp

from generative_ai_toolkit.agent import BedrockConverseAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Agent starting - logger configured")


def validate_environment_variables():
    """Validate required environment variables are present."""
    required_vars = {
        "AWS_REGION": "AWS region for Bedrock service",
        "BEDROCK_MODEL_ID": "Bedrock model identifier",
    }

    missing_vars = []
    for var, description in required_vars.items():
        if not os.environ.get(var):
            missing_vars.append(f"{var} ({description})")

    if missing_vars:
        logger.error("Missing required environment variables:")
        for var in missing_vars:
            logger.error(f"  - {var}")
        logger.error("Please set these environment variables before running the agent.")
        sys.exit(1)

    # Log key configuration
    logger.info("Key agent configuration:")
    logger.info(f"  AWS_REGION: {os.environ.get('AWS_REGION')}")
    logger.info(f"  BEDROCK_MODEL_ID: {os.environ.get('BEDROCK_MODEL_ID')}")


# Validate environment on startup
validate_environment_variables()

app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload: dict[str, object]) -> dict[str, str]:
    """Process agent invocation from AgentCore Runtime."""
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
        region_name = os.environ["AWS_REGION"]  # Required env var, validated at startup
        model_id = os.environ[
            "BEDROCK_MODEL_ID"
        ]  # Required env var, validated at startup

        session = boto3.Session(region_name=region_name)

        agent = BedrockConverseAgent(
            model_id=model_id,
            session=session,
            system_prompt="You are a helpful weather assistant. Provide accurate weather information when asked.",
        )

        # Get response from agent
        logger.info("Calling Bedrock Converse API")
        response = agent.converse(user_message)

        logger.info(f"Sending response: {response[:100]}...")

        return {"result": response}

    except Exception as e:
        # Enhanced error handling for common Bedrock issues
        error_msg = str(e)
        if "ValidationException" in error_msg:
            if "model identifier is invalid" in error_msg:
                logger.error(
                    f"Model '{os.environ.get('BEDROCK_MODEL_ID')}' is not available in region '{os.environ.get('AWS_REGION')}'"
                )
                logger.error(
                    "Please check available models with: aws bedrock list-foundation-models --region <your-region>"
                )
                return {
                    "result": f"Model configuration error: {os.environ.get('BEDROCK_MODEL_ID')} is not available in {os.environ.get('AWS_REGION')}"
                }
            elif "on-demand throughput isn't supported" in error_msg:
                logger.error(
                    f"Model '{os.environ.get('BEDROCK_MODEL_ID')}' requires an inference profile for access"
                )
                logger.error(
                    "Please check available inference profiles with: aws bedrock list-inference-profiles --region <your-region>"
                )
                logger.error(
                    "Use an inference profile ID instead of the direct model ID"
                )
                return {
                    "result": f"Model access error: {os.environ.get('BEDROCK_MODEL_ID')} requires an inference profile. Use an inference profile ID instead."
                }

        logger.error(f"Error processing invocation: {e}", exc_info=True)
        return {"result": f"Error: {error_msg}"}


if __name__ == "__main__":
    app.run()
