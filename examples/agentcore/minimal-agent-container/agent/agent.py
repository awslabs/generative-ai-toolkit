#!/usr/bin/env python3
"""Minimal AgentCore Runtime agent using Generative AI Toolkit."""

import logging
import os

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from generative_ai_toolkit.agent import BedrockConverseAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get version from environment
agent_version = os.environ.get("AGENT_VERSION", "unknown")
logger.info(f"Agent starting - logger configured - version: {agent_version}")

app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload: dict[str, object]) -> dict[str, str]:
    """Process agent invocation from AgentCore Runtime.

    Args:
        payload: Request payload containing user input

    Returns:
        Dictionary with agent response
    """
    logger.info(f"Agent version: {agent_version}")
    logger.info(f"Received invocation: {payload}")

    user_message = str(payload.get("input", {}).get("prompt", "No prompt provided"))
    logger.info(f"Processing message: {user_message}")

    region_name = os.environ.get("AWS_REGION")
    logger.info(f"Selected region: {region_name}")

    # Create Generative AI Toolkit agent with detected region
    import boto3
    session = boto3.Session(region_name=region_name)
    agent = BedrockConverseAgent(
        model_id="eu.amazon.nova-micro-v1:0",
        session=session,
        system_prompt=f"You are a helpful assistant. Always mention that you are running as agent version {agent_version}.",
    )

    # Get response from agent
    response = agent.converse(user_message)
    logger.info(f"Sending response: {response}")

    return {"result": response}


if __name__ == "__main__":
    app.run()
