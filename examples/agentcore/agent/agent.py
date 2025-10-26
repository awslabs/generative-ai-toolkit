#!/usr/bin/env python3
"""Weather Agent for AgentCore Runtime using Generative AI Toolkit with MCP integration."""

import asyncio
import base64
import json
import logging
import os
import sys

import boto3
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.runtime.context import RequestContext
from mcp_tool_manager import McpToolManager

from generative_ai_toolkit.agent import BedrockConverseAgent
from generative_ai_toolkit.tracer import InMemoryTracer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Agent starting - logger configured")


def validate_environment_variables():
    """Validate required environment variables are present."""
    required_vars = {
        "AWS_REGION": "AWS region for Bedrock service",
        "BEDROCK_MODEL_ID": "Bedrock model identifier",
        "MCP_SERVER_RUNTIME_ARN": "MCP server runtime ARN for tool integration",
        "OAUTH_CREDENTIALS_SECRET_NAME": "OAuth credentials secret name for authentication",
        "OAUTH_USER_POOL_ID": "OAuth Cognito User Pool ID for authentication",
        "OAUTH_USER_POOL_CLIENT_ID": "OAuth Cognito User Pool Client ID for authentication",
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
    logger.info(f"  MCP_SERVER_RUNTIME_ARN: {os.environ.get('MCP_SERVER_RUNTIME_ARN')}")
    logger.info(
        f"  OAUTH_CREDENTIALS_SECRET_NAME: {os.environ.get('OAUTH_CREDENTIALS_SECRET_NAME')}"
    )
    logger.info(f"  OAUTH_USER_POOL_ID: {os.environ.get('OAUTH_USER_POOL_ID')}")
    logger.info(
        f"  OAUTH_USER_POOL_CLIENT_ID: {os.environ.get('OAUTH_USER_POOL_CLIENT_ID')}"
    )


# Validate environment on startup
validate_environment_variables()

app = BedrockAgentCoreApp()

# Global MCP tool manager instance
mcp_manager = McpToolManager()


def extract_session_info(payload: dict[str, object]) -> tuple[str, str]:
    """Extract session ID and user message from AgentCore payload."""
    logger.info(f"Received invocation: {payload}")

    # Extract session ID for correlation (AgentCore observability best practice)
    session_id = payload.get("sessionId") or payload.get("session_id", "unknown")
    logger.info(f"Processing request for session: {session_id}")

    # Extract prompt from AgentCore format
    if "input" not in payload or "prompt" not in payload["input"]:
        raise ValueError(
            "Invalid payload format. Expected: {'input': {'prompt': 'message'}}"
        )

    user_message = str(payload["input"]["prompt"])
    logger.info(f"Processing message: {user_message[:100]}...")

    return session_id, user_message


def create_bedrock_agent(tracer=None) -> BedrockConverseAgent:
    """Create and configure the Bedrock Converse Agent."""
    region_name = os.environ["AWS_REGION"]  # Required env var, validated at startup
    model_id = os.environ["BEDROCK_MODEL_ID"]  # Required env var, validated at startup

    session = boto3.Session(region_name=region_name)

    agent_kwargs = {
        "model_id": model_id,
        "session": session,
        "system_prompt": "You are a helpful weather assistant. You have access to weather tools to provide accurate, real-time weather information. Use the available tools when users ask about weather conditions, forecasts, or related information.",
    }

    if tracer:
        agent_kwargs["tracer"] = tracer

    return BedrockConverseAgent(**agent_kwargs)


def register_mcp_tools_safely(agent: BedrockConverseAgent) -> bool:
    """Register MCP tools with the agent, handling errors gracefully."""
    try:
        # Use a simple approach - just use asyncio.run and let the MCP manager handle connection state
        tools_registered = asyncio.run(mcp_manager.register_mcp_tools(agent))

        if tools_registered:
            logger.info("MCP tools available for this request")
            return True
        else:
            logger.warning("MCP tools not available - using agent without tools")
            return False
    except Exception as mcp_error:
        logger.error(f"MCP tool registration failed: {mcp_error}")
        logger.info("Continuing without MCP tools")
        return False


def handle_bedrock_validation_error(error_msg: str) -> dict[str, str]:
    """Handle specific Bedrock ValidationException errors."""
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
        logger.error("Use an inference profile ID instead of the direct model ID")
        return {
            "result": f"Model access error: {os.environ.get('BEDROCK_MODEL_ID')} requires an inference profile. Use an inference profile ID instead."
        }
    return None


def handle_error(e: Exception) -> dict[str, str]:
    """Handle and categorize different types of errors."""
    error_msg = str(e)

    # Handle Bedrock ValidationException errors
    if "ValidationException" in error_msg:
        bedrock_error_response = handle_bedrock_validation_error(error_msg)
        if bedrock_error_response:
            return bedrock_error_response

    logger.error(f"Error processing invocation: {e}", exc_info=True)

    # Check if this is an MCP-related error
    if "mcp" in error_msg.lower() or "tool" in error_msg.lower():
        return {
            "result": "I'm sorry, but I'm currently unable to access weather tools. Please try again later or contact support if the issue persists."
        }

    return {"result": f"Error: {error_msg}"}


# Module-level agent instance - created once and reused
bedrock_agent: BedrockConverseAgent = create_bedrock_agent(tracer=InMemoryTracer())

# Register MCP tools
register_mcp_tools_safely(bedrock_agent)


@app.entrypoint
def invoke(payload: dict[str, object], context: RequestContext) -> dict[str, str]:
    """Process agent invocation from AgentCore Runtime."""

    logger.info("Agent invoked.")

    try:

        # Check for Authorization header and extract user info from JWT
        if context.request_headers and "Authorization" in context.request_headers:
            auth_header = context.request_headers["Authorization"]

            # Extract JWT token (remove "Bearer " prefix)
            if auth_header.startswith("Bearer "):
                jwt_token = auth_header[7:]  # Remove "Bearer " prefix

                # Decode JWT token to extract user information (without verification for info extraction)
                try:

                    # Decode JWT payload (second part after splitting by '.')
                    parts = jwt_token.split(".")
                    if len(parts) >= 2:
                        # Add padding if needed for base64 decoding
                        payload_b64 = parts[1]
                        payload_b64 += "=" * (4 - len(payload_b64) % 4)
                        payload_bytes = base64.urlsafe_b64decode(payload_b64)
                        jwt_claims = json.loads(payload_bytes)

                        # Extract user information
                        user_id = jwt_claims.get("sub")
                        username = jwt_claims.get("username")

                        if user_id:
                            logger.info(
                                f"Processing request for user: {user_id} (username: {username})"
                            )
                        else:
                            logger.info("No user ID found in JWT claims")
                    else:
                        logger.warning("Invalid JWT token format")

                except Exception as e:
                    logger.error(f"Error decoding JWT token: {e}")
            else:
                logger.warning("Authorization header doesn't start with 'Bearer '")
        else:
            logger.info("No Authorization header found in request_headers")
            if context.request_headers:
                logger.info(
                    f"Available headers: {list(context.request_headers.keys())}"
                )
            else:
                logger.info("request_headers is None")

        # Extract session information and user message
        session_id, user_message = extract_session_info(payload)

        # Get response from agent (with or without tools)
        logger.info("Calling Bedrock Converse API")
        response = bedrock_agent.converse(user_message)

        logger.info(f"Sending response: {response[:100]}...")
        return {"result": response}

    except Exception as e:
        return handle_error(e)


if __name__ == "__main__":
    app.run()
