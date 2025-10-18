#!/usr/bin/env python3
"""Weather Agent for AgentCore Runtime using Generative AI Toolkit with MCP integration."""

import asyncio
import logging
import os
import sys

import boto3
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from simple_mcp_client import SimpleMcpClient

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

    # Log MCP configuration (optional)
    mcp_arn = os.environ.get("MCP_SERVER_RUNTIME_ARN")
    if mcp_arn:
        logger.info(f"  MCP_SERVER_RUNTIME_ARN: {mcp_arn}")
        logger.info("  MCP integration: ENABLED")
    else:
        logger.info("  MCP integration: DISABLED (no MCP_SERVER_RUNTIME_ARN provided)")


# Validate environment on startup
validate_environment_variables()

app = BedrockAgentCoreApp()


class McpToolManager:
    """Manages MCP client and tool registration."""

    def __init__(self):
        self.mcp_client: SimpleMcpClient | None = None
        self.tools_registered = False

    async def get_mcp_client(self) -> SimpleMcpClient | None:
        """Get or create MCP client with lazy initialization."""
        mcp_arn = os.environ.get("MCP_SERVER_RUNTIME_ARN")
        if not mcp_arn:
            logger.warning(
                "No MCP_SERVER_RUNTIME_ARN provided - MCP integration disabled"
            )
            return None

        if self.mcp_client is None:
            try:
                logger.info(f"Initializing MCP client for runtime: {mcp_arn}")
                self.mcp_client = SimpleMcpClient(runtime_arn=mcp_arn)
                await self.mcp_client.connect()
                logger.info("MCP client connected successfully")
            except Exception as e:
                logger.error(f"Failed to initialize MCP client: {e}")
                self.mcp_client = None
                raise

        return self.mcp_client

    async def register_mcp_tools(self, agent: BedrockConverseAgent) -> bool:
        """Register MCP tools with the Generative AI Toolkit agent."""
        if self.tools_registered:
            return True

        try:
            mcp_client = await self.get_mcp_client()
            if not mcp_client:
                return False

            # List available tools from MCP server
            tools_result = await mcp_client.list_tools()

            if not tools_result.tools:
                logger.warning("No tools available from MCP server")
                return False

            logger.info(f"Registering {len(tools_result.tools)} MCP tools with agent")

            # Register each MCP tool with the Generative AI Toolkit
            for mcp_tool in tools_result.tools:
                logger.info(f"Registering tool: {mcp_tool.name}")

                # Create a wrapper function for the MCP tool
                def create_tool_wrapper(tool_name: str):
                    async def tool_wrapper(**kwargs) -> str:
                        """Wrapper function to call MCP tool."""
                        try:
                            client = await self.get_mcp_client()
                            if not client:
                                return "Error: MCP server not available"

                            result = await client.call_tool(tool_name, kwargs)

                            # Extract text content from MCP result
                            if hasattr(result, "content") and result.content:
                                if hasattr(result.content[0], "text"):
                                    return result.content[0].text
                                else:
                                    return str(result.content[0])
                            else:
                                return str(result)

                        except Exception as e:
                            logger.error(f"Error calling MCP tool {tool_name}: {e}")
                            return f"Error calling {tool_name}: {str(e)}"

                    return tool_wrapper

                # Create the wrapper and register it
                tool_func = create_tool_wrapper(mcp_tool.name)
                tool_func.__name__ = mcp_tool.name
                tool_func.__doc__ = mcp_tool.description

                # Register with the agent
                agent.register_tool(tool_func)

            self.tools_registered = True
            logger.info("MCP tools registered successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to register MCP tools: {e}")
            return False


# Global MCP tool manager instance
mcp_manager = McpToolManager()


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
            system_prompt="You are a helpful weather assistant. You have access to weather tools to provide accurate, real-time weather information. Use the available tools when users ask about weather conditions, forecasts, or related information.",
        )

        # Register MCP tools with lazy initialization
        try:
            # Run async tool registration in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                tools_registered = loop.run_until_complete(
                    mcp_manager.register_mcp_tools(agent)
                )
                if tools_registered:
                    logger.info("MCP tools available for this request")
                else:
                    logger.warning(
                        "MCP tools not available - using agent without tools"
                    )
            finally:
                loop.close()
        except Exception as mcp_error:
            logger.error(f"MCP tool registration failed: {mcp_error}")
            logger.info("Continuing without MCP tools")

        # Get response from agent (with or without tools)
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

        # Check if this is an MCP-related error
        if "mcp" in error_msg.lower() or "tool" in error_msg.lower():
            return {
                "result": "I'm sorry, but I'm currently unable to access weather tools. Please try again later or contact support if the issue persists."
            }

        return {"result": f"Error: {error_msg}"}


if __name__ == "__main__":
    app.run()
