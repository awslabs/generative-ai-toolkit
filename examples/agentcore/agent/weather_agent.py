"""
Copyright 2024 Amazon.com, Inc. and its affiliates. All Rights Reserved.

Licensed under the Amazon Software License (the "License").
You may not use this file except in compliance with the License.
A copy of the License is located at

  http://aws.amazon.com/asl/

or in the "license" file accompanying this file. This file is distributed
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
express or implied. See the License for the specific language governing
permissions and limitations under the License.
"""

"""
Weather Agent with MCP Client Integration

This module implements a weather agent that uses the Generative AI Toolkit's
BedrockConverseAgent with MCP (Model Context Protocol) client integration.

The agent demonstrates:
1. Integration with BedrockConverseAgent for LLM interactions
2. MCP client for accessing weather tools from a separate MCP server
3. Session management and conversation history
4. Proper error handling and logging
5. Tracing and observability features
"""

import os
import logging
import asyncio
import textwrap
import boto3
from typing import Dict, Any, Optional, List
from datetime import datetime

from generative_ai_toolkit.agent import BedrockConverseAgent
from generative_ai_toolkit.tracer import InMemoryTracer, TeeTracer
from generative_ai_toolkit.context import AgentContext

try:
    from generative_ai_toolkit.tracer.otlp import OtlpTracer
except ImportError:
    OtlpTracer = None

from models import AgentCoreRequest, AgentCoreResponse
from agentcore_mcp_client import AgentCoreMCPClient, AgentCoreMCPToolWrapper


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WeatherAgent:
    """
    Weather agent that integrates BedrockConverseAgent with MCP tools.

    This agent provides weather information by:
    1. Using BedrockConverseAgent for natural language processing
    2. Connecting to an MCP server for weather tool access
    3. Managing conversation sessions and history
    4. Maintaining tracing and observability
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        mcp_server_endpoint: Optional[str] = None,
        region: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ):
        """
        Initialize the weather agent.

        Args:
            model_id: The Bedrock model ID to use (defaults to BEDROCK_MODEL_ID env var)
            mcp_server_endpoint: ARN of the MCP server runtime endpoint (defaults to MCP_SERVER_ENDPOINT env var)
            region: AWS region for Bedrock (defaults to AWS_REGION env var)
            max_tokens: Maximum tokens for model responses (defaults to MAX_TOKENS env var)
            temperature: Temperature setting for model responses (defaults to TEMPERATURE env var)
        """
        # Use environment variables as defaults if parameters not provided
        self.model_id = model_id or os.getenv(
            "BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0"
        )
        self.mcp_server_endpoint = mcp_server_endpoint or os.getenv(
            "MCP_SERVER_ENDPOINT"
        )
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        self.max_tokens = max_tokens or int(os.getenv("MAX_TOKENS", "4096"))
        self.temperature = temperature or float(os.getenv("TEMPERATURE", "0.1"))

        # Session storage for conversation history
        self._sessions: Dict[str, List[Dict[str, Any]]] = {}

        # Initialize tracer for observability
        self.tracer = self._initialize_tracer()

        # Initialize MCP client
        self.mcp_client = None

        # Initialize the agent
        self._initialize_agent()

        logger.info(f"WeatherAgent initialized with model {self.model_id}")
        if self.mcp_server_endpoint:
            logger.info(f"MCP Server Endpoint: {self.mcp_server_endpoint}")
        else:
            logger.warning(
                "No MCP Server Endpoint configured - agent will run without MCP tools"
            )

    def _initialize_tracer(self):
        """Initialize tracer with support for multiple backends."""
        try:
            # Always include in-memory tracer for local access
            tracers = [InMemoryTracer()]

            # Add OTLP tracer for X-Ray and CloudWatch if configured
            otlp_endpoint = os.getenv("OTLP_ENDPOINT")
            if otlp_endpoint and OtlpTracer:
                try:
                    otlp_tracer = OtlpTracer(host="localhost", port=4318)
                    tracers.append(otlp_tracer)
                    logger.info(
                        f"OTLP tracer configured with endpoint: {otlp_endpoint}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize OTLP tracer: {str(e)}")

            # Use TeeTracer if multiple tracers, otherwise single tracer
            if len(tracers) > 1:
                return TeeTracer(tracers)
            else:
                return tracers[0]

        except Exception as e:
            logger.error(f"Failed to initialize tracer: {str(e)}")
            # Fallback to in-memory tracer
            return InMemoryTracer()

    def _initialize_agent(self):
        """Initialize the BedrockConverseAgent with MCP tools."""
        try:
            # Create AWS session for the specified region
            session = boto3.Session()
            bedrock_client = session.client("bedrock-runtime", region_name=self.region)

            # Create the agent with basic configuration
            self.agent = BedrockConverseAgent(
                model_id=self.model_id,
                bedrock_client=bedrock_client,
                tracer=self.tracer,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system_prompt=self._get_system_prompt(),
            )

            # Initialize MCP client for tool access
            self._initialize_mcp_client()

            logger.info("BedrockConverseAgent initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize agent: {str(e)}")
            raise

    def _initialize_mcp_client(self):
        """Initialize the MCP client and register tools with the agent."""
        try:
            # Check if MCP server endpoint is configured
            if not self.mcp_server_endpoint:
                logger.warning(
                    "MCP_SERVER_ENDPOINT not configured, skipping MCP client initialization"
                )
                self.mcp_client = None
                return

            # Create AgentCore MCP client
            self.mcp_client = AgentCoreMCPClient(
                mcp_server_endpoint_arn=self.mcp_server_endpoint, region=self.region
            )

            # Connect to MCP server and get available tools
            # Note: MCP setup will be done lazily on first use to avoid event loop issues

            logger.info("AgentCore MCP client initialized and tools registered")

        except Exception as e:
            logger.error(f"Failed to initialize MCP client: {str(e)}")
            # Continue without MCP tools - agent can still function
            self.mcp_client = None

    async def _setup_mcp_tools(self):
        """Set up MCP tools by connecting to the server and registering them."""
        if not self.mcp_client:
            return

        try:
            # Connect to MCP server
            await self.mcp_client.connect()

            # Get available tools from MCP server
            tools = await self.mcp_client.list_tools()

            # Register each tool with the agent
            for tool in tools:
                self._register_mcp_tool(tool)

            logger.info(f"Registered {len(tools)} MCP tools with agent")

        except Exception as e:
            logger.error(f"Failed to setup MCP tools: {str(e)}")
            raise

    async def _ensure_mcp_tools_setup(self):
        """Ensure MCP tools are set up (lazy initialization)."""
        if self.mcp_client and not self.mcp_client.is_connected():
            await self._setup_mcp_tools()

    def _register_mcp_tool(self, tool_info: Dict[str, Any]):
        """Register an MCP tool with the BedrockConverseAgent."""
        tool_name = tool_info.get("name")
        tool_description = tool_info.get("description", "")
        tool_schema = tool_info.get("inputSchema", {})

        # Create AgentCore MCP tool wrapper
        tool_wrapper = AgentCoreMCPToolWrapper(
            mcp_client=self.mcp_client,
            tool_name=tool_name,
            tool_description=tool_description,
            tool_schema=tool_schema,
        )

        def traced_tool_wrapper(**kwargs):
            """Wrapper that adds tracing to MCP tool calls."""
            try:
                # Add trace information
                context = AgentContext.current()
                if context and context.tracer:
                    context.tracer.trace(
                        "mcp_tool_call",
                        {
                            "tool_name": tool_name,
                            "arguments": kwargs,
                            "mcp_server_endpoint": self.mcp_server_endpoint,
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                    )

                # Call the tool
                result = tool_wrapper(**kwargs)

                # Add result to trace
                if context and context.tracer:
                    context.tracer.trace(
                        "mcp_tool_result",
                        {
                            "tool_name": tool_name,
                            "result": result,
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                    )

                return result

            except Exception as e:
                logger.error(f"MCP tool call failed for {tool_name}: {str(e)}")
                return {"error": f"Tool call failed: {str(e)}"}

        # Set function metadata for the agent
        traced_tool_wrapper.__name__ = tool_name
        traced_tool_wrapper.__doc__ = tool_description

        # Register the tool with the agent
        self.agent.register_tool(traced_tool_wrapper)

        logger.debug(f"Registered AgentCore MCP tool: {tool_name}")

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the weather agent."""
        return textwrap.dedent(
            """
            You are a helpful weather assistant that provides accurate and timely weather information.

            When users ask about weather:
            1. Use the appropriate weather tools to get current information
            2. Provide clear, helpful responses with specific details
            3. Include relevant safety information for severe weather
            4. Ask for clarification if the location is unclear

            Always be helpful, accurate, and prioritize user safety when discussing weather conditions.
            """
        )

    def get_session_history(
        self, session_id: str, last_k_turns: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session.

        Args:
            session_id: The session identifier
            last_k_turns: Number of recent turns to include

        Returns:
            List of conversation turns
        """
        if session_id not in self._sessions:
            return []

        # Return the last k turns
        history = self._sessions[session_id]
        return history[-last_k_turns:] if last_k_turns > 0 else history

    def update_session_history(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Update the conversation history for a session.

        Args:
            session_id: The session identifier
            user_message: The user's message
            assistant_response: The assistant's response
            metadata: Optional metadata about the interaction
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = []

        # Add user message
        self._sessions[session_id].append(
            {
                "role": "user",
                "content": user_message,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        # Add assistant response
        self._sessions[session_id].append(
            {
                "role": "assistant",
                "content": assistant_response,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": metadata,
            }
        )

        # Keep only recent history to prevent memory issues
        max_history = 100  # Keep last 100 messages
        if len(self._sessions[session_id]) > max_history:
            self._sessions[session_id] = self._sessions[session_id][-max_history:]

    async def process_request(self, request: AgentCoreRequest) -> AgentCoreResponse:
        """
        Process an AgentCore request and return a response.

        Args:
            request: The AgentCore request containing prompt and session info

        Returns:
            AgentCore response with the agent's reply
        """
        try:
            logger.info(f"Processing request for session {request.session_id}")

            # Ensure MCP tools are set up
            await self._ensure_mcp_tools_setup()

            # Get conversation history
            history = self.get_session_history(request.session_id, request.last_k_turns)

            # Prepare conversation context
            conversation = []

            # Add history to conversation
            for turn in history:
                conversation.append(
                    {"role": turn["role"], "content": [{"text": turn["content"]}]}
                )

            # Add current user message
            conversation.append({"role": "user", "content": [{"text": request.prompt}]})

            # Create agent context with tracing
            with AgentContext(tracer=self.tracer):
                # Get response from agent
                response = await self.agent.arun(
                    conversation=conversation, session_id=request.session_id
                )

            # Extract response text
            response_text = response.get("content", [{}])[0].get("text", "")

            # Update session history
            self.update_session_history(
                request.session_id,
                request.prompt,
                response_text,
                {"model_id": self.model_id, "timestamp": datetime.utcnow().isoformat()},
            )

            # Get trace information for metadata
            traces = self.tracer.get_traces()
            metadata = {
                "trace_count": len(traces),
                "model_id": self.model_id,
                "timestamp": datetime.utcnow().isoformat(),
            }

            logger.info(
                f"Successfully processed request for session {request.session_id}"
            )

            return AgentCoreResponse(
                response=response_text, session_id=request.session_id, metadata=metadata
            )

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")

            # Return error response
            return AgentCoreResponse(
                response=f"I apologize, but I encountered an error while processing your request: {str(e)}",
                session_id=request.session_id,
                metadata={"error": str(e), "timestamp": datetime.utcnow().isoformat()},
            )

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of the agent and its components.

        Returns:
            Dictionary with health status information
        """
        health_info = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "agent_initialized": self.agent is not None,
            "model_id": self.model_id,
            "mcp_client_connected": self.mcp_client is not None,
            "active_sessions": len(self._sessions),
            "tracer_type": type(self.tracer).__name__,
            "tracing_enabled": True,
        }

        # Check MCP client health if available
        if self.mcp_client:
            try:
                # Simple connectivity check
                health_info["mcp_server_endpoint"] = self.mcp_server_endpoint
                health_info["mcp_status"] = (
                    "connected" if self.mcp_client.is_connected() else "disconnected"
                )
                health_info["mcp_session_id"] = self.mcp_client.get_session_id()
            except Exception as e:
                health_info["mcp_status"] = f"error: {str(e)}"
        else:
            health_info["mcp_status"] = "not_configured"

        return health_info

    def cleanup(self):
        """Clean up resources when shutting down."""
        try:
            if self.mcp_client:
                asyncio.run(self.mcp_client.disconnect())
            logger.info("WeatherAgent cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
