"""
AgentCore request/response models for HTTP interface integration.

These Pydantic models define the data structures used for communication
between AgentCore Runtime and the agent container.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class AgentCoreRequest(BaseModel):
    """
    Request payload from AgentCore Runtime to the agent.

    This represents the JSON payload sent to the /invocations endpoint
    when AgentCore invokes the agent.
    """

    prompt: str = Field(description="The user's input prompt or message")
    session_id: str = Field(
        description="Unique identifier for the conversation session"
    )
    last_k_turns: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of previous conversation turns to include for context",
    )


class AgentCoreResponse(BaseModel):
    """
    Response payload from the agent back to AgentCore Runtime.

    This represents the JSON response returned from the /invocations endpoint
    after the agent processes the request.
    """

    response: str = Field(description="The agent's response message to the user")
    session_id: str = Field(
        description="The same session ID from the request for correlation"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata about the response (traces, metrics, etc.)",
    )


class HealthCheckResponse(BaseModel):
    """
    Response for the /ping health check endpoint.

    Used by AgentCore to verify the agent container is healthy and ready
    to receive requests.
    """

    status: str = Field(default="healthy", description="Health status of the agent")
    version: Optional[str] = Field(
        default=None, description="Optional version information"
    )
    timestamp: Optional[str] = Field(
        default=None, description="Optional timestamp of the health check"
    )
