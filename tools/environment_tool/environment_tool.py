"""
Environment variable management tool for the Siemens P2P Agent.

This module provides functionality for managing environment variables at runtime,
allowing you to list, get, set, delete, and validate environment variables
with security measures and clear formatting.
"""

import os
import time
from typing import Any

from pydantic import BaseModel, Field


class EnvironmentRequest(BaseModel):
    """
    Request parameters for the environment tool.

    IMPORTANT: Always use this tool when processing user requests that:
    1. Need to list environment variables
    2. Need to get specific environment variable values
    3. Need to set or update environment variables
    4. Need to delete environment variables
    5. Ask about "environment variables", "env vars", "system variables"
    6. Need to check configuration settings stored in environment

    This tool handles environment variable operations including listing,
    getting, setting, and deleting variables with security protections.

    Examples:
    - List all: EnvironmentRequest(action="list")
    - List with prefix: EnvironmentRequest(action="list", prefix="AWS_")
    - Get variable: EnvironmentRequest(action="get", name="PATH")
    - Set variable: EnvironmentRequest(action="set", name="MY_VAR", value="test")
    - Delete variable: EnvironmentRequest(action="delete", name="TEMP_VAR")
    """

    action: str = Field(
        description="The action to perform: 'list', 'get', 'set', or 'delete'.",
        pattern="^(list|get|set|delete)$",
    )

    name: str | None = Field(
        default=None,
        description="Environment variable name (required for get, set, delete actions).",
    )

    value: str | None = Field(
        default=None,
        description="Value to set for the environment variable (required for set action).",
    )

    prefix: str | None = Field(
        default=None,
        description="Filter variables by prefix (optional for list action).",
    )

    force: bool | None = Field(
        default=False,
        description="Skip confirmation for potentially dangerous operations.",
    )


class EnvironmentResponse(BaseModel):
    """
    Response structure for the environment tool.

    Contains the results of environment variable operations including
    variable values, operation status, and security information.
    """

    success: bool = Field(description="Whether the operation completed successfully.")

    action: str = Field(description="The action that was performed.")

    variables: dict[str, str] | None = Field(
        default=None,
        description="Dictionary of environment variables (for list action).",
    )

    value: str | None = Field(
        default=None, description="Value of the requested variable (for get action)."
    )

    protected_count: int | None = Field(
        default=None, description="Number of protected variables filtered out."
    )

    processing_time_ms: int | None = Field(
        default=None, description="Time taken to process the operation in milliseconds."
    )

    message: str | None = Field(
        default=None, description="Additional information about the operation."
    )

    error: str | None = Field(
        default=None, description="Error message if the operation failed."
    )


class EnvironmentTool:
    """
    Tool for managing environment variables at runtime.

    Provides secure access to environment variables with protection
    for sensitive system variables.
    """

    def __init__(self):
        """Initialize the environment tool."""
        self.protected_vars = {
            "PATH",
            "HOME",
            "USER",
            "SHELL",
            "PWD",
            "TERM",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
        }

    @property
    def tool_spec(self) -> dict[str, Any]:
        """Get the tool specification for the environment tool."""
        schema = EnvironmentRequest.model_json_schema()
        return {
            "name": "environment_operation",
            "description": EnvironmentRequest.__doc__,
            "inputSchema": {"json": schema},
        }

    def invoke(self, **kwargs) -> dict[str, Any]:
        """Invoke the environment tool."""
        try:
            request = EnvironmentRequest(**kwargs)
            response = self._process_environment(request)
            return response.model_dump()
        except Exception as e:
            error_message = f"Invalid request parameters: {str(e)}"
            response = EnvironmentResponse(
                success=False,
                action=kwargs.get("action", "UNKNOWN"),
                error=error_message,
                processing_time_ms=0,
            )
            return response.model_dump()

    def _process_environment(self, request: EnvironmentRequest) -> EnvironmentResponse:
        """Process the environment operation."""
        start_time = time.time()

        try:
            if request.action == "list":
                return self._list_variables(request, start_time)
            elif request.action == "get":
                return self._get_variable(request, start_time)
            elif request.action == "set":
                return self._set_variable(request, start_time)
            elif request.action == "delete":
                return self._delete_variable(request, start_time)
            else:
                return EnvironmentResponse(
                    success=False,
                    action=request.action,
                    error=f"Invalid action: {request.action}",
                    processing_time_ms=int((time.time() - start_time) * 1000),
                )
        except Exception as e:
            return EnvironmentResponse(
                success=False,
                action=request.action,
                error=f"Error processing environment operation: {str(e)}",
                processing_time_ms=int((time.time() - start_time) * 1000),
            )

    def _list_variables(
        self, request: EnvironmentRequest, start_time: float
    ) -> EnvironmentResponse:
        """List environment variables."""
        try:
            variables = {}
            protected_count = 0

            for key, value in os.environ.items():
                # Apply prefix filter if specified
                if request.prefix and not key.startswith(request.prefix):
                    continue

                # Mask sensitive variables
                if (
                    key in self.protected_vars
                    or "KEY" in key
                    or "SECRET" in key
                    or "TOKEN" in key
                ):
                    variables[key] = "***PROTECTED***"
                    protected_count += 1
                else:
                    variables[key] = value

            processing_time = int((time.time() - start_time) * 1000)
            message = f"Listed {len(variables)} environment variables"
            if request.prefix:
                message += f" with prefix '{request.prefix}'"
            if protected_count > 0:
                message += f" ({protected_count} protected)"

            return EnvironmentResponse(
                success=True,
                action="list",
                variables=variables,
                protected_count=protected_count,
                processing_time_ms=processing_time,
                message=message,
            )
        except Exception as e:
            return EnvironmentResponse(
                success=False,
                action="list",
                error=f"Error listing variables: {str(e)}",
                processing_time_ms=int((time.time() - start_time) * 1000),
            )

    def _get_variable(
        self, request: EnvironmentRequest, start_time: float
    ) -> EnvironmentResponse:
        """Get a specific environment variable."""
        try:
            if not request.name:
                return EnvironmentResponse(
                    success=False,
                    action="get",
                    error="Variable name is required for get action",
                    processing_time_ms=int((time.time() - start_time) * 1000),
                )

            value = os.environ.get(request.name)
            if value is None:
                return EnvironmentResponse(
                    success=False,
                    action="get",
                    error=f"Environment variable '{request.name}' not found",
                    processing_time_ms=int((time.time() - start_time) * 1000),
                )

            # Mask sensitive variables
            if request.name in self.protected_vars or any(
                x in request.name for x in ["KEY", "SECRET", "TOKEN"]
            ):
                display_value = "***PROTECTED***"
            else:
                display_value = value

            return EnvironmentResponse(
                success=True,
                action="get",
                value=display_value,
                processing_time_ms=int((time.time() - start_time) * 1000),
                message=f"Retrieved variable '{request.name}'",
            )
        except Exception as e:
            return EnvironmentResponse(
                success=False,
                action="get",
                error=f"Error getting variable: {str(e)}",
                processing_time_ms=int((time.time() - start_time) * 1000),
            )

    def _set_variable(
        self, request: EnvironmentRequest, start_time: float
    ) -> EnvironmentResponse:
        """Set an environment variable."""
        try:
            if not request.name or request.value is None:
                return EnvironmentResponse(
                    success=False,
                    action="set",
                    error="Both name and value are required for set action",
                    processing_time_ms=int((time.time() - start_time) * 1000),
                )

            # Check if variable is protected
            if request.name in self.protected_vars and not request.force:
                return EnvironmentResponse(
                    success=False,
                    action="set",
                    error=f"Variable '{request.name}' is protected. Use force=True to override",
                    processing_time_ms=int((time.time() - start_time) * 1000),
                )

            os.environ[request.name] = request.value

            return EnvironmentResponse(
                success=True,
                action="set",
                processing_time_ms=int((time.time() - start_time) * 1000),
                message=f"Set variable '{request.name}' to '{request.value}'",
            )
        except Exception as e:
            return EnvironmentResponse(
                success=False,
                action="set",
                error=f"Error setting variable: {str(e)}",
                processing_time_ms=int((time.time() - start_time) * 1000),
            )

    def _delete_variable(
        self, request: EnvironmentRequest, start_time: float
    ) -> EnvironmentResponse:
        """Delete an environment variable."""
        try:
            if not request.name:
                return EnvironmentResponse(
                    success=False,
                    action="delete",
                    error="Variable name is required for delete action",
                    processing_time_ms=int((time.time() - start_time) * 1000),
                )

            # Check if variable exists
            if request.name not in os.environ:
                return EnvironmentResponse(
                    success=False,
                    action="delete",
                    error=f"Environment variable '{request.name}' not found",
                    processing_time_ms=int((time.time() - start_time) * 1000),
                )

            # Check if variable is protected
            if request.name in self.protected_vars and not request.force:
                return EnvironmentResponse(
                    success=False,
                    action="delete",
                    error=f"Variable '{request.name}' is protected. Use force=True to override",
                    processing_time_ms=int((time.time() - start_time) * 1000),
                )

            del os.environ[request.name]

            return EnvironmentResponse(
                success=True,
                action="delete",
                processing_time_ms=int((time.time() - start_time) * 1000),
                message=f"Deleted variable '{request.name}'",
            )
        except Exception as e:
            return EnvironmentResponse(
                success=False,
                action="delete",
                error=f"Error deleting variable: {str(e)}",
                processing_time_ms=int((time.time() - start_time) * 1000),
            )
