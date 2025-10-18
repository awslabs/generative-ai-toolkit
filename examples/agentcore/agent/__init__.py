"""
Agent package for MCP OAuth authentication.

Simple configuration management and OAuth authentication helpers.
"""

from .config_loader import ConfigLoader, UserCredentials
from .simple_auth import AuthenticationError, SimpleAuth
from .simple_mcp_client import SimpleMcpClient

__all__ = [
    "SimpleAuth",
    "AuthenticationError",
    "ConfigLoader",
    "UserCredentials",
    "SimpleMcpClient",
]
