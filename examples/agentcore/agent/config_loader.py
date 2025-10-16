"""
Simple configuration loader for MCP OAuth authentication.

Loads CDK stack name from .env file and retrieves credentials from AWS Secrets Manager.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import boto3


@dataclass
class UserCredentials:
    """User credentials from AWS Secrets Manager."""

    username: str
    password: str


class ConfigLoader:
    """Simple configuration loader for MCP OAuth authentication."""

    def __init__(self, region: str = None):
        self.region = region or boto3.Session().region_name

    def get_cdk_stack_name(self) -> str:
        """Get CDK stack name from .env file in project root."""
        # Check environment variable first
        if stack_name := os.getenv("CDK_STACK_NAME"):
            return stack_name

        # Read from .env file relative to this script's location
        script_dir = Path(__file__).parent
        env_path = script_dir / "../.env"
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if line.startswith("CDK_STACK_NAME="):
                        return line.split("=", 1)[1].strip()

        raise RuntimeError("CDK_STACK_NAME not found in environment or .env file")

    def get_credentials(self) -> UserCredentials:
        """Get user credentials from AWS Secrets Manager."""
        stack_name = self.get_cdk_stack_name()
        secret_name = f"{stack_name}/test-user/credentials"

        client = boto3.client("secretsmanager", region_name=self.region)
        response = client.get_secret_value(SecretId=secret_name)

        credentials_data = json.loads(response["SecretString"])
        return UserCredentials(
            username=credentials_data["username"], password=credentials_data["password"]
        )
