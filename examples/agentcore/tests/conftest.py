"""Unified pytest configuration for AgentCore tests."""

import json
import os
import sys
from pathlib import Path
from typing import Any

import boto3
import pytest


def get_cdk_stack_name() -> str:
    """Get CDK stack name from .env file."""

    # First try environment variable
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


# Set required environment variables to prevent SystemExit during test collection
# This must be done before any agent module imports
def _setup_required_environment_variables():
    """Set up required environment variables from CDK outputs or defaults."""

    # Set defaults for basic configuration
    os.environ["AWS_REGION"] = boto3.Session().region_name or "eu-central-1"
    os.environ["BEDROCK_MODEL_ID"] = "eu.anthropic.claude-sonnet-4-20250514-v1:0"

    # Get CDK stack name
    stack_name = get_cdk_stack_name()

    # Get CDK outputs
    cf_client = boto3.client("cloudformation")
    response = cf_client.describe_stacks(StackName=stack_name)
    stack_outputs = response["Stacks"][0]["Outputs"]

    # Set environment variables from CDK outputs
    for output in stack_outputs:
        key = output["OutputKey"]
        value = output["OutputValue"]

        # MCP Server Runtime ARN
        if "McpServerRuntimeArn" in key and "Endpoint" not in key:
            os.environ["MCP_SERVER_RUNTIME_ARN"] = value
        # Agent OAuth Credentials Secret Name (legacy output name)
        elif "AgentOAuthCredentialsSecretName" in key:
            os.environ["OAUTH_CREDENTIALS_SECRET_NAME"] = value
        # Agent User Credentials Secret Name (new output name)
        elif "AgentUserCredentialsSecretName" in key:
            os.environ["OAUTH_CREDENTIALS_SECRET_NAME"] = value
        # Cognito User Pool ID
        elif "CognitoAuthUserPoolId" in key:
            os.environ["OAUTH_USER_POOL_ID"] = value
        # Cognito User Pool Client ID
        elif "CognitoAuthUserPoolClientId" in key:
            os.environ["OAUTH_USER_POOL_CLIENT_ID"] = value
        # Client User Credentials Secret Name
        elif "ClientUserCredentialsSecretName" in key:
            os.environ["CLIENT_USER_CREDENTIALS_SECRET_NAME"] = value
        # Agent Runtime ARN (not endpoint)
        elif "AgentRuntimeArn" in key and "Endpoint" not in key:
            os.environ["AGENT_RUNTIME_ARN"] = value
        # Agent Runtime Endpoint ARN
        elif "AgentRuntimeEndpointArn" in key:
            os.environ["AGENT_RUNTIME_ENDPOINT_ARN"] = value
        # MCP Server Runtime Endpoint ARN
        elif "McpServerRuntimeEndpointArn" in key:
            os.environ["MCP_SERVER_RUNTIME_ENDPOINT_ARN"] = value


# Set up environment variables before any imports
_setup_required_environment_variables()

# Add agent directory to Python path for imports
agent_dir = Path(__file__).parent.parent / "agent"
sys.path.insert(0, str(agent_dir))


@pytest.fixture(scope="session")
def cdk_outputs() -> dict[str, Any]:
    """Get CDK stack outputs for testing deployed resources."""
    try:
        stack_name = get_cdk_stack_name()

        # Create CloudFormation client (uses environment variables for region)
        cf_client = boto3.client("cloudformation")

        # Get stack outputs using AWS SDK
        response = cf_client.describe_stacks(StackName=stack_name)

        outputs = {}
        if "Stacks" in response and len(response["Stacks"]) > 0:
            stack_outputs = response["Stacks"][0].get("Outputs", [])
            for output in stack_outputs:
                outputs[output["OutputKey"]] = output["OutputValue"]

        return outputs

    except Exception as e:
        raise RuntimeError(
            f"Failed to get CDK outputs for stack '{stack_name}': {e}. "
            f"Ensure CDK stack is deployed with 'cdk deploy'."
        ) from e


@pytest.fixture(scope="session")
def bedrock_agentcore_control_client():
    """Create Bedrock AgentCore Control client for testing."""
    return boto3.client("bedrock-agentcore-control")


@pytest.fixture(scope="session")
def bedrock_agentcore_client():
    """Create Bedrock AgentCore client for testing runtime invocation."""
    return boto3.client("bedrock-agentcore")


# JWT Authentication fixtures


@pytest.fixture(scope="session")
def client_credentials() -> dict[str, str]:
    """Retrieve client user credentials from AWS Secrets Manager."""
    secret_name = os.environ["CLIENT_USER_CREDENTIALS_SECRET_NAME"]
    region = os.environ["AWS_REGION"]

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region)

    get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    secret = json.loads(get_secret_value_response["SecretString"])
    return {"username": secret["username"], "password": secret["password"]}


@pytest.fixture(scope="session")
def jwt_token(client_credentials: dict[str, str]) -> str:
    """Authenticate with Cognito and get JWT access token."""
    client_id = os.environ["OAUTH_USER_POOL_CLIENT_ID"]
    region = os.environ["AWS_REGION"]

    # Create Cognito Identity Provider client
    cognito_client = boto3.client("cognito-idp", region_name=region)

    response = cognito_client.initiate_auth(
        ClientId=client_id,
        AuthFlow="USER_PASSWORD_AUTH",
        AuthParameters={
            "USERNAME": client_credentials["username"],
            "PASSWORD": client_credentials["password"],
        },
    )

    access_token = response["AuthenticationResult"]["AccessToken"]
    assert access_token, "JWT access token should not be empty"
    return access_token


# Agent-specific fixtures - now using environment variables


# MCP Server-specific fixtures - now using environment variables
