"""
Integration tests for the auth_helper module.

Tests McpAuthHelper with real AWS Cognito services.
These tests require a deployed CDK stack with Cognito configuration.
"""

import boto3
from agent.auth_helper import McpAuthHelper, TokenResponse
from agent.config_loader import ConfigLoader


def get_cognito_config():
    """Load Cognito configuration from CloudFormation stack outputs."""
    config_loader = ConfigLoader()
    stack_name = config_loader.get_cdk_stack_name()

    # Get stack outputs from CloudFormation
    cf_client = boto3.client("cloudformation", region_name=config_loader.region)

    try:
        response = cf_client.describe_stacks(StackName=stack_name)
        stacks = response.get("Stacks", [])

        if not stacks:
            raise RuntimeError(
                f"CloudFormation stack '{stack_name}' not found. Ensure CDK stack is deployed."
            )

        stack = stacks[0]
        outputs = stack.get("Outputs", [])

        # Convert outputs to dict
        output_dict = {}
        for output in outputs:
            output_dict[output["OutputKey"]] = output["OutputValue"]

        # Find outputs by key patterns (CDK adds unique suffixes)
        user_pool_id = None
        client_id = None
        discovery_url = None

        for key, value in output_dict.items():
            if "UserPoolId" in key and "ClientId" not in key:
                user_pool_id = value
            elif "UserPoolClientId" in key:
                client_id = value
            elif "UserPoolDiscoveryUrl" in key:
                discovery_url = value

        return {
            "user_pool_id": user_pool_id,
            "client_id": client_id,
            "discovery_url": discovery_url,
        }

    except Exception as e:
        raise RuntimeError(
            f"Failed to get CloudFormation outputs for stack '{stack_name}': {e}. "
            f"Ensure CDK stack is deployed with 'cdk deploy'."
        ) from e


class TestMcpAuthHelperIntegration:
    """Integration tests with real Cognito."""

    def test_get_bearer_token_with_real_cognito(self):
        """Test Bearer token retrieval with real Cognito."""
        # Load real configuration
        config_loader = ConfigLoader()
        credentials = config_loader.get_credentials()
        cognito_config = get_cognito_config()

        # Validate configuration
        if not cognito_config["user_pool_id"]:
            raise RuntimeError(
                "UserPoolId not found in CDK outputs. Ensure CDK stack is deployed."
            )
        if not cognito_config["client_id"]:
            raise RuntimeError(
                "UserPoolClientId not found in CDK outputs. Ensure CDK stack is deployed."
            )

        # Test authentication
        helper = McpAuthHelper()
        token = helper.get_bearer_token(
            user_pool_id=cognito_config["user_pool_id"],
            client_id=cognito_config["client_id"],
            username=credentials.username,
            password=credentials.password,
        )

        # Verify token
        assert token is not None, "Bearer token should not be None"
        assert isinstance(token, str), "Bearer token should be a string"
        assert len(token) > 0, "Bearer token should not be empty"
        assert "." in token, "Bearer token should be a JWT with dots"

        # Verify token is stored
        current_token = helper.get_current_token()
        assert current_token is not None, "Current token should be stored"
        assert current_token.access_token == token, (
            "Stored token should match returned token"
        )

    def test_create_authenticated_headers(self):
        """Test creation of authenticated headers."""
        helper = McpAuthHelper()
        token = "test-bearer-token"

        headers = helper.create_authenticated_headers(token)

        assert headers["Authorization"] == "Bearer test-bearer-token"
        assert headers["Content-Type"] == "application/json"

    def test_token_refresh_with_real_cognito(self):
        """Test token refresh with real Cognito."""
        # First get a token to have a refresh token
        config_loader = ConfigLoader()
        credentials = config_loader.get_credentials()
        cognito_config = get_cognito_config()

        helper = McpAuthHelper()

        # Get initial token
        initial_token = helper.get_bearer_token(
            user_pool_id=cognito_config["user_pool_id"],
            client_id=cognito_config["client_id"],
            username=credentials.username,
            password=credentials.password,
        )

        # Get refresh token
        current_token = helper.get_current_token()
        assert current_token is not None, (
            "Should have current token after authentication"
        )
        assert current_token.refresh_token, "Should have refresh token"

        # Test refresh
        refreshed_token = helper.refresh_token(
            refresh_token=current_token.refresh_token,
            client_id=cognito_config["client_id"],
        )

        # Verify refreshed token
        assert refreshed_token is not None, "Refreshed token should not be None"
        assert isinstance(refreshed_token, str), "Refreshed token should be a string"
        assert refreshed_token != initial_token, (
            "Refreshed token should be different from initial"
        )

    def test_token_response_creation(self):
        """Test TokenResponse creation and expiration calculation."""
        token = TokenResponse(
            access_token="test-token", refresh_token="refresh-token", expires_in=3600
        )

        assert token.access_token == "test-token"
        assert token.refresh_token == "refresh-token"
        assert token.expires_in == 3600
        assert token.token_type == "Bearer"
        assert token.expires_at is not None
