"""
Integration tests for the simple_auth module.

Tests SimpleAuth with real AWS Cognito services.
These tests require a deployed CDK stack with Cognito configuration.
"""

import json
import os
from dataclasses import dataclass

import boto3
import pytest
from simple_auth import AuthenticationError, SimpleAuth


class TestSimpleAuthIntegration:
    """Integration tests for SimpleAuth with real Cognito services."""

    @pytest.fixture(scope="class")
    def simple_auth(self):
        """Create SimpleAuth instance."""
        region = boto3.Session().region_name
        return SimpleAuth(region=region)

    @pytest.fixture(scope="class")
    def cognito_config(self):
        """Get Cognito configuration from environment variables."""
        return (
            os.environ["OAUTH_USER_POOL_ID"],
            os.environ["OAUTH_USER_POOL_CLIENT_ID"],
        )

    @pytest.fixture(scope="class")
    def credentials(self):
        """Get user credentials from Secrets Manager using environment variables."""

        @dataclass
        class UserCredentials:
            username: str
            password: str

        secret_name = os.environ["OAUTH_CREDENTIALS_SECRET_NAME"]
        region = boto3.Session().region_name
        client = boto3.client("secretsmanager", region_name=region)
        response = client.get_secret_value(SecretId=secret_name)
        credentials_data = json.loads(response["SecretString"])

        return UserCredentials(
            username=credentials_data["username"], password=credentials_data["password"]
        )

    def test_simple_auth_initialization(self, simple_auth):
        """Test SimpleAuth initialization."""
        assert simple_auth.region is not None
        assert simple_auth.cognito_client is not None

    def test_get_bearer_token_with_valid_credentials(
        self, simple_auth, cognito_config, credentials
    ):
        """Test getting Bearer token with valid credentials."""
        user_pool_id, client_id = cognito_config

        try:
            token = simple_auth.get_bearer_token(
                user_pool_id=user_pool_id,
                client_id=client_id,
                username=credentials.username,
                password=credentials.password,
            )

            # Verify we got a token
            assert token is not None
            assert isinstance(token, str)
            assert len(token) > 0

            # Basic JWT token format check (should have 3 parts separated by dots)
            token_parts = token.split(".")
            assert len(token_parts) == 3, "Bearer token should be a valid JWT"

            print(f"✅ Successfully obtained Bearer token (length: {len(token)})")

        except AuthenticationError as e:
            pytest.fail(f"Authentication failed with valid credentials: {e}")

    def test_get_bearer_token_with_invalid_credentials(
        self, simple_auth, cognito_config
    ):
        """Test getting Bearer token with invalid credentials."""
        user_pool_id, client_id = cognito_config

        with pytest.raises(AuthenticationError) as exc_info:
            simple_auth.get_bearer_token(
                user_pool_id=user_pool_id,
                client_id=client_id,
                username="invalid_user",
                password="invalid_password",
            )

        # Verify the error message indicates invalid credentials
        error_message = str(exc_info.value).lower()
        assert "invalid credentials" in error_message or "unauthorized" in error_message

    def test_get_bearer_token_with_invalid_user_pool(self, simple_auth):
        """Test getting Bearer token with invalid User Pool ID."""
        with pytest.raises(AuthenticationError) as exc_info:
            simple_auth.get_bearer_token(
                user_pool_id="invalid_pool_id",
                client_id="invalid_client_id",
                username="test_user",
                password="test_password",
            )

        # Should get an authentication error
        assert "Authentication failed" in str(exc_info.value)

    def test_end_to_end_authentication_flow(
        self, simple_auth, cognito_config, credentials
    ):
        """Test complete end-to-end authentication flow."""
        try:
            # Step 1: Get configuration from environment variables
            user_pool_id, client_id = cognito_config

            # Step 2: Authenticate and get Bearer token
            token = simple_auth.get_bearer_token(
                user_pool_id=user_pool_id,
                client_id=client_id,
                username=credentials.username,
                password=credentials.password,
            )

            # Step 3: Verify token is valid
            assert token is not None
            assert isinstance(token, str)
            assert len(token) > 0

            # Basic JWT format check
            token_parts = token.split(".")
            assert len(token_parts) == 3

            print("✅ End-to-end authentication flow completed successfully")

        except Exception as e:
            pytest.fail(f"End-to-end authentication flow failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
