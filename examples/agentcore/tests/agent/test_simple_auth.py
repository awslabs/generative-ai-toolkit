"""
Integration tests for the simple_auth module.

Tests SimpleAuth with real AWS Cognito services.
These tests require a deployed CDK stack with Cognito configuration.
"""

import pytest
from agent.config_loader import ConfigLoader
from agent.simple_auth import AuthenticationError, SimpleAuth


class TestSimpleAuthIntegration:
    """Integration tests for SimpleAuth with real Cognito services."""

    @pytest.fixture(scope="class")
    def config_loader(self):
        """Create ConfigLoader instance."""
        return ConfigLoader()

    @pytest.fixture(scope="class")
    def simple_auth(self, config_loader):
        """Create SimpleAuth instance."""
        return SimpleAuth(region=config_loader.region)

    @pytest.fixture(scope="class")
    def cognito_config(self, config_loader):
        """Get Cognito configuration from deployed CDK stack."""
        return config_loader.get_cognito_config()

    @pytest.fixture(scope="class")
    def credentials(self, config_loader):
        """Get user credentials from Secrets Manager."""
        return config_loader.get_credentials()

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

    def test_config_loader_get_cognito_config(self, config_loader):
        """Test that ConfigLoader can get Cognito configuration from CDK stack."""
        try:
            user_pool_id, client_id = config_loader.get_cognito_config()

            # Verify we got valid configuration
            assert user_pool_id is not None
            assert client_id is not None
            assert isinstance(user_pool_id, str)
            assert isinstance(client_id, str)
            assert len(user_pool_id) > 0
            assert len(client_id) > 0

            # Basic format checks
            assert user_pool_id.startswith(config_loader.region), (
                "User Pool ID should start with region"
            )
            assert "_" in user_pool_id, "User Pool ID should contain underscore"

            print("✅ Successfully loaded Cognito config:")
            print(f"  User Pool ID: {user_pool_id}")
            print(f"  Client ID: {client_id}")

        except Exception as e:
            pytest.fail(f"Failed to get Cognito configuration from CDK stack: {e}")

    def test_config_loader_get_credentials(self, config_loader):
        """Test that ConfigLoader can get credentials from Secrets Manager."""
        try:
            credentials = config_loader.get_credentials()

            # Verify we got valid credentials
            assert credentials is not None
            assert credentials.username is not None
            assert credentials.password is not None
            assert isinstance(credentials.username, str)
            assert isinstance(credentials.password, str)
            assert len(credentials.username) > 0
            assert len(credentials.password) > 0

            print(
                f"✅ Successfully loaded credentials for user: {credentials.username}"
            )

        except Exception as e:
            pytest.fail(f"Failed to get credentials from Secrets Manager: {e}")

    def test_end_to_end_authentication_flow(self, config_loader, simple_auth):
        """Test complete end-to-end authentication flow."""
        try:
            # Step 1: Load configuration from CDK stack
            user_pool_id, client_id = config_loader.get_cognito_config()
            credentials = config_loader.get_credentials()

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
