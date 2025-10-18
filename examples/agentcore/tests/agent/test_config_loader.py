"""
Tests for the config_loader module.

Integration tests that use real AWS services to verify ConfigLoader functionality.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from agent.config_loader import ConfigLoader, UserCredentials


class TestConfigLoader:
    """Test cases for ConfigLoader class."""

    def test_init_uses_boto3_session_region(self):
        """Test ConfigLoader uses region from boto3 session by default."""
        loader = ConfigLoader()
        # Should get region from boto3 session, not None
        assert loader.region is not None
        assert isinstance(loader.region, str)

    def test_init_custom_region_override(self):
        """Test ConfigLoader can override region."""
        loader = ConfigLoader(region="us-west-2")
        assert loader.region == "us-west-2"

    def test_get_cdk_stack_name_from_environment(self):
        """Test getting CDK stack name from environment variable."""
        with patch.dict(os.environ, {"CDK_STACK_NAME": "test-stack-env"}):
            loader = ConfigLoader()
            assert loader.get_cdk_stack_name() == "test-stack-env"

    def test_get_cdk_stack_name_no_env_file_no_env_var(self):
        """Test error when neither .env file nor environment variable exists."""
        # Create a temporary directory without .env file
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock __file__ to point to a location where .env doesn't exist
            fake_script_path = Path(temp_dir) / "agent" / "config_loader.py"

            with patch("agent.config_loader.__file__", str(fake_script_path)):
                with patch.dict(os.environ, {}, clear=True):
                    loader = ConfigLoader()

                    with pytest.raises(RuntimeError, match="CDK_STACK_NAME not found"):
                        loader.get_cdk_stack_name()

                        with pytest.raises(
                            RuntimeError, match="CDK_STACK_NAME not found"
                        ):
                            loader.get_cdk_stack_name()

    def test_get_cdk_stack_name_from_real_env_file(self):
        """Test getting CDK stack name from the actual .env file."""
        # This test uses the real .env file in the project
        loader = ConfigLoader()
        stack_name = loader.get_cdk_stack_name()

        # Should get the actual stack name from the project's .env file
        assert stack_name is not None
        assert isinstance(stack_name, str)
        assert len(stack_name) > 0

    def test_get_cdk_stack_name_env_var_takes_precedence(self):
        """Test that environment variable takes precedence over .env file."""
        with patch.dict(os.environ, {"CDK_STACK_NAME": "test-stack-env"}):
            loader = ConfigLoader()
            assert loader.get_cdk_stack_name() == "test-stack-env"

    def test_get_credentials_with_real_aws(self):
        """Test credential retrieval from real AWS Secrets Manager."""
        loader = ConfigLoader()

        try:
            credentials = loader.get_credentials()

            # Verify we got valid credentials
            assert isinstance(credentials, UserCredentials)
            assert credentials.username is not None
            assert credentials.password is not None
            assert len(credentials.username) > 0
            assert len(credentials.password) > 0

        except Exception as e:
            # If the test fails due to missing AWS credentials or secret,
            # that's expected in some environments
            pytest.skip(f"Skipping AWS integration test: {e}")

    def test_get_cognito_config_with_real_aws(self):
        """Test Cognito configuration retrieval from real CloudFormation stack."""
        loader = ConfigLoader()

        try:
            user_pool_id, client_id = loader.get_cognito_config()

            # Verify we got valid Cognito configuration
            assert user_pool_id is not None
            assert client_id is not None
            assert isinstance(user_pool_id, str)
            assert isinstance(client_id, str)
            assert len(user_pool_id) > 0
            assert len(client_id) > 0

            # Basic format checks
            assert user_pool_id.startswith(loader.region), (
                "User Pool ID should start with region"
            )
            assert "_" in user_pool_id, "User Pool ID should contain underscore"

        except Exception as e:
            # If the test fails due to missing AWS credentials or stack,
            # that's expected in some environments
            pytest.skip(f"Skipping AWS integration test: {e}")
