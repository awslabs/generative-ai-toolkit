"""
Simple authentication for MCP OAuth integration with AWS Cognito.

Provides basic OAuth Bearer token authentication for short-running examples.
No token refresh needed for examples that run for just a few seconds.
"""

import boto3
from botocore.exceptions import ClientError


class AuthenticationError(Exception):
    """Authentication failed."""

    pass


class SimpleAuth:
    """Simple authentication for MCP OAuth integration."""

    def __init__(self, region: str = None):
        """Initialize with AWS region."""
        self.region = region or boto3.Session().region_name
        self.cognito_client = boto3.client("cognito-idp", region_name=self.region)

    def get_bearer_token(
        self, user_pool_id: str, client_id: str, username: str, password: str
    ) -> str:
        """
        Get Bearer token from Cognito using username/password.

        Args:
            user_pool_id: Cognito User Pool ID
            client_id: Cognito App Client ID
            username: Username
            password: Password

        Returns:
            Bearer token string

        Raises:
            AuthenticationError: When authentication fails
        """
        try:
            response = self.cognito_client.initiate_auth(
                ClientId=client_id,
                AuthFlow="USER_PASSWORD_AUTH",
                AuthParameters={"USERNAME": username, "PASSWORD": password},
            )

            if "AuthenticationResult" in response:
                return response["AuthenticationResult"]["AccessToken"]
            else:
                raise AuthenticationError("Authentication failed - no token received")

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            error_message = e.response["Error"]["Message"]

            if error_code in ["NotAuthorizedException", "UserNotFoundException"]:
                raise AuthenticationError(
                    f"Invalid credentials: {error_message}"
                ) from e
            else:
                raise AuthenticationError(
                    f"Authentication failed: {error_message}"
                ) from e
