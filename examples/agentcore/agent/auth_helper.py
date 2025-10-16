"""
Authentication helper for MCP OAuth integration with AWS Cognito.

This module provides OAuth Bearer token authentication for MCP servers
using AWS Cognito User Pools with the USER_PASSWORD_AUTH flow.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)


@dataclass
class TokenResponse:
    """Response from Cognito authentication containing tokens and metadata."""

    access_token: str
    refresh_token: str
    expires_in: int
    token_type: str = "Bearer"
    expires_at: datetime | None = None

    def __post_init__(self):
        """Calculate expiration time based on expires_in."""
        if self.expires_at is None and self.expires_in:
            self.expires_at = datetime.now(UTC) + timedelta(seconds=self.expires_in)


class AuthenticationError(Exception):
    """Base class for authentication-related errors."""

    pass


class InvalidCredentialsError(AuthenticationError):
    """Raised when username/password authentication fails."""

    pass


class TokenExpiredError(AuthenticationError):
    """Raised when Bearer token has expired."""

    pass


class ConfigurationError(AuthenticationError):
    """Raised when Cognito configuration is invalid."""

    pass


class McpAuthHelper:
    """
    Authentication helper for MCP OAuth integration with AWS Cognito.

    Provides methods for obtaining Bearer tokens, refreshing tokens,
    and creating authenticated headers for MCP server requests.
    """

    def __init__(self, region: str = None) -> None:
        """
        Initialize the authentication helper.

        Args:
            region: AWS region for Cognito service
        """
        self.region = region or boto3.Session().region_name
        self._cognito_client = None
        self._current_token: TokenResponse | None = None

    @property
    def cognito_client(self):
        """Lazy initialization of Cognito client."""
        if self._cognito_client is None:
            try:
                self._cognito_client = boto3.client(
                    "cognito-idp", region_name=self.region
                )
            except NoCredentialsError as e:
                raise ConfigurationError(
                    "AWS credentials not configured. Please configure AWS credentials."
                ) from e
        return self._cognito_client

    def get_bearer_token(
        self, user_pool_id: str, client_id: str, username: str, password: str
    ) -> str:
        """
        Obtain a Bearer token using USER_PASSWORD_AUTH flow.

        Args:
            user_pool_id: Cognito User Pool ID
            client_id: Cognito App Client ID
            username: Username for authentication
            password: Password for authentication

        Returns:
            Bearer token string

        Raises:
            InvalidCredentialsError: When authentication fails
            ConfigurationError: When Cognito configuration is invalid
            AuthenticationError: For other authentication issues
        """
        try:
            logger.info(f"Attempting authentication for user: {username}")

            response = self.cognito_client.initiate_auth(
                ClientId=client_id,
                AuthFlow="USER_PASSWORD_AUTH",
                AuthParameters={"USERNAME": username, "PASSWORD": password},
            )

            # Handle successful authentication
            if "AuthenticationResult" in response:
                auth_result = response["AuthenticationResult"]

                self._current_token = TokenResponse(
                    access_token=auth_result["AccessToken"],
                    refresh_token=auth_result.get("RefreshToken", ""),
                    expires_in=auth_result.get("ExpiresIn", 3600),
                    token_type=auth_result.get("TokenType", "Bearer"),
                )

                logger.info("Authentication successful")
                return self._current_token.access_token

            # Handle challenge responses (e.g., temporary password)
            elif "ChallengeName" in response:
                challenge_name = response["ChallengeName"]
                session = response.get("Session")

                if challenge_name == "NEW_PASSWORD_REQUIRED":
                    # Handle temporary password automatically
                    return self._handle_new_password_required(
                        client_id, username, password, session, response
                    )
                else:
                    raise AuthenticationError(
                        f"Authentication challenge required: {challenge_name}. "
                        f"Session: {session[:20]}..."
                        if session
                        else "No session provided."
                    )
            else:
                raise AuthenticationError("Unexpected authentication response format")

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            error_message = e.response["Error"]["Message"]

            if error_code == "NotAuthorizedException":
                raise InvalidCredentialsError(
                    f"Invalid username or password: {error_message}"
                ) from e
            elif error_code == "UserNotFoundException":
                raise InvalidCredentialsError(f"User not found: {error_message}") from e
            elif error_code == "ResourceNotFoundException":
                raise ConfigurationError(
                    f"Cognito configuration invalid - User Pool or Client not found: {error_message}"
                ) from e
            elif error_code == "InvalidParameterException":
                raise ConfigurationError(
                    f"Invalid Cognito parameters: {error_message}"
                ) from e
            else:
                raise AuthenticationError(
                    f"Authentication failed with error {error_code}: {error_message}"
                ) from e
        except Exception as e:
            # Provide diagnostic information for unexpected errors
            diagnostic_info = self.get_diagnostic_info()
            raise AuthenticationError(
                f"Unexpected authentication error: {str(e)}. "
                f"Diagnostic info: {diagnostic_info}"
            ) from e

    def refresh_token(self, refresh_token: str, client_id: str) -> str:
        """
        Refresh an expired access token using refresh token.

        Args:
            refresh_token: Refresh token from previous authentication
            client_id: Cognito App Client ID

        Returns:
            New Bearer token string

        Raises:
            TokenExpiredError: When refresh token is invalid or expired
            ConfigurationError: When Cognito configuration is invalid
            AuthenticationError: For other refresh issues
        """
        try:
            logger.info("Attempting token refresh")

            response = self.cognito_client.initiate_auth(
                ClientId=client_id,
                AuthFlow="REFRESH_TOKEN_AUTH",
                AuthParameters={"REFRESH_TOKEN": refresh_token},
            )

            if "AuthenticationResult" in response:
                auth_result = response["AuthenticationResult"]

                # Update current token with refreshed values
                if self._current_token:
                    self._current_token.access_token = auth_result["AccessToken"]
                    self._current_token.expires_in = auth_result.get("ExpiresIn", 3600)
                    self._current_token.expires_at = datetime.now(UTC) + timedelta(
                        seconds=self._current_token.expires_in
                    )
                else:
                    self._current_token = TokenResponse(
                        access_token=auth_result["AccessToken"],
                        refresh_token=refresh_token,  # Keep original refresh token
                        expires_in=auth_result.get("ExpiresIn", 3600),
                        token_type=auth_result.get("TokenType", "Bearer"),
                    )

                logger.info("Token refresh successful")
                return self._current_token.access_token
            else:
                raise AuthenticationError("Unexpected token refresh response format")

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            error_message = e.response["Error"]["Message"]

            if error_code == "NotAuthorizedException":
                raise TokenExpiredError(
                    f"Refresh token expired or invalid: {error_message}"
                ) from e
            elif error_code == "ResourceNotFoundException":
                raise ConfigurationError(
                    f"Cognito configuration invalid: {error_message}"
                ) from e
            else:
                raise AuthenticationError(
                    f"Token refresh failed with error {error_code}: {error_message}"
                ) from e
        except Exception as e:
            # Provide diagnostic information for unexpected refresh errors
            diagnostic_info = self.get_diagnostic_info()
            raise AuthenticationError(
                f"Unexpected token refresh error: {str(e)}. "
                f"Diagnostic info: {diagnostic_info}"
            ) from e

    def create_authenticated_headers(self, bearer_token: str) -> dict[str, str]:
        """
        Create HTTP headers with Bearer token authentication.

        Args:
            bearer_token: Bearer token for authentication

        Returns:
            Dictionary containing Authorization header
        """
        return {
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json",
        }

    def validate_token(self, bearer_token: str) -> bool:
        """
        Validate if a Bearer token is still valid (basic format check).

        Note: This performs basic validation. Full validation requires
        calling Cognito or checking JWT signature.

        Args:
            bearer_token: Bearer token to validate

        Returns:
            True if token appears valid, False otherwise
        """
        if not bearer_token or not isinstance(bearer_token, str):
            return False

        # Basic JWT format check (3 parts separated by dots)
        parts = bearer_token.split(".")
        if len(parts) != 3:
            return False

        # Check if current token exists and matches
        if self._current_token and self._current_token.access_token == bearer_token:
            # Check expiration if we have the information
            if self._current_token.expires_at:
                return datetime.utcnow() < self._current_token.expires_at
            return True

        return True  # Basic format is valid, but we can't verify expiration

    def is_token_expired(self) -> bool:
        """
        Check if the current token is expired.

        Returns:
            True if token is expired or doesn't exist, False otherwise
        """
        if not self._current_token or not self._current_token.expires_at:
            return True

        return datetime.utcnow() >= self._current_token.expires_at

    def get_current_token(self) -> TokenResponse | None:
        """
        Get the current token response if available.

        Returns:
            Current TokenResponse or None if no token available
        """
        return self._current_token

    def clear_tokens(self) -> None:
        """
        Clear stored tokens for security cleanup.
        """
        logger.info("Clearing stored authentication tokens")
        self._current_token = None

    def _handle_new_password_required(
        self,
        client_id: str,
        username: str,
        new_password: str,
        session: str,
        challenge_response: dict[str, Any],
    ) -> str:
        """
        Handle NEW_PASSWORD_REQUIRED challenge by setting the same password.

        This automatically handles temporary password scenarios by setting
        the provided password as the permanent password.

        Args:
            client_id: Cognito App Client ID
            username: Username for authentication
            new_password: Password to set as permanent password
            session: Session token from challenge response
            challenge_response: Original challenge response

        Returns:
            Bearer token string after password change

        Raises:
            AuthenticationError: When password change fails
        """
        try:
            logger.info(
                f"Handling NEW_PASSWORD_REQUIRED challenge for user: {username}"
            )

            # Get required attributes from challenge parameters
            challenge_params = challenge_response.get("ChallengeParameters", {})
            required_attributes = challenge_params.get("requiredAttributes")
            user_attributes = challenge_params.get("userAttributes", {})

            # Prepare challenge response
            challenge_responses = {"USERNAME": username, "NEW_PASSWORD": new_password}

            # Add any required attributes
            if required_attributes:
                logger.info(
                    f"Required attributes for password change: {required_attributes}"
                )
                # For now, we'll use existing user attributes
                for attr_name in required_attributes.split(","):
                    attr = attr_name.strip()
                    if attr in user_attributes:
                        challenge_responses[f"userAttributes.{attr}"] = user_attributes[
                            attr
                        ]

            response = self.cognito_client.respond_to_auth_challenge(
                ClientId=client_id,
                ChallengeName="NEW_PASSWORD_REQUIRED",
                Session=session,
                ChallengeResponses=challenge_responses,
            )

            # Handle successful password change
            if "AuthenticationResult" in response:
                auth_result = response["AuthenticationResult"]

                self._current_token = TokenResponse(
                    access_token=auth_result["AccessToken"],
                    refresh_token=auth_result.get("RefreshToken", ""),
                    expires_in=auth_result.get("ExpiresIn", 3600),
                    token_type=auth_result.get("TokenType", "Bearer"),
                )

                logger.info("Password change and authentication successful")
                return self._current_token.access_token
            else:
                raise AuthenticationError("Unexpected response after password change")

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            error_message = e.response["Error"]["Message"]

            if error_code == "InvalidPasswordException":
                raise AuthenticationError(
                    f"Password does not meet requirements: {error_message}. "
                    "Please ensure password meets Cognito password policy requirements."
                ) from e
            elif error_code == "NotAuthorizedException":
                raise AuthenticationError(
                    f"Not authorized to change password: {error_message}"
                ) from e
            else:
                raise AuthenticationError(
                    f"Password change failed with error {error_code}: {error_message}"
                ) from e
        except Exception as e:
            raise AuthenticationError(
                f"Unexpected error during password change: {str(e)}"
            ) from e

    def get_diagnostic_info(self) -> dict[str, Any]:
        """
        Get diagnostic information for troubleshooting authentication issues.

        Returns:
            Dictionary containing diagnostic information
        """
        current_token = self.get_current_token()

        diagnostic_info = {
            "region": self.region,
            "has_current_token": current_token is not None,
            "aws_credentials_available": self._check_aws_credentials(),
            "cognito_client_initialized": self._cognito_client is not None,
        }

        if current_token:
            diagnostic_info.update(
                {
                    "token_type": current_token.token_type,
                    "token_expires_at": (
                        current_token.expires_at.isoformat()
                        if current_token.expires_at
                        else None
                    ),
                    "token_expired": self.is_token_expired(),
                    "has_refresh_token": bool(current_token.refresh_token),
                }
            )

        return diagnostic_info

    def _check_aws_credentials(self) -> bool:
        """
        Check if AWS credentials are properly configured.

        Returns:
            True if credentials are available, False otherwise
        """
        try:
            # Try to create a client to test credentials
            boto3.client("sts", region_name=self.region).get_caller_identity()
            return True
        except Exception:
            return False
