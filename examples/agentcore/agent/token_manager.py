"""
Token lifecycle management for MCP OAuth authentication.

This module provides automatic token refresh, expiration detection,
and cleanup functionality for OAuth Bearer tokens.
"""

import atexit
import logging
import threading
from collections.abc import Callable
from datetime import datetime, timedelta

from agent.auth_helper import (
    AuthenticationError,
    McpAuthHelper,
    TokenExpiredError,
    TokenResponse,
)

logger = logging.getLogger(__name__)


class TokenManager:
    """
    Manages OAuth token lifecycle including automatic refresh and cleanup.

    Provides automatic token refresh before expiration and handles
    token cleanup on session termination.
    """

    def __init__(
        self,
        auth_helper: McpAuthHelper,
        user_pool_id: str,
        client_id: str,
        username: str,
        password: str,
        auto_refresh: bool = True,
        refresh_buffer_seconds: int = 300,  # Refresh 5 minutes before expiration
    ) -> None:
        """
        Initialize the token manager.

        Args:
            auth_helper: McpAuthHelper instance for authentication
            user_pool_id: Cognito User Pool ID
            client_id: Cognito App Client ID
            username: Username for authentication
            password: Password for authentication
            auto_refresh: Whether to automatically refresh tokens
            refresh_buffer_seconds: Seconds before expiration to refresh token
        """
        self.auth_helper = auth_helper
        self.user_pool_id = user_pool_id
        self.client_id = client_id
        self.username = username
        self.password = password
        self.auto_refresh = auto_refresh
        self.refresh_buffer_seconds = refresh_buffer_seconds

        self._refresh_thread: threading.Thread | None = None
        self._stop_refresh = threading.Event()
        self._token_lock = threading.Lock()
        self._refresh_callback: Callable[[str], None] | None = None

        # Register cleanup on exit
        atexit.register(self.cleanup_tokens)

    def get_valid_token(self) -> str:
        """
        Get a valid Bearer token, refreshing if necessary.

        Returns:
            Valid Bearer token string

        Raises:
            AuthenticationError: When unable to obtain valid token
        """
        with self._token_lock:
            current_token = self.auth_helper.get_current_token()

            # If no token exists, get initial token
            if not current_token:
                logger.info("No current token, obtaining initial token")
                token = self.auth_helper.get_bearer_token(
                    self.user_pool_id, self.client_id, self.username, self.password
                )

                # Start auto-refresh if enabled
                if self.auto_refresh:
                    self._start_auto_refresh()

                return token

            # Check if token needs refresh
            if self._needs_refresh(current_token):
                logger.info("Token needs refresh")
                return self.refresh_if_needed()

            return current_token.access_token

    def refresh_if_needed(self) -> str:
        """
        Refresh token if it's expired or close to expiration.

        Returns:
            Valid Bearer token string

        Raises:
            TokenExpiredError: When refresh fails
            AuthenticationError: For other authentication issues
        """
        with self._token_lock:
            current_token = self.auth_helper.get_current_token()

            if not current_token:
                raise AuthenticationError("No token available for refresh")

            if not self._needs_refresh(current_token):
                return current_token.access_token

            try:
                # Try to refresh using refresh token
                if current_token.refresh_token:
                    logger.info("Refreshing token using refresh token")
                    new_token = self.auth_helper.refresh_token(
                        current_token.refresh_token, self.client_id
                    )

                    # Notify callback if registered
                    if self._refresh_callback:
                        self._refresh_callback(new_token)

                    return new_token
                else:
                    # Fall back to re-authentication
                    logger.info("No refresh token available, re-authenticating")
                    new_token = self.auth_helper.get_bearer_token(
                        self.user_pool_id, self.client_id, self.username, self.password
                    )

                    # Notify callback if registered
                    if self._refresh_callback:
                        self._refresh_callback(new_token)

                    return new_token

            except TokenExpiredError:
                # Refresh token expired, need to re-authenticate
                logger.warning("Refresh token expired, re-authenticating")
                new_token = self.auth_helper.get_bearer_token(
                    self.user_pool_id, self.client_id, self.username, self.password
                )

                # Notify callback if registered
                if self._refresh_callback:
                    self._refresh_callback(new_token)

                return new_token

    def _needs_refresh(self, token: TokenResponse) -> bool:
        """
        Check if token needs to be refreshed.

        Args:
            token: Token to check

        Returns:
            True if token needs refresh, False otherwise
        """
        if not token.expires_at:
            return False

        # Check if token expires within the buffer time
        refresh_time = token.expires_at - timedelta(seconds=self.refresh_buffer_seconds)
        return datetime.utcnow() >= refresh_time

    def _start_auto_refresh(self) -> None:
        """Start automatic token refresh in background thread."""
        if self._refresh_thread and self._refresh_thread.is_alive():
            return

        logger.info("Starting automatic token refresh")
        self._stop_refresh.clear()
        self._refresh_thread = threading.Thread(
            target=self._auto_refresh_worker, daemon=True, name="TokenRefreshWorker"
        )
        self._refresh_thread.start()

    def _auto_refresh_worker(self) -> None:
        """Background worker for automatic token refresh."""
        while not self._stop_refresh.is_set():
            try:
                current_token = self.auth_helper.get_current_token()

                if current_token and self._needs_refresh(current_token):
                    logger.info("Auto-refreshing token")
                    self.refresh_if_needed()

                # Check every 60 seconds
                self._stop_refresh.wait(60)

            except Exception as e:
                logger.error(f"Error in auto-refresh worker: {e}")
                # Wait before retrying
                self._stop_refresh.wait(60)

    def stop_auto_refresh(self) -> None:
        """Stop automatic token refresh."""
        logger.info("Stopping automatic token refresh")
        self._stop_refresh.set()

        if self._refresh_thread and self._refresh_thread.is_alive():
            self._refresh_thread.join(timeout=5)

    def cleanup_tokens(self) -> None:
        """
        Clean up tokens and stop background processes.

        This method is automatically called on program exit.
        """
        logger.info("Cleaning up authentication tokens")

        # Stop auto-refresh
        self.stop_auto_refresh()

        # Clear tokens from auth helper
        self.auth_helper.clear_tokens()

    def set_refresh_callback(self, callback: Callable[[str], None]) -> None:
        """
        Set callback function to be called when token is refreshed.

        Args:
            callback: Function that takes new token as parameter
        """
        self._refresh_callback = callback

    def get_token_info(self) -> dict | None:
        """
        Get information about current token.

        Returns:
            Dictionary with token information or None if no token
        """
        current_token = self.auth_helper.get_current_token()

        if not current_token:
            return None

        return {
            "expires_at": (
                current_token.expires_at.isoformat()
                if current_token.expires_at
                else None
            ),
            "expires_in_seconds": (
                int((current_token.expires_at - datetime.utcnow()).total_seconds())
                if current_token.expires_at
                else None
            ),
            "needs_refresh": self._needs_refresh(current_token),
            "token_type": current_token.token_type,
        }

    def force_refresh(self) -> str:
        """
        Force token refresh regardless of expiration status.

        Returns:
            New Bearer token string

        Raises:
            AuthenticationError: When refresh fails
        """
        logger.info("Forcing token refresh")
        return self.refresh_if_needed()


class SessionManager:
    """
    Manages authentication sessions with automatic cleanup.

    Provides context manager interface for secure session handling.
    """

    def __init__(self, token_manager: TokenManager) -> None:
        """
        Initialize session manager.

        Args:
            token_manager: TokenManager instance to manage
        """
        self.token_manager = token_manager
        self._session_active = False

    def __enter__(self) -> TokenManager:
        """Start authentication session."""
        logger.info("Starting authentication session")
        self._session_active = True
        return self.token_manager

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """End authentication session and cleanup."""
        logger.info("Ending authentication session")
        self._session_active = False
        self.token_manager.cleanup_tokens()

    def is_active(self) -> bool:
        """Check if session is currently active."""
        return self._session_active
