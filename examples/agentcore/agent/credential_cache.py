"""
Secure credential caching for MCP OAuth authentication.

This module provides secure in-memory credential caching with automatic
cleanup and security best practices.
"""

import atexit
import logging
import threading
import weakref
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CachedCredential:
    """Cached credential with expiration and security metadata."""

    username: str
    password: str
    cached_at: datetime
    expires_at: datetime
    access_count: int = 0

    def is_expired(self) -> bool:
        """Check if the cached credential has expired."""
        return datetime.utcnow() >= self.expires_at

    def increment_access(self) -> None:
        """Increment access counter for monitoring."""
        self.access_count += 1

    def time_to_expiry(self) -> timedelta:
        """Get time remaining until expiration."""
        return self.expires_at - datetime.utcnow()


class SecureCredentialCache:
    """
    Secure in-memory credential cache with automatic cleanup.

    Features:
    - Automatic expiration and cleanup
    - Thread-safe operations
    - Memory-only storage (no disk persistence)
    - Automatic cleanup on process exit
    - Access monitoring and logging
    """

    def __init__(self, default_ttl_minutes: int = 30) -> None:
        """
        Initialize the secure credential cache.

        Args:
            default_ttl_minutes: Default time-to-live for cached credentials in minutes
        """
        self.default_ttl = timedelta(minutes=default_ttl_minutes)
        self._cache: dict[str, CachedCredential] = {}
        self._lock = threading.RLock()
        self._cleanup_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()

        # Register cleanup on exit
        atexit.register(self.cleanup_all)

        # Start background cleanup thread
        self._start_cleanup_thread()

        # Keep a weak reference to track instances
        self._instances = weakref.WeakSet()
        self._instances.add(self)

    def cache_credentials(
        self, key: str, username: str, password: str, ttl_minutes: int | None = None
    ) -> None:
        """
        Cache credentials securely in memory.

        Args:
            key: Unique key for the credentials (e.g., secret name)
            username: Username to cache
            password: Password to cache (stored in memory only)
            ttl_minutes: Time-to-live in minutes (uses default if None)
        """
        ttl = timedelta(minutes=ttl_minutes) if ttl_minutes else self.default_ttl
        expires_at = datetime.utcnow() + ttl

        with self._lock:
            # Clear any existing credential for this key
            if key in self._cache:
                self._secure_clear_credential(key)

            # Cache new credential
            self._cache[key] = CachedCredential(
                username=username,
                password=password,
                cached_at=datetime.utcnow(),
                expires_at=expires_at,
            )

            logger.info(f"Cached credentials for key: {key}, expires at: {expires_at}")

    def get_credentials(self, key: str) -> tuple[str, str] | None:
        """
        Retrieve cached credentials if available and not expired.

        Args:
            key: Unique key for the credentials

        Returns:
            Tuple of (username, password) if available, None otherwise
        """
        with self._lock:
            cached = self._cache.get(key)

            if not cached:
                logger.debug(f"No cached credentials found for key: {key}")
                return None

            if cached.is_expired():
                logger.info(f"Cached credentials expired for key: {key}")
                self._secure_clear_credential(key)
                return None

            # Increment access counter and return credentials
            cached.increment_access()
            logger.debug(
                f"Retrieved cached credentials for key: {key} "
                f"(access count: {cached.access_count})"
            )

            return (cached.username, cached.password)

    def is_cached(self, key: str) -> bool:
        """
        Check if credentials are cached and not expired.

        Args:
            key: Unique key for the credentials

        Returns:
            True if credentials are cached and valid
        """
        with self._lock:
            cached = self._cache.get(key)
            return cached is not None and not cached.is_expired()

    def get_cache_info(self, key: str) -> dict[str, Any] | None:
        """
        Get cache information for a key without retrieving credentials.

        Args:
            key: Unique key for the credentials

        Returns:
            Dictionary with cache metadata or None if not cached
        """
        with self._lock:
            cached = self._cache.get(key)

            if not cached:
                return None

            return {
                "username": cached.username,
                "cached_at": cached.cached_at.isoformat(),
                "expires_at": cached.expires_at.isoformat(),
                "is_expired": cached.is_expired(),
                "time_to_expiry_seconds": cached.time_to_expiry().total_seconds(),
                "access_count": cached.access_count,
            }

    def invalidate(self, key: str) -> bool:
        """
        Invalidate cached credentials for a specific key.

        Args:
            key: Unique key for the credentials

        Returns:
            True if credentials were cached and removed, False otherwise
        """
        with self._lock:
            if key in self._cache:
                self._secure_clear_credential(key)
                logger.info(f"Invalidated cached credentials for key: {key}")
                return True
            return False

    def cleanup_expired(self) -> int:
        """
        Clean up expired credentials from cache.

        Returns:
            Number of expired credentials removed
        """
        expired_keys = []

        with self._lock:
            for key, cached in self._cache.items():
                if cached.is_expired():
                    expired_keys.append(key)

            for key in expired_keys:
                self._secure_clear_credential(key)

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired credentials")

        return len(expired_keys)

    def cleanup_all(self) -> None:
        """Clean up all cached credentials."""
        with self._lock:
            keys_to_clear = list(self._cache.keys())
            for key in keys_to_clear:
                self._secure_clear_credential(key)

            logger.info(
                f"Cleaned up all cached credentials ({len(keys_to_clear)} items)"
            )

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_cached = len(self._cache)
            expired_count = sum(
                1 for cached in self._cache.values() if cached.is_expired()
            )
            active_count = total_cached - expired_count

            total_accesses = sum(cached.access_count for cached in self._cache.values())

            return {
                "total_cached": total_cached,
                "active_credentials": active_count,
                "expired_credentials": expired_count,
                "total_accesses": total_accesses,
                "cache_keys": list(self._cache.keys()),
            }

    def shutdown(self) -> None:
        """Shutdown the cache and cleanup resources."""
        logger.info("Shutting down credential cache")

        # Signal shutdown to cleanup thread
        self._shutdown_event.set()

        # Wait for cleanup thread to finish
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5.0)

        # Clean up all credentials
        self.cleanup_all()

    def _secure_clear_credential(self, key: str) -> None:
        """
        Securely clear a credential from cache.

        This method attempts to overwrite the password string in memory
        before removing it from the cache.
        """
        if key in self._cache:
            cached = self._cache[key]

            # Attempt to overwrite password in memory (best effort)
            try:
                # This is a best-effort attempt to clear sensitive data
                # Python strings are immutable, so this may not always work
                if hasattr(cached, "password") and cached.password:
                    # Create a new string of the same length with zeros
                    password_length = len(cached.password)
                    cached.password = "\0" * password_length
            except Exception:
                # If overwriting fails, just continue with removal
                pass

            # Remove from cache
            del self._cache[key]

    def _start_cleanup_thread(self) -> None:
        """Start background thread for periodic cleanup of expired credentials."""

        def cleanup_worker():
            while not self._shutdown_event.is_set():
                try:
                    self.cleanup_expired()
                    # Wait for 5 minutes or until shutdown
                    self._shutdown_event.wait(timeout=300)  # 5 minutes
                except Exception as e:
                    logger.error(f"Error in credential cache cleanup thread: {str(e)}")
                    # Continue running even if cleanup fails
                    self._shutdown_event.wait(timeout=60)  # Wait 1 minute before retry

        self._cleanup_thread = threading.Thread(
            target=cleanup_worker, name="CredentialCacheCleanup", daemon=True
        )
        self._cleanup_thread.start()
        logger.info("Started credential cache cleanup thread")


# Global cache instance
_global_cache: SecureCredentialCache | None = None
_cache_lock = threading.Lock()


def get_global_cache() -> SecureCredentialCache:
    """
    Get the global credential cache instance.

    Returns:
        Global SecureCredentialCache instance
    """
    global _global_cache  # noqa: PLW0603

    with _cache_lock:
        if _global_cache is None:
            _global_cache = SecureCredentialCache()
            logger.info("Created global credential cache")

        return _global_cache


def cleanup_global_cache() -> None:
    """Clean up the global credential cache."""
    global _global_cache  # noqa: PLW0603

    with _cache_lock:
        if _global_cache is not None:
            _global_cache.shutdown()
            _global_cache = None
            logger.info("Cleaned up global credential cache")


# Register global cleanup on exit
atexit.register(cleanup_global_cache)
