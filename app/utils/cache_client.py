"""Redis client wrapper for the caching layer.

Uses `redis.asyncio` (bundled with the `redis` package ≥ 4.2).
All operations are async and wrapped in try/except so a Redis outage
degrades gracefully — cache misses rather than 500 errors.

Design decisions:
  - Single process-level connection pool (not one connection per request)
  - All values serialized as JSON (human-readable, debuggable in redis-cli)
  - `CacheClient.get()` returns None on miss OR on any Redis error
  - `CacheClient.set()` silently swallows errors — never raises to callers
  - `CacheClient.is_available()` for health-check endpoint
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from app.config.settings import get_settings
from app.core.logging import get_logger

log = get_logger("cache.redis")

# Sentinel value stored in cache to represent "this query returned empty"
# (distinguishes "cache miss" from "cached empty result")
_EMPTY_SENTINEL = "__EMPTY__"


class CacheClient:
    """Async Redis client with JSON serialization and graceful degradation."""

    def __init__(self) -> None:
        self._client = None
        self._settings = get_settings()
        self._enabled = bool(self._settings.redis_host)

    async def connect(self) -> None:
        """Open the connection pool. Called once at app startup."""
        if not self._enabled:
            return
        try:
            import redis.asyncio as aioredis
            self._client = aioredis.from_url(
                self._settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
                retry_on_timeout=True,
                health_check_interval=30,
            )
            # Verify connectivity at startup
            await self._client.ping()
            log.info(
                f"Redis connected: {self._settings.redis_host}:{self._settings.redis_port}"
            )
        except Exception as exc:
            log.warning(f"Redis unavailable at startup — caching disabled: {exc}")
            self._client = None

    async def disconnect(self) -> None:
        """Close the connection pool. Called at app shutdown."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def get(self, key: str) -> Any | None:
        """Return the cached value or None on miss/error."""
        if self._client is None:
            return None
        try:
            raw = await self._client.get(key)
            if raw is None:
                return None
            if raw == _EMPTY_SENTINEL:
                return []   # cached empty result
            return json.loads(raw)
        except Exception as exc:
            log.debug(f"Cache GET error for key '{key}': {exc}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        *,
        ttl: int | None = None,
    ) -> None:
        """Store a value. Silently ignores errors."""
        if self._client is None:
            return
        try:
            ttl = ttl or self._settings.cache_ttl_seconds
            if value == [] or value is None:
                serialized = _EMPTY_SENTINEL
            else:
                serialized = json.dumps(value, default=str)
            await self._client.setex(key, ttl, serialized)
        except Exception as exc:
            log.debug(f"Cache SET error for key '{key}': {exc}")

    async def delete(self, key: str) -> None:
        if self._client is None:
            return
        try:
            await self._client.delete(key)
        except Exception as exc:
            log.debug(f"Cache DELETE error for key '{key}': {exc}")

    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching a glob pattern. Returns count deleted."""
        if self._client is None:
            return 0
        try:
            keys = await self._client.keys(pattern)
            if keys:
                return await self._client.delete(*keys)
            return 0
        except Exception as exc:
            log.debug(f"Cache DELETE pattern error for '{pattern}': {exc}")
            return 0

    async def is_available(self) -> bool:
        """Return True if Redis is reachable."""
        if self._client is None:
            return False
        try:
            return await self._client.ping()
        except Exception:
            return False


@lru_cache
def get_cache_client() -> CacheClient:
    """Return the process-level CacheClient singleton."""
    return CacheClient()
