"""Simple in-process rate limiting middleware.

Uses a per-IP / per-user sliding-window counter stored in a module-level
dict. This is intentionally single-process — for multi-process or
multi-instance deployments, swap the counter store for Redis (Phase 14
adds Redis caching; rate limiting can be moved there then).

Limits are enforced on all `/api/v1/` routes. The `/health` endpoint
and Swagger UI are exempt.
"""

import time
from collections import defaultdict, deque

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.config.settings import get_settings
from app.core.logging import get_logger

log = get_logger("middleware.rate_limit")

# {identifier: deque of request timestamps within the window}
_windows: dict[str, deque] = defaultdict(deque)
_WINDOW_SECONDS = 60


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter applied to /api/v1/ routes."""

    def __init__(self, app, requests_per_minute: int | None = None) -> None:
        super().__init__(app)
        settings = get_settings()
        self._limit = requests_per_minute or settings.rate_limit_requests_per_minute

    async def dispatch(self, request: Request, call_next) -> Response:
        # Only rate-limit API routes
        if not request.url.path.startswith("/api/"):
            return await call_next(request)

        identifier = self._get_identifier(request)
        now = time.monotonic()
        window = _windows[identifier]

        # Evict timestamps outside the 60-second window
        while window and window[0] < now - _WINDOW_SECONDS:
            window.popleft()

        if len(window) >= self._limit:
            log.bind(identifier=identifier, count=len(window)).warning(
                "Rate limit exceeded"
            )
            return Response(
                content='{"error":"rate_limit_exceeded","message":"Too many requests. '
                        'Please wait before sending another request."}',
                status_code=429,
                media_type="application/json",
            )

        window.append(now)
        return await call_next(request)

    @staticmethod
    def _get_identifier(request: Request) -> str:
        """Use authenticated user ID if available, else fall back to IP."""
        # The JWT is parsed in the endpoint dependency, not here — so we
        # use the raw forwarded IP as the identifier at middleware level.
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        client = request.client
        return client.host if client else "unknown"
