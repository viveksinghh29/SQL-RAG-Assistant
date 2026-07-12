"""Cached query executor (Phase 14).

Wraps `QueryExecutor` with a Redis read-through cache. The cache is
keyed on the validated SQL + user role so that:

  - The same query by the same role hits the cache on repeat
  - Role changes invalidate the cache (employee vs admin see same SQL
    but the validator guarantees they never share a query that touches
    restricted tables, so keying on role is belt-and-suspenders)

Cache behaviour:
  - HIT  → return cached ExecutionResult, skip DB entirely
  - MISS → execute against DB, cache the result, return it
  - ERROR (Redis down, serialisation failure) → execute against DB,
    log warning, return result (cache degradation is transparent)

Cache TTL is `CACHE_TTL_SECONDS` from settings (default 300s / 5 min).
Queries that return errors are NOT cached — only SUCCESS and EMPTY.
"""

import time

from app.auth.roles import Role
from app.core.logging import get_logger
from app.query_executor.result import ExecutionResult, ExecutionStatus
from app.query_executor.service import QueryExecutor
from app.utils.cache_client import CacheClient
from app.utils.cache_keys import query_result_key

log = get_logger("cache.query_executor")


class CachedQueryExecutor:
    """Drop-in replacement for QueryExecutor with Redis caching.

    The API layer (chat.py) can switch between QueryExecutor and
    CachedQueryExecutor by changing one line in `_build_services()`.
    Both implement the same `.execute(sql)` interface.
    """

    def __init__(self, executor: QueryExecutor, cache: CacheClient, role: Role) -> None:
        self._executor = executor
        self._cache = cache
        self._role = role

    async def execute(self, sql: str) -> ExecutionResult:
        """Execute SQL with Redis read-through caching."""
        cache_key = query_result_key(sql, self._role.value)
        t0 = time.monotonic()

        # ── Cache read ────────────────────────────────────────────────────
        cached = await self._cache.get(cache_key)
        if cached is not None:
            hit_ms = (time.monotonic() - t0) * 1000
            log.bind(
                cache_key=cache_key[:20] + "…",
                hit_ms=round(hit_ms, 2),
                role=self._role.value,
            ).info("Query cache HIT")
            # Reconstruct ExecutionResult from the cached dict
            return ExecutionResult(**cached)

        # ── Cache miss — execute against DB ───────────────────────────────
        result = await self._executor.execute(sql)
        miss_ms = (time.monotonic() - t0) * 1000

        log.bind(
            cache_key=cache_key[:20] + "…",
            miss_ms=round(miss_ms, 2),
            status=result.status.value,
            role=self._role.value,
        ).info("Query cache MISS")

        # Only cache successful results — never cache errors or timeouts
        if result.status in (ExecutionStatus.SUCCESS, ExecutionStatus.EMPTY):
            await self._cache.set(cache_key, result.model_dump())

        return result
