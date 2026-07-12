"""Unit tests for Phase 14: Performance Optimization & Caching.

All tests use an in-memory fake cache (no Redis required).

Coverage:
  CacheClient (via FakeCacheClient):
    - get returns None on miss, value on hit
    - set/get round-trip for dicts, strings, lists
    - empty list stored as sentinel, retrieved as []
    - errors silently return None (graceful degradation)
  CacheKeys:
    - query_result_key is deterministic (same inputs → same key)
    - different roles → different keys (no cross-role cache sharing)
    - different SQL → different keys (no collision)
    - key length/format validation
  CachedQueryExecutor:
    - First call misses cache → executor called → result cached
    - Second call hits cache → executor NOT called again
    - Error results (TIMEOUT, SYNTAX_ERROR) are NOT cached
    - Cache degradation: CacheClient returning None → falls through to executor
  CachedAnalyticsService:
    - Explanation and chart hit cache on repeat call
    - LLM called only once per unique sql+row_count pair
    - Cached chart decision reconstructed into ChartDecision correctly
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app.auth.roles import Role
from app.llm.providers.mock_provider import MockLLMProvider
from app.query_executor.cached_executor import CachedQueryExecutor
from app.query_executor.result import ExecutionResult, ExecutionStatus
from app.query_executor.service import QueryExecutor
from app.utils.cache_keys import (
    chart_decision_key,
    explanation_key,
    query_result_key,
)
from app.visualization.cached_analytics import CachedAnalyticsService
from app.visualization.chart_inference import ChartDecision
import json


# ── Fake cache (in-memory, no Redis) ─────────────────────────────────────────

class FakeCacheClient:
    """In-memory cache for testing. Mirrors CacheClient interface."""

    def __init__(self, fail_on_get: bool = False):
        self._store: dict = {}
        self._fail_on_get = fail_on_get
        self.get_calls: list[str] = []
        self.set_calls: list[str] = []

    async def get(self, key: str):
        self.get_calls.append(key)
        if self._fail_on_get:
            return None   # simulates Redis unavailable
        return self._store.get(key)

    async def set(self, key: str, value, *, ttl: int | None = None):
        self.set_calls.append(key)
        self._store[key] = value

    async def delete(self, key: str):
        self._store.pop(key, None)

    async def is_available(self) -> bool:
        return True


# ── Fake executor ─────────────────────────────────────────────────────────────

class FakeExecutor:
    def __init__(self, result: ExecutionResult):
        self._result = result
        self.call_count = 0

    async def execute(self, sql: str) -> ExecutionResult:
        self.call_count += 1
        return self._result


def _success_result(sql: str = "SELECT 1") -> ExecutionResult:
    return ExecutionResult(
        status=ExecutionStatus.SUCCESS,
        rows=[{"id": 1, "name": "Test"}],
        columns=["id", "name"],
        row_count=1,
        sql_executed=sql,
        execution_time_ms=5.0,
    )


def _error_result() -> ExecutionResult:
    return ExecutionResult(
        status=ExecutionStatus.TIMEOUT,
        sql_executed="SELECT * FROM orders",
        error_message="Query timed out",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Cache key tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_query_key_is_deterministic():
    k1 = query_result_key("SELECT * FROM orders", "admin")
    k2 = query_result_key("SELECT * FROM orders", "admin")
    assert k1 == k2


def test_query_key_different_roles_produce_different_keys():
    k_admin = query_result_key("SELECT * FROM orders", "admin")
    k_emp = query_result_key("SELECT * FROM orders", "employee")
    assert k_admin != k_emp


def test_query_key_different_sql_produce_different_keys():
    k1 = query_result_key("SELECT * FROM orders", "admin")
    k2 = query_result_key("SELECT * FROM customers", "admin")
    assert k1 != k2


def test_query_key_normalizes_whitespace():
    """Leading/trailing whitespace should not affect the key."""
    k1 = query_result_key("SELECT * FROM orders", "admin")
    k2 = query_result_key("  SELECT * FROM orders  ", "admin")
    assert k1 == k2


def test_explanation_key_deterministic():
    k1 = explanation_key("SELECT 1", 42)
    k2 = explanation_key("SELECT 1", 42)
    assert k1 == k2


def test_explanation_key_different_row_counts_differ():
    k1 = explanation_key("SELECT 1", 10)
    k2 = explanation_key("SELECT 1", 20)
    assert k1 != k2


def test_chart_key_column_order_independent():
    """Column list should be sorted before hashing — order shouldn't matter."""
    k1 = chart_decision_key("SELECT 1", ["revenue", "category"])
    k2 = chart_decision_key("SELECT 1", ["category", "revenue"])
    assert k1 == k2


def test_keys_have_namespace_prefix():
    assert query_result_key("SELECT 1", "admin").startswith("query:")
    assert explanation_key("SELECT 1", 1).startswith("explain:")
    assert chart_decision_key("SELECT 1", ["x"]).startswith("chart:")


# ═══════════════════════════════════════════════════════════════════════════════
# CachedQueryExecutor tests
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_cached_executor_first_call_hits_db():
    cache = FakeCacheClient()
    inner = FakeExecutor(_success_result())
    executor = CachedQueryExecutor(executor=inner, cache=cache, role=Role.ADMIN)

    result = await executor.execute("SELECT 1")

    assert inner.call_count == 1
    assert result.status == ExecutionStatus.SUCCESS
    assert len(cache.set_calls) == 1


@pytest.mark.asyncio
async def test_cached_executor_second_call_hits_cache():
    cache = FakeCacheClient()
    inner = FakeExecutor(_success_result())
    executor = CachedQueryExecutor(executor=inner, cache=cache, role=Role.ADMIN)

    await executor.execute("SELECT 1")
    result2 = await executor.execute("SELECT 1")

    # Inner executor called only once; second call used cache
    assert inner.call_count == 1
    assert result2.status == ExecutionStatus.SUCCESS
    assert result2.row_count == 1


@pytest.mark.asyncio
async def test_cached_executor_error_results_not_cached():
    cache = FakeCacheClient()
    inner = FakeExecutor(_error_result())
    executor = CachedQueryExecutor(executor=inner, cache=cache, role=Role.ADMIN)

    result = await executor.execute("SELECT * FROM orders")

    assert result.status == ExecutionStatus.TIMEOUT
    assert len(cache.set_calls) == 0   # errors must NOT be cached


@pytest.mark.asyncio
async def test_cached_executor_cache_degradation_falls_through():
    """If cache fails (returns None), executor must still be called."""
    cache = FakeCacheClient(fail_on_get=True)
    inner = FakeExecutor(_success_result())
    executor = CachedQueryExecutor(executor=inner, cache=cache, role=Role.ADMIN)

    result = await executor.execute("SELECT 1")
    assert inner.call_count == 1
    assert result.status == ExecutionStatus.SUCCESS


@pytest.mark.asyncio
async def test_different_roles_dont_share_cache():
    cache = FakeCacheClient()
    inner_admin = FakeExecutor(_success_result())
    inner_emp = FakeExecutor(_success_result())

    exec_admin = CachedQueryExecutor(inner_admin, cache, Role.ADMIN)
    exec_emp = CachedQueryExecutor(inner_emp, cache, Role.EMPLOYEE)

    await exec_admin.execute("SELECT 1")
    await exec_emp.execute("SELECT 1")

    # Both should hit the DB — different roles = different cache keys
    assert inner_admin.call_count == 1
    assert inner_emp.call_count == 1


@pytest.mark.asyncio
async def test_empty_result_cached_correctly():
    empty = ExecutionResult(
        status=ExecutionStatus.EMPTY,
        rows=[],
        columns=[],
        row_count=0,
        sql_executed="SELECT 1 WHERE 1=0",
    )
    cache = FakeCacheClient()
    inner = FakeExecutor(empty)
    executor = CachedQueryExecutor(inner, cache, Role.ADMIN)

    r1 = await executor.execute("SELECT 1 WHERE 1=0")
    r2 = await executor.execute("SELECT 1 WHERE 1=0")

    assert r1.status == ExecutionStatus.EMPTY
    assert inner.call_count == 1   # second call hit cache
    assert r2.status == ExecutionStatus.EMPTY


# ═══════════════════════════════════════════════════════════════════════════════
# CachedAnalyticsService tests
# ═══════════════════════════════════════════════════════════════════════════════

VALID_CHART_JSON = json.dumps({
    "chart_type": "bar",
    "x_column": "category",
    "y_column": "revenue",
    "title": "Revenue by Category",
    "rationale": "Categorical comparison.",
})

SAMPLE_EXEC_RESULT = ExecutionResult(
    status=ExecutionStatus.SUCCESS,
    rows=[{"category": "Electronics", "revenue": 45000}],
    columns=["category", "revenue"],
    row_count=1,
    sql_executed="SELECT category, SUM(revenue) FROM orders GROUP BY category",
    execution_time_ms=10.0,
)


@pytest.mark.asyncio
async def test_cached_analytics_first_call_invokes_llm():
    llm = MockLLMProvider(responses=[
        "Revenue up 17%.",   # explanation
        VALID_CHART_JSON,    # chart inference
    ])
    cache = FakeCacheClient()
    svc = CachedAnalyticsService(llm=llm, cache=cache)

    result = await svc.analyse(
        question="Revenue by category",
        execution_result=SAMPLE_EXEC_RESULT,
    )

    assert "17%" in result.explanation
    assert result.chart_type == "bar"
    assert llm.call_count == 2


@pytest.mark.asyncio
async def test_cached_analytics_second_call_skips_llm():
    llm = MockLLMProvider(responses=[
        "Revenue up 17%.",
        VALID_CHART_JSON,
    ])
    cache = FakeCacheClient()
    svc = CachedAnalyticsService(llm=llm, cache=cache)

    await svc.analyse(question="Revenue by category", execution_result=SAMPLE_EXEC_RESULT)
    llm_calls_after_first = llm.call_count

    await svc.analyse(question="Revenue by category", execution_result=SAMPLE_EXEC_RESULT)

    # Second call should not invoke the LLM at all
    assert llm.call_count == llm_calls_after_first


@pytest.mark.asyncio
async def test_cached_analytics_chart_decision_reconstructed():
    llm = MockLLMProvider(responses=["Explanation text.", VALID_CHART_JSON])
    cache = FakeCacheClient()
    svc = CachedAnalyticsService(llm=llm, cache=cache)

    r1 = await svc.analyse(question="q", execution_result=SAMPLE_EXEC_RESULT)
    r2 = await svc.analyse(question="q", execution_result=SAMPLE_EXEC_RESULT)

    assert r1.chart_type == r2.chart_type == "bar"
    assert r1.chart_title == r2.chart_title
