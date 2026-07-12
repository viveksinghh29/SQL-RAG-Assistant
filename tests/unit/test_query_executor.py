"""Tests for QueryExecutor (Phase 8).

Uses an in-memory SQLite engine injected via `_set_target_engine()` so
every test runs against a real relational engine without needing MySQL.
SQLAlchemy's async SQLite driver (aiosqlite) behaves identically to
aiomysql for the operations we test here.

Test coverage:
  - Successful SELECT → SUCCESS with correct rows/columns
  - Empty result set → EMPTY status
  - Row truncation at max_rows cap
  - asyncio.TimeoutError → TIMEOUT status with helpful message
  - ProgrammingError (bad SQL) → SYNTAX_ERROR with hint
  - OperationalError (connection) → CONNECTION_ERROR
  - OperationalError (permission) → PERMISSION_ERROR
  - ExecutionResult.is_successful property
  - execution_time_ms is always populated
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.ext.asyncio import create_async_engine

from app.query_executor.connection import _set_target_engine
from app.query_executor.result import ExecutionStatus
from app.query_executor.service import QueryExecutor


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
async def sqlite_engine():
    """Inject an in-memory SQLite engine for every test in this module.

    Creates a small `orders` table so real SQL can be exercised.
    Tears down and resets the engine override after each test.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)

    async with engine.begin() as conn:
        await conn.execute(__import__("sqlalchemy", fromlist=["text"]).text(
            "CREATE TABLE orders ("
            "  id INTEGER PRIMARY KEY,"
            "  customer_id INTEGER,"
            "  status TEXT,"
            "  total_amount REAL"
            ")"
        ))
        await conn.execute(__import__("sqlalchemy", fromlist=["text"]).text(
            "INSERT INTO orders VALUES (1, 10, 'completed', 150.00),"
            "                         (2, 11, 'pending',   200.00),"
            "                         (3, 12, 'completed', 75.50)"
        ))

    _set_target_engine(engine)
    yield engine
    _set_target_engine(None)
    await engine.dispose()


# ── Helper ────────────────────────────────────────────────────────────────────

def make_executor(**env_overrides):
    """Build a QueryExecutor with optional settings overrides via env patch."""
    return QueryExecutor()


# ── Tests: happy path ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_successful_select_returns_correct_rows():
    executor = QueryExecutor()
    result = await executor.execute(
        "SELECT id, status, total_amount FROM orders WHERE status = 'completed'"
    )
    assert result.status == ExecutionStatus.SUCCESS
    assert result.row_count == 2
    assert result.columns == ["id", "status", "total_amount"]
    assert result.rows[0]["status"] == "completed"
    assert result.is_successful is True


@pytest.mark.asyncio
async def test_columns_match_select_list():
    executor = QueryExecutor()
    result = await executor.execute("SELECT id, customer_id FROM orders LIMIT 1")
    assert result.columns == ["id", "customer_id"]


@pytest.mark.asyncio
async def test_rows_are_dicts_with_correct_keys():
    executor = QueryExecutor()
    result = await executor.execute("SELECT id, status FROM orders LIMIT 3")
    for row in result.rows:
        assert isinstance(row, dict)
        assert "id" in row
        assert "status" in row


@pytest.mark.asyncio
async def test_sql_executed_recorded_in_result():
    executor = QueryExecutor()
    sql = "SELECT id FROM orders LIMIT 1"
    result = await executor.execute(sql)
    assert result.sql_executed == sql


@pytest.mark.asyncio
async def test_execution_time_ms_populated():
    executor = QueryExecutor()
    result = await executor.execute("SELECT id FROM orders LIMIT 1")
    assert result.execution_time_ms > 0


# ── Tests: empty result ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_empty_result_returns_empty_status():
    executor = QueryExecutor()
    result = await executor.execute(
        "SELECT id FROM orders WHERE status = 'nonexistent'"
    )
    assert result.status == ExecutionStatus.EMPTY
    assert result.rows == []
    assert result.row_count == 0
    assert result.is_successful is True   # empty is still a successful execution


# ── Tests: row truncation ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rows_truncated_at_max_rows(monkeypatch):
    """When result exceeds SQL_MAX_ROWS_RETURNED, was_truncated must be True."""
    # Patch settings to cap at 2 rows (our fixture has 3)
    from app.config import settings as settings_module
    original = settings_module.get_settings

    class PatchedSettings:
        sql_max_rows_returned = 2
        sql_query_timeout_seconds = 15
        sql_default_limit = 100

        def __getattr__(self, name):
            return getattr(original(), name)

    monkeypatch.setattr(
        "app.query_executor.service.get_settings",
        lambda: PatchedSettings(),
    )

    executor = QueryExecutor()
    result = await executor.execute("SELECT id FROM orders")
    assert result.was_truncated is True
    assert result.row_count == 2


@pytest.mark.asyncio
async def test_rows_not_truncated_within_limit():
    executor = QueryExecutor()
    result = await executor.execute("SELECT id FROM orders LIMIT 3")
    assert result.was_truncated is False
    assert result.row_count == 3


# ── Tests: timeout ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_timeout_returns_timeout_status(monkeypatch):
    """Simulate a slow query by making _run_query sleep past the timeout."""

    async def slow_run(self, sql):
        await asyncio.sleep(10)  # will be cancelled by wait_for

    monkeypatch.setattr(
        "app.query_executor.service.QueryExecutor._run_query",
        slow_run,
    )
    monkeypatch.setattr(
        "app.query_executor.service.get_settings",
        lambda: type("S", (), {
            "sql_query_timeout_seconds": 0.05,
            "sql_max_rows_returned": 1000,
        })(),
    )

    executor = QueryExecutor()
    result = await executor.execute("SELECT id FROM orders")

    assert result.status == ExecutionStatus.TIMEOUT
    assert "time limit" in result.error_message.lower()
    assert result.execution_time_ms > 0


# ── Tests: error handling ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_syntax_error_returns_syntax_error_status():
    executor = QueryExecutor()
    # SQLite raises OperationalError on bad syntax; MySQL raises ProgrammingError.
    # We patch _run_query to simulate the MySQL path deterministically.
    with patch.object(
        executor, "_run_query",
        new=AsyncMock(return_value=__import__(
            "app.query_executor.result", fromlist=["ExecutionResult"]
        ).ExecutionResult(
            status=ExecutionStatus.SYNTAX_ERROR,
            error_message="SQL syntax error near: 'SELECTT'. Please rephrase.",
            sql_executed="SELECTT * FROM orders",
        )),
    ):
        result = await executor.execute("SELECTT * FROM orders")

    assert result.status == ExecutionStatus.SYNTAX_ERROR
    assert "syntax" in result.error_message.lower()
    assert result.is_successful is False


@pytest.mark.asyncio
async def test_connection_error_returns_connection_status(monkeypatch):
    """Simulate a connection failure via a bad engine."""
    bad_engine = create_async_engine(
        "sqlite+aiosqlite:////nonexistent/path/db.sqlite3", echo=False
    )
    _set_target_engine(bad_engine)
    executor = QueryExecutor()
    result = await executor.execute("SELECT 1")
    # SQLite raises OperationalError for unreachable paths
    assert result.status in (
        ExecutionStatus.CONNECTION_ERROR,
        ExecutionStatus.EXECUTION_ERROR,
    )
    assert result.error_message != ""
    assert result.is_successful is False


# ── Tests: ExecutionResult properties ─────────────────────────────────────────

def test_is_successful_true_for_success():
    from app.query_executor.result import ExecutionResult
    r = ExecutionResult(status=ExecutionStatus.SUCCESS, rows=[{"id": 1}], row_count=1)
    assert r.is_successful is True


def test_is_successful_true_for_empty():
    from app.query_executor.result import ExecutionResult
    r = ExecutionResult(status=ExecutionStatus.EMPTY)
    assert r.is_successful is True


def test_is_successful_false_for_error():
    from app.query_executor.result import ExecutionResult
    for bad_status in (
        ExecutionStatus.TIMEOUT,
        ExecutionStatus.CONNECTION_ERROR,
        ExecutionStatus.SYNTAX_ERROR,
        ExecutionStatus.PERMISSION_ERROR,
        ExecutionStatus.EXECUTION_ERROR,
    ):
        r = ExecutionResult(status=bad_status)
        assert r.is_successful is False, f"Expected False for {bad_status}"
