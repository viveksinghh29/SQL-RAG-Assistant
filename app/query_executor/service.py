"""Query Execution Engine (Phase 8).

Executes a validated SQL query against the target (read-only) database
and returns a structured `ExecutionResult`.

Responsibilities:
  - Enforce a hard query timeout via `asyncio.wait_for`
  - Cap result rows at `SQL_MAX_ROWS_RETURNED` (configurable via .env)
  - Classify DB exceptions into typed `ExecutionStatus` values so the
    API layer never leaks raw DB error strings to clients
  - Track and return execution time in milliseconds
  - Never raise — all errors are captured in `ExecutionResult.status`

What this service does NOT do:
  - Validate the SQL (Phase 7 — SQLValidationService)
  - Generate explanations or charts (Phase 9)
  - Cache results (Phase 14 — Redis caching layer wraps this service)
"""

import asyncio
import re
import time

from sqlalchemy import text
from sqlalchemy.exc import (
    DBAPIError,
    OperationalError,
    ProgrammingError,
)

from app.config.settings import get_settings
from app.core.logging import get_logger
from app.query_executor.connection import get_target_engine
from app.query_executor.result import ExecutionResult, ExecutionStatus

log = get_logger("query_executor")


class QueryExecutor:
    """Executes validated SQL against the target database.

    Designed to be instantiated per-request (lightweight — no state
    beyond settings). Uses the process-level cached engine from
    `get_target_engine()`.
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    async def execute(self, sql: str) -> ExecutionResult:
        """Execute `sql` and return a structured result.

        Args:
            sql: A fully validated SELECT statement (output of
                 SQLValidationService.validate()).  Passing unvalidated
                 SQL here is a caller error; this service trusts its
                 input but still classifies any DB-level errors cleanly.

        Returns:
            ExecutionResult — always, never raises.
        """
        t0 = time.monotonic()

        try:
            result = await asyncio.wait_for(
                self._run_query(sql),
                timeout=self._settings.sql_query_timeout_seconds,
            )
            result.execution_time_ms = (time.monotonic() - t0) * 1000
            return result

        except asyncio.TimeoutError:
            elapsed_ms = (time.monotonic() - t0) * 1000
            log.bind(sql_preview=sql[:120], elapsed_ms=round(elapsed_ms, 1)).warning(
                "Query execution timed out"
            )
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                sql_executed=sql,
                execution_time_ms=elapsed_ms,
                error_message=(
                    f"Query exceeded the {self._settings.sql_query_timeout_seconds}s "
                    "time limit. Try narrowing your filter conditions or adding a "
                    "stricter date range."
                ),
            )

    async def _run_query(self, sql: str) -> ExecutionResult:
        """Inner query runner — separated from `execute` so the timeout
        wrapper in `execute` can cleanly cancel this coroutine."""
        max_rows = self._settings.sql_max_rows_returned
        engine = get_target_engine()

        try:
            async with engine.connect() as conn:
                cursor_result = await conn.execute(text(sql))
                columns = list(cursor_result.keys())

                # Fetch one extra row beyond the cap to detect truncation
                # without issuing a second COUNT query.
                raw_rows = cursor_result.fetchmany(max_rows + 1)
                was_truncated = len(raw_rows) > max_rows
                rows = raw_rows[:max_rows]

                dicts = [dict(zip(columns, row)) for row in rows]

                status = ExecutionStatus.EMPTY if not dicts else ExecutionStatus.SUCCESS

                log.bind(
                    row_count=len(dicts),
                    was_truncated=was_truncated,
                    columns=columns,
                ).info("Query executed successfully")

                return ExecutionResult(
                    status=status,
                    rows=dicts,
                    columns=columns,
                    row_count=len(dicts),
                    total_rows=len(dicts) + (1 if was_truncated else 0),
                    was_truncated=was_truncated,
                    sql_executed=sql,
                )

        except ProgrammingError as exc:
            return self._handle_db_error(
                exc, sql,
                status=ExecutionStatus.SYNTAX_ERROR,
                user_message=self._extract_syntax_hint(str(exc)),
            )

        except OperationalError as exc:
            raw = str(exc).lower()
            if any(kw in raw for kw in ("access denied", "command denied", "permission")):
                status = ExecutionStatus.PERMISSION_ERROR
                msg = "Database permission denied. Contact your administrator."
            else:
                status = ExecutionStatus.CONNECTION_ERROR
                msg = "Could not connect to the database. Please try again shortly."
            return self._handle_db_error(exc, sql, status=status, user_message=msg)

        except DBAPIError as exc:
            return self._handle_db_error(
                exc, sql,
                status=ExecutionStatus.EXECUTION_ERROR,
                user_message="An unexpected database error occurred. Please try again.",
            )

        except Exception as exc:  # noqa: BLE001
            log.exception("Unexpected non-DBAPIError during query execution")
            return ExecutionResult(
                status=ExecutionStatus.EXECUTION_ERROR,
                sql_executed=sql,
                error_message="An unexpected error occurred during query execution.",
            )

    @staticmethod
    def _handle_db_error(
        exc: Exception,
        sql: str,
        *,
        status: ExecutionStatus,
        user_message: str,
    ) -> ExecutionResult:
        """Log the raw exception (for debugging) and return a clean result."""
        log.bind(status=status.value, exc_type=type(exc).__name__).warning(
            f"Query execution error: {exc}"
        )
        return ExecutionResult(
            status=status,
            sql_executed=sql,
            error_message=user_message,
        )

    @staticmethod
    def _extract_syntax_hint(raw_error: str) -> str:
        """Extract a usable hint from a MySQL syntax error message.

        MySQL error format:
          "(...) You have an error in your SQL syntax; check ... near '...' at line N"
        """
        match = re.search(r"near ['\"](.{0,60})['\"]", raw_error, re.IGNORECASE)
        if match:
            return (
                f"SQL syntax error near: '{match.group(1)}'. "
                "Please rephrase your question."
            )
        return (
            "The generated SQL contained a syntax error. "
            "Please rephrase your question."
        )
