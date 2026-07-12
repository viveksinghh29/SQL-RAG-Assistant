"""Execution result schema for the query executor.

Kept in its own module so the visualization engine (Phase 9), the API
layer (Phase 11), and the frontend (Phase 12) can all import the result
type without pulling in execution-engine internals.
"""

from enum import StrEnum

from pydantic import BaseModel


class ExecutionStatus(StrEnum):
    SUCCESS = "success"
    EMPTY = "empty"           # query ran fine but returned zero rows
    TIMEOUT = "timeout"       # exceeded SQL_QUERY_TIMEOUT_SECONDS
    CONNECTION_ERROR = "connection_error"
    SYNTAX_ERROR = "syntax_error"
    PERMISSION_ERROR = "permission_error"
    EXECUTION_ERROR = "execution_error"   # catch-all for unexpected DB errors


class ExecutionResult(BaseModel):
    """Structured output from a query execution, regardless of success or failure."""

    status: ExecutionStatus
    rows: list[dict] = []
    columns: list[str] = []
    row_count: int = 0           # total rows returned (may be capped)
    total_rows: int = 0          # pre-cap count if rows were truncated
    was_truncated: bool = False  # True when row_count < total_rows
    execution_time_ms: float = 0.0
    error_message: str = ""      # user-friendly; never raw DB exception text
    sql_executed: str = ""       # the exact SQL that was sent to the DB

    @property
    def is_successful(self) -> bool:
        return self.status in (ExecutionStatus.SUCCESS, ExecutionStatus.EMPTY)
