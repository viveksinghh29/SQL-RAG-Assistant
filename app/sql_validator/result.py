"""Validation result schema for structured SQL validation outcomes, including rejection reasons and detailed error information."""

from enum import StrEnum

from pydantic import BaseModel


class ValidationStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"        # syntax / semantic error — not safe to execute
    BLOCKED = "blocked"        # dangerous operation or RBAC violation — hard reject


class RejectionReason(StrEnum):
    DANGEROUS_OPERATION = "dangerous_operation"
    SQL_INJECTION = "sql_injection"
    SYNTAX_ERROR = "syntax_error"
    UNKNOWN_TABLE = "unknown_table"
    UNKNOWN_COLUMN = "unknown_column"
    RBAC_VIOLATION = "rbac_violation"
    MISSING_LIMIT = "missing_limit"
    MULTIPLE_STATEMENTS = "multiple_statements"
    EMPTY_QUERY = "empty_query"


class ValidationResult(BaseModel):
    """Outcome of a full SQL validation pass."""

    status: ValidationStatus
    sql: str = ""                      # normalised SQL (stripped / upper-cased keywords)
    reason: RejectionReason | None = None
    message: str = ""                  # human-readable explanation
    details: dict = {}                 # structured context (which table, which keyword, …)
