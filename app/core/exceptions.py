"""Application-wide exception hierarchy.

All custom exceptions inherit from `AppException` so the centralized
FastAPI exception handler (registered in `app/main.py`, Phase 2/11) can
catch one base type and map it to a consistent JSON error response,
instead of leaking raw tracebacks or library-specific exceptions to
clients.
"""


class AppException(Exception):
    """Base class for all expected, handled application errors."""

    status_code: int = 500
    error_code: str = "internal_error"

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(message)


# --- Auth ---
class AuthenticationError(AppException):
    status_code = 401
    error_code = "authentication_error"


class AuthorizationError(AppException):
    status_code = 403
    error_code = "authorization_error"


# --- RAG ---
class RetrievalError(AppException):
    status_code = 502
    error_code = "retrieval_error"


# --- LLM ---
class LLMProviderError(AppException):
    status_code = 502
    error_code = "llm_provider_error"


class LLMResponseParsingError(AppException):
    status_code = 502
    error_code = "llm_response_parsing_error"


# --- SQL generation / validation ---
class SQLGenerationError(AppException):
    status_code = 422
    error_code = "sql_generation_error"


class SQLValidationError(AppException):
    status_code = 422
    error_code = "sql_validation_error"


class UnsafeSQLError(SQLValidationError):
    """Raised when SQL is rejected for containing a dangerous operation."""

    error_code = "unsafe_sql_rejected"


class PermissionDeniedSQLError(SQLValidationError):
    """Raised when SQL references tables/columns the user's role cannot access."""

    status_code = 403
    error_code = "sql_permission_denied"


# --- Query execution ---
class QueryExecutionError(AppException):
    status_code = 502
    error_code = "query_execution_error"


class QueryTimeoutError(QueryExecutionError):
    status_code = 504
    error_code = "query_timeout"


# --- Generic resource errors ---
class NotFoundError(AppException):
    status_code = 404
    error_code = "not_found"


class ValidationError(AppException):
    status_code = 422
    error_code = "validation_error"


class RateLimitExceededError(AppException):
    status_code = 429
    error_code = "rate_limit_exceeded"
