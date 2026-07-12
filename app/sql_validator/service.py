"""SQL validation and security pipeline that enforces query safety, RBAC, schema integrity, and LIMIT constraints before execution."""

import re

import sqlparse

from app.auth.roles import Role, assert_tables_allowed, get_restricted_tables
from app.config.settings import get_settings
from app.core.exceptions import PermissionDeniedSQLError
from app.core.logging import get_logger
from app.sql_validator.parser_utils import (
    count_statements,
    extract_select_columns,
    extract_table_names,
    get_first_keyword,
    has_limit_clause,
)
from app.sql_validator.result import RejectionReason, ValidationResult, ValidationStatus
from app.sql_validator.rules import FORBIDDEN_OPERATIONS, INJECTION_REGEXES

log = get_logger("sql_validator")


class SQLValidationService:
    """Validates a generated SQL query before execution.

    `known_tables` maps table name (lower-cased) to a set of column names
    (lower-cased). It is populated from the schema that was parsed in
    Phase 4 (RAG ingestion) and passed in by the orchestration layer
    (Phase 11) so the validator has no dependency on the RAG layer itself.

    Passing an empty `known_tables` dict disables table/column checks —
    useful in tests that only want to exercise the security checks.
    """

    def __init__(self, known_tables: dict[str, set[str]] | None = None) -> None:
        self._known_tables = {k.lower(): {c.lower() for c in v}
                              for k, v in (known_tables or {}).items()}
        self._settings = get_settings()

    async def validate(
        self,
        sql: str,
        *,
        role: Role,
    ) -> ValidationResult:
        """Run the full validation pipeline. Returns on first failure."""

        # ── 1. Empty query ────────────────────────────────────────────────
        stripped = sql.strip()
        if not stripped:
            return ValidationResult(
                status=ValidationStatus.BLOCKED,
                reason=RejectionReason.EMPTY_QUERY,
                message="Empty SQL query received.",
            )

        # ── 2. Multiple statements (stacked injection) ────────────────────
        if self._has_multiple_statements(stripped):
            log.bind(role=role.value).warning("Blocked: multiple SQL statements detected")
            return ValidationResult(
                status=ValidationStatus.BLOCKED,
                sql=stripped,
                reason=RejectionReason.MULTIPLE_STATEMENTS,
                message="Only a single SQL statement is allowed per query.",
            )

        # ── 3. Dangerous operation blocklist ──────────────────────────────
        first_kw = get_first_keyword(stripped)
        if first_kw in FORBIDDEN_OPERATIONS:
            log.bind(role=role.value, keyword=first_kw).warning(
                "Blocked: forbidden SQL operation"
            )
            return ValidationResult(
                status=ValidationStatus.BLOCKED,
                sql=stripped,
                reason=RejectionReason.DANGEROUS_OPERATION,
                message=(
                    f"The operation '{first_kw}' is not permitted. "
                    "Only SELECT queries are allowed."
                ),
                details={"forbidden_keyword": first_kw},
            )

        # ── 4. SQL injection pattern detection ────────────────────────────
        injection_check = self._check_injection_patterns(stripped)
        if injection_check is not None:
            log.bind(role=role.value, pattern=injection_check).warning(
                "Blocked: SQL injection pattern detected"
            )
            return ValidationResult(
                status=ValidationStatus.BLOCKED,
                sql=stripped,
                reason=RejectionReason.SQL_INJECTION,
                message="The query contains a pattern that is not permitted.",
                details={"pattern_type": injection_check},
            )

        # ── 5. Must be a SELECT statement ─────────────────────────────────
        if first_kw != "SELECT":
            log.bind(role=role.value, first_kw=first_kw).warning(
                "Blocked: non-SELECT statement"
            )
            return ValidationResult(
                status=ValidationStatus.BLOCKED,
                sql=stripped,
                reason=RejectionReason.DANGEROUS_OPERATION,
                message=(
                    f"Only SELECT statements are permitted. "
                    f"Received a '{first_kw}' statement."
                ),
                details={"first_keyword": first_kw},
            )

        # ── 6. RBAC — hard-reject restricted tables ───────────────────────
        referenced_tables = extract_table_names(stripped)
        try:
            assert_tables_allowed(role, referenced_tables)
        except PermissionDeniedSQLError as exc:
            log.bind(role=role.value, tables=sorted(referenced_tables)).warning(
                "Blocked: RBAC violation"
            )
            return ValidationResult(
                status=ValidationStatus.BLOCKED,
                sql=stripped,
                reason=RejectionReason.RBAC_VIOLATION,
                message=exc.message,
                details=exc.details,
            )

        # ── 7. Schema check — table names ─────────────────────────────────
        if self._known_tables:
            unknown_tables = referenced_tables - self._known_tables.keys()
            # Remove restricted tables from unknown check — they may not be
            # in known_tables if they were RBAC-filtered during RAG ingestion.
            restricted = get_restricted_tables(role)
            unknown_tables -= restricted
            if unknown_tables:
                log.bind(role=role.value, unknown=sorted(unknown_tables)).warning(
                    "Invalid: unknown table(s) referenced"
                )
                return ValidationResult(
                    status=ValidationStatus.INVALID,
                    sql=stripped,
                    reason=RejectionReason.UNKNOWN_TABLE,
                    message=(
                        f"The query references table(s) that do not exist: "
                        f"{', '.join(sorted(unknown_tables))}."
                    ),
                    details={"unknown_tables": sorted(unknown_tables)},
                )

        # ── 8. Schema check — column names (best-effort) ──────────────────
        if self._known_tables:
            col_result = self._check_columns(stripped, referenced_tables, role)
            if col_result is not None:
                return col_result

        # ── 9. LIMIT enforcement (soft — auto-add if missing) ─────────────
        final_sql = stripped
        if not has_limit_clause(stripped):
            default_limit = self._settings.sql_default_limit
            final_sql = self._append_limit(stripped, default_limit)
            log.bind(role=role.value, default_limit=default_limit).info(
                "LIMIT clause auto-appended to query"
            )

        log.bind(role=role.value, tables=sorted(referenced_tables)).info(
            "SQL validation passed"
        )
        return ValidationResult(
            status=ValidationStatus.VALID,
            sql=final_sql,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _has_multiple_statements(self, sql: str) -> bool:
        """Return True if the SQL contains more than one statement."""
        # sqlparse count_statements approach
        statements = [s for s in sqlparse.parse(sql) if s.get_type() is not None]
        if len(statements) > 1:
            return True
        # Secondary check: semicolon followed by a word character
        # (covers cases sqlparse might miss)
        return bool(re.search(r";\s*\w", sql))

    def _check_injection_patterns(self, sql: str) -> str | None:
        """Return the pattern label if an injection pattern is found, else None."""
        for pattern, label in INJECTION_REGEXES:
            if pattern.search(sql):
                return label
        return None

    def _check_columns(
        self,
        sql: str,
        referenced_tables: set[str],
        role: Role,
    ) -> ValidationResult | None:
        """Check SELECT-list columns against the known schema.

        Returns a ValidationResult on failure, or None on success.
        Skips the check if SELECT * is used (columns set will be empty).
        """
        columns = extract_select_columns(sql)
        if not columns:
            return None  # SELECT * or extraction failed — skip

        restricted_cols = {}
        from app.auth.roles import get_restricted_columns
        for table in referenced_tables:
            rc = get_restricted_columns(role, table)
            if rc:
                restricted_cols[table] = rc

        for col in columns:
            # Check if column is restricted for this role
            for table, restricted in restricted_cols.items():
                if col in restricted:
                    return ValidationResult(
                        status=ValidationStatus.BLOCKED,
                        sql=sql,
                        reason=RejectionReason.RBAC_VIOLATION,
                        message=(
                            f"Column '{col}' on table '{table}' is not "
                            f"accessible for role '{role.value}'."
                        ),
                        details={"column": col, "table": table, "role": role.value},
                    )

            # Check if column exists in any referenced table
            col_found = any(
                col in self._known_tables.get(table, set())
                for table in referenced_tables
            )
            # Also allow aggregate function aliases and expressions
            if not col_found and not self._is_likely_alias_or_expr(sql, col):
                log.bind(role=role.value, column=col).debug(
                    "Unknown column detected (may be alias)"
                )
                # Warn in logs but do NOT reject — column could be an alias
                # defined in the same query (e.g. SUM(...) AS revenue).
                # Hard column rejection causes too many false positives.

        return None

    def _is_likely_alias_or_expr(self, sql: str, col_name: str) -> bool:
        """Heuristic: return True if col_name appears as an alias in this SQL."""
        return bool(re.search(
            rf"\bAS\s+[`\"]?{re.escape(col_name)}[`\"]?\b",
            sql, re.IGNORECASE
        ))

    def _append_limit(self, sql: str, limit: int) -> str:
        """Append a LIMIT clause to a SQL statement that lacks one."""
        # Strip trailing semicolons before appending
        clean = sql.rstrip().rstrip(";").rstrip()
        return f"{clean}\nLIMIT {limit};"
