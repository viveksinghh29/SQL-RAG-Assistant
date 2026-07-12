"""SQL parsing utilities for extracting query structure via lexical analysis without requiring a database connection."""

import re

import sqlparse
from sqlparse.sql import Identifier, IdentifierList
from sqlparse.tokens import Keyword, DML, DDL, Punctuation

from app.core.logging import get_logger

log = get_logger("sql_validator.parser")

# Two-pass regex: reliable for the exact blocklist keyword set.
# sqlparse occasionally assigns the DDL ttype to the *object* token (TABLE)
# rather than the *action* token (DROP/CREATE/ALTER), so regex wins here.
_FIRST_KEYWORD_RE = re.compile(
    r"^\s*(SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|REPLACE"
    r"|RENAME|GRANT|REVOKE|LOCK|UNLOCK|CALL|EXEC|EXECUTE|LOAD|IMPORT"
    r"|FLUSH|KILL|SHUTDOWN|SHOW|DESCRIBE|EXPLAIN)\b",
    re.IGNORECASE,
)


def get_first_keyword(sql: str) -> str:
    """Return the first DML/DDL keyword in the SQL string (upper-cased).

    Uses a two-pass strategy:
      1. Regex pre-scan  — reliable for the known-bad keyword set.
      2. sqlparse walk   — catches anything the regex misses.

    Examples:
        "SELECT * FROM orders"  → "SELECT"
        "  drop table users"    → "DROP"
        "GRANT ALL ON ..."      → "GRANT"
        ""                      → ""
    """
    match = _FIRST_KEYWORD_RE.match(sql.strip())
    if match:
        return match.group(1).upper()

    parsed = sqlparse.parse(sql.strip())
    if not parsed:
        return ""
    for token in parsed[0].flatten():
        if token.ttype in (DML, DDL, Keyword):
            return token.normalized.upper()
    return ""


def count_statements(sql: str) -> int:
    """Count the number of distinct SQL statements separated by semicolons."""
    statements = [s for s in sqlparse.parse(sql) if s.get_type() is not None]
    return max(len(statements), 1)


def extract_table_names(sql: str) -> set[str]:
    """Extract table names from a SQL SELECT statement (lower-cased).

    Handles:
    - Simple FROM clause:    FROM orders
    - Aliased tables:        FROM orders o
    - JOIN clauses:          JOIN order_items oi ON ...
    - Subqueries are skipped (we validate the outer query's tables).
    """
    tables: set[str] = set()
    parsed = sqlparse.parse(sql.strip())
    if not parsed:
        return tables

    statement = parsed[0]
    from_seen = False

    for token in statement.flatten():
        ttype = token.ttype
        value = token.value.strip()

        if not value:
            continue

        if ttype in (DML, DDL, Keyword):
            upper = value.upper()
            if upper == "FROM":
                from_seen = True
                continue
            if upper in ("JOIN", "INNER", "LEFT", "RIGHT", "OUTER", "CROSS", "FULL"):
                from_seen = True
                continue
            if upper in ("WHERE", "GROUP", "ORDER", "HAVING", "LIMIT", "ON", "SET"):
                from_seen = False
                continue

        if from_seen and ttype not in (Punctuation,) and not _is_keyword(ttype):
            if value not in ("(", ")", ",") and not value.startswith("'"):
                clean = value.strip("`").split(".")[-1].lower()
                if clean and re.match(r"^[a-z_][a-z0-9_]*$", clean):
                    tables.add(clean)
                from_seen = False

    return tables


def extract_select_columns(sql: str) -> set[str]:
    """Extract explicitly named columns from a SELECT list (lower-cased).

    Returns an empty set for SELECT * or when extraction fails.
    """
    columns: set[str] = set()
    parsed = sqlparse.parse(sql.strip())
    if not parsed:
        return columns

    statement = parsed[0]
    in_select = False

    for token in statement.tokens:
        ttype = token.ttype

        if ttype is DML and token.normalized.upper() == "SELECT":
            in_select = True
            continue

        if in_select:
            if isinstance(token, (Identifier, IdentifierList)):
                _collect_identifiers(token, columns)
            in_select = False

    return columns


def _collect_identifiers(token, columns: set[str]) -> None:
    """Recursively collect column name strings into `columns`."""
    if isinstance(token, IdentifierList):
        for item in token.get_identifiers():
            _collect_identifiers(item, columns)
    elif isinstance(token, Identifier):
        name = token.get_real_name()
        if name and name != "*":
            columns.add(name.strip("`").lower())


def _is_keyword(ttype) -> bool:
    """Return True if the token type is a SQL keyword type."""
    from sqlparse import tokens as T
    return ttype in (
        T.Keyword, T.Keyword.DML, T.Keyword.DDL,
        T.Keyword.CTE, T.Comment.Single, T.Comment.Multiline,
        T.Whitespace, T.Newline, T.Punctuation,
    )


def has_limit_clause(sql: str) -> bool:
    """Return True if the SQL contains a LIMIT clause."""
    return bool(re.search(r"\bLIMIT\s+\d+", sql, re.IGNORECASE))
