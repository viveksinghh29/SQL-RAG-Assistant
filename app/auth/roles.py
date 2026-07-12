"""Minimal role-based access control (RBAC) foundation with role definitions, permission mapping, and authorization utilities."""

from enum import StrEnum

from app.core.exceptions import PermissionDeniedSQLError


class Role(StrEnum):
    ADMIN = "admin"
    MANAGER = "manager"
    EMPLOYEE = "employee"


# Tables that are fully invisible to a role — the SQL generator (Phase 6)
# will exclude these from retrieved schema context, and the SQL validator
# (Phase 7) will hard-reject any generated query that references them
# regardless of what the LLM produced.
RESTRICTED_TABLES: dict[Role, set[str]] = {
    Role.ADMIN: set(),  # no restrictions
    Role.MANAGER: {"payroll"},
    Role.EMPLOYEE: {"payroll", "employees"},
}

# Column-level restrictions for tables that ARE visible but contain some
# sensitive columns (e.g. an employee can see their own table's existence
# referenced via joins, but never salary figures).
RESTRICTED_COLUMNS: dict[Role, dict[str, set[str]]] = {
    Role.ADMIN: {},
    Role.MANAGER: {},
    Role.EMPLOYEE: {"employees": {"hire_date"}},
}


def get_restricted_tables(role: Role) -> set[str]:
    """Return the set of table names this role may never query."""
    return RESTRICTED_TABLES.get(role, set())


def get_restricted_columns(role: Role, table: str) -> set[str]:
    """Return the set of column names on `table` this role may never select."""
    return RESTRICTED_COLUMNS.get(role, {}).get(table, set())


def assert_tables_allowed(role: Role, tables: set[str]) -> None:
    """Raise PermissionDeniedSQLError if any of `tables` is restricted for `role`.

    Called by the SQL validator (Phase 7) after parsing table names out of
    a generated query, as a hard backstop independent of what context the
    LLM was given — defense in depth against prompt injection or retrieval
    mistakes that might surface a restricted table anyway.
    """
    forbidden = tables & get_restricted_tables(role)
    if forbidden:
        raise PermissionDeniedSQLError(
            f"Role '{role.value}' is not permitted to access table(s): {', '.join(sorted(forbidden))}",
            details={"role": role.value, "forbidden_tables": sorted(forbidden)},
        )
