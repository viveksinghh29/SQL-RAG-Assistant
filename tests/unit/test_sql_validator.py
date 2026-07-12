"""Unit tests for SQLValidationService (Phase 7).

Every validation check is tested with:
  - A passing case (VALID)
  - One or more failing cases (BLOCKED or INVALID)

No database, no LLM, no network — pure SQL string analysis.
"""

import pytest

from app.auth.roles import Role
from app.sql_validator.result import RejectionReason, ValidationStatus
from app.sql_validator.service import SQLValidationService

# ── Fixture: schema-aware validator ──────────────────────────────────────────

SAMPLE_SCHEMA: dict[str, set[str]] = {
    "orders": {"id", "customer_id", "order_date", "status", "total_amount"},
    "order_items": {"id", "order_id", "product_id", "quantity", "unit_price"},
    "customers": {"id", "name", "email", "region", "segment", "signup_date"},
    "products": {"id", "name", "category", "price", "cost", "supplier_id"},
    "employees": {"id", "name", "role", "department", "hire_date"},
    "payroll": {"employee_id", "salary", "bonus"},
    "suppliers": {"id", "name", "country", "contact_email"},
}


@pytest.fixture
def validator() -> SQLValidationService:
    return SQLValidationService(known_tables=SAMPLE_SCHEMA)


@pytest.fixture
def validator_no_schema() -> SQLValidationService:
    """Validator with no schema — skips table/column existence checks."""
    return SQLValidationService()


# ── 1. Happy path ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_valid_select_passes(validator):
    sql = "SELECT id, total_amount FROM orders WHERE status = 'completed' LIMIT 10;"
    result = await validator.validate(sql, role=Role.ADMIN)
    assert result.status == ValidationStatus.VALID


@pytest.mark.asyncio
async def test_valid_join_passes(validator):
    sql = (
        "SELECT o.id, c.name, o.total_amount "
        "FROM orders o "
        "JOIN customers c ON c.id = o.customer_id "
        "WHERE o.status = 'completed' "
        "LIMIT 50;"
    )
    result = await validator.validate(sql, role=Role.MANAGER)
    assert result.status == ValidationStatus.VALID


@pytest.mark.asyncio
async def test_valid_aggregation_passes(validator):
    sql = (
        "SELECT category, SUM(price) AS total_price "
        "FROM products "
        "GROUP BY category "
        "LIMIT 20;"
    )
    result = await validator.validate(sql, role=Role.EMPLOYEE)
    assert result.status == ValidationStatus.VALID


# ── 2. Empty query ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_empty_query_blocked(validator):
    result = await validator.validate("", role=Role.ADMIN)
    assert result.status == ValidationStatus.BLOCKED
    assert result.reason == RejectionReason.EMPTY_QUERY


@pytest.mark.asyncio
async def test_whitespace_only_blocked(validator):
    result = await validator.validate("   \n\t  ", role=Role.ADMIN)
    assert result.status == ValidationStatus.BLOCKED
    assert result.reason == RejectionReason.EMPTY_QUERY


# ── 3. Dangerous operations ────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("sql,keyword", [
    ("DROP TABLE orders;", "DROP"),
    ("DELETE FROM orders WHERE id = 1;", "DELETE"),
    ("UPDATE orders SET status = 'cancelled';", "UPDATE"),
    ("INSERT INTO orders VALUES (1, 2, '2024-01-01', 'new', 100);", "INSERT"),
    ("ALTER TABLE orders ADD COLUMN foo INT;", "ALTER"),
    ("TRUNCATE TABLE orders;", "TRUNCATE"),
    ("CREATE TABLE evil (id INT);", "CREATE"),
    ("GRANT ALL ON orders TO user;", "GRANT"),
    ("EXEC sp_helpdb;", "EXEC"),
])
async def test_dangerous_operations_blocked(validator, sql, keyword):
    result = await validator.validate(sql, role=Role.ADMIN)
    assert result.status == ValidationStatus.BLOCKED
    assert result.reason == RejectionReason.DANGEROUS_OPERATION
    assert keyword in result.message or keyword in str(result.details)


# ── 4. SQL injection patterns ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_union_select_injection_blocked(validator_no_schema):
    sql = "SELECT id FROM orders UNION SELECT password FROM users"
    result = await validator_no_schema.validate(sql, role=Role.ADMIN)
    assert result.status == ValidationStatus.BLOCKED
    assert result.reason == RejectionReason.SQL_INJECTION
    assert result.details.get("pattern_type") == "union_select"


@pytest.mark.asyncio
async def test_comment_injection_blocked(validator_no_schema):
    sql = "SELECT * FROM orders WHERE id = 1 -- DROP TABLE orders"
    result = await validator_no_schema.validate(sql, role=Role.ADMIN)
    assert result.status == ValidationStatus.BLOCKED
    assert result.reason == RejectionReason.SQL_INJECTION


@pytest.mark.asyncio
async def test_sleep_injection_blocked(validator_no_schema):
    sql = "SELECT * FROM orders WHERE id = SLEEP(5)"
    result = await validator_no_schema.validate(sql, role=Role.ADMIN)
    assert result.status == ValidationStatus.BLOCKED
    assert result.reason == RejectionReason.SQL_INJECTION


@pytest.mark.asyncio
async def test_information_schema_probing_blocked(validator_no_schema):
    sql = "SELECT table_name FROM information_schema.tables"
    result = await validator_no_schema.validate(sql, role=Role.ADMIN)
    assert result.status == ValidationStatus.BLOCKED
    assert result.reason == RejectionReason.SQL_INJECTION


@pytest.mark.asyncio
async def test_file_write_injection_blocked(validator_no_schema):
    sql = "SELECT * FROM orders INTO OUTFILE '/tmp/data.csv'"
    result = await validator_no_schema.validate(sql, role=Role.ADMIN)
    assert result.status == ValidationStatus.BLOCKED
    assert result.reason == RejectionReason.SQL_INJECTION


# ── 5. Non-SELECT first keyword ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_non_select_blocked(validator_no_schema):
    """A statement whose first keyword is not SELECT must be blocked
    even if it slipped past the FORBIDDEN_OPERATIONS check."""
    sql = "SHOW TABLES"
    result = await validator_no_schema.validate(sql, role=Role.ADMIN)
    assert result.status == ValidationStatus.BLOCKED
    assert result.reason == RejectionReason.DANGEROUS_OPERATION


# ── 6. RBAC — table restrictions ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_employee_cannot_access_payroll(validator):
    sql = "SELECT salary FROM payroll LIMIT 10;"
    result = await validator.validate(sql, role=Role.EMPLOYEE)
    assert result.status == ValidationStatus.BLOCKED
    assert result.reason == RejectionReason.RBAC_VIOLATION
    assert "payroll" in result.message


@pytest.mark.asyncio
async def test_employee_cannot_access_employees(validator):
    sql = "SELECT name, hire_date FROM employees LIMIT 10;"
    result = await validator.validate(sql, role=Role.EMPLOYEE)
    assert result.status == ValidationStatus.BLOCKED
    assert result.reason == RejectionReason.RBAC_VIOLATION


@pytest.mark.asyncio
async def test_manager_cannot_access_payroll(validator):
    sql = "SELECT employee_id, salary FROM payroll LIMIT 10;"
    result = await validator.validate(sql, role=Role.MANAGER)
    assert result.status == ValidationStatus.BLOCKED
    assert result.reason == RejectionReason.RBAC_VIOLATION


@pytest.mark.asyncio
async def test_admin_can_access_payroll(validator):
    sql = "SELECT employee_id, salary FROM payroll LIMIT 10;"
    result = await validator.validate(sql, role=Role.ADMIN)
    assert result.status == ValidationStatus.VALID


@pytest.mark.asyncio
async def test_manager_can_access_employees(validator):
    """Manager can access employees table (only payroll is restricted)."""
    sql = "SELECT id, name, department FROM employees LIMIT 20;"
    result = await validator.validate(sql, role=Role.MANAGER)
    assert result.status == ValidationStatus.VALID


# ── 7. Unknown table detection ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_hallucinated_table_rejected(validator):
    sql = "SELECT * FROM sales_summary LIMIT 10;"
    result = await validator.validate(sql, role=Role.ADMIN)
    assert result.status == ValidationStatus.INVALID
    assert result.reason == RejectionReason.UNKNOWN_TABLE
    assert "sales_summary" in result.message


@pytest.mark.asyncio
async def test_known_table_passes(validator):
    sql = "SELECT id, name FROM customers LIMIT 10;"
    result = await validator.validate(sql, role=Role.ADMIN)
    assert result.status == ValidationStatus.VALID


# ── 8. LIMIT auto-enforcement ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_missing_limit_auto_appended(validator):
    sql = "SELECT id, name FROM customers WHERE region = 'North';"
    result = await validator.validate(sql, role=Role.ADMIN)
    assert result.status == ValidationStatus.VALID
    assert "LIMIT" in result.sql.upper()


@pytest.mark.asyncio
async def test_existing_limit_preserved(validator):
    sql = "SELECT id FROM customers LIMIT 5;"
    result = await validator.validate(sql, role=Role.ADMIN)
    assert result.status == ValidationStatus.VALID
    assert result.sql.upper().count("LIMIT") == 1  # not doubled


# ── 9. Multiple statements ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_multiple_statements_blocked(validator_no_schema):
    sql = "SELECT 1; DROP TABLE orders;"
    result = await validator_no_schema.validate(sql, role=Role.ADMIN)
    assert result.status == ValidationStatus.BLOCKED
    assert result.reason == RejectionReason.MULTIPLE_STATEMENTS


# ── 10. Validation result fields ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_valid_result_contains_normalised_sql(validator):
    sql = "SELECT id FROM orders LIMIT 10;"
    result = await validator.validate(sql, role=Role.ADMIN)
    assert result.sql  # non-empty
    assert result.reason is None
    assert result.message == ""


@pytest.mark.asyncio
async def test_blocked_result_contains_message(validator):
    sql = "DROP TABLE orders;"
    result = await validator.validate(sql, role=Role.ADMIN)
    assert result.status == ValidationStatus.BLOCKED
    assert len(result.message) > 0
    assert result.reason is not None


# ── 11. RBAC column restriction ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_employee_restricted_column_blocked(validator):
    """Employee role cannot select hire_date from employees."""
    sql = "SELECT name, hire_date FROM employees LIMIT 10;"
    result = await validator.validate(sql, role=Role.EMPLOYEE)
    # Employee is blocked from employees table entirely first
    assert result.status == ValidationStatus.BLOCKED
    assert result.reason == RejectionReason.RBAC_VIOLATION
