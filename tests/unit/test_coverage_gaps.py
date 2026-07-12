"""Coverage gap tests for parser_utils, SQL validator, and cache client (Phase 15)."""

import pytest

from app.auth.roles import Role
from app.sql_validator.parser_utils import (
    extract_select_columns,
    extract_table_names,
    get_first_keyword,
    has_limit_clause,
    count_statements,
)
from app.sql_validator.result import ValidationStatus, RejectionReason
from app.sql_validator.service import SQLValidationService


# ── parser_utils ──────────────────────────────────────────────────────────────

def test_get_first_keyword_select():
    assert get_first_keyword("SELECT * FROM orders") == "SELECT"


def test_get_first_keyword_case_insensitive():
    assert get_first_keyword("select id from orders") == "SELECT"


def test_get_first_keyword_with_leading_whitespace():
    assert get_first_keyword("   DROP TABLE orders") == "DROP"


def test_get_first_keyword_empty():
    assert get_first_keyword("") == ""


def test_get_first_keyword_truncate():
    assert get_first_keyword("TRUNCATE TABLE orders") == "TRUNCATE"


def test_get_first_keyword_grant():
    assert get_first_keyword("GRANT ALL ON orders TO user") == "GRANT"


def test_get_first_keyword_create():
    assert get_first_keyword("CREATE TABLE new_table (id INT)") == "CREATE"


def test_extract_table_names_simple_from():
    tables = extract_table_names("SELECT * FROM orders")
    assert "orders" in tables


def test_extract_table_names_join():
    sql = "SELECT o.id, c.name FROM orders o JOIN customers c ON c.id = o.customer_id"
    tables = extract_table_names(sql)
    assert "orders" in tables
    assert "customers" in tables


def test_extract_table_names_multiple_joins():
    sql = (
        "SELECT * FROM orders o "
        "JOIN order_items oi ON oi.order_id = o.id "
        "JOIN products p ON p.id = oi.product_id"
    )
    tables = extract_table_names(sql)
    assert "orders" in tables
    assert "order_items" in tables
    assert "products" in tables


def test_extract_table_names_empty_sql():
    tables = extract_table_names("")
    assert tables == set()


def test_extract_select_columns_star_returns_empty():
    cols = extract_select_columns("SELECT * FROM orders")
    assert cols == set()


def test_has_limit_clause_present():
    assert has_limit_clause("SELECT * FROM orders LIMIT 10") is True


def test_has_limit_clause_absent():
    assert has_limit_clause("SELECT * FROM orders") is False


def test_has_limit_clause_case_insensitive():
    assert has_limit_clause("SELECT * FROM orders limit 5") is True


def test_count_statements_single():
    assert count_statements("SELECT 1") == 1


# ── SQL validator extended paths ──────────────────────────────────────────────

SCHEMA = {
    "orders": {"id", "customer_id", "order_date", "status", "total_amount"},
    "customers": {"id", "name", "email", "region"},
    "payroll": {"employee_id", "salary", "bonus"},
}


@pytest.mark.asyncio
async def test_validator_accepts_aggregate_query():
    validator = SQLValidationService(known_tables=SCHEMA)
    sql = "SELECT COUNT(*) as total, SUM(total_amount) as revenue FROM orders WHERE status='completed' LIMIT 1;"
    result = await validator.validate(sql, role=Role.ADMIN)
    assert result.status == ValidationStatus.VALID


@pytest.mark.asyncio
async def test_validator_accepts_subquery():
    validator = SQLValidationService(known_tables=SCHEMA)
    sql = (
        "SELECT * FROM orders WHERE customer_id IN "
        "(SELECT id FROM customers WHERE region = 'North') LIMIT 10;"
    )
    result = await validator.validate(sql, role=Role.ADMIN)
    assert result.status == ValidationStatus.VALID


@pytest.mark.asyncio
async def test_validator_appends_limit_without_trailing_semicolon():
    validator = SQLValidationService(known_tables=SCHEMA)
    sql = "SELECT id FROM customers"
    result = await validator.validate(sql, role=Role.ADMIN)
    assert result.status == ValidationStatus.VALID
    assert "LIMIT" in result.sql.upper()
    # Must not double-apply limit
    assert result.sql.upper().count("LIMIT") == 1


@pytest.mark.asyncio
async def test_validator_hex_encoding_blocked():
    validator = SQLValidationService()
    sql = "SELECT 0x41424344 FROM orders LIMIT 1"
    result = await validator.validate(sql, role=Role.ADMIN)
    assert result.status == ValidationStatus.BLOCKED
    assert result.reason == RejectionReason.SQL_INJECTION


@pytest.mark.asyncio
async def test_validator_benchmark_injection_blocked():
    validator = SQLValidationService()
    sql = "SELECT BENCHMARK(1000000, MD5('a')) FROM orders LIMIT 1"
    result = await validator.validate(sql, role=Role.ADMIN)
    assert result.status == ValidationStatus.BLOCKED


@pytest.mark.asyncio
async def test_validator_no_schema_skips_table_check():
    """Without a schema, table existence check is skipped — not blocked."""
    validator = SQLValidationService()  # no known_tables
    sql = "SELECT * FROM totally_fake_table LIMIT 10;"
    result = await validator.validate(sql, role=Role.ADMIN)
    # Should pass since no schema to check against
    assert result.status == ValidationStatus.VALID


@pytest.mark.asyncio
async def test_validator_load_file_injection_blocked():
    validator = SQLValidationService()
    sql = "SELECT LOAD_FILE('/etc/passwd') LIMIT 1"
    result = await validator.validate(sql, role=Role.ADMIN)
    assert result.status == ValidationStatus.BLOCKED
    assert result.reason == RejectionReason.SQL_INJECTION


@pytest.mark.asyncio
async def test_validator_replace_statement_blocked():
    validator = SQLValidationService()
    result = await validator.validate(
        "REPLACE INTO orders VALUES (1, 1, '2024-01-01', 'done', 100);",
        role=Role.ADMIN
    )
    assert result.status == ValidationStatus.BLOCKED


# ── Cache client (in-memory fake) ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cache_client_connect_without_redis():
    """CacheClient must not raise when Redis is unavailable at startup."""
    from app.utils.cache_client import CacheClient
    import sys
    # Simulate redis not installed by hiding the module
    original = sys.modules.pop("redis", None)
    original_async = sys.modules.pop("redis.asyncio", None)
    try:
        client = CacheClient()
        client._enabled = True
        await client.connect()   # must not raise even if redis import fails
        assert client._client is None
    finally:
        if original is not None:
            sys.modules["redis"] = original
        if original_async is not None:
            sys.modules["redis.asyncio"] = original_async


@pytest.mark.asyncio
async def test_cache_client_get_returns_none_when_unavailable():
    from app.utils.cache_client import CacheClient
    client = CacheClient()
    client._client = None  # simulate unavailable
    result = await client.get("any-key")
    assert result is None


@pytest.mark.asyncio
async def test_cache_client_set_is_noop_when_unavailable():
    from app.utils.cache_client import CacheClient
    client = CacheClient()
    client._client = None
    await client.set("key", {"data": 1})  # must not raise


@pytest.mark.asyncio
async def test_cache_client_is_available_false_when_no_client():
    from app.utils.cache_client import CacheClient
    client = CacheClient()
    client._client = None
    result = await client.is_available()
    assert result is False


# ── LLM parser edge cases ─────────────────────────────────────────────────────

def test_extract_sql_with_backtick_table_names():
    from app.llm.parser import extract_sql
    response = "Fetches orders.\n```sql\nSELECT `id`, `status` FROM `orders` LIMIT 10;\n```"
    sql = extract_sql(response)
    assert "SELECT" in sql.upper()
    assert "orders" in sql


def test_extract_sql_multiline_query():
    from app.llm.parser import extract_sql
    response = (
        "Multi-join query.\n```sql\n"
        "SELECT o.id,\n       c.name,\n       o.total_amount\n"
        "FROM orders o\nJOIN customers c ON c.id = o.customer_id\n"
        "LIMIT 50;\n```"
    )
    sql = extract_sql(response)
    assert "JOIN" in sql.upper()
    assert "LIMIT 50" in sql


def test_extract_json_embedded_in_prose():
    from app.llm.parser import extract_json
    response = (
        'Here is my decision: {"chart_type": "pie", "x_column": "region", '
        '"y_column": "revenue", "title": "By Region", "rationale": "Part-of-whole."} '
        "I chose pie because it shows proportions."
    )
    data = extract_json(response)
    assert data["chart_type"] == "pie"
