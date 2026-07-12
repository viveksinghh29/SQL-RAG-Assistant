"""Unit test configuration - per-module isolation fixtures."""

import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def reset_login_limiter_unit():
    """Reset login rate limiter before and after every unit test."""
    from app.auth import login_limiter
    login_limiter._attempts.clear()
    yield
    login_limiter._attempts.clear()


@pytest.fixture(autouse=True)
def ensure_schema_registry_loaded():
    """Ensure schema registry is loaded for tests that need it."""
    from app.sql_validator.schema_registry import is_loaded, load_from_sql_file
    schema_path = Path("data/seed/schema.sql")
    if not is_loaded() and schema_path.exists():
        load_from_sql_file(schema_path)
    yield
