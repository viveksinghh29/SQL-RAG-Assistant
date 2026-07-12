"""Integration test configuration."""
import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def reset_login_limiter_integration():
    from app.auth import login_limiter
    login_limiter._attempts.clear()
    yield
    login_limiter._attempts.clear()


@pytest.fixture(autouse=True)
def ensure_schema_loaded_integration():
    from app.sql_validator.schema_registry import load_from_sql_file
    path = Path("data/seed/schema.sql")
    if path.exists():
        load_from_sql_file(path)
    yield
