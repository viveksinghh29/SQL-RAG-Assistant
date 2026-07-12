"""API test configuration."""
import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def reset_login_limiter_api():
    from app.auth import login_limiter
    login_limiter._attempts.clear()
    yield
    login_limiter._attempts.clear()


@pytest.fixture(autouse=True)
def load_schema_for_api_tests():
    from app.sql_validator.schema_registry import load_from_sql_file
    path = Path("data/seed/schema.sql")
    if path.exists():
        load_from_sql_file(path)
    yield
