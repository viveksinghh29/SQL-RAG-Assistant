"""Pytest configuration and shared fixtures.

`asyncio_mode = auto` is set in pyproject.toml, but we still need this
conftest to establish the event_loop scope and provide fixtures that span
multiple test modules (db session, mock LLM, etc.) as the test suite
grows through Phase 15.
"""

import os

import pytest

# Provide a minimal .env so Settings can be imported without a real .env file
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-ci-only-32chars!")
os.environ.setdefault("GROQ_API_KEY", "fake-key-for-tests")
os.environ.setdefault("APP_DB_PASSWORD", "test")
os.environ.setdefault("TARGET_DB_PASSWORD", "test")


@pytest.fixture(autouse=True)
def reset_login_limiter():
    """Isolate login rate limiter state for every test in the suite."""
    from app.auth import login_limiter
    login_limiter._attempts.clear()
    yield
    login_limiter._attempts.clear()


@pytest.fixture(autouse=True)
def load_schema_globally():
    """Ensure schema registry is loaded for all tests."""
    from pathlib import Path
    from app.sql_validator.schema_registry import is_loaded, load_from_sql_file
    path = Path("data/seed/schema.sql")
    if path.exists() and not is_loaded():
        load_from_sql_file(path)
    yield
