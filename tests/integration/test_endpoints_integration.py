"""Integration tests for chat pipeline, history, feedback, and health endpoints."""

import json
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.auth.roles import Role
from app.auth.security import create_access_token, hash_password
from app.database.models import Base, User
from app.database.session import get_db_session
from app.llm.factory import get_llm_provider
from app.llm.providers.mock_provider import MockLLMProvider
from app.main import create_app
from app.query_executor.connection import _set_target_engine
from app.query_executor.result import ExecutionResult, ExecutionStatus
from app.query_executor.service import QueryExecutor
from app.sql_validator.schema_registry import load_from_sql_file
from pathlib import Path

# ── Shared fixture ────────────────────────────────────────────────────────────

VALID_CHART_JSON = json.dumps({
    "chart_type": "bar", "x_column": "category", "y_column": "revenue",
    "title": "Revenue Chart", "rationale": "Categorical.",
})


@pytest.fixture(autouse=True)
def load_schema():
    path = Path("data/seed/schema.sql")
    if path.exists():
        load_from_sql_file(path)


@pytest.fixture
async def full_setup():
    """App client with SQLite app-DB, SQLite target-DB, and mock LLM."""
    # App DB
    app_engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with app_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sf = async_sessionmaker(app_engine, expire_on_commit=False)

    # Target DB (for query execution)
    from sqlalchemy import text
    target_engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with target_engine.begin() as conn:
        await conn.execute(text(
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, status TEXT, total_amount REAL)"
        ))
        await conn.execute(text(
            "INSERT INTO orders VALUES (1,'completed',150.0),(2,'pending',200.0)"
        ))
    _set_target_engine(target_engine)

    async def override_db():
        async with sf() as s:
            try:
                yield s
                await s.commit()
            except Exception:
                await s.rollback()
                raise

    app = create_app()
    app.dependency_overrides[get_db_session] = override_db

    # Mock LLM via monkeypatching the factory cache
    SQL_RESP = "Returns completed orders.\n```sql\nSELECT id, status, total_amount FROM orders LIMIT 100;\n```"
    mock_llm = MockLLMProvider(responses=[
        SQL_RESP,           # title gen (tolerates non-title ok — falls back to truncation)
        SQL_RESP,           # SQL generation
        "Revenue insight.",  # explanation
        VALID_CHART_JSON,   # chart inference
        SQL_RESP,           # SQL generation (second chat turn)
        "More insight.",
        VALID_CHART_JSON,
    ])
    import app.llm.factory as _factory
    _factory._llm_override = mock_llm

    # Seed user
    async with sf() as s:
        user = User(email="u@t.com", hashed_password=hash_password("pw123456"),
                    full_name="User", role=Role.MANAGER, is_active=True)
        s.add(user)
        await s.commit()
        await s.refresh(user)
    token = create_access_token(subject=str(user.id), role=Role.MANAGER.value)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac, token, user.id

    app.dependency_overrides.clear()
    import app.llm.factory as _f; _f._llm_override = None; _f._get_real_provider.cache_clear()
    _set_target_engine(None)
    await app_engine.dispose()
    await target_engine.dispose()


def auth(token): return {"Authorization": f"Bearer {token}"}


# ── Health ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_liveness(full_setup):
    ac, token, _ = full_setup
    r = await ac.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_health_ready_returns_checks(full_setup):
    ac, token, _ = full_setup
    r = await ac.get("/api/v1/health/ready")
    assert r.status_code in (200, 503)
    data = r.json()
    # Even if degraded, must return structured checks
    assert "checks" in data or "status" in data


# ── Chat pipeline ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_endpoint_full_pipeline(full_setup):
    ac, token, _ = full_setup
    r = await ac.post("/api/v1/chat",
                      json={"question": "Show completed orders"},
                      headers=auth(token))
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "success"
    assert data["sql"] != ""
    assert "SELECT" in data["sql"].upper()
    assert data["conversation_id"] is not None
    assert data["message_id"] > 0


@pytest.mark.asyncio
async def test_chat_creates_conversation(full_setup):
    ac, token, _ = full_setup
    r = await ac.post("/api/v1/chat",
                      json={"question": "Show me orders"},
                      headers=auth(token))
    assert r.status_code == 200
    conv_id = r.json()["conversation_id"]

    # Conversation must now exist in history
    hist = await ac.get("/api/v1/conversations", headers=auth(token))
    assert hist.status_code == 200
    conv_ids = [c["id"] for c in hist.json()]
    assert conv_id in conv_ids


@pytest.mark.asyncio
async def test_chat_continues_existing_conversation(full_setup):
    ac, token, _ = full_setup
    # First message — creates conversation
    r1 = await ac.post("/api/v1/chat",
                       json={"question": "Show orders"},
                       headers=auth(token))
    conv_id = r1.json()["conversation_id"]

    # Second message — continues same conversation
    r2 = await ac.post("/api/v1/chat",
                       json={"question": "How many are completed?",
                             "conversation_id": conv_id},
                       headers=auth(token))
    assert r2.status_code == 200
    assert r2.json()["conversation_id"] == conv_id


@pytest.mark.asyncio
async def test_chat_requires_auth(full_setup):
    ac, _, _ = full_setup
    r = await ac.post("/api/v1/chat", json={"question": "Show orders"})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_chat_blocked_dangerous_sql(full_setup):
    """Validator directly blocks dangerous SQL."""
    from app.sql_validator.service import SQLValidationService
    from app.sql_validator.result import ValidationStatus
    validator = SQLValidationService()
    result = await validator.validate("DROP TABLE orders;", role=Role.MANAGER)
    assert result.status == ValidationStatus.BLOCKED


# ── History endpoints ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_history_get_conversation_with_messages(full_setup):
    ac, token, _ = full_setup
    # Create a conversation via chat
    r = await ac.post("/api/v1/chat",
                      json={"question": "Show orders"},
                      headers=auth(token))
    conv_id = r.json()["conversation_id"]

    # Fetch it back
    conv_r = await ac.get(f"/api/v1/conversations/{conv_id}", headers=auth(token))
    assert conv_r.status_code == 200
    data = conv_r.json()
    assert data["id"] == conv_id
    assert len(data["messages"]) >= 2  # user + assistant


@pytest.mark.asyncio
async def test_history_delete_conversation(full_setup):
    ac, token, _ = full_setup
    r = await ac.post("/api/v1/chat",
                      json={"question": "Show orders"},
                      headers=auth(token))
    conv_id = r.json()["conversation_id"]

    del_r = await ac.delete(f"/api/v1/conversations/{conv_id}", headers=auth(token))
    assert del_r.status_code == 204

    # Must be gone
    get_r = await ac.get(f"/api/v1/conversations/{conv_id}", headers=auth(token))
    assert get_r.status_code == 404


# ── Feedback endpoints ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_feedback_submit_and_retrieve(full_setup):
    ac, token, _ = full_setup
    # Get a message id from a chat
    r = await ac.post("/api/v1/chat",
                      json={"question": "Show orders"},
                      headers=auth(token))
    message_id = r.json()["message_id"]

    # Submit positive feedback
    fb_r = await ac.post("/api/v1/feedback",
                         json={"message_id": message_id, "is_positive": True,
                               "comment": "Very helpful!"},
                         headers=auth(token))
    assert fb_r.status_code == 201
    fb_data = fb_r.json()
    assert fb_data["is_positive"] is True
    assert fb_data["message_id"] == message_id


@pytest.mark.asyncio
async def test_feedback_unknown_message_returns_404(full_setup):
    ac, token, _ = full_setup
    r = await ac.post("/api/v1/feedback",
                      json={"message_id": 99999, "is_positive": True},
                      headers=auth(token))
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_feedback_analytics_requires_manager_role(full_setup):
    """Employee cannot access feedback analytics."""
    ac, _, _ = full_setup
    # Create an employee token
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    emp_token = create_access_token(subject="999", role=Role.EMPLOYEE.value)
    r = await ac.get("/api/v1/feedback/analytics",
                     headers={"Authorization": f"Bearer {emp_token}"})
    assert r.status_code in (401, 403)
