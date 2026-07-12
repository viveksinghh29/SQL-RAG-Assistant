"""API integration tests for Phase 11.

Uses httpx.AsyncClient with ASGITransport (httpx >= 0.20) and an
in-memory SQLite DB injected via FastAPI dependency_overrides.
"""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.auth.security import create_access_token, hash_password
from app.auth.roles import Role
from app.database.models import Base, User
from app.database.session import get_db_session
from app.main import create_app


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
async def test_session_factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


@pytest.fixture
async def app_client(test_session_factory):
    """FastAPI app with SQLite override, returns (AsyncClient, session_factory)."""
    async def override_get_db():
        async with test_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app = create_app()
    app.dependency_overrides[get_db_session] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac, test_session_factory

    app.dependency_overrides.clear()


@pytest.fixture
async def admin_token(test_session_factory):
    async with test_session_factory() as session:
        user = User(
            email="admin@test.com",
            hashed_password=hash_password("password123"),
            full_name="Admin User",
            role=Role.ADMIN,
            is_active=True,
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
    return user.id, create_access_token(subject=str(user.id), role=Role.ADMIN.value)


@pytest.fixture
async def employee_token(test_session_factory):
    async with test_session_factory() as session:
        user = User(
            email="employee@test.com",
            hashed_password=hash_password("password123"),
            full_name="Employee User",
            role=Role.EMPLOYEE,
            is_active=True,
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
    return user.id, create_access_token(subject=str(user.id), role=Role.EMPLOYEE.value)


def auth(token): return {"Authorization": f"Bearer {token}"}


@pytest.fixture(autouse=True)
def load_schema_registry():
    """Pre-load schema registry for every test that calls /api/v1/schema."""
    from pathlib import Path
    from app.sql_validator.schema_registry import load_from_sql_file, _SCHEMA
    schema_path = Path("data/seed/schema.sql")
    if schema_path.exists():
        load_from_sql_file(schema_path)
    yield
    _SCHEMA.clear()


# ── Health ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_root_health_always_200(app_client):
    ac, _ = app_client
    r = await ac.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_api_health_liveness(app_client):
    ac, _ = app_client
    r = await ac.get("/api/v1/health")
    assert r.status_code == 200


# ── Auth: login ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_login_success(app_client, admin_token):
    ac, _ = app_client
    r = await ac.post("/api/v1/auth/login",
                      json={"email": "admin@test.com", "password": "password123"})
    assert r.status_code == 200
    data = r.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["email"] == "admin@test.com"
    assert data["user"]["role"] == "admin"
    assert "hashed_password" not in data["user"]


@pytest.mark.asyncio
async def test_login_wrong_password(app_client, admin_token):
    ac, _ = app_client
    r = await ac.post("/api/v1/auth/login",
                      json={"email": "admin@test.com", "password": "wrongpass"})
    assert r.status_code == 401
    assert r.json()["error"] == "authentication_error"


@pytest.mark.asyncio
async def test_login_unknown_email(app_client):
    ac, _ = app_client
    r = await ac.post("/api/v1/auth/login",
                      json={"email": "nobody@test.com", "password": "anything"})
    assert r.status_code == 401


# ── Auth: /me ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_me_with_valid_token(app_client, admin_token):
    ac, _ = app_client
    _, token = admin_token
    r = await ac.get("/api/v1/auth/me", headers=auth(token))
    assert r.status_code == 200
    assert r.json()["email"] == "admin@test.com"


@pytest.mark.asyncio
async def test_get_me_without_token_returns_401(app_client):
    ac, _ = app_client
    r = await ac.get("/api/v1/auth/me")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_get_me_with_invalid_token_returns_401(app_client):
    ac, _ = app_client
    r = await ac.get("/api/v1/auth/me",
                     headers={"Authorization": "Bearer not.a.valid.token"})
    assert r.status_code == 401


# ── Schema: RBAC filtering ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_schema_admin_sees_all_tables(app_client, admin_token):
    ac, _ = app_client
    _, token = admin_token
    r = await ac.get("/api/v1/schema", headers=auth(token))
    assert r.status_code == 200
    table_names = [t["name"] for t in r.json()["tables"]]
    assert "payroll" in table_names
    assert "employees" in table_names


@pytest.mark.asyncio
async def test_schema_employee_cannot_see_restricted_tables(app_client, employee_token):
    ac, _ = app_client
    _, token = employee_token
    r = await ac.get("/api/v1/schema", headers=auth(token))
    assert r.status_code == 200
    table_names = [t["name"] for t in r.json()["tables"]]
    assert "payroll" not in table_names
    assert "employees" not in table_names


@pytest.mark.asyncio
async def test_schema_requires_auth(app_client):
    ac, _ = app_client
    r = await ac.get("/api/v1/schema")
    assert r.status_code == 401


# ── Conversations ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_conversations_empty_for_new_user(app_client, admin_token):
    ac, _ = app_client
    _, token = admin_token
    r = await ac.get("/api/v1/conversations", headers=auth(token))
    assert r.status_code == 200
    assert r.json() == []


@pytest.mark.asyncio
async def test_get_unknown_conversation_returns_404(app_client, admin_token):
    ac, _ = app_client
    _, token = admin_token
    r = await ac.get("/api/v1/conversations/9999", headers=auth(token))
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_delete_unknown_conversation_returns_404(app_client, admin_token):
    ac, _ = app_client
    _, token = admin_token
    r = await ac.delete("/api/v1/conversations/9999", headers=auth(token))
    assert r.status_code == 404


# ── Users (RBAC) ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_user_as_admin(app_client, admin_token):
    ac, _ = app_client
    _, token = admin_token
    r = await ac.post(
        "/api/v1/users",
        json={"email": "new@test.com", "password": "newpass99",
              "full_name": "New User", "role": "employee"},
        headers=auth(token),
    )
    assert r.status_code == 201
    data = r.json()
    assert data["email"] == "new@test.com"
    assert data["role"] == "employee"
    assert "hashed_password" not in data


@pytest.mark.asyncio
async def test_create_user_as_employee_returns_403(app_client, employee_token):
    ac, _ = app_client
    _, token = employee_token
    r = await ac.post(
        "/api/v1/users",
        json={"email": "other@test.com", "password": "pass1234",
              "full_name": "Other", "role": "employee"},
        headers=auth(token),
    )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_list_users_admin_only(app_client, admin_token):
    ac, _ = app_client
    _, token = admin_token
    r = await ac.get("/api/v1/users", headers=auth(token))
    assert r.status_code == 200
    assert isinstance(r.json(), list)


@pytest.mark.asyncio
async def test_response_never_exposes_hashed_password(app_client, admin_token):
    ac, _ = app_client
    _, token = admin_token
    r = await ac.get("/api/v1/users", headers=auth(token))
    for user_obj in r.json():
        assert "hashed_password" not in user_obj
        assert "password" not in user_obj
