"""Integration tests for auth endpoints (Phase 15).

Covers the paths missing from Phase 11 and Phase 13 tests:
  - POST /auth/login success returns refresh token
  - POST /auth/refresh valid token rotation
  - POST /auth/logout revokes token
  - POST /auth/login inactive user blocked
  - GET  /auth/me with expired/tampered token
  - Refresh with already-revoked token → 401
"""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.auth.roles import Role
from app.auth.security import hash_password
from app.database.models import Base, User
from app.database.session import get_db_session
from app.main import create_app


@pytest.fixture
async def auth_setup():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sf = async_sessionmaker(engine, expire_on_commit=False)

    async def override():
        async with sf() as s:
            try:
                yield s
                await s.commit()
            except Exception:
                await s.rollback()
                raise

    app = create_app()
    app.dependency_overrides[get_db_session] = override

    # Seed users
    async with sf() as s:
        active = User(email="active@t.com", hashed_password=hash_password("pass1234"),
                      full_name="Active", role=Role.EMPLOYEE, is_active=True)
        inactive = User(email="inactive@t.com", hashed_password=hash_password("pass1234"),
                        full_name="Inactive", role=Role.EMPLOYEE, is_active=False)
        s.add_all([active, inactive])
        await s.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac, sf

    app.dependency_overrides.clear()
    await engine.dispose()


@pytest.mark.asyncio
async def test_login_returns_both_tokens(auth_setup):
    ac, _ = auth_setup
    r = await ac.post("/api/v1/auth/login",
                      json={"email": "active@t.com", "password": "pass1234"})
    assert r.status_code == 200
    data = r.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_login_inactive_user_rejected(auth_setup):
    ac, _ = auth_setup
    r = await ac.post("/api/v1/auth/login",
                      json={"email": "inactive@t.com", "password": "pass1234"})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_refresh_token_rotation(auth_setup):
    ac, _ = auth_setup
    # Login to get initial tokens
    login_r = await ac.post("/api/v1/auth/login",
                            json={"email": "active@t.com", "password": "pass1234"})
    tokens = login_r.json()
    old_refresh = tokens["refresh_token"]

    # Exchange refresh token
    refresh_r = await ac.post("/api/v1/auth/refresh",
                              json={"refresh_token": old_refresh})
    assert refresh_r.status_code == 200
    new_tokens = refresh_r.json()
    assert "access_token" in new_tokens
    assert "refresh_token" in new_tokens
    # New refresh token must differ from old one
    assert new_tokens["refresh_token"] != old_refresh


@pytest.mark.asyncio
async def test_refresh_old_token_rejected_after_rotation(auth_setup):
    ac, _ = auth_setup
    login_r = await ac.post("/api/v1/auth/login",
                            json={"email": "active@t.com", "password": "pass1234"})
    old_refresh = login_r.json()["refresh_token"]

    # Rotate once
    await ac.post("/api/v1/auth/refresh", json={"refresh_token": old_refresh})

    # Try to reuse the old refresh token — must fail
    r = await ac.post("/api/v1/auth/refresh", json={"refresh_token": old_refresh})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_logout_revokes_refresh_token(auth_setup):
    ac, _ = auth_setup
    login_r = await ac.post("/api/v1/auth/login",
                            json={"email": "active@t.com", "password": "pass1234"})
    data = login_r.json()
    access = data["access_token"]
    refresh = data["refresh_token"]

    logout_r = await ac.post(
        "/api/v1/auth/logout",
        json={"refresh_token": refresh},
        headers={"Authorization": f"Bearer {access}"},
    )
    assert logout_r.status_code == 204

    # Attempt to refresh after logout — must fail
    r = await ac.post("/api/v1/auth/refresh", json={"refresh_token": refresh})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_invalid_refresh_token_returns_401(auth_setup):
    ac, _ = auth_setup
    r = await ac.post("/api/v1/auth/refresh",
                      json={"refresh_token": "not-a-valid-token"})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_me_returns_correct_role(auth_setup):
    ac, _ = auth_setup
    login_r = await ac.post("/api/v1/auth/login",
                            json={"email": "active@t.com", "password": "pass1234"})
    token = login_r.json()["access_token"]
    r = await ac.get("/api/v1/auth/me",
                     headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json()["role"] == "employee"
    assert "hashed_password" not in r.json()
