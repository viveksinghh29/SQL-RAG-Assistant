"""Authentication endpoints — Phase 13 complete implementation.

POST /auth/login     — credentials → access token + refresh token
POST /auth/refresh   — refresh token → new access token + rotated refresh token
POST /auth/logout    — revoke active refresh token
GET  /auth/me        — current user profile

Security properties:
  - Login rate-limited per email (5 failures / 15 min)
  - Refresh tokens stored as SHA-256 hashes, never raw values
  - Token rotation: every /refresh issues a new refresh token and
    revokes the old one (prevents replay after theft)
  - /logout revokes the server-side refresh token immediately
  - No user enumeration: invalid email and wrong password return
    identical 401 responses after the same code path
  - hashed_password never appears in any response
"""

from fastapi import APIRouter, Header, Request
from fastapi import Depends
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import get_current_user
from app.auth.login_limiter import check_login_allowed, record_failure, record_success
from app.auth.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_password,
)
from app.core.exceptions import AuthenticationError
from app.core.logging import get_logger
from app.database.models.user import User
from app.database.repositories.token_repository import RefreshTokenRepository
from app.database.repositories.user_repository import UserRepository
from app.database.session import get_db_session
from app.schemas.user import UserRead

router = APIRouter(prefix="/auth", tags=["auth"])
log = get_logger("api.auth")


# ── Request / response schemas ────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: UserRead


class RefreshRequest(BaseModel):
    refresh_token: str


class AccessTokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/login", response_model=TokenResponse)
async def login(
    payload: LoginRequest,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> TokenResponse:
    """Authenticate and receive JWT access + refresh tokens.

    Rate-limited per email: 5 failed attempts per 15 minutes.
    Identical error message for unknown email and wrong password
    to prevent user enumeration.
    """
    email = payload.email.strip().lower()

    # Check rate limit BEFORE touching the DB (prevents timing oracle)
    check_login_allowed(email)

    user_repo = UserRepository(session)
    token_repo = RefreshTokenRepository(session)
    user = await user_repo.get_by_email(email)

    # Constant-time path: always verify even if user is None
    # (uses a dummy hash to prevent timing-based enumeration)
    _DUMMY_HASH = "$2b$12$KIXBjQpD4JKp7N0lF6KCGuehvCKdYB0Tv5.HIMVuaUiT4CnrJmpDi"
    candidate_hash = user.hashed_password if user else _DUMMY_HASH
    password_ok = verify_password(payload.password, candidate_hash)

    if user is None or not password_ok or not user.is_active:
        record_failure(email)
        raise AuthenticationError("Invalid email or password.")

    record_success(email)

    # Issue tokens
    access_token = create_access_token(
        subject=str(user.id), role=user.role.value
    )
    raw_refresh, refresh_hash = create_refresh_token()
    user_agent = request.headers.get("User-Agent", "")[:255]
    await token_repo.create(user.id, refresh_hash, user_agent=user_agent)
    await session.commit()

    log.bind(user_id=user.id, role=user.role.value).info("User logged in")

    return TokenResponse(
        access_token=access_token,
        refresh_token=raw_refresh,
        token_type="bearer",
        user=UserRead.model_validate(user),
    )


@router.post("/refresh", response_model=AccessTokenResponse)
async def refresh_token(
    payload: RefreshRequest,
    session: AsyncSession = Depends(get_db_session),
) -> AccessTokenResponse:
    """Exchange a valid refresh token for a new access token.

    Implements token rotation: the submitted refresh token is revoked and
    a new one is issued. Replay attacks (submitting an already-used token)
    are rejected because the old token is immediately marked revoked.
    """
    token_repo = RefreshTokenRepository(session)
    user_repo = UserRepository(session)

    db_token = await token_repo.get_valid_token(payload.refresh_token)
    if db_token is None:
        raise AuthenticationError(
            "Refresh token is invalid, expired, or has already been used."
        )

    user = await user_repo.get_by_id(db_token.user_id)
    if user is None or not user.is_active:
        await token_repo.revoke(db_token)
        await session.commit()
        raise AuthenticationError("Associated user account is inactive.")

    # Rotate: revoke old, issue new
    await token_repo.revoke(db_token)
    new_access = create_access_token(subject=str(user.id), role=user.role.value)
    raw_new_refresh, new_hash = create_refresh_token()
    await token_repo.create(user.id, new_hash)
    await session.commit()

    log.bind(user_id=user.id).info("Refresh token rotated")

    return AccessTokenResponse(
        access_token=new_access,
        refresh_token=raw_new_refresh,
    )


@router.post("/logout", status_code=204)
async def logout(
    payload: RefreshRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> None:
    """Revoke the supplied refresh token, effectively ending the session.

    Requires a valid access token (so only the legitimate session owner
    can revoke their own token). The access token itself expires naturally
    after JWT_ACCESS_TOKEN_EXPIRE_MINUTES.
    """
    token_repo = RefreshTokenRepository(session)
    db_token = await token_repo.get_valid_token(payload.refresh_token)

    if db_token is not None:
        # Extra check: only allow revoking own tokens
        if db_token.user_id != current_user.id:
            raise AuthenticationError("Cannot revoke another user's token.")
        await token_repo.revoke(db_token)
        await session.commit()

    log.bind(user_id=current_user.id).info("User logged out")


@router.get("/me", response_model=UserRead)
async def get_me(current_user: User = Depends(get_current_user)) -> UserRead:
    """Return the authenticated user's profile."""
    return UserRead.model_validate(current_user)
