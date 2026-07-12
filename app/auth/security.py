"""Password hashing, JWT authentication, and refresh token utilities with secure token generation, hashing, and validation."""

import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.config.settings import get_settings
from app.core.exceptions import AuthenticationError

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Raw refresh tokens are 32 URL-safe random bytes (256 bits of entropy)
_REFRESH_TOKEN_BYTES = 32


def hash_password(plain_password: str) -> str:
    return _pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return _pwd_context.verify(plain_password, hashed_password)


def hash_token(raw_token: str) -> str:
    """SHA-256 hash a raw token for safe DB storage."""
    return hashlib.sha256(raw_token.encode()).hexdigest()


def create_access_token(
    subject: str,
    role: str,
    *,
    expires_delta: timedelta | None = None,
) -> str:
    """Issue a signed JWT access token."""
    settings = get_settings()
    expire = datetime.now(UTC) + (
        expires_delta or timedelta(minutes=settings.jwt_access_token_expire_minutes)
    )
    payload = {
        "sub": subject,
        "role": role,
        "exp": expire,
        "type": "access",
        "jti": secrets.token_hex(8),   # unique token id for future revocation
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def create_refresh_token() -> tuple[str, str]:
    """Generate a refresh token. Returns (raw_token, token_hash).

    Only the hash is stored in the database. The raw token is returned
    once and sent to the client — it is never retrievable from the DB.
    """
    raw = secrets.token_urlsafe(_REFRESH_TOKEN_BYTES)
    return raw, hash_token(raw)


def decode_token(token: str) -> dict[str, Any]:
    """Decode and verify a JWT, raising AuthenticationError if invalid/expired."""
    settings = get_settings()
    try:
        return jwt.decode(
            token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )
    except JWTError as exc:
        raise AuthenticationError("Invalid or expired token") from exc
