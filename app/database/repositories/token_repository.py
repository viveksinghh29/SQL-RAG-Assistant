"""Refresh token repository.

Handles all DB interactions for refresh tokens: issuing, looking up,
rotating (revoke old + issue new), and revoking on logout.
"""

from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.security import hash_token
from app.config.settings import get_settings
from app.database.models.token import RefreshToken


class RefreshTokenRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._settings = get_settings()

    async def create(
        self,
        user_id: int,
        token_hash: str,
        user_agent: str | None = None,
    ) -> RefreshToken:
        """Persist a new hashed refresh token."""
        expires_at = datetime.now(UTC) + timedelta(
            days=self._settings.jwt_refresh_token_expire_days
        )
        token = RefreshToken(
            user_id=user_id,
            token_hash=token_hash,
            expires_at=expires_at,
            user_agent=user_agent,
        )
        self._session.add(token)
        await self._session.flush()
        await self._session.refresh(token)
        return token

    async def get_valid_token(self, raw_token: str) -> RefreshToken | None:
        """Look up a refresh token by raw value.

        Returns None if:
          - The token hash doesn't exist in the DB
          - The token has been revoked
          - The token has expired
        """
        token_hash = hash_token(raw_token)
        result = await self._session.execute(
            select(RefreshToken).where(
                RefreshToken.token_hash == token_hash,
                RefreshToken.is_revoked == False,  # noqa: E712
                RefreshToken.expires_at > datetime.now(UTC),
            )
        )
        return result.scalar_one_or_none()

    async def revoke(self, token: RefreshToken) -> None:
        """Mark a token as revoked."""
        token.is_revoked = True
        await self._session.flush()

    async def revoke_all_for_user(self, user_id: int) -> int:
        """Revoke every active token for a user (e.g. on password change).
        Returns the number of tokens revoked.
        """
        result = await self._session.execute(
            select(RefreshToken).where(
                RefreshToken.user_id == user_id,
                RefreshToken.is_revoked == False,  # noqa: E712
            )
        )
        tokens = result.scalars().all()
        for token in tokens:
            token.is_revoked = True
        await self._session.flush()
        return len(tokens)

    async def purge_expired(self) -> int:
        """Delete expired tokens (maintenance — run periodically).
        Returns count deleted.
        """
        from sqlalchemy import delete
        result = await self._session.execute(
            delete(RefreshToken).where(
                RefreshToken.expires_at <= datetime.now(UTC)
            )
        )
        return result.rowcount
