"""FastAPI authentication and authorization dependencies for user authentication and role-based access control."""

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.auth.roles import Role
from app.auth.security import decode_token
from app.core.exceptions import AuthenticationError, AuthorizationError
from app.core.logging import get_logger
from app.database.models.user import User
from app.database.repositories.user_repository import UserRepository
from app.database.session import get_db_session
from sqlalchemy.ext.asyncio import AsyncSession

log = get_logger("auth.dependencies")

_bearer = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    session: AsyncSession = Depends(get_db_session),
) -> User:
    """Extract and validate the JWT bearer token, return the active User.

    Raises `AuthenticationError` (→ 401) if:
      - No Authorization header present
      - Token is invalid or expired
      - User does not exist in the database
      - User account is inactive
    """
    if credentials is None:
        raise AuthenticationError("Authorization header is required.")

    payload = decode_token(credentials.credentials)

    user_id_str = payload.get("sub")
    if not user_id_str:
        raise AuthenticationError("Token payload missing 'sub' claim.")

    try:
        user_id = int(user_id_str)
    except ValueError:
        raise AuthenticationError("Token 'sub' claim is not a valid user ID.")

    repo = UserRepository(session)
    user = await repo.get_by_id(user_id)

    if user is None:
        raise AuthenticationError("User not found.")
    if not user.is_active:
        raise AuthenticationError("User account is inactive.")

    return user


def require_role(*roles: Role):
    """Return a dependency that enforces one of the given roles.

    Usage:
        Depends(require_role(Role.ADMIN))
        Depends(require_role(Role.ADMIN, Role.MANAGER))

    Raises `AuthorizationError` (→ 403) if the user's role is not in
    the allowed set.
    """
    allowed = frozenset(roles)

    async def _check_role(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed:
            raise AuthorizationError(
                f"This action requires one of these roles: "
                f"{', '.join(r.value for r in sorted(allowed))}.",
                details={"required_roles": [r.value for r in sorted(allowed)],
                         "current_role": current_user.role.value},
            )
        return current_user

    return _check_role
