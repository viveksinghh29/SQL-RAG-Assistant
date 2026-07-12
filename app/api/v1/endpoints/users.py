"""User management endpoints.

POST /users          — register a new user (admin only in production)
GET  /users          — list all users (admin only)
GET  /users/{id}     — get a specific user (admin only)
PUT  /users/{id}     — update role / active status (admin only)
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import require_role
from app.auth.roles import Role
from app.core.exceptions import NotFoundError
from app.core.logging import get_logger
from app.database.models.user import User
from app.database.repositories.user_repository import UserRepository
from app.database.session import get_db_session
from app.schemas.user import UserCreate, UserRead, UserUpdate

router = APIRouter(prefix="/users", tags=["users"])
log = get_logger("api.users")


@router.post("", response_model=UserRead, status_code=201)
async def create_user(
    payload: UserCreate,
    _: User = Depends(require_role(Role.ADMIN)),
    session: AsyncSession = Depends(get_db_session),
) -> UserRead:
    """Register a new user. Requires Admin role."""
    repo = UserRepository(session)
    user = await repo.create(payload)
    await session.commit()
    log.bind(new_user_id=user.id, role=payload.role.value).info("User created")
    return UserRead.model_validate(user)


@router.get("", response_model=list[UserRead])
async def list_users(
    skip: int = 0,
    limit: int = 50,
    _: User = Depends(require_role(Role.ADMIN)),
    session: AsyncSession = Depends(get_db_session),
) -> list[UserRead]:
    """List all users. Requires Admin role."""
    repo = UserRepository(session)
    users = await repo.list(skip=skip, limit=min(limit, 200))
    return [UserRead.model_validate(u) for u in users]


@router.get("/{user_id}", response_model=UserRead)
async def get_user(
    user_id: int,
    _: User = Depends(require_role(Role.ADMIN)),
    session: AsyncSession = Depends(get_db_session),
) -> UserRead:
    """Get a user by ID. Requires Admin role."""
    repo = UserRepository(session)
    user = await repo.get_by_id(user_id)
    if user is None:
        raise NotFoundError(f"User {user_id} not found.")
    return UserRead.model_validate(user)


@router.put("/{user_id}", response_model=UserRead)
async def update_user(
    user_id: int,
    payload: UserUpdate,
    _: User = Depends(require_role(Role.ADMIN)),
    session: AsyncSession = Depends(get_db_session),
) -> UserRead:
    """Update a user's role or active status. Requires Admin role."""
    repo = UserRepository(session)
    user = await repo.update(user_id, payload)
    if user is None:
        raise NotFoundError(f"User {user_id} not found.")
    await session.commit()
    log.bind(user_id=user_id).info("User updated")
    return UserRead.model_validate(user)
