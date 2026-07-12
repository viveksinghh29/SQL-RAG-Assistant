"""User repository — concrete SQLAlchemy implementation of AbstractRepository."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.security import hash_password
from app.core.repository import AbstractRepository
from app.database.models.user import User
from app.schemas.user import UserCreate, UserUpdate


class UserRepository(AbstractRepository[User, UserCreate, UserUpdate]):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_id(self, entity_id: int) -> User | None:
        return await self._session.get(User, entity_id)

    async def get_by_email(self, email: str) -> User | None:
        result = await self._session.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    async def list(self, *, skip: int = 0, limit: int = 100) -> list[User]:
        result = await self._session.execute(select(User).offset(skip).limit(limit))
        return list(result.scalars().all())

    async def create(self, data: UserCreate) -> User:
        user = User(
            email=data.email,
            hashed_password=hash_password(data.password),
            full_name=data.full_name,
            role=data.role,
        )
        self._session.add(user)
        await self._session.flush()
        await self._session.refresh(user)
        return user

    async def update(self, entity_id: int, data: UserUpdate) -> User | None:
        user = await self.get_by_id(entity_id)
        if user is None:
            return None
        for field, value in data.model_dump(exclude_unset=True).items():
            setattr(user, field, value)
        await self._session.flush()
        await self._session.refresh(user)
        return user

    async def delete(self, entity_id: int) -> bool:
        user = await self.get_by_id(entity_id)
        if user is None:
            return False
        await self._session.delete(user)
        await self._session.flush()
        return True
