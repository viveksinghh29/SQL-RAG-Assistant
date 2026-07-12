"""Generic repository pattern interfaces.

Every data-access class in this project implements `AbstractRepository`
rather than letting services talk to SQLAlchemy directly. This keeps
services testable (mock the repository, not the database) and keeps the
ORM swappable in principle, though SQLAlchemy is the only implementation
planned.

This module deliberately has zero dependency on SQLAlchemy — it is a pure
contract. The concrete SQLAlchemy implementation lives in
`app/database/repositories/` (built in Phase 3).
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

ModelType = TypeVar("ModelType")
CreateSchemaType = TypeVar("CreateSchemaType")
UpdateSchemaType = TypeVar("UpdateSchemaType")


class AbstractRepository(ABC, Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """Contract for CRUD access to a single entity type.

    Generic over the ORM model and its Pydantic create/update schemas so
    concrete repositories (UserRepository, ChatHistoryRepository, ...)
    get full type checking without duplicating boilerplate.
    """

    @abstractmethod
    async def get_by_id(self, entity_id: int) -> ModelType | None:
        """Fetch a single entity by primary key, or None if not found."""
        raise NotImplementedError

    @abstractmethod
    async def list(self, *, skip: int = 0, limit: int = 100) -> list[ModelType]:
        """Fetch a paginated list of entities."""
        raise NotImplementedError

    @abstractmethod
    async def create(self, data: CreateSchemaType) -> ModelType:
        """Persist a new entity and return it."""
        raise NotImplementedError

    @abstractmethod
    async def update(self, entity_id: int, data: UpdateSchemaType) -> ModelType | None:
        """Update an existing entity, returning the updated row or None."""
        raise NotImplementedError

    @abstractmethod
    async def delete(self, entity_id: int) -> bool:
        """Delete an entity. Returns True if a row was deleted."""
        raise NotImplementedError
