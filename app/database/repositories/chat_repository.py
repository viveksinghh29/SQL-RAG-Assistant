"""Conversation/ChatMessage repositories — concrete SQLAlchemy implementations."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.repository import AbstractRepository
from app.database.models.chat import ChatMessage, Conversation
from app.schemas.chat import ConversationCreate


class ConversationRepository(AbstractRepository[Conversation, ConversationCreate, ConversationCreate]):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_id(self, entity_id: int) -> Conversation | None:
        return await self._session.get(Conversation, entity_id)

    async def get_with_messages(self, entity_id: int) -> Conversation | None:
        """Eager-load messages — used by conversation memory (Phase 10) to
        reconstruct multi-turn context without N+1 queries."""
        result = await self._session.execute(
            select(Conversation)
            .where(Conversation.id == entity_id)
            .options(selectinload(Conversation.messages))
        )
        return result.scalar_one_or_none()

    async def list_for_user(self, user_id: int, *, skip: int = 0, limit: int = 50) -> list[Conversation]:
        result = await self._session.execute(
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def list(self, *, skip: int = 0, limit: int = 100) -> list[Conversation]:
        result = await self._session.execute(select(Conversation).offset(skip).limit(limit))
        return list(result.scalars().all())

    async def create(self, data: ConversationCreate, *, user_id: int | None = None) -> Conversation:
        conversation = Conversation(title=data.title, user_id=user_id)
        self._session.add(conversation)
        await self._session.flush()
        await self._session.refresh(conversation)
        return conversation

    async def update(self, entity_id: int, data: ConversationCreate) -> Conversation | None:
        conversation = await self.get_by_id(entity_id)
        if conversation is None:
            return None
        conversation.title = data.title
        await self._session.flush()
        return conversation

    async def delete(self, entity_id: int) -> bool:
        conversation = await self.get_by_id(entity_id)
        if conversation is None:
            return False
        await self._session.delete(conversation)
        await self._session.flush()
        return True


class ChatMessageRepository:
    """Not a full AbstractRepository — messages are only ever appended to a
    conversation, never independently listed/updated, so a narrower
    interface is more honest than forcing the generic CRUD shape."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add_message(self, message: ChatMessage) -> ChatMessage:
        self._session.add(message)
        await self._session.flush()
        await self._session.refresh(message)
        return message
