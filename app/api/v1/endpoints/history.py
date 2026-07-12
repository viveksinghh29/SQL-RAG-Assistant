"""Conversation history endpoints.

GET /conversations            — list current user's conversations
GET /conversations/{id}       — fetch a conversation with all messages
DELETE /conversations/{id}    — delete a conversation
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import get_current_user
from app.conversation_memory.manager import ConversationManager
from app.core.exceptions import AuthorizationError, NotFoundError
from app.core.logging import get_logger
from app.database.models.user import User
from app.database.repositories.chat_repository import (
    ChatMessageRepository,
    ConversationRepository,
)
from app.database.session import get_db_session
from app.schemas.chat import ConversationRead, ConversationWithMessages

router = APIRouter(prefix="/conversations", tags=["history"])
log = get_logger("api.history")


def _get_manager(session: AsyncSession) -> ConversationManager:
    return ConversationManager(
        ConversationRepository(session),
        ChatMessageRepository(session),
    )


@router.get("", response_model=list[ConversationRead])
async def list_conversations(
    skip: int = 0,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> list[ConversationRead]:
    """List the current user's conversations, most recent first."""
    manager = _get_manager(session)
    return await manager.list_conversations(
        current_user.id, skip=skip, limit=min(limit, 100)
    )


@router.get("/{conversation_id}", response_model=ConversationWithMessages)
async def get_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> ConversationWithMessages:
    """Fetch a conversation with all its messages."""
    manager = _get_manager(session)
    conversation = await manager.get_conversation_with_messages(conversation_id)

    if conversation is None:
        raise NotFoundError(f"Conversation {conversation_id} not found.")

    if conversation.user_id != current_user.id:
        raise AuthorizationError("You do not have access to this conversation.")

    return ConversationWithMessages.model_validate(conversation)


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> None:
    """Delete a conversation and all its messages."""
    conv_repo = ConversationRepository(session)
    conversation = await conv_repo.get_by_id(conversation_id)

    if conversation is None:
        raise NotFoundError(f"Conversation {conversation_id} not found.")
    if conversation.user_id != current_user.id:
        raise AuthorizationError("You do not have access to this conversation.")

    await conv_repo.delete(conversation_id)
    await session.commit()
    log.bind(conversation_id=conversation_id, user_id=current_user.id).info(
        "Conversation deleted"
    )
