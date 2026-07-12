"""Pydantic schemas for conversations and chat messages."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: MessageRole
    content: str
    generated_sql: str | None = None
    execution_time_ms: int | None = None
    created_at: datetime


class ConversationCreate(BaseModel):
    title: str = "New Conversation"


class ConversationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationWithMessages(ConversationRead):
    messages: list[ChatMessageRead] = []
