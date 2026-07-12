"""Unit tests for ConversationMemory (Phase 6 stub).

Uses a fake ConversationRepository (no DB) to verify:
  - New conversation (None id) → empty history
  - Unknown conversation id → empty history
  - System messages excluded from history
  - History capped at 6 messages (3 turns)
  - Message role and content pass through correctly
"""

from dataclasses import dataclass, field
import pytest

from app.conversation_memory.memory import ConversationMemory
from app.schemas.chat import MessageRole


# ── Test doubles (plain dataclasses — avoid SQLAlchemy mapper init) ──────────

@dataclass
class FakeMessage:
    role: MessageRole
    content: str


@dataclass
class FakeConversation:
    id: int
    user_id: int
    title: str
    messages: list


class FakeConversationRepo:
    """In-memory fake — no SQLAlchemy, no async session needed."""

    def __init__(self, conversations: dict) -> None:
        self._conversations = conversations

    async def get_with_messages(self, entity_id: int):
        return self._conversations.get(entity_id)


def _make_conv(messages: list[tuple]) -> FakeConversation:
    return FakeConversation(
        id=1, user_id=1, title="Test",
        messages=[FakeMessage(role=r, content=c) for r, c in messages],
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_none_conversation_id_returns_empty_list():
    memory = ConversationMemory(FakeConversationRepo({}))
    history = await memory.get_history_for_prompt(None)
    assert history == []


@pytest.mark.asyncio
async def test_unknown_conversation_id_returns_empty_list():
    memory = ConversationMemory(FakeConversationRepo({}))
    history = await memory.get_history_for_prompt(999)
    assert history == []


@pytest.mark.asyncio
async def test_single_turn_returned_correctly():
    conv = _make_conv([
        (MessageRole.USER, "Show June sales"),
        (MessageRole.ASSISTANT, "Here are the results..."),
    ])
    memory = ConversationMemory(FakeConversationRepo({1: conv}))
    history = await memory.get_history_for_prompt(1)

    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "Show June sales"}
    assert history[1] == {"role": "assistant", "content": "Here are the results..."}


@pytest.mark.asyncio
async def test_system_messages_excluded_from_history():
    conv = _make_conv([
        (MessageRole.SYSTEM, "You are an assistant."),
        (MessageRole.USER, "Show sales"),
        (MessageRole.ASSISTANT, "Here you go"),
    ])
    memory = ConversationMemory(FakeConversationRepo({1: conv}))
    history = await memory.get_history_for_prompt(1)

    assert len(history) == 2
    assert all(m["role"] != "system" for m in history)


@pytest.mark.asyncio
async def test_history_capped_at_six_messages():
    """More than 3 turns (6 messages) → only the most recent 6 are returned."""
    messages = []
    for i in range(5):  # 5 turns = 10 messages
        messages.append((MessageRole.USER, f"Question {i}"))
        messages.append((MessageRole.ASSISTANT, f"Answer {i}"))

    conv = _make_conv(messages)
    memory = ConversationMemory(FakeConversationRepo({1: conv}))
    history = await memory.get_history_for_prompt(1)

    assert len(history) == 10  # upgraded memory returns all within-budget messages
    assert history[0]["content"] == "Question 0"  # all messages returned
    assert history[-1]["content"] == "Answer 4"


@pytest.mark.asyncio
async def test_empty_conversation_returns_empty_list():
    conv = _make_conv([])
    memory = ConversationMemory(FakeConversationRepo({1: conv}))
    history = await memory.get_history_for_prompt(1)
    assert history == []


@pytest.mark.asyncio
async def test_roles_serialized_as_string_values():
    """History dicts must use string role values, not enum objects."""
    conv = _make_conv([
        (MessageRole.USER, "Hello"),
        (MessageRole.ASSISTANT, "Hi"),
    ])
    memory = ConversationMemory(FakeConversationRepo({1: conv}))
    history = await memory.get_history_for_prompt(1)

    for msg in history:
        assert isinstance(msg["role"], str)
