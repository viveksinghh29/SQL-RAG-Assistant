"""Unit tests for Phase 10: Conversation Memory & Context Management.

All tests use fakes (no DB, no real LLM).

Coverage:
  TokenBudget:
    - estimate_tokens accuracy
    - estimate_messages_tokens
    - trim_to_budget respects limit, preserves recent messages
    - trim_to_budget handles empty list
  ContextSummariser:
    - Returns formatted summary from LLM
    - LLM failure → fallback string, no exception
    - Empty message list → empty string
  ConversationMemory (upgraded):
    - Short history (within budget) returned in full
    - Long history (exceeds budget) → summarised older + trimmed recent
    - No LLM provided → older messages dropped, recent kept
    - None conversation_id → []
    - Unknown conversation_id → []
    - System messages excluded from history
  ConversationManager:
    - get_or_create_conversation creates new with generated title
    - get_or_create_conversation returns existing for known id
    - Title truncation fallback when LLM is None
    - persist_turn saves user + assistant messages
    - _format_assistant_content combines description + explanation
"""

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.conversation_memory.manager import ConversationManager
from app.conversation_memory.memory import ConversationMemory
from app.conversation_memory.summariser import ContextSummariser
from app.conversation_memory.token_budget import (
    HISTORY_TOKEN_BUDGET,
    estimate_messages_tokens,
    estimate_tokens,
    trim_to_budget,
)
from app.core.exceptions import LLMProviderError
from app.llm.providers.mock_provider import MockLLMProvider
from app.schemas.chat import MessageRole
from app.schemas.pipeline import ChatResponse, GenerationStatus


# ── Shared test doubles ───────────────────────────────────────────────────────

@dataclass
class FakeMessage:
    role: MessageRole
    content: str


@dataclass
class FakeConversation:
    id: int
    user_id: int
    title: str
    messages: list = field(default_factory=list)


class FakeConversationRepo:
    def __init__(self, conversations: dict | None = None):
        self._convs: dict[int, FakeConversation] = conversations or {}
        self._next_id = max(self._convs.keys(), default=0) + 1
        self.created: list[FakeConversation] = []

    async def get_by_id(self, entity_id: int):
        return self._convs.get(entity_id)

    async def get_with_messages(self, entity_id: int):
        return self._convs.get(entity_id)

    async def list_for_user(self, user_id: int, *, skip=0, limit=50):
        return [c for c in self._convs.values() if c.user_id == user_id]

    async def create(self, data, *, user_id=None):
        conv = FakeConversation(
            id=self._next_id, user_id=user_id or 0, title=data.title
        )
        self._convs[conv.id] = conv
        self.created.append(conv)
        self._next_id += 1
        return conv


class FakeMsgRepo:
    def __init__(self):
        self.saved: list = []
        self._next_id = 1

    async def add_message(self, msg):
        msg.id = self._next_id
        self._next_id += 1
        self.saved.append(msg)
        return msg


def _make_conv(messages: list[tuple]) -> FakeConversation:
    return FakeConversation(
        id=1, user_id=1, title="Test",
        messages=[FakeMessage(role=r, content=c) for r, c in messages]
    )


def _make_chat_response(**kwargs) -> ChatResponse:
    defaults = dict(
        conversation_id=1, message_id=1, question="test",
        status=GenerationStatus.SUCCESS, description="Desc", explanation="Explains",
    )
    defaults.update(kwargs)
    return ChatResponse(**defaults)


# ═══════════════════════════════════════════════════════════════════════════════
# TokenBudget
# ═══════════════════════════════════════════════════════════════════════════════

def test_estimate_tokens_basic():
    # 40 chars ÷ 4 = 10 tokens
    assert estimate_tokens("a" * 40) == 10


def test_estimate_tokens_minimum_one():
    assert estimate_tokens("") == 1


def test_estimate_messages_tokens():
    messages = [
        {"role": "user",      "content": "a" * 100},
        {"role": "assistant", "content": "b" * 200},
    ]
    est = estimate_messages_tokens(messages)
    # (300 chars // 4) + (2 * 4 overhead) = 75 + 8 = 83
    assert est.tokens == 83
    assert est.chars == 300


def test_trim_to_budget_keeps_recent():
    messages = [{"role": "user", "content": f"message {i}" * 10} for i in range(20)]
    trimmed = trim_to_budget(messages, budget=50)
    # Recent messages kept, oldest dropped
    assert len(trimmed) < len(messages)
    assert trimmed[-1] == messages[-1]   # most recent always kept


def test_trim_to_budget_empty_list():
    assert trim_to_budget([], budget=100) == []


def test_trim_to_budget_single_message_exceeds_budget():
    msg = {"role": "user", "content": "x" * 10_000}
    result = trim_to_budget([msg], budget=10)
    assert result == []   # even the single message exceeds budget


def test_within_budget_flag():
    small = estimate_messages_tokens([{"role": "user", "content": "short"}])
    assert small.within_budget is True

    large_messages = [{"role": "user", "content": "x" * 1000} for _ in range(50)]
    large = estimate_messages_tokens(large_messages)
    assert large.within_budget is False


# ═══════════════════════════════════════════════════════════════════════════════
# ContextSummariser
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_summariser_returns_formatted_summary():
    llm = MockLLMProvider(responses=["• User explored sales data\n• Filtered by region"])
    summariser = ContextSummariser(llm)
    messages = [
        {"role": "user",      "content": "Show June sales"},
        {"role": "assistant", "content": "Here are June sales by region"},
    ]
    result = await summariser.summarise(messages)
    assert "[Earlier conversation summary]" in result
    assert "sales" in result.lower()
    assert llm.call_count == 1


@pytest.mark.asyncio
async def test_summariser_llm_failure_returns_fallback():
    llm = MockLLMProvider(raise_on_call=LLMProviderError("rate limit"))
    summariser = ContextSummariser(llm)
    result = await summariser.summarise([{"role": "user", "content": "test"}])
    assert isinstance(result, str)
    assert len(result) > 0
    assert "summarised" in result.lower()


@pytest.mark.asyncio
async def test_summariser_empty_messages_returns_empty():
    llm = MockLLMProvider()
    summariser = ContextSummariser(llm)
    result = await summariser.summarise([])
    assert result == ""
    assert llm.call_count == 0


# ═══════════════════════════════════════════════════════════════════════════════
# ConversationMemory (upgraded)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_memory_none_conversation_returns_empty():
    memory = ConversationMemory(FakeConversationRepo())
    result = await memory.get_history_for_prompt(None)
    assert result == []


@pytest.mark.asyncio
async def test_memory_unknown_conversation_returns_empty():
    memory = ConversationMemory(FakeConversationRepo())
    result = await memory.get_history_for_prompt(999)
    assert result == []


@pytest.mark.asyncio
async def test_memory_short_history_returned_in_full():
    conv = _make_conv([
        (MessageRole.USER, "Show June sales"),
        (MessageRole.ASSISTANT, "Here are June sales"),
        (MessageRole.USER, "Compare with May"),
        (MessageRole.ASSISTANT, "Here is the May comparison"),
    ])
    repo = FakeConversationRepo({1: conv})
    memory = ConversationMemory(repo)
    result = await memory.get_history_for_prompt(1)
    assert len(result) == 4
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "Show June sales"


@pytest.mark.asyncio
async def test_memory_system_messages_excluded():
    conv = _make_conv([
        (MessageRole.SYSTEM, "System prompt here"),
        (MessageRole.USER, "Show sales"),
        (MessageRole.ASSISTANT, "Here are sales"),
    ])
    repo = FakeConversationRepo({1: conv})
    memory = ConversationMemory(repo)
    result = await memory.get_history_for_prompt(1)
    assert all(m["role"] != "system" for m in result)
    assert len(result) == 2


@pytest.mark.asyncio
async def test_memory_long_history_summarises_older_turns():
    """A conversation with many long messages should trigger summarisation."""
    # Create enough messages to exceed the token budget
    messages = []
    for i in range(30):
        messages.append((MessageRole.USER,      "x" * 200))
        messages.append((MessageRole.ASSISTANT, "y" * 200))

    conv = _make_conv(messages)
    repo = FakeConversationRepo({1: conv})

    llm = MockLLMProvider(responses=["• User explored lots of data"])
    memory = ConversationMemory(repo, llm=llm)
    result = await memory.get_history_for_prompt(1)

    # Result should be shorter than the original 60 messages
    assert len(result) < 60
    # Should contain the summary system message
    summary_messages = [m for m in result if m["role"] == "system"]
    assert len(summary_messages) == 1
    assert "summary" in summary_messages[0]["content"].lower()


@pytest.mark.asyncio
async def test_memory_long_history_no_llm_drops_older_turns():
    """Without an LLM, long history should drop older turns gracefully."""
    messages = [(MessageRole.USER, "x" * 200), (MessageRole.ASSISTANT, "y" * 200)] * 30
    conv = _make_conv(messages)
    repo = FakeConversationRepo({1: conv})

    memory = ConversationMemory(repo, llm=None)
    result = await memory.get_history_for_prompt(1)

    # Should be trimmed — no exception
    assert isinstance(result, list)
    assert len(result) < 60


# ═══════════════════════════════════════════════════════════════════════════════
# ConversationManager
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_manager_creates_new_conversation_with_generated_title():
    conv_repo = FakeConversationRepo()
    msg_repo = FakeMsgRepo()
    llm = MockLLMProvider(responses=["Monthly Sales by Region"])
    manager = ConversationManager(conv_repo, msg_repo, llm=llm)

    conv = await manager.get_or_create_conversation(
        user_id=1,
        conversation_id=None,
        first_question="Show me total sales by region last month",
    )

    assert conv.id is not None
    assert conv.title == "Monthly Sales by Region"
    assert len(conv_repo.created) == 1


@pytest.mark.asyncio
async def test_manager_returns_existing_conversation():
    existing = FakeConversation(id=42, user_id=1, title="Existing")
    conv_repo = FakeConversationRepo({42: existing})
    msg_repo = FakeMsgRepo()
    manager = ConversationManager(conv_repo, msg_repo)

    conv = await manager.get_or_create_conversation(
        user_id=1,
        conversation_id=42,
        first_question="anything",
    )

    assert conv.id == 42
    assert len(conv_repo.created) == 0   # no new conversation created


@pytest.mark.asyncio
async def test_manager_title_truncation_when_no_llm():
    conv_repo = FakeConversationRepo()
    msg_repo = FakeMsgRepo()
    manager = ConversationManager(conv_repo, msg_repo, llm=None)

    long_question = "Show me " + "very detailed " * 10 + "sales data"
    conv = await manager.get_or_create_conversation(
        user_id=1, conversation_id=None, first_question=long_question
    )

    assert len(conv.title) <= 62   # 60 chars + possible "…"
    assert conv.title.endswith("…")


@pytest.mark.asyncio
async def test_manager_persist_turn_saves_both_messages():
    conv_repo = FakeConversationRepo()
    msg_repo = FakeMsgRepo()
    manager = ConversationManager(conv_repo, msg_repo)

    response = _make_chat_response(
        sql="SELECT 1",
        description="Test query",
        explanation="Result is 1",
        execution_time_ms=42.0,
    )

    user_msg, asst_msg = await manager.persist_turn(
        conversation_id=1,
        user_question="What is 1?",
        response=response,
    )

    assert len(msg_repo.saved) == 2
    assert user_msg.role == MessageRole.USER
    assert user_msg.content == "What is 1?"
    assert asst_msg.role == MessageRole.ASSISTANT
    assert asst_msg.generated_sql == "SELECT 1"
    assert asst_msg.execution_time_ms == 42


@pytest.mark.asyncio
async def test_manager_format_assistant_content_combines_parts():
    response = _make_chat_response(
        description="Returns order count",
        explanation="There were 150 orders in June",
    )
    content = ConversationManager._format_assistant_content(response)
    assert "Returns order count" in content
    assert "150 orders" in content


@pytest.mark.asyncio
async def test_manager_format_assistant_content_fallback():
    response = _make_chat_response(description="", explanation="")
    content = ConversationManager._format_assistant_content(response)
    assert len(content) > 0   # fallback message, not empty
