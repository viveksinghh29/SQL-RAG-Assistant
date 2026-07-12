"""Unit tests for SQLGenerationService (Phase 6).

All collaborators are mocked — no database, no network, no FAISS index.
Tests cover:

  - Happy path: question → retrieved context → LLM → SQL extracted
  - LLM provider failure → FAILED status, no exception raised to caller
  - LLM response parse failure → FAILED status with user-friendly message
  - Multi-turn history: memory returns prior turns, injected into prompt
  - New conversation (conversation_id=None): empty history
  - RBAC role flows through to retriever and prompt
"""

import pytest

from app.auth.roles import Role
from app.conversation_memory.memory import ConversationMemory
from app.core.exceptions import LLMProviderError
from app.llm.base import LLMMessage, LLMResponse
from app.llm.providers.mock_provider import MockLLMProvider
from app.rag.base import DocumentType, RetrievedDocument
from app.rag.retriever.retriever import RetrievalResult
from app.schemas.pipeline import GenerationStatus
from app.sql_generator.service import SQLGenerationService


# ── Test doubles ─────────────────────────────────────────────────────────────

class MockRetriever:
    """Fake retriever that returns deterministic context, tracks calls."""

    def __init__(self, documents: list[RetrievedDocument] | None = None) -> None:
        self._documents = documents or _default_docs()
        self.last_role: Role | None = None
        self.last_query: str | None = None

    async def retrieve_for_sql_generation(
        self, query: str, *, role: Role, top_k: int = 5
    ) -> RetrievalResult:
        self.last_role = role
        self.last_query = query
        return RetrievalResult(documents=self._documents)


class MockMemory:
    """Fake ConversationMemory that returns a pre-set history list."""

    def __init__(self, history: list[dict] | None = None) -> None:
        self._history = history or []
        self.last_conversation_id: int | None = None

    async def get_history_for_prompt(
        self, conversation_id: int | None
    ) -> list[dict]:
        self.last_conversation_id = conversation_id
        return self._history


def _default_docs() -> list[RetrievedDocument]:
    return [
        RetrievedDocument(
            content="Table orders has columns: id, customer_id, order_date, total_amount, status",
            document_type=DocumentType.SCHEMA,
            source="schema.sql",
            score=0.95,
            metadata={"table": "orders"},
        ),
        RetrievedDocument(
            content="Table order_items has columns: id, order_id, product_id, quantity, unit_price",
            document_type=DocumentType.SCHEMA,
            source="schema.sql",
            score=0.90,
            metadata={"table": "order_items"},
        ),
    ]


def _make_service(
    *,
    llm_responses: list[str] | None = None,
    llm_raise: Exception | None = None,
    history: list[dict] | None = None,
    documents: list[RetrievedDocument] | None = None,
) -> tuple[SQLGenerationService, MockRetriever, MockLLMProvider, MockMemory]:
    retriever = MockRetriever(documents=documents)
    llm = MockLLMProvider(
        responses=llm_responses
        or [
            "Returns all completed orders.\n"
            "```sql\n"
            "SELECT * FROM orders WHERE status = 'completed' LIMIT 100;\n"
            "```"
        ],
        raise_on_call=llm_raise,
    )
    memory = MockMemory(history=history)
    service = SQLGenerationService(retriever=retriever, llm=llm, memory=memory)
    return service, retriever, llm, memory


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_successful_generation_returns_sql():
    """Happy path: service returns a SUCCESS result containing the generated SQL."""
    service, retriever, llm, memory = _make_service()

    result = await service.generate(
        "Show me all completed orders",
        role=Role.MANAGER,
        conversation_id=42,
    )

    assert result.status == GenerationStatus.SUCCESS
    assert "SELECT" in result.sql.upper()
    assert "orders" in result.sql.lower()
    assert result.description != ""
    assert result.prompt_tokens > 0
    assert result.latency_ms > 0


@pytest.mark.asyncio
async def test_role_is_passed_to_retriever():
    """The authenticated role must reach the retriever for RBAC filtering."""
    service, retriever, _, _ = _make_service()

    await service.generate("Show revenue", role=Role.EMPLOYEE)

    assert retriever.last_role == Role.EMPLOYEE


@pytest.mark.asyncio
async def test_question_is_passed_to_retriever():
    """The exact question must reach the retriever unchanged."""
    service, retriever, _, _ = _make_service()

    question = "What is the total revenue by region for Q2?"
    await service.generate(question, role=Role.MANAGER)

    assert retriever.last_query == question


@pytest.mark.asyncio
async def test_conversation_id_passed_to_memory():
    """The conversation_id must reach the memory layer for history assembly."""
    service, _, _, memory = _make_service(history=[])

    await service.generate("Show sales", role=Role.ADMIN, conversation_id=99)

    assert memory.last_conversation_id == 99


@pytest.mark.asyncio
async def test_none_conversation_id_returns_empty_history():
    """New conversations (conversation_id=None) must produce empty history."""
    service, _, llm, memory = _make_service()

    await service.generate("Show sales", role=Role.ADMIN, conversation_id=None)

    assert memory.last_conversation_id is None
    # System prompt + user turn only (no history injected)
    assert llm.last_messages is not None
    assert len(llm.last_messages) == 2


@pytest.mark.asyncio
async def test_conversation_history_injected_into_messages():
    """Prior conversation turns must appear in the LLM message list."""
    history = [
        {"role": "user", "content": "Show June sales"},
        {"role": "assistant", "content": "Here are June sales..."},
    ]
    service, _, llm, _ = _make_service(history=history)

    await service.generate("Compare with May", role=Role.MANAGER, conversation_id=1)

    # system + 2 history + current user turn = 4
    assert llm.last_messages is not None
    assert len(llm.last_messages) == 4
    roles = [m.role for m in llm.last_messages]
    assert roles == ["system", "user", "assistant", "user"]


@pytest.mark.asyncio
async def test_llm_failure_returns_failed_status_not_raise():
    """An LLM provider error must produce FAILED status — must not propagate."""
    service, _, _, _ = _make_service(llm_raise=LLMProviderError("quota exceeded"))

    result = await service.generate("Show sales", role=Role.ADMIN)

    assert result.status == GenerationStatus.FAILED
    assert "quota exceeded" in result.error_message.lower()
    assert result.sql == ""


@pytest.mark.asyncio
async def test_unparseable_llm_response_returns_failed_status():
    """If the LLM returns text with no SQL, status must be FAILED with user message."""
    service, _, _, _ = _make_service(
        llm_responses=["I cannot answer that question."]
    )

    result = await service.generate("Show sales", role=Role.ADMIN)

    assert result.status == GenerationStatus.FAILED
    assert result.sql == ""
    # Message must be user-friendly, not a raw exception string
    assert "rephrase" in result.error_message.lower()


@pytest.mark.asyncio
async def test_retrieved_context_included_in_result():
    """The rendered RAG context must be returned so the API/UI can surface it."""
    service, _, _, _ = _make_service()

    result = await service.generate("Show orders", role=Role.MANAGER)

    assert result.status == GenerationStatus.SUCCESS
    assert len(result.retrieved_context) > 0
    assert "orders" in result.retrieved_context.lower()


@pytest.mark.asyncio
async def test_token_counts_populated_on_success():
    """Token counts from the LLM response must be carried through to the result."""
    service, _, _, _ = _make_service()

    result = await service.generate("Show sales", role=Role.ADMIN)

    assert result.prompt_tokens > 0
    assert result.completion_tokens > 0


@pytest.mark.asyncio
async def test_admin_role_passes_through_correctly():
    """Admin role must be forwarded to retriever without modification."""
    service, retriever, _, _ = _make_service()

    await service.generate("Show payroll data", role=Role.ADMIN)

    assert retriever.last_role == Role.ADMIN


@pytest.mark.asyncio
async def test_generation_result_contains_description():
    """The one-sentence query description must be extracted from the LLM response."""
    llm_response = (
        "This query returns total revenue per region.\n"
        "```sql\n"
        "SELECT region, SUM(total_amount) FROM orders GROUP BY region LIMIT 10;\n"
        "```"
    )
    service, _, _, _ = _make_service(llm_responses=[llm_response])

    result = await service.generate("Revenue by region", role=Role.MANAGER)

    assert result.status == GenerationStatus.SUCCESS
    assert "revenue" in result.description.lower()
