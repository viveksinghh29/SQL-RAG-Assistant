"""Service for orchestrating RAG retrieval, conversation memory, prompt construction, LLM-based SQL generation, and response parsing."""

import time

from app.auth.roles import Role
from app.config.settings import get_settings
from app.conversation_memory.memory import ConversationMemory
from app.core.exceptions import LLMProviderError, LLMResponseParsingError, SQLGenerationError
from app.core.logging import get_logger
from app.llm.base import LLMProvider
from app.llm.parser import extract_description, extract_sql
from app.prompts.sql_generation import build_sql_generation_messages
from app.rag.retriever.retriever import Retriever
from app.schemas.pipeline import GenerationStatus, SQLGenerationResult

log = get_logger("sql_generator")


class SQLGenerationService:
    """Converts a natural-language question into a SQL query.

    All dependencies are injected — no module-level singletons — so the
    service is straightforwardly unit-testable with mocks. FastAPI will
    inject real implementations via `Depends()` in Phase 11.

    Args:
        retriever:   RBAC-aware RAG retriever (Phase 4).
        llm:         LLM provider — Groq or mock for tests (Phase 5).
        memory:      Conversation history adapter (Phase 6 stub).
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMProvider,
        memory: ConversationMemory,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._memory = memory
        self._settings = get_settings()

    async def generate(
        self,
        question: str,
        *,
        role: Role,
        conversation_id: int | None = None,
    ) -> SQLGenerationResult:
        """Generate a SQL query for `question`.

        Returns a `SQLGenerationResult` in all cases — never raises to
        the caller. Errors are captured in `status=FAILED` and
        `error_message` so the orchestration layer (Phase 11) can decide
        whether to surface the error or retry.

        Args:
            question:        The user's natural-language question.
            role:            Authenticated user's role (for RBAC filtering).
            conversation_id: Active conversation for multi-turn context,
                             or None to start a fresh conversation.
        """
        t0 = time.monotonic()

        # ── Step 1: Retrieve RBAC-filtered context from the RAG layer ──────
        try:
            retrieval_result = await self._retriever.retrieve_for_sql_generation(
                question, role=role, top_k=self._settings.vector_store_top_k
            )
        except Exception as exc:
            log.exception("RAG retrieval failed during SQL generation")
            return SQLGenerationResult(
                status=GenerationStatus.FAILED,
                error_message=f"Context retrieval failed: {exc}",
                latency_ms=(time.monotonic() - t0) * 1000,
            )

        retrieved_context = retrieval_result.to_prompt_context()

        # ── Step 2: Fetch conversation history for multi-turn context ───────
        history = await self._memory.get_history_for_prompt(conversation_id)

        # ── Step 3: Build the prompt message list ────────────────────────────
        messages = build_sql_generation_messages(
            user_question=question,
            retrieved_context=retrieved_context,
            role=role,
            conversation_history=history,
            default_limit=self._settings.sql_default_limit,
        )

        # ── Step 4: Call the LLM ─────────────────────────────────────────────
        try:
            llm_response = await self._llm.generate(messages)
        except LLMProviderError as exc:
            log.bind(role=role.value).error(f"LLM call failed: {exc.message}")
            return SQLGenerationResult(
                status=GenerationStatus.FAILED,
                retrieved_context=retrieved_context,
                error_message=f"LLM provider error: {exc.message}",
                latency_ms=(time.monotonic() - t0) * 1000,
            )

        # ── Step 5: Extract SQL and description from the response ────────────
        try:
            sql = extract_sql(llm_response.content)
            description = extract_description(llm_response.content)
        except LLMResponseParsingError as exc:
            log.bind(role=role.value).warning(
                f"SQL extraction failed: {exc.message}",
                raw_preview=llm_response.content[:200],
            )
            return SQLGenerationResult(
                status=GenerationStatus.FAILED,
                retrieved_context=retrieved_context,
                prompt_tokens=llm_response.prompt_tokens,
                completion_tokens=llm_response.completion_tokens,
                error_message="The AI did not return a valid SQL query. Please rephrase your question.",
                latency_ms=(time.monotonic() - t0) * 1000,
            )

        latency_ms = (time.monotonic() - t0) * 1000

        log.bind(
            role=role.value,
            conversation_id=conversation_id,
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
            latency_ms=round(latency_ms, 1),
        ).info("SQL generated successfully")

        return SQLGenerationResult(
            status=GenerationStatus.SUCCESS,
            sql=sql,
            description=description,
            retrieved_context=retrieved_context,
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
            latency_ms=latency_ms,
        )
