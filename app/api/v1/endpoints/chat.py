"""Chat endpoint — the core pipeline orchestration.

POST /chat

Orchestrates the full request lifecycle:
  1.  Resolve or create a Conversation           (ConversationManager)
  2.  Generate SQL from the NL question          (SQLGenerationService)
  3.  Validate & secure the generated SQL        (SQLValidationService)
  4.  Execute the validated SQL                  (QueryExecutor)
  5.  Generate explanation + chart               (AnalyticsService)
  6.  Persist the user + assistant messages      (ConversationManager)
  7.  Return the structured ChatResponse

Every stage is independently error-handled — a failure at stage N returns
a partial ChatResponse (status=FAILED/BLOCKED) rather than a 500, so the
frontend always has a structured, displayable response.
"""

import time

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import get_current_user
from app.auth.roles import Role
from app.conversation_memory.manager import ConversationManager
from app.conversation_memory.memory import ConversationMemory
from app.core.logging import get_logger
from app.database.models.user import User
from app.database.repositories.chat_repository import (
    ChatMessageRepository,
    ConversationRepository,
)
from app.database.session import get_db_session
from app.llm.factory import get_llm_provider
from app.query_executor.service import QueryExecutor
from app.query_executor.result import ExecutionStatus
from app.rag.retriever.retriever import Retriever
from app.rag.embeddings.embedder import Embedder
from app.rag.vector_store.faiss_store import FAISSVectorStore
from app.schemas.pipeline import (
    ChatRequest,
    ChatResponse,
    GenerationStatus,
)
from app.sql_generator.service import SQLGenerationService
from app.sql_validator.schema_registry import get_known_tables
from app.sql_validator.service import SQLValidationService
from app.visualization.analytics_service import AnalyticsService
from app.visualization.cached_analytics import CachedAnalyticsService
from app.visualization.chart_renderer import ChartRenderer
from app.query_executor.cached_executor import CachedQueryExecutor
from app.utils.cache_client import get_cache_client

router = APIRouter(prefix="/chat", tags=["chat"])
log = get_logger("api.chat")


def _build_services(session: AsyncSession):
    """Construct all request-scoped services.

    Heavy shared resources (FAISS index, LLM client) are process-level
    singletons obtained via their own cached factories. Request-scoped
    objects (DB session, per-request service instances) are created here.
    """
    llm = get_llm_provider()

    # RAG retriever — uses process-level cached FAISS index
    from app.sql_generator.factory import _get_shared_retriever
    retriever = _get_shared_retriever()

    conv_repo = ConversationRepository(session)
    msg_repo = ChatMessageRepository(session)

    memory = ConversationMemory(conv_repo, llm=llm)
    manager = ConversationManager(conv_repo, msg_repo, llm=llm)
    generator = SQLGenerationService(retriever=retriever, llm=llm, memory=memory)
    validator = SQLValidationService(known_tables=get_known_tables())
    cache = get_cache_client()
    executor = QueryExecutor()      # wrapped with role in endpoint body
    analytics = CachedAnalyticsService(llm=llm, cache=cache, renderer=ChartRenderer())

    return manager, generator, validator, executor, analytics


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> ChatResponse:
    """Process a natural-language question through the full AI pipeline."""
    t0 = time.monotonic()
    role = Role(current_user.role.value)

    manager, generator, validator, _base_executor, analytics = _build_services(session)
    from app.utils.cache_client import get_cache_client as _gcache
    executor = CachedQueryExecutor(
        executor=_base_executor, cache=_gcache(), role=role
    )

    # ── 1. Resolve / create conversation ─────────────────────────────────
    conversation = await manager.get_or_create_conversation(
        user_id=current_user.id,
        conversation_id=request.conversation_id,
        first_question=request.question,
    )
    await session.commit()

    # ── 2. Generate SQL ───────────────────────────────────────────────────
    gen_result = await generator.generate(
        request.question,
        role=role,
        conversation_id=conversation.id,
    )

    if gen_result.status == GenerationStatus.FAILED:
        response = ChatResponse(
            conversation_id=conversation.id,
            message_id=0,
            question=request.question,
            status=GenerationStatus.FAILED,
            error_message=gen_result.error_message,
        )
        _, asst_msg = await manager.persist_turn(
            conversation.id, request.question, response
        )
        await session.commit()
        response.message_id = asst_msg.id
        return response

    # ── 3. Validate SQL ───────────────────────────────────────────────────
    from app.sql_validator.result import ValidationStatus
    val_result = await validator.validate(gen_result.sql, role=role)

    if val_result.status != ValidationStatus.VALID:
        log.bind(
            reason=val_result.reason,
            role=role.value,
            user_id=current_user.id,
        ).warning("SQL rejected by validator")
        response = ChatResponse(
            conversation_id=conversation.id,
            message_id=0,
            question=request.question,
            sql=gen_result.sql,
            status=GenerationStatus.BLOCKED,
            error_message=val_result.message,
        )
        _, asst_msg = await manager.persist_turn(
            conversation.id, request.question, response
        )
        await session.commit()
        response.message_id = asst_msg.id
        return response

    validated_sql = val_result.sql

    # ── 4. Execute SQL ────────────────────────────────────────────────────
    exec_result = await executor.execute(validated_sql)

    if not exec_result.is_successful:
        response = ChatResponse(
            conversation_id=conversation.id,
            message_id=0,
            question=request.question,
            sql=validated_sql,
            description=gen_result.description,
            status=GenerationStatus.FAILED,
            error_message=exec_result.error_message,
            execution_time_ms=exec_result.execution_time_ms,
        )
        _, asst_msg = await manager.persist_turn(
            conversation.id, request.question, response
        )
        await session.commit()
        response.message_id = asst_msg.id
        return response

    # ── 5. Generate explanation + chart ───────────────────────────────────
    analytics_result = await analytics.analyse(
        question=request.question,
        execution_result=exec_result,
    )

    total_latency_ms = (time.monotonic() - t0) * 1000

    log.bind(
        user_id=current_user.id,
        role=role.value,
        conversation_id=conversation.id,
        row_count=exec_result.row_count,
        chart_type=analytics_result.chart_type,
        total_latency_ms=round(total_latency_ms, 1),
    ).info("Chat request completed successfully")

    # ── 6. Build and persist response ─────────────────────────────────────
    response = ChatResponse(
        conversation_id=conversation.id,
        message_id=0,
        question=request.question,
        sql=validated_sql,
        description=gen_result.description,
        explanation=analytics_result.explanation,
        rows=exec_result.rows,
        row_count=exec_result.row_count,
        columns=exec_result.columns,
        chart_type=analytics_result.chart_type,
        chart_json=analytics_result.chart_json,
        execution_time_ms=exec_result.execution_time_ms,
        total_latency_ms=total_latency_ms,
        status=GenerationStatus.SUCCESS,
    )

    _, asst_msg = await manager.persist_turn(
        conversation.id, request.question, response
    )
    await session.commit()
    response.message_id = asst_msg.id
    return response
