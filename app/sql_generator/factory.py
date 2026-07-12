"""Dependency injection factory for constructing the SQL generation service with retrieval, LLM, and conversation memory components."""

from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncSession

from app.conversation_memory.memory import ConversationMemory
from app.database.repositories.chat_repository import ConversationRepository
from app.llm.factory import get_llm_provider
from app.rag.embeddings.embedder import Embedder
from app.rag.retriever.retriever import Retriever
from app.rag.vector_store.faiss_store import FAISSVectorStore
from app.sql_generator.service import SQLGenerationService


@lru_cache
def _get_shared_retriever() -> Retriever:
    """Build and cache the Retriever (loads the FAISS index once per process).

    The FAISS index and Embedder are heavy — they must not be re-created
    per request. `lru_cache` guarantees a single shared instance.

    In tests, do NOT call this; construct `Retriever` directly with a
    `MockVectorStore` and `FakeEmbedder`.
    """
    embedder = Embedder()
    store = FAISSVectorStore(embedder=embedder)

    # Load the persisted index — built by `scripts/build_index.py`.
    # If the index doesn't exist yet (first run), the retriever will
    # return empty context. The caller (SQLGenerationService) handles
    # this gracefully: the LLM gets no context and responds accordingly.
    import asyncio
    try:
        asyncio.get_event_loop().run_until_complete(store.load())
    except Exception:
        pass  # index not yet built; first-run or test environment

    return Retriever(store)


def get_sql_generation_service(session: AsyncSession) -> SQLGenerationService:
    """Construct a SQLGenerationService with real collaborators.

    `session` is a per-request async DB session (injected by FastAPI via
    `Depends(get_db_session)` in Phase 11). Everything else is process-
    scoped and cached.
    """
    retriever = _get_shared_retriever()
    llm = get_llm_provider()
    conv_repo = ConversationRepository(session)
    memory = ConversationMemory(conv_repo)
    return SQLGenerationService(retriever=retriever, llm=llm, memory=memory)
