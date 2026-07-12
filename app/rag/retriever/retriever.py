"""RAG retriever that queries the vector store and assembles relevant context for SQL generation."""

from dataclasses import dataclass

from app.auth.roles import Role, get_restricted_tables
from app.core.logging import get_logger
from app.rag.base import DocumentType, RetrievedDocument, VectorStore

log = get_logger("retriever")

# Document types relevant to grounding a SQL-generation prompt. FAQs are
# deliberately excluded here — they're useful for a general chat-support
# experience but would dilute SQL-generation context with question/answer
# pairs that aren't schema-relevant.
_SQL_GENERATION_TYPES = [
    DocumentType.SCHEMA,
    DocumentType.TABLE_DESCRIPTION,
    DocumentType.COLUMN_DESCRIPTION,
    DocumentType.RELATIONSHIP,
    DocumentType.BUSINESS_RULE,
    DocumentType.KPI_DEFINITION,
    DocumentType.SQL_EXAMPLE,
]


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    documents: list[RetrievedDocument]

    def to_prompt_context(self) -> str:
        """Render retrieved documents as a single context block for
        injection into the SQL generation prompt, grouped by type so the
        LLM sees schema before business rules before examples."""
        if not self.documents:
            return "No relevant context retrieved."

        grouped: dict[DocumentType, list[RetrievedDocument]] = {}
        for doc in self.documents:
            grouped.setdefault(doc.document_type, []).append(doc)

        sections = []
        for doc_type, docs in grouped.items():
            lines = "\n".join(f"- {d.content}" for d in docs)
            sections.append(f"## {doc_type.value.replace('_', ' ').title()}\n{lines}")
        return "\n\n".join(sections)


class Retriever:
    def __init__(self, vector_store: VectorStore, *, top_k: int = 5) -> None:
        self._vector_store = vector_store
        self._top_k = top_k

    async def retrieve_for_sql_generation(
        self, query: str, *, role: Role, top_k: int | None = None
    ) -> RetrievalResult:
        """Retrieve schema + business context for a natural-language query,
        filtering out anything the user's role isn't permitted to see.

        This is the FIRST line of defense against restricted tables ever
        reaching the LLM's context — `assert_tables_allowed` in the SQL
        validator (Phase 7) is the second, independent line of defense.
        """
        k = top_k or self._top_k
        documents = await self._vector_store.similarity_search(
            query, top_k=k * 2, document_types=_SQL_GENERATION_TYPES
        )

        restricted = get_restricted_tables(role)
        filtered = [doc for doc in documents if doc.metadata.get("table") not in restricted]

        if len(filtered) < len(documents):
            log.bind(role=role.value).info(
                f"Filtered {len(documents) - len(filtered)} restricted-table document(s) "
                f"from retrieval context"
            )

        return RetrievalResult(documents=filtered[:k])

    async def retrieve_faqs(self, query: str, *, top_k: int = 3) -> RetrievalResult:
        documents = await self._vector_store.similarity_search(
            query, top_k=top_k, document_types=[DocumentType.FAQ]
        )
        return RetrievalResult(documents=documents)
