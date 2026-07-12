"""Abstract vector store contract for the RAG layer.

Concrete implementation (FAISS) lands in Phase 4. Defining the interface
now lets the SQL generator and retriever services (Phases 4-6) be coded
and unit-tested against a mock implementation before FAISS integration
exists.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum


class DocumentType(StrEnum):
    """Source category of a RAG document, used for filtering retrieval."""

    SCHEMA = "schema"
    TABLE_DESCRIPTION = "table_description"
    COLUMN_DESCRIPTION = "column_description"
    RELATIONSHIP = "relationship"
    BUSINESS_RULE = "business_rule"
    KPI_DEFINITION = "kpi_definition"
    SQL_EXAMPLE = "sql_example"
    FAQ = "faq"
    DOCUMENTATION = "documentation"


@dataclass(frozen=True, slots=True)
class RetrievedDocument:
    content: str
    document_type: DocumentType
    source: str
    score: float
    metadata: dict


class VectorStore(ABC):
    """Contract for storing and semantically retrieving RAG documents."""

    @abstractmethod
    async def add_documents(
        self,
        texts: list[str],
        document_types: list[DocumentType],
        sources: list[str],
        metadatas: list[dict],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def similarity_search(
        self,
        query: str,
        *,
        top_k: int = 5,
        document_types: list[DocumentType] | None = None,
    ) -> list[RetrievedDocument]:
        """Return the top_k most relevant documents, optionally filtered by type."""
        raise NotImplementedError

    @abstractmethod
    async def persist(self) -> None:
        """Flush the index to disk."""
        raise NotImplementedError

    @abstractmethod
    async def load(self) -> None:
        """Load a previously persisted index from disk."""
        raise NotImplementedError
