"""FAISS-based vector store implementation for efficient semantic search with persistent metadata storage."""

import json
from pathlib import Path

import faiss
import numpy as np

from app.config.settings import get_settings
from app.core.logging import get_logger
from app.rag.base import DocumentType, RetrievedDocument, VectorStore
from app.rag.embeddings.embedder import Embedder

log = get_logger("faiss_store")


class FAISSVectorStore(VectorStore):
    def __init__(self, embedder: Embedder | None = None) -> None:
        self._embedder = embedder or Embedder()
        self._index: faiss.Index | None = None
        self._metadata: list[dict] = []  # parallel to FAISS vector ids

    def _ensure_index(self) -> faiss.Index:
        if self._index is None:
            self._index = faiss.IndexFlatIP(self._embedder.dimension)
        return self._index

    async def add_documents(
        self,
        texts: list[str],
        document_types: list[DocumentType],
        sources: list[str],
        metadatas: list[dict],
    ) -> None:
        if not (len(texts) == len(document_types) == len(sources) == len(metadatas)):
            raise ValueError("texts, document_types, sources, and metadatas must be equal length")
        if not texts:
            return

        vectors = self._embedder.embed(texts)
        index = self._ensure_index()
        index.add(vectors)

        for text, doc_type, source, meta in zip(texts, document_types, sources, metadatas, strict=True):
            self._metadata.append(
                {"content": text, "document_type": doc_type.value, "source": source, "metadata": meta}
            )

        log.info(f"Added {len(texts)} documents to vector store (total: {len(self._metadata)})")

    async def similarity_search(
        self,
        query: str,
        *,
        top_k: int = 5,
        document_types: list[DocumentType] | None = None,
    ) -> list[RetrievedDocument]:
        if self._index is None or self._index.ntotal == 0:
            return []

        query_vector = self._embedder.embed_one(query).reshape(1, -1)

        # Over-fetch when filtering by type, since the top-k nearest
        # neighbors overall may not all match the requested type filter.
        fetch_k = top_k * 5 if document_types else top_k
        fetch_k = min(fetch_k, self._index.ntotal)

        scores, indices = self._index.search(query_vector, fetch_k)

        allowed_types = {dt.value for dt in document_types} if document_types else None
        results: list[RetrievedDocument] = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx == -1:
                continue
            doc = self._metadata[idx]
            if allowed_types is not None and doc["document_type"] not in allowed_types:
                continue
            results.append(
                RetrievedDocument(
                    content=doc["content"],
                    document_type=DocumentType(doc["document_type"]),
                    source=doc["source"],
                    score=float(score),
                    metadata=doc["metadata"],
                )
            )
            if len(results) >= top_k:
                break

        return results

    async def persist(self) -> None:
        settings = get_settings()
        store_dir = Path(settings.vector_store_path)
        store_dir.mkdir(parents=True, exist_ok=True)

        if self._index is not None:
            faiss.write_index(self._index, str(store_dir / "index.faiss"))
        (store_dir / "metadata.json").write_text(json.dumps(self._metadata, default=str))
        log.info(f"Persisted vector store ({len(self._metadata)} docs) to {store_dir}")

    async def load(self) -> None:
        settings = get_settings()
        store_dir = Path(settings.vector_store_path)
        index_path = store_dir / "index.faiss"
        metadata_path = store_dir / "metadata.json"

        if not index_path.exists() or not metadata_path.exists():
            log.warning(f"No persisted vector store found at {store_dir}; starting empty")
            return

        self._index = faiss.read_index(str(index_path))
        self._metadata = json.loads(metadata_path.read_text())
        log.info(f"Loaded vector store ({len(self._metadata)} docs) from {store_dir}")

    @property
    def size(self) -> int:
        return len(self._metadata)
