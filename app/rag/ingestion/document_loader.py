"""Document loaders for business rules, FAQs, and curated SQL examples used in RAG knowledge retrieval."""

import json
from pathlib import Path

from app.rag.base import DocumentType
from app.rag.ingestion.chunker import chunk_text


def load_markdown_document(
    path: Path, document_type: DocumentType
) -> list[tuple[str, DocumentType, str, dict]]:
    """Chunk a markdown file into RAG documents."""
    text = path.read_text(encoding="utf-8")
    chunks = chunk_text(text)
    return [(chunk, document_type, path.name, {"chunk_index": i}) for i, chunk in enumerate(chunks)]


def load_sql_examples(path: Path) -> list[tuple[str, DocumentType, str, dict]]:
    """Load curated NL-question -> SQL pairs as one document per example.

    Kept as one chunk per example (not further chunked) since splitting a
    question/SQL pair would destroy the thing that makes it useful as a
    few-shot reference.
    """
    examples = json.loads(path.read_text(encoding="utf-8"))
    documents = []
    for example in examples:
        content = f"Example question: {example['question']}\nExample SQL: {example['sql']}"
        documents.append(
            (content, DocumentType.SQL_EXAMPLE, path.name, {"question": example["question"]})
        )
    return documents
