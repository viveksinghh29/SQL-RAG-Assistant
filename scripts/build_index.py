"""Build the FAISS RAG index from the target DB's schema and the business
documents in data/documents/.

Usage:
    python -m scripts.build_index

Run this once after schema changes or document updates, then the running
application loads the persisted index at startup (Phase 11's lifespan
hook) rather than rebuilding it on every boot.
"""

import asyncio
from pathlib import Path

from app.core.logging import configure_logging, get_logger
from app.rag.base import DocumentType
from app.rag.ingestion.document_loader import load_markdown_document, load_sql_examples
from app.rag.ingestion.schema_loader import build_schema_documents
from app.rag.vector_store.faiss_store import FAISSVectorStore

log = get_logger("build_index")

DATA_DIR = Path("data")


async def main() -> None:
    configure_logging()

    schema_sql = (DATA_DIR / "seed" / "schema.sql").read_text(encoding="utf-8")
    schema_docs = build_schema_documents(schema_sql)
    log.info(f"Parsed {len(schema_docs)} schema/KPI/relationship documents from schema.sql")

    business_rules_docs = load_markdown_document(
        DATA_DIR / "documents" / "business_rules.md", DocumentType.BUSINESS_RULE
    )
    faq_docs = load_markdown_document(DATA_DIR / "documents" / "faqs.md", DocumentType.FAQ)
    sql_example_docs = load_sql_examples(DATA_DIR / "documents" / "sql_examples.json")

    all_docs = schema_docs + business_rules_docs + faq_docs + sql_example_docs
    log.info(f"Total documents to index: {len(all_docs)}")

    store = FAISSVectorStore()
    texts = [d[0] for d in all_docs]
    types = [d[1] for d in all_docs]
    sources = [d[2] for d in all_docs]
    metadatas = [d[3] for d in all_docs]

    await store.add_documents(texts, types, sources, metadatas)
    await store.persist()

    log.info("Index build complete.")


if __name__ == "__main__":
    asyncio.run(main())
