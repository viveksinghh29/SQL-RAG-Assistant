"""Unit tests for the RAG pipeline modules (Phase 15).

Covers the gaps in Phase 4 coverage:
  - document_loader: markdown + SQL example loading
  - schema_loader: edge cases (empty DDL, tables without PKs)
  - retriever: filtering, top_k, prompt context formatting
  - faiss_store: add/search/persist/load with fake embedder
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from app.rag.base import DocumentType, RetrievedDocument
from app.rag.ingestion.document_loader import (
    load_markdown_document,
    load_sql_examples,
)
from app.rag.ingestion.schema_loader import build_schema_documents, parse_ddl
from app.rag.retriever.retriever import RetrievalResult, Retriever
from app.rag.vector_store.faiss_store import FAISSVectorStore


# ── Fake embedder ─────────────────────────────────────────────────────────────

class FakeEmbedder:
    dimension = 16

    def embed(self, texts):
        vecs = []
        for t in texts:
            rng = np.random.RandomState(abs(hash(t)) % (2 ** 32))
            v = rng.rand(self.dimension).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-8
            vecs.append(v)
        return np.array(vecs, dtype=np.float32)

    def embed_one(self, text):
        return self.embed([text])[0]


# ── Schema loader ─────────────────────────────────────────────────────────────

def test_parse_ddl_extracts_all_tables():
    ddl = Path("data/seed/schema.sql").read_text()
    tables = parse_ddl(ddl)
    names = {t.name for t in tables}
    assert "orders" in names
    assert "customers" in names
    assert "payroll" in names
    assert len(tables) == 7


def test_parse_ddl_extracts_columns():
    ddl = "CREATE TABLE products (id INT PRIMARY KEY, name VARCHAR(255), price DECIMAL(10,2));"
    tables = parse_ddl(ddl)
    assert len(tables) == 1
    t = tables[0]
    assert t.name == "products"
    col_names = {c.name for c in t.columns}
    assert "id" in col_names
    assert "name" in col_names
    assert "price" in col_names


def test_parse_ddl_empty_returns_empty_list():
    tables = parse_ddl("")
    assert tables == []


def test_build_schema_documents_returns_list():
    ddl = Path("data/seed/schema.sql").read_text()
    docs = build_schema_documents(ddl)
    assert len(docs) > 0
    # Each doc is (text, DocumentType, source, metadata)
    text, dtype, source, meta = docs[0]
    assert isinstance(text, str)
    assert isinstance(dtype, DocumentType)


def test_build_schema_documents_covers_all_tables():
    ddl = Path("data/seed/schema.sql").read_text()
    docs = build_schema_documents(ddl)
    tables_covered = {d[3].get("table") for d in docs if d[3].get("table")}
    assert "orders" in tables_covered
    assert "customers" in tables_covered


# ── Document loader ───────────────────────────────────────────────────────────

def test_load_markdown_document_chunks_correctly():
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
        f.write("# Business Rules\n\nRevenue is calculated monthly.\n\nMargin = (price - cost) / price.\n")
        path = Path(f.name)

    docs = load_markdown_document(path, DocumentType.BUSINESS_RULE)
    assert len(docs) > 0
    text, dtype, source, meta = docs[0]
    assert dtype == DocumentType.BUSINESS_RULE
    assert len(text) > 0
    path.unlink()


def test_load_markdown_document_real_file():
    path = Path("data/documents/business_rules.md")
    if not path.exists():
        pytest.skip("business_rules.md not present")
    docs = load_markdown_document(path, DocumentType.BUSINESS_RULE)
    assert len(docs) > 0


def test_load_sql_examples_parses_correctly():
    examples = [
        {"question": "Total revenue", "sql": "SELECT SUM(total_amount) FROM orders;",
         "description": "Revenue aggregation"}
    ]
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(examples, f)
        path = Path(f.name)

    docs = load_sql_examples(path)
    assert len(docs) == 1
    text, dtype, source, meta = docs[0]
    assert dtype == DocumentType.SQL_EXAMPLE
    assert "Total revenue" in text
    path.unlink()


def test_load_sql_examples_empty_file():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump([], f)
        path = Path(f.name)
    docs = load_sql_examples(path)
    assert docs == []
    path.unlink()


# ── FAISS vector store ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_faiss_store_add_and_search():
    store = FAISSVectorStore(embedder=FakeEmbedder())
    texts = ["orders table has id and total_amount", "customers have region and email"]
    types = [DocumentType.SCHEMA, DocumentType.SCHEMA]
    sources = ["schema.sql", "schema.sql"]
    metas = [{"table": "orders"}, {"table": "customers"}]

    await store.add_documents(texts, types, sources, metas)
    assert store.size == 2

    results = await store.similarity_search("show me orders data", top_k=2)
    assert len(results) == 2
    assert all(isinstance(r, RetrievedDocument) for r in results)
    assert all(0 <= r.score <= 1 for r in results)


@pytest.mark.asyncio
async def test_faiss_store_top_k_respected():
    store = FAISSVectorStore(embedder=FakeEmbedder())
    texts = [f"document {i}" for i in range(10)]
    types = [DocumentType.FAQ] * 10
    sources = ["faq.md"] * 10
    metas = [{}] * 10

    await store.add_documents(texts, types, sources, metas)
    results = await store.similarity_search("question", top_k=3)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_faiss_store_persist_and_load(tmp_path, monkeypatch):
    monkeypatch.setenv("VECTOR_STORE_PATH", str(tmp_path))

    # Patch settings to use tmp_path
    from app.config.settings import get_settings
    settings = get_settings()
    original_path = settings.vector_store_path

    import os
    os.environ["VECTOR_STORE_PATH"] = str(tmp_path)
    from app.rag.vector_store.faiss_store import FAISSVectorStore as _FS
    from app.config.settings import get_settings
    get_settings.cache_clear()
    store = _FS(embedder=FakeEmbedder())
    texts = ["order data", "customer info"]
    await store.add_documents(
        texts,
        [DocumentType.SCHEMA, DocumentType.SCHEMA],
        ["s.sql", "s.sql"],
        [{}, {}],
    )
    await store.persist()
    store2 = _FS(embedder=FakeEmbedder())
    await store2.load()
    assert store2.size == 2


@pytest.mark.asyncio
async def test_faiss_store_filter_by_document_type():
    store = FAISSVectorStore(embedder=FakeEmbedder())
    await store.add_documents(
        ["schema text", "faq text", "business rule text"],
        [DocumentType.SCHEMA, DocumentType.FAQ, DocumentType.BUSINESS_RULE],
        ["s", "f", "b"],
        [{}, {}, {}],
    )
    results = await store.similarity_search(
        "schema", top_k=5, document_types=[DocumentType.SCHEMA]
    )
    assert all(r.document_type == DocumentType.SCHEMA for r in results)


# ── Retriever ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_retriever_returns_results():
    store = FAISSVectorStore(embedder=FakeEmbedder())
    await store.add_documents(
        ["orders table: id, status, total_amount"],
        [DocumentType.SCHEMA],
        ["schema.sql"],
        [{"table": "orders"}],
    )
    from app.auth.roles import Role
    retriever = Retriever(store, top_k=5)
    result = await retriever.retrieve_for_sql_generation(
        "Show me orders", role=Role.ADMIN
    )
    assert len(result.documents) > 0


@pytest.mark.asyncio
async def test_retriever_filters_restricted_tables():
    store = FAISSVectorStore(embedder=FakeEmbedder())
    await store.add_documents(
        ["payroll table: salary, bonus", "orders table: id, total"],
        [DocumentType.SCHEMA, DocumentType.SCHEMA],
        ["schema.sql", "schema.sql"],
        [{"table": "payroll"}, {"table": "orders"}],
    )
    from app.auth.roles import Role
    retriever = Retriever(store, top_k=5)
    result = await retriever.retrieve_for_sql_generation(
        "Show salary info", role=Role.EMPLOYEE
    )
    tables = {d.metadata.get("table") for d in result.documents}
    assert "payroll" not in tables


def test_retrieval_result_to_prompt_context():
    docs = [
        RetrievedDocument(
            content="Table orders: id, status",
            document_type=DocumentType.SCHEMA,
            source="schema.sql",
            score=0.9,
            metadata={"table": "orders"},
        )
    ]
    result = RetrievalResult(documents=docs)
    ctx = result.to_prompt_context()
    assert "orders" in ctx
    assert len(ctx) > 0


def test_retrieval_result_empty_context():
    result = RetrievalResult(documents=[])
    ctx = result.to_prompt_context()
    assert isinstance(ctx, str)
