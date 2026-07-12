# Architecture

## Folder structure and responsibilities

```
app/
  api/v1/endpoints/     REST endpoints only — no business logic. Calls services.
  core/                 Cross-cutting contracts: repository interface, exceptions, logging.
  config/               Pydantic Settings — the only place that reads environment variables.
  database/
    models/             SQLAlchemy ORM models (app DB: users, chat history, feedback).
    repositories/       Concrete repository implementations (AbstractRepository).
  schemas/              Pydantic request/response/DTO schemas — never expose ORM models directly.
  crud/                 Thin CRUD helpers used by repositories where generic CRUD suffices.
  services/             Business logic / orchestration. Endpoints call services, never repositories directly.
  llm/
    base.py             LLMProvider abstract interface.
    providers/           Concrete Groq / OpenAI implementations.
  rag/
    base.py             VectorStore abstract interface, DocumentType taxonomy.
    embeddings/          Sentence-transformer embedding wrapper.
    vector_store/        FAISS implementation.
    retriever/            Retrieval + ranking logic, prompt context assembly.
    ingestion/            Schema/doc loaders and chunking.
  sql_generator/         NL -> SQL, using retrieved context + conversation memory.
  sql_validator/         Syntax/semantic validation, permission checks, dangerous-operation blocking.
  query_executor/        Safe execution against the target DB, timeout/error handling.
  conversation_memory/   Multi-turn context store.
  visualization/         Chart-type inference + Plotly figure generation.
  auth/                  JWT issuance/verification, role definitions, permission checks.
  middleware/            Rate limiting, request logging, auth middleware.
  utils/                 Stateless helpers (no business logic).
  prompts/               Prompt templates, kept as data, not inline strings in code.
tests/
  unit/                  Pure logic, no I/O, mocked dependencies.
  integration/            Real DB/Redis/vector store, run against test instances.
  api/                    FastAPI TestClient end-to-end request tests.
alembic/                  DB schema migrations for the app DB.
data/seed/                Synthetic seed data for the sample retail DB.
data/documents/           Business rules, FAQs, KPI definitions — raw RAG source documents.
```

## Two databases, on purpose

The system talks to **two separate MySQL databases**:

- **App DB** (`APP_DB_*`): owned by this application, writable, stores users, chat history,
  and feedback. Standard SQLAlchemy ORM models + Alembic migrations.
- **Target DB** (`TARGET_DB_*`): the company's actual business data that users ask questions
  about. Accessed with a **read-only** credential, queried only through the validated,
  generated SQL pipeline — never through the ORM, and never with a writable user.

Keeping these separate means a bug in chat-history persistence can never touch business
data, and the blast radius of a SQL-generation mistake is capped at SELECT-only,
read-only-credentialed access.

## Dependency flow

```
API endpoint → Service → Repository / LLMProvider / VectorStore (interfaces)
                              ↓
                    Concrete implementation (injected via FastAPI Depends)
```

Services depend on abstract interfaces (`AbstractRepository`, `LLMProvider`, `VectorStore`),
not concrete classes. Concrete implementations are wired up via FastAPI's `Depends()` in
Phase 11, which means every service can be unit tested with a mock/fake implementation
with zero database, Redis, or network calls.

## Sequencing decision (Phase 1 review)

Per the Phase 1 architecture review: a minimal auth/RBAC stub (role enum + permission
check utility, not the full JWT flow) will be introduced alongside Phase 3 (Database Layer),
and a minimal conversation memory stub before Phase 6 (NL-to-SQL engine), so the SQL
generator and validator are built role-aware and context-aware from the start rather than
retrofitted in Phases 10 and 13. The full implementations still land in their named phases.

## Coding standards

- All I/O-bound code is `async`/`await` (SQLAlchemy async engine, async HTTP clients).
- Every function has type hints; `mypy --strict` is enforced (see `pyproject.toml`).
- Linting via `ruff` (PEP8 + import sorting + bugbear checks).
- No `print()` — use `app.core.logging.get_logger`.
- No raw `os.environ` access outside `app/config/settings.py`.
- Endpoints never catch exceptions silently — raise `AppException` subclasses and let the
  centralized handler in `app/main.py` format the response.
