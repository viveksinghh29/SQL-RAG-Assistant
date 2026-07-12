"""Pydantic schemas for the SQL generation pipeline.

These cross module boundaries (generator → validator → executor → API),
so they live in `app/schemas/` rather than inside any one module.
Keeping them here prevents circular imports and makes the data contract
between pipeline stages explicit and type-checked.
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class GenerationStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"   # unsafe SQL or RBAC violation — never executed


class SQLGenerationResult(BaseModel):
    """Output of the SQL generation stage, before validation or execution."""

    status: GenerationStatus
    sql: str = ""
    description: str = ""          # one-sentence summary from the LLM
    retrieved_context: str = ""    # rendered RAG context injected into prompt
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    error_message: str = ""


class ChatRequest(BaseModel):
    """Inbound request to the /chat endpoint (Phase 11)."""

    question: str = Field(min_length=1, max_length=2000)
    conversation_id: int | None = None   # None → start a new conversation


class ChatResponse(BaseModel):
    """Full response returned by the /chat endpoint."""

    conversation_id: int
    message_id: int
    question: str
    sql: str = ""
    description: str = ""
    explanation: str = ""           # business-language insight from LLM
    rows: list[dict] = []
    row_count: int = 0
    columns: list[str] = []
    chart_type: str = "table"
    chart_json: str = ""            # Plotly figure serialized as JSON
    execution_time_ms: float = 0.0
    total_latency_ms: float = 0.0
    status: GenerationStatus = GenerationStatus.SUCCESS
    error_message: str = ""
