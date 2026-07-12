"""Cache key construction for every cached artifact.

All keys follow the pattern:  `{namespace}:{version}:{hash}`

  namespace  — identifies the type of cached data
  version    — bumped when the cached data format changes (invalidates all
               old entries without needing to flush Redis manually)
  hash       — SHA-256 of the inputs that determine cache identity

Keeping key construction here means:
  - Key format is auditable in one place
  - Version bumps are one-line changes
  - Collision between namespaces is impossible
"""

import hashlib
import json


# Bump these when the cached data format changes
_VERSIONS = {
    "query":      "v1",   # QueryExecutor results
    "explain":    "v1",   # LLM explanation text
    "chart":      "v1",   # ChartDecision JSON
    "embedding":  "v1",   # sentence-transformer vectors
    "schema":     "v1",   # schema registry contents
}

_SEP = ":"


def _sha256(*parts: str) -> str:
    """SHA-256 of joined parts, hex-encoded (first 16 chars = 64 bits, enough for cache keys)."""
    combined = _SEP.join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()[:32]


def query_result_key(sql: str, role: str) -> str:
    """Cache key for a validated SQL query result.

    Keyed on both SQL and role because the same SQL executed by different
    roles could theoretically see different data if row-level security
    were enabled. Currently our security is query-level (blocked before
    execution), but keying on role is future-proof and free.
    """
    return f"query{_SEP}{_VERSIONS['query']}{_SEP}{_sha256(sql.strip().lower(), role)}"


def explanation_key(sql: str, row_count: int) -> str:
    """Cache key for an LLM explanation.

    Explanations depend on the SQL (determines the question intent) and
    the row count (the LLM references specific numbers). Row values
    themselves are NOT included — the explanation is already cached with
    the same SQL+count that produced it, so the values are implicitly
    fixed.
    """
    return (
        f"explain{_SEP}{_VERSIONS['explain']}{_SEP}"
        f"{_sha256(sql.strip().lower(), str(row_count))}"
    )


def chart_decision_key(sql: str, columns: list[str]) -> str:
    """Cache key for a chart type inference decision."""
    cols_str = json.dumps(sorted(columns))
    return (
        f"chart{_SEP}{_VERSIONS['chart']}{_SEP}"
        f"{_sha256(sql.strip().lower(), cols_str)}"
    )


def embedding_key(text: str) -> str:
    """Cache key for a sentence embedding vector."""
    return f"embedding{_SEP}{_VERSIONS['embedding']}{_SEP}{_sha256(text)}"


def schema_key(db_name: str) -> str:
    """Cache key for the schema registry for a given database."""
    return f"schema{_SEP}{_VERSIONS['schema']}{_SEP}{_sha256(db_name)}"
