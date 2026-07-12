"""Schema registry for caching database tables and columns from DDL to enable offline SQL validation without a live database connection."""

from pathlib import Path

from app.core.logging import get_logger
from app.rag.ingestion.schema_loader import parse_ddl

log = get_logger("sql_validator.registry")

# Process-level schema cache: {table_name_lower: {column_name_lower, ...}}
_SCHEMA: dict[str, set[str]] = {}


def load_from_ddl(ddl_sql: str) -> None:
    """Parse `ddl_sql` and populate the in-memory schema registry.

    Idempotent — calling multiple times replaces the registry.
    """
    global _SCHEMA
    _SCHEMA = {}
    tables = parse_ddl(ddl_sql)
    for table in tables:
        _SCHEMA[table.name.lower()] = {col.name.lower() if hasattr(col, "name") else str(col).lower() for col in table.columns}
    log.info(f"Schema registry loaded: {len(_SCHEMA)} table(s) — {sorted(_SCHEMA.keys())}")


def load_from_sql_file(path: Path | str) -> None:
    """Convenience wrapper: read DDL from a file and call `load_from_ddl`."""
    ddl = Path(path).read_text(encoding="utf-8")
    load_from_ddl(ddl)


def get_known_tables() -> dict[str, set[str]]:
    """Return a copy of the current registry.

    Returns an empty dict if the registry has not been loaded — the
    validator will skip schema-existence checks in that case.
    """
    return dict(_SCHEMA)


def is_loaded() -> bool:
    return bool(_SCHEMA)
