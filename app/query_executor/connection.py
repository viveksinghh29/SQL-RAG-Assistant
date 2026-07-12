"""Target database connection factory."""

from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.config.settings import get_settings
from app.core.logging import get_logger
from app.utils.pool_config import TARGET_DB_POOL

log = get_logger("query_executor.connection")

_override_engine: AsyncEngine | None = None


def _set_target_engine(engine: AsyncEngine | None) -> None:
    global _override_engine
    _override_engine = engine


@lru_cache
def _build_engine() -> AsyncEngine:
    settings = get_settings()
    engine = create_async_engine(
        settings.target_database_url,
        **TARGET_DB_POOL,
    )
    log.info(
        f"Target DB engine created: "
        f"{settings.target_db_host}:{settings.target_db_port}"
        f"/{settings.target_db_name}"
    )
    return engine


def get_target_engine() -> AsyncEngine:
    if _override_engine is not None:
        return _override_engine
    return _build_engine()
