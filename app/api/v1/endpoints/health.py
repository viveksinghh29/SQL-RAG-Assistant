"""Health check endpoints.

GET /health          — liveness probe (always fast)
GET /health/ready    — readiness probe (checks DB, vector store, LLM)
"""

from fastapi import APIRouter

from app.config.settings import get_settings
from app.core.logging import get_logger
from app.sql_validator.schema_registry import is_loaded

router = APIRouter(prefix="/health", tags=["health"])
log = get_logger("api.health")


@router.get("")
async def liveness() -> dict:
    """Liveness probe — returns immediately. Used by Docker/k8s to detect crashes."""
    settings = get_settings()
    return {
        "status": "ok",
        "app": settings.app_name,
        "env": settings.app_env,
    }


@router.get("/ready")
async def readiness() -> dict:
    """Readiness probe — checks that the app is ready to serve traffic.

    Checks:
      - Schema registry loaded (RAG index built)
      - App DB reachable (optional — skips gracefully if not yet connected)
      - LLM provider configured

    Returns 200 if ready, 503 if any critical check fails.
    """
    from fastapi import HTTPException
    settings = get_settings()

    checks: dict[str, str] = {}

    # Schema registry
    checks["schema_registry"] = "ok" if is_loaded() else "not_loaded"

    # LLM provider configured
    if settings.llm_provider == "groq":
        checks["llm"] = "configured" if settings.groq_api_key else "missing_api_key"
    else:
        checks["llm"] = "configured" if settings.openai_api_key else "missing_api_key"

    # App DB connectivity (best-effort)
    try:
        from app.database.session import get_engine
        async with get_engine().connect() as conn:
            await conn.execute(__import__("sqlalchemy", fromlist=["text"]).text("SELECT 1"))
        checks["app_db"] = "ok"
    except Exception:
        checks["app_db"] = "unreachable"

    # Redis connectivity
    try:
        from app.utils.cache_client import get_cache_client
        checks["redis"] = "ok" if await get_cache_client().is_available() else "unreachable"
    except Exception:
        checks["redis"] = "unreachable"

    all_ok = all(v in ("ok", "configured", "not_loaded") for v in checks.values())
    status = "ready" if all_ok else "degraded"

    if checks.get("llm") == "missing_api_key":
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "checks": checks},
        )

    return {"status": status, "checks": checks}
