"""FastAPI application factory with Redis caching and health monitoring."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config.settings import get_settings
from app.core.exceptions import AppException
from app.core.logging import configure_logging, get_logger
from app.middleware.rate_limit import RateLimitMiddleware

log = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging()
    log.info(f"Starting {settings.app_name} | env={settings.app_env}")

    # Schema registry
    from pathlib import Path
    from app.sql_validator.schema_registry import load_from_sql_file
    schema_path = Path("data/seed/schema.sql")
    if schema_path.exists():
        load_from_sql_file(schema_path)
    else:
        log.warning("Schema DDL not found — SQL validator will skip table checks")

    # Redis cache
    #from app.utils.cache_client import get_cache_client
    #cache = get_cache_client()
    #await cache.connect()

    yield

    #await cache.disconnect()
    log.info("Shutting down application")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description=(
            "Enterprise AI SQL RAG Assistant — natural language to SQL, "
            "grounded in retrieved schema and business context."
        ),
        version="0.1.0",
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        allow_headers=["*"],
    )
    app.add_middleware(RateLimitMiddleware)

    _register_exception_handlers(app)

    from app.api.v1.router import api_router
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    @app.get("/health", tags=["health"], include_in_schema=False)
    async def root_health() -> dict:
        return {"status": "ok", "app": settings.app_name, "env": settings.app_env}

    return app


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppException)
    async def handle_app_exception(request: Request, exc: AppException) -> JSONResponse:
        log.bind(error_code=exc.error_code, path=str(request.url)).warning(exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.error_code,
                "message": exc.message,
                "details": exc.details,
            },
        )

    

    @app.exception_handler(Exception)
    async def handle_unexpected_exception(request: Request, exc: Exception):
        import traceback

        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "message": str(exc),
                "details": {
                    "type": type(exc).__name__
                }
            },
        )

app = create_app()
