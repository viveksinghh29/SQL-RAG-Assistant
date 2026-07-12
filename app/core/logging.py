"""Structured logging configuration using Loguru.

All application code should `from app.core.logging import get_logger` and
use the returned logger — never the stdlib `logging` module directly and
never `print()`. Centralizing this lets us change log format, sinks
(file/stdout/external), and structured fields in one place.
"""

import sys
from pathlib import Path
from typing import Any

from loguru import logger

from app.config.settings import get_settings

_LOG_DIR = Path("logs")


def configure_logging() -> None:
    """Configure Loguru sinks. Call once at application startup."""
    settings = get_settings()
    _LOG_DIR.mkdir(exist_ok=True)

    logger.remove()  # remove default handler so we control format/sinks explicitly

    # Console sink: human-readable, colorized in development
    logger.add(
        sys.stdout,
        level=settings.log_level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[component]}</cyan> | "
            "{message}"
        ),
        backtrace=False,
        diagnose=not settings.is_production,  # never leak variable values in prod logs
    )

    # File sink: JSON structured logs for ingestion by external log tools
    logger.add(
        _LOG_DIR / "app.jsonl",
        level=settings.log_level,
        serialize=True,
        rotation="50 MB",
        retention="14 days",
        compression="zip",
        backtrace=False,
        diagnose=False,
    )

    # Separate sink for errors only, kept longer
    logger.add(
        _LOG_DIR / "errors.jsonl",
        level="ERROR",
        serialize=True,
        rotation="50 MB",
        retention="90 days",
        compression="zip",
    )

    logger.bind(component="startup").info(
        f"Logging configured | env={settings.app_env} level={settings.log_level}"
    )


def get_logger(component: str, **extra_context: Any):
    """Return a logger bound to a named component (e.g. 'sql_generator').

    Usage:
        log = get_logger("sql_validator")
        log.info("Rejected query", reason="DROP statement detected")
    """
    return logger.bind(component=component, **extra_context)
