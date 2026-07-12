"""Declarative base and shared mixins for app-DB ORM models.

Only the app DB (users, chat history, feedback) gets ORM models. The
target business database is intentionally never modeled with the ORM —
see docs/ARCHITECTURE.md for why. It is queried only through the
validated, generated-SQL pipeline (Phases 6-8) using raw SQL execution.
"""

from datetime import datetime

from sqlalchemy import DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all app-DB ORM models."""


class TimestampMixin:
    """Adds created_at / updated_at columns, managed by the database."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
