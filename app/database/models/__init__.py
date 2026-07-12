"""Import every ORM model here so `Base.metadata` is fully populated —
required for Alembic `--autogenerate` to detect all tables."""

from app.database.models.base import Base
from app.database.models.chat import ChatMessage, Conversation
from app.database.models.feedback import Feedback
from app.database.models.token import RefreshToken
from app.database.models.user import User

__all__ = ["Base", "User", "Conversation", "ChatMessage", "Feedback", "RefreshToken"]
