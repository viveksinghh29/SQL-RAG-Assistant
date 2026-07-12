"""ConversationManager — conversation lifecycle management
"""

from app.core.logging import get_logger
from app.database.models.chat import ChatMessage, Conversation
from app.database.repositories.chat_repository import (
    ChatMessageRepository,
    ConversationRepository,
)
from app.llm.base import LLMMessage, LLMProvider
from app.schemas.chat import ConversationCreate, ConversationRead, MessageRole
from app.schemas.pipeline import ChatResponse

log = get_logger("conversation_memory.manager")

_TITLE_SYSTEM_PROMPT = """\
Generate a short, descriptive title (4-7 words) for a database query
conversation based on the user's first message. Return ONLY the title —
no quotes, no punctuation at the end, no explanation.

Examples:
  User: Show me total sales by region last month
  Title: Monthly Sales by Region

  User: Which products have the highest return rate?
  Title: Products with Highest Return Rate
"""


class ConversationManager:
    """Manages the write side of conversation persistence.

    Args:
        conv_repo:    ConversationRepository for creating/updating conversations.
        msg_repo:     ChatMessageRepository for appending messages.
        llm:          Optional LLM provider for title generation. If None,
                      the first 60 chars of the user's question are used.
    """

    def __init__(
        self,
        conv_repo: ConversationRepository,
        msg_repo: ChatMessageRepository,
        llm: LLMProvider | None = None,
    ) -> None:
        self._conv_repo = conv_repo
        self._msg_repo = msg_repo
        self._llm = llm

    async def get_or_create_conversation(
        self,
        user_id: int,
        conversation_id: int | None,
        first_question: str,
    ) -> Conversation:
        """Return an existing conversation or create a new one.

        For a new conversation (conversation_id is None), generates a
        descriptive title from the first user question.
        """
        if conversation_id is not None:
            conv = await self._conv_repo.get_by_id(conversation_id)
            if conv is not None:
                return conv
            log.warning(
                f"conversation_id={conversation_id} not found — creating new"
            )

        title = await self._generate_title(first_question)
        conv = await self._conv_repo.create(
            ConversationCreate(title=title),
            user_id=user_id,
        )
        log.bind(
            conversation_id=conv.id,
            user_id=user_id,
            title=title,
        ).info("New conversation created")
        return conv

    async def persist_turn(
        self,
        conversation_id: int,
        user_question: str,
        response: ChatResponse,
    ) -> tuple[ChatMessage, ChatMessage]:
        """Persist a user message and its assistant response to the DB.

        Returns:
            Tuple of (user_message, assistant_message) ORM objects.
        """
        user_msg = await self._msg_repo.add_message(
            ChatMessage(
                conversation_id=conversation_id,
                role=MessageRole.USER,
                content=user_question,
            )
        )

        # Build the assistant message content from the response
        assistant_content = self._format_assistant_content(response)

        assistant_msg = await self._msg_repo.add_message(
            ChatMessage(
                conversation_id=conversation_id,
                role=MessageRole.ASSISTANT,
                content=assistant_content,
                generated_sql=response.sql or None,
                execution_time_ms=int(response.execution_time_ms)
                if response.execution_time_ms
                else None,
            )
        )

        log.bind(
            conversation_id=conversation_id,
            user_msg_id=user_msg.id,
            assistant_msg_id=assistant_msg.id,
        ).debug("Turn persisted to database")

        return user_msg, assistant_msg

    async def list_conversations(
        self,
        user_id: int,
        *,
        skip: int = 0,
        limit: int = 50,
    ) -> list[ConversationRead]:
        """List a user's conversations ordered by most recent first."""
        conversations = await self._conv_repo.list_for_user(
            user_id, skip=skip, limit=limit
        )
        return [ConversationRead.model_validate(c) for c in conversations]

    async def get_conversation_with_messages(
        self, conversation_id: int
    ) -> Conversation | None:
        """Load a full conversation including all messages."""
        return await self._conv_repo.get_with_messages(conversation_id)

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _generate_title(self, question: str) -> str:
        """Generate a short descriptive title using the LLM, or fall back
        to the first 60 chars of the question if the LLM is unavailable."""
        if self._llm is None:
            return self._truncate_title(question)

        try:
            response = await self._llm.generate(
                [
                    LLMMessage(role="system", content=_TITLE_SYSTEM_PROMPT),
                    LLMMessage(role="user", content=question),
                ],
                max_tokens=32,
                temperature=0.3,
            )
            title = response.content.strip().strip('"').strip("'")
            if title:
                return title[:120]   # guard against unexpectedly long output
        except Exception as exc:  # noqa: BLE001
            log.warning(f"Title generation failed, using truncated question: {exc}")

        return self._truncate_title(question)

    @staticmethod
    def _truncate_title(question: str, max_len: int = 60) -> str:
        q = question.strip()
        return q if len(q) <= max_len else q[:max_len].rsplit(" ", 1)[0] + "…"

    @staticmethod
    def _format_assistant_content(response: ChatResponse) -> str:
        """Build the assistant message content that gets stored in the DB.

        The stored content is the explanation (natural language insight),
        not the raw SQL — the SQL is stored separately in `generated_sql`.
        This keeps the history readable in the UI and usable as LLM context.
        """
        parts = []
        if response.description:
            parts.append(response.description)
        if response.explanation:
            parts.append(response.explanation)
        if not parts:
            parts.append("Query executed successfully.")
        return "\n\n".join(parts)
