"""Conversation memory module for retrieving and summarising prior chat turns to provide context for LLM prompts.
"""

from app.conversation_memory.summariser import ContextSummariser
from app.conversation_memory.token_budget import (
    HISTORY_TOKEN_BUDGET,
    estimate_messages_tokens,
    trim_to_budget,
)
from app.core.logging import get_logger
from app.database.repositories.chat_repository import ConversationRepository
from app.llm.base import LLMProvider
from app.schemas.chat import MessageRole

log = get_logger("conversation_memory")


class ConversationMemory:
    """Retrieves and formats conversation history for LLM prompts.

    Behaviour:
      - If the full non-system history fits within HISTORY_TOKEN_BUDGET,
        return it all (no truncation, no summarisation).
      - If it exceeds the budget, summarise the oldest half and return
        [summary_message] + [recent messages that fit the budget].
      - If conversation_id is None (new chat), return [].

    This class is the single place that decides what prior context the
    LLM sees — keeping it isolated makes it straightforward to swap the
    summarisation strategy (e.g. switch from LLM summary to extractive
    summary) without touching the SQL generator.
    """

    def __init__(
        self,
        conversation_repo: ConversationRepository,
        llm: LLMProvider | None = None,
    ) -> None:
        self._repo = conversation_repo
        self._summariser = ContextSummariser(llm) if llm else None

    async def get_history_for_prompt(
        self,
        conversation_id: int | None,
    ) -> list[dict[str, str]]:
        """Return prior conversation turns formatted for the LLM message list.

        Args:
            conversation_id: ID of the active conversation, or None for
                             a brand-new chat.

        Returns:
            List of {"role": ..., "content": ...} dicts ready to pass
            directly into `build_sql_generation_messages()`.
        """
        if conversation_id is None:
            return []

        conversation = await self._repo.get_with_messages(conversation_id)
        if conversation is None or not conversation.messages:
            return []

        # Convert ORM messages to plain dicts, excluding system messages
        all_messages = [
            {"role": m.role.value, "content": m.content}
            for m in conversation.messages
            if m.role != MessageRole.SYSTEM
        ]

        if not all_messages:
            return []

        estimate = estimate_messages_tokens(all_messages)

        # Fast path: everything fits — no summarisation needed
        if estimate.within_budget:
            log.bind(
                conversation_id=conversation_id,
                messages=len(all_messages),
                tokens=estimate.tokens,
            ).debug("Full history fits within token budget")
            return all_messages

        # Slow path: history exceeds budget — summarise the oldest half,
        # keep the most recent messages that fit the remaining budget.
        midpoint = len(all_messages) // 2
        older_messages = all_messages[:midpoint]
        recent_messages = all_messages[midpoint:]

        # Trim recent messages to the full budget
        recent_trimmed = trim_to_budget(recent_messages, budget=HISTORY_TOKEN_BUDGET)

        log.bind(
            conversation_id=conversation_id,
            total_messages=len(all_messages),
            older_to_summarise=len(older_messages),
            recent_kept=len(recent_trimmed),
        ).info("History exceeds token budget — summarising older turns")

        # Summarise older turns if we have an LLM; otherwise drop them
        if self._summariser is not None and older_messages:
            summary_text = await self._summariser.summarise(older_messages)
            summary_message = {"role": "system", "content": summary_text}
            return [summary_message] + recent_trimmed

        return recent_trimmed
