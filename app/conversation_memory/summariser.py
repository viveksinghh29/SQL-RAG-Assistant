"""Context summariser for long conversations
"""

from app.core.exceptions import LLMProviderError
from app.core.logging import get_logger
from app.llm.base import LLMMessage, LLMProvider

log = get_logger("conversation_memory.summariser")

_SUMMARY_SYSTEM_PROMPT = """\
You are a conversation summariser for an AI SQL assistant.
Given a sequence of user questions and assistant SQL answers,
produce a concise summary (3-5 bullet points) capturing:
- What data the user has been exploring
- Key filters, time ranges, or dimensions used
- Any follow-up patterns (comparisons, drill-downs)

Return ONLY the bullet points. No preamble, no explanation.
"""

_FALLBACK_SUMMARY = (
    "[Earlier conversation summarised — user was exploring database queries.]"
)


class ContextSummariser:
    """Summarises older conversation turns using the LLM.

    Args:
        llm: Any LLMProvider implementation. If summarisation fails
             (LLM error, timeout), a static fallback string is used
             rather than propagating the error.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def summarise(self, messages: list[dict]) -> str:
        """Return a compact summary of `messages`.

        Args:
            messages: List of {role, content} dicts to be summarised.
                      These are the *older* turns that will be replaced
                      by this summary in the prompt context.

        Returns:
            A multi-line bullet-point string, or a static fallback if
            the LLM call fails.
        """
        if not messages:
            return ""

        # Format the turns into a readable transcript for the LLM
        transcript_lines = []
        for msg in messages:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")[:500]   # cap very long messages
            transcript_lines.append(f"{role}: {content}")
        transcript = "\n".join(transcript_lines)

        try:
            response = await self._llm.generate(
                [
                    LLMMessage(role="system", content=_SUMMARY_SYSTEM_PROMPT),
                    LLMMessage(
                        role="user",
                        content=f"Summarise this conversation:\n\n{transcript}",
                    ),
                ],
                max_tokens=256,
                temperature=0.0,   # summaries should be deterministic
            )
            summary = response.content.strip()
            log.bind(
                original_messages=len(messages),
                summary_chars=len(summary),
            ).info("Conversation history summarised")
            return f"[Earlier conversation summary]\n{summary}"

        except LLMProviderError as exc:
            log.warning(f"Summarisation LLM call failed: {exc.message}")
            return _FALLBACK_SUMMARY

        except Exception as exc:  # noqa: BLE001
            log.exception(f"Unexpected error during summarisation: {exc}")
            return _FALLBACK_SUMMARY
